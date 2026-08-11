"""Indicator, backtest-metric, trade-plan and classification tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.classify import sector_from_sic
from quant import backtest as BT
from quant import factors as FA
from quant import fundamentals as F
from quant import tradeplan as TP


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────

def test_wilder_rsi_all_gains_is_100():
    prices = pd.Series(np.arange(1, 40, dtype=float))
    assert FA.wilder_rsi(prices) == pytest.approx(100.0)


def test_wilder_rsi_bounded_and_midrange_on_noise():
    rng = np.random.RandomState(7)
    prices = pd.Series(100 + np.cumsum(rng.randn(200)))
    rsi = FA.wilder_rsi(prices)
    assert 0 <= rsi <= 100
    assert 20 < rsi < 80


def test_wilder_rsi_differs_from_simple_moving_average_version():
    """
    The original claimed Wilder's RSI in its docstring but used a simple
    rolling mean, which produces a different number.
    """
    rng = np.random.RandomState(3)
    prices = pd.Series(100 + np.cumsum(rng.randn(120)))

    wilder = FA.wilder_rsi(prices, period=14)

    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    simple = 100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1])

    assert abs(wilder - simple) > 0.5, 'the two smoothings must not coincide'


def test_atr_is_positive_and_scales_with_range():
    n = 60
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    close = pd.Series(np.full(n, 100.0), index=idx)
    tight = TP  # noqa: F841  (keeps import used if ATR moves)

    narrow = FA.atr(close + 0.5, close - 0.5, close)
    wide = FA.atr(close + 5.0, close - 5.0, close)
    assert narrow > 0 and wide > narrow * 5


# ─────────────────────────────────────────────
# BACKTEST METRICS
# ─────────────────────────────────────────────

def test_metrics_on_analytically_known_curve():
    """Exactly 1% per month for five years."""
    idx = pd.date_range('2020-01-31', periods=61, freq='ME')
    equity = pd.Series(1.01 ** np.arange(61), index=idx)

    m = BT.compute_metrics(equity)

    assert m['total_return'] == pytest.approx(1.01 ** 60 - 1, rel=1e-9)
    assert m['cagr'] == pytest.approx(1.01 ** 12 - 1, abs=1e-3)
    assert m['max_drawdown'] == pytest.approx(0.0)
    assert m['hit_rate'] == pytest.approx(1.0)


def test_max_drawdown_matches_hand_computation():
    idx = pd.date_range('2020-01-31', periods=5, freq='ME')
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 1.5], index=idx)
    m = BT.compute_metrics(equity)
    assert m['max_drawdown'] == pytest.approx(-0.25)


def test_metrics_empty_input_is_safe():
    assert BT.compute_metrics(pd.Series(dtype=float)) == {}


# ─────────────────────────────────────────────
# TRADE PLANS
# ─────────────────────────────────────────────

def test_plan_levels_are_ordered_and_sized_by_risk():
    plan = TP.build_plan('TEST', date(2026, 1, 5), price=100.0, atr_value=2.0,
                         ma50=95.0, ma200=90.0, low_series=None,
                         setup='breakout')
    assert plan is not None
    assert plan.stop < plan.entry < plan.target

    risk = plan.entry - plan.stop
    reward = plan.target - plan.entry
    assert reward / risk == pytest.approx(plan.r_multiple, rel=1e-6)


def test_position_weight_is_inverse_to_stop_distance():
    """
    Fixed-fractional sizing: weight = risk budget / stop distance, so a wider
    stop earns a smaller position.
    """
    params = TP.PlanParams(risk_budget_pct=1.0, max_weight_pct=25.0,
                           atr_stop_mult=2.0, entry_pullback_atr=0.0)

    # 2 x ATR of 4.0 on a $100 price => an 8% stop, comfortably under the cap.
    moderate = TP.build_plan('A', date(2026, 1, 5), 100.0, 4.0, None, None,
                             None, 'breakout', params)
    wide = TP.build_plan('B', date(2026, 1, 5), 100.0, 6.0, None, None,
                         None, 'breakout', params)

    assert moderate.weight_pct > wide.weight_pct
    assert moderate.weight_pct == pytest.approx(
        params.risk_budget_pct / (moderate.stop_pct / 100.0), rel=1e-6)


def test_position_weight_is_capped():
    """A very tight stop would imply an oversized position; the cap binds."""
    params = TP.PlanParams(risk_budget_pct=1.0, max_weight_pct=25.0,
                           atr_stop_mult=2.0, entry_pullback_atr=0.0)
    tight = TP.build_plan('A', date(2026, 1, 5), 100.0, 1.0, None, None,
                          None, 'breakout', params)
    uncapped = params.risk_budget_pct / (tight.stop_pct / 100.0)
    assert uncapped > params.max_weight_pct, 'precondition: cap should bind'
    assert tight.weight_pct == pytest.approx(params.max_weight_pct)


def test_plan_rejected_when_stop_would_be_too_wide():
    params = TP.PlanParams(max_stop_pct=10.0, atr_stop_mult=2.0)
    plan = TP.build_plan('VOLATILE', date(2026, 1, 5), price=100.0,
                         atr_value=20.0, ma50=None, ma200=None,
                         low_series=None, params=params)
    assert plan is None, 'a stop beyond the risk limit must not produce a plan'


def test_plan_uses_the_wider_of_atr_and_swing_low():
    """
    The tighter level looks better on paper but gets stopped by noise, so the
    wider of the two is used.
    """
    idx = pd.date_range('2026-01-01', periods=25, freq='B')
    lows = pd.Series(np.full(25, 90.0), index=idx)      # swing low well below

    params = TP.PlanParams(atr_stop_mult=2.0, entry_pullback_atr=0.0)
    plan = TP.build_plan('T', date(2026, 2, 5), price=100.0, atr_value=1.0,
                         ma50=None, ma200=None, low_series=lows,
                         params=params)
    assert plan.stop_basis == 'swing_low'
    assert plan.stop < 98.0


def test_portfolio_risk_aggregates():
    plans = pd.DataFrame({'weight_pct': [10.0, 10.0], 'stop_pct': [5.0, 5.0]},
                         index=['A', 'B'])
    risk = TP.portfolio_risk(plans)
    assert risk['n_positions'] == 2
    assert risk['total_weight_pct'] == pytest.approx(20.0)
    assert risk['portfolio_risk_pct'] == pytest.approx(1.0)
    assert risk['cash_pct'] == pytest.approx(80.0)


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

@pytest.mark.parametrize('sic,expected', [
    (3571, 'Information Technology'),   # Apple — electronic computers
    (7372, 'Information Technology'),   # prepackaged software
    (2834, 'Health Care'),              # pharmaceutical preparations
    (6021, 'Financials'),               # national commercial banks
    (2911, 'Energy'),                   # petroleum refining
    (4911, 'Utilities'),                # electric services
    (6798, 'Real Estate'),              # REIT
    (2840, 'Consumer Staples'),         # P&G — soaps and detergents
    (3021, 'Consumer Discretionary'),   # Nike — rubber footwear
    (2821, 'Materials'),                # plastics
    (3721, 'Industrials'),              # aircraft
    (4813, 'Communication Services'),   # telephone communications
])
def test_sic_maps_to_expected_sector(sic, expected):
    assert sector_from_sic(sic) == expected


def test_sic_unknown_is_handled():
    assert sector_from_sic(None) == 'Unknown'
    assert sector_from_sic('') == 'Unknown'
    assert sector_from_sic('not-a-code') == 'Unknown'


def test_narrow_ranges_take_priority_over_broad_ones():
    """
    Regression guard: 2840 sits inside the broad chemicals band and 3021
    inside the broad rubber band. Ordering the list wrong put Procter & Gamble
    and Nike in Materials.
    """
    assert sector_from_sic(2840) != 'Materials'
    assert sector_from_sic(3021) != 'Materials'


# ─────────────────────────────────────────────
# SHARE COUNT / TAG DRIFT
# ─────────────────────────────────────────────

def _share_facts(rows):
    """Minimal fact frame for the share-count helpers."""
    import pandas as pd
    df = pd.DataFrame(rows)
    for c in ('period_start', 'period_end', 'filed'):
        df[c] = pd.to_datetime(df[c])
    return df


def test_share_count_prefers_diluted_when_both_current():
    facts = _share_facts([
        {'concept': 'shares_diluted', 'period_start': '2026-01-01',
         'period_end': '2026-03-31', 'filed': '2026-04-30', 'val': 1000.0},
        {'concept': 'shares_basic', 'period_start': '2026-01-01',
         'period_end': '2026-03-31', 'filed': '2026-04-30', 'val': 990.0},
    ])
    assert F.share_count(facts) == 1000.0


def test_share_count_falls_back_when_the_diluted_tag_goes_stale():
    """Exxon stopped filing the diluted tag in 2014.

    Taking diluted unconditionally pinned its share count — and so its market
    cap and every yield factor — to a twelve-year-old figure.
    """
    facts = _share_facts([
        {'concept': 'shares_diluted', 'period_start': '2013-01-01',
         'period_end': '2013-12-31', 'filed': '2014-02-26', 'val': 4419.0},
        {'concept': 'shares_basic', 'period_start': '2026-04-01',
         'period_end': '2026-06-30', 'filed': '2026-08-03', 'val': 4174.0},
    ])
    assert F.share_count(facts) == 4174.0


def test_share_count_handles_a_missing_tag_either_way():
    only_basic = _share_facts([
        {'concept': 'shares_basic', 'period_start': '2026-01-01',
         'period_end': '2026-03-31', 'filed': '2026-04-30', 'val': 500.0}])
    assert F.share_count(only_basic) == 500.0

    only_diluted = _share_facts([
        {'concept': 'shares_diluted', 'period_start': '2026-01-01',
         'period_end': '2026-03-31', 'filed': '2026-04-30', 'val': 500.0}])
    assert F.share_count(only_diluted) == 500.0


def test_share_count_is_nan_when_neither_tag_exists():
    import numpy as np
    facts = _share_facts([
        {'concept': 'revenue', 'period_start': '2026-01-01',
         'period_end': '2026-03-31', 'filed': '2026-04-30', 'val': 1.0}])
    assert np.isnan(F.share_count(facts))


# ─────────────────────────────────────────────
# STRATEGY DOCUMENTATION
# ─────────────────────────────────────────────

def test_every_strategy_has_notes():
    """A strategy nobody can distinguish from its neighbour is not usable."""
    from quant import strategy_notes as SN
    c = SN.coverage()
    assert c['missing'] == [], f'strategies without notes: {c["missing"]}'
    assert c['orphan'] == [], f'notes for strategies that do not exist: {c["orphan"]}'


def test_notes_are_substantive():
    """Guards against placeholder text creeping in.

    The regime fields are naturally terse ("Steady trending markets"), so they
    carry a lower bar than the explanatory ones.
    """
    from quant import strategy_notes as SN
    minimums = {'differentiator': 60, 'benefit': 60, 'drawback': 60,
                'risk': 60, 'best_when': 15, 'worst_when': 15}
    for key, n in SN.NOTES.items():
        for field_name, floor in minimums.items():
            text = getattr(n, field_name)
            assert len(text) >= floor, (
                f'{key}.{field_name} is too short ({len(text)} < {floor}): {text!r}')
            assert 'TODO' not in text.upper(), f'{key}.{field_name} has a TODO'


# ─────────────────────────────────────────────
# PURE FUNCTIONS  (no database)
# ─────────────────────────────────────────────

def _views(spec, days=90):
    """Tidy pageview frame from {ticker: (baseline, recent)}."""
    rows = []
    for t, (base, recent) in spec.items():
        for i in range(days):
            rows.append({'ticker': t,
                         'date': pd.Timestamp('2026-01-01') + pd.Timedelta(days=i),
                         'wiki_views': (recent if i >= days - 14 else base) + (i % 5)})
    return pd.DataFrame(rows)


def test_attention_zscore_flags_a_spike():
    from quant.factors import attention_zscore
    z = attention_zscore(_views({'AAPL': (100, 400), 'MSFT': (100, 100)}))
    assert z['AAPL'] > 5
    assert abs(z['MSFT']) < 2


def test_attention_zscore_needs_enough_history():
    """A ticker with a few days of data cannot have a meaningful baseline."""
    from quant.factors import attention_zscore
    z = attention_zscore(_views({'AAPL': (100, 400)}, days=90).pipe(
        lambda d: pd.concat([d, _views({'NEW': (50, 50)}, days=10)])))
    assert 'NEW' not in (z.index if z is not None else [])


def test_attention_zscore_handles_empty_and_flat():
    from quant.factors import attention_zscore
    assert attention_zscore(pd.DataFrame()) is None
    assert attention_zscore(None) is None
    # Zero variance must not divide by zero.
    flat = pd.DataFrame({'ticker': ['A'] * 90,
                         'date': pd.date_range('2026-01-01', periods=90),
                         'wiki_views': [10.0] * 90})
    assert attention_zscore(flat) is None


def test_last_close_is_pure():
    """Takes a frame, returns a series — no I/O, forward-filled."""
    from data.yahoo import last_close
    px = pd.DataFrame({'AAPL': [1.0, 2.0, None], 'MSFT': [5.0, None, None]},
                      index=pd.date_range('2026-01-01', periods=3))
    out = last_close(px)
    assert out['AAPL'] == 2.0 and out['MSFT'] == 5.0
    assert last_close(pd.DataFrame()).empty


# ─────────────────────────────────────────────
# 8-K INTERPRETATION
# ─────────────────────────────────────────────

def test_every_configured_item_code_has_a_meaning():
    """A code we classify but cannot explain is a table cell with no answer."""
    import config
    from nlp import eightk as EK
    missing = [c for c in config.EIGHTK_ITEMS if EK.get(c) is None]
    assert missing == [], f'item codes without an interpretation: {missing}'


def test_restatement_is_flagged_as_material():
    """4.02 invalidates the fundamentals every factor is built from."""
    from nlp import eightk as EK
    assert EK.significance('4.02') == EK.MATERIAL
    out = EK.summarize(pd.DataFrame({'item_code': ['4.02']}))
    assert any('unreliable' in o or 'disowned' in o for o in out['observations'])


def test_repeated_management_turnover_is_called_out():
    from nlp import eightk as EK
    out = EK.summarize(pd.DataFrame({'item_code': ['5.02'] * 3}))
    assert any('turnover' in o for o in out['observations'])


def test_routine_filings_produce_a_routine_reading():
    from nlp import eightk as EK
    out = EK.summarize(pd.DataFrame({'item_code': ['5.07', '8.01', '7.01']}))
    assert out['material'] == 0
    assert any('Nothing structurally unusual' in o for o in out['observations'])


def test_summarize_handles_empty():
    from nlp import eightk as EK
    out = EK.summarize(pd.DataFrame())
    assert out['total'] == 0 and out['observations'] == []


# ─────────────────────────────────────────────
# DATA VALIDITY
# ─────────────────────────────────────────────

def _facts(rows):
    """Fact frame with the columns quarterly_flows expects."""
    df = pd.DataFrame(rows)
    for c in ('period_start', 'period_end', 'filed'):
        df[c] = pd.to_datetime(df[c])
    for c in ('fy', 'fp'):
        if c not in df.columns:
            df[c] = None
    return df


def test_equity_ratios_are_dropped_when_equity_is_negligible():
    """Buybacks can erase book equity without the business being distressed.

    GoDaddy carried $0.01B of equity on $0.91B of earnings — an ROE of
    13,587%. Arithmetically right, and meaningless.
    """
    assert np.isnan(F._meaningful_equity(-1.4e8, 4.0e9))     # negative
    assert np.isnan(F._meaningful_equity(1e7, 4.0e9))        # 0.25% of assets
    assert F._meaningful_equity(4.4e11, 5.6e11) == 4.4e11    # healthy


def test_invested_capital_is_nan_without_usable_equity():
    """Otherwise ROIC divides by debt alone; Marriott reached 185."""
    assert np.isnan(F._invested_capital(np.nan, 5e9))
    assert F._invested_capital(1e10, 5e9) == 1.5e10


def test_one_tag_is_chosen_per_concept_not_pooled():
    """A filer reporting a total and a component tags both.

    Camden Property files ~$5M of contract revenue alongside ~$390M of lease
    income; pooling them produced a net margin of 2,565%.
    """
    facts = _facts([
        {'concept': 'revenue', 'tag': 'OperatingLeaseLeaseIncome',
         'period_start': '2026-01-01', 'period_end': '2026-03-31',
         'filed': '2026-05-01', 'val': 390e6},
        {'concept': 'revenue', 'tag': 'RevenueFromContractWithCustomerExcludingAssessedTax',
         'period_start': '2026-01-01', 'period_end': '2026-03-31',
         'filed': '2026-05-01', 'val': 5e6},
    ])
    memo = {}
    out = F._concept_frame(facts, 'revenue', memo)
    assert out['tag'].nunique() == 1
    assert out.iloc[0]['val'] == 390e6, 'the larger current line must win'


def test_retired_tags_lose_to_current_ones():
    """Microsoft's obsolete `Revenues` history outnumbered its current line."""
    facts = _facts([
        {'concept': 'revenue', 'tag': 'Revenues',
         'period_start': '2015-01-01', 'period_end': '2015-03-31',
         'filed': '2015-05-01', 'val': 900e9},
        {'concept': 'revenue', 'tag': 'RevenueFromContractWithCustomerExcludingAssessedTax',
         'period_start': '2026-01-01', 'period_end': '2026-03-31',
         'filed': '2026-05-01', 'val': 70e9},
    ])
    out = F._concept_frame(facts, 'revenue', {})
    assert out.iloc[0]['tag'].startswith('RevenueFromContract')


def test_share_count_rejected_when_eps_contradicts_it():
    """Berkshire's cover page reports Class A only — a thousandth of the base.

    Multiplied by a Class B price that gave a market cap under $1B against a
    real ~$1.1T, and an earnings yield of 9,894%. An independent count derived
    from net income / diluted EPS disagrees by three orders of magnitude, so
    the reported figure is refused.
    """
    from datetime import date as _d

    rows = [{'concept': 'shares_outstanding', 'tag': 'dei',
             'period_start': '2026-06-30', 'period_end': '2026-06-30',
             'filed': '2026-07-30', 'val': 1.6e6}]
    # Four quarters, so TTM resolves rather than refusing to extrapolate.
    for i, (s_, e_, f_) in enumerate([
            ('2025-07-01', '2025-09-30', '2025-10-30'),
            ('2025-10-01', '2025-12-31', '2026-01-30'),
            ('2026-01-01', '2026-03-31', '2026-04-30'),
            ('2026-04-01', '2026-06-30', '2026-07-30')]):
        rows.append({'concept': 'net_income', 'tag': 'NetIncomeLoss',
                     'period_start': s_, 'period_end': e_, 'filed': f_,
                     'val': 5e9})
        rows.append({'concept': 'eps_diluted', 'tag': 'EarningsPerShareDiluted',
                     'period_start': s_, 'period_end': e_, 'filed': f_,
                     'val': 3.25})

    facts = _facts(rows)
    # 20e9 / 13.0 => ~1.54e9 implied shares vs 1.6e6 reported: a 1000x gap.
    assert np.isfinite(F.implied_share_count(facts, {}))
    assert np.isnan(F.share_count(facts, {}, as_of=_d(2026, 8, 1)))

    # The same count passes when the two agree.
    ok = _facts([r for r in rows if r['concept'] != 'shares_outstanding'] +
                [{'concept': 'shares_outstanding', 'tag': 'dei',
                  'period_start': '2026-06-30', 'period_end': '2026-06-30',
                  'filed': '2026-07-30', 'val': 1.54e9}])
    assert np.isfinite(F.share_count(ok, {}, as_of=_d(2026, 8, 1)))


def test_validate_flags_impossible_ratios():
    from quant import validate as V
    factors = pd.DataFrame({'EARNINGS_YIELD': [0.05, 899.0]},
                           index=['AAPL', 'ERIE'])
    found = V.check_factor_bounds(factors)
    assert found and found[0].count == 1
    assert 'ERIE' in [t for t, _v in found[0].examples]
