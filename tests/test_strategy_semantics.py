"""Strategy semantics: signal tiers, avoid lists, and the drift factor."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from core import market_calendar as MC
from quant import backtest as BT
from quant import engine as EN
from quant import factors as FA
from quant import strategies as ST


def _scores(scale: float, n: int = 40) -> pd.DataFrame:
    idx = [f'T{i}' for i in range(n)]
    return pd.DataFrame({'composite': np.linspace(1, -1, n) * scale,
                         'sector': 'X'}, index=idx)


# ── signal tiers ─────────────────────────────────────────────────

@pytest.mark.parametrize('horizon', ['long', 'mid'])
def test_signal_tiers_do_not_depend_on_composite_scale(horizon):
    """A many-factor strategy's composite is compressed; its top decile is
    still its top decile."""
    narrow = EN.generate_signals(_scores(0.3), pd.DataFrame(), horizon)
    wide = EN.generate_signals(_scores(3.0), pd.DataFrame(), horizon)
    assert narrow['signal'].tolist() == wide['signal'].tolist()
    assert 'Top-decile' in narrow['reasons'].iloc[0]


def test_buys_are_a_bounded_share_of_any_universe():
    sig = EN.generate_signals(_scores(0.2, 100), pd.DataFrame(), 'long')
    buys = sig['signal'].str.contains('Buy').mean()
    assert 0.05 <= buys <= 0.30


# ── avoid lists ──────────────────────────────────────────────────

def test_avoid_list_never_says_buy_or_enter():
    sig = EN.generate_signals(_scores(1.0), pd.DataFrame(), 'mid', invert=True)
    assert sig['signal'].iloc[0].startswith('🔴 Avoid')
    assert not sig['signal'].str.contains('Enter|Buy').any()


def test_avoid_list_reasons_name_the_problems():
    raw = pd.DataFrame({'PIOTROSKI_F': [2.0], 'PCT_VS_MA200': [-0.18],
                        'INSIDER_NET_BUY': [-0.8], 'EVENT_MANAGEMENT': [1.0]},
                       index=['T0'])
    sig = EN.generate_signals(_scores(1.0, 1), raw, 'mid', invert=True)
    why = sig['reasons'].iloc[0]
    assert 'Piotroski 2/9' in why and 'below 200-day' in why
    assert 'Insiders net sellers' in why and 'Management change' in why


def test_avoid_lists_are_not_backtested_as_longs():
    assert ST.get('quality_breakdown').invert
    with pytest.raises(ValueError, match='avoid list'):
        BT.run_trade_backtest(None, 'quality_breakdown', '2025-01-01', '2025-06-30')


def test_avoid_list_is_kept_out_of_the_consensus(fresh_db, monkeypatch, tmp_path):
    import config
    from data.universe import UniverseSpec
    from reports import builder as RB
    monkeypatch.setattr(config, 'REPORT_DIR', tmp_path)
    monkeypatch.setattr(RB.yahoo, 'update_prices', lambda *a, **k: 0)
    monkeypatch.setattr(RB, '_auto_run_prior', lambda *a, **k: (None, None))
    keys = ['buffett', 'momentum_trend', 'quality_breakdown']
    monkeypatch.setattr(RB, '_all_strategy_keys', lambda: keys)
    spec = UniverseSpec(preset='SPY')
    d = date(2026, 11, 6)
    for k in keys:
        order = ['BAD', 'AAA', 'BBB'] if k == 'quality_breakdown' else ['AAA', 'BBB', 'BAD']
        sc = pd.DataFrame({'composite': [1.0, 0.5, 0.0], 'rank': [1, 2, 3],
                           'signal': ['🔴 Avoid — deteriorating' if k == 'quality_breakdown'
                                      else '🟢 Buy'] * 3,
                           'signal_score': 1.0, 'reasons': 'Piotroski 2/9',
                           'coverage': 1.0, 'sector': 'X'}, index=order)
        if k != 'quality_breakdown':
            sc = sc.drop(index='BAD')
        EN._persist(f'{k}-r', d, spec, ST.get(k), order, sc, pd.DataFrame())
    payload = RB._gather_cross_strategy(date(2026, 11, 9), spec_json=spec.to_json())
    assert 'BAD' not in set(payload['consensus']['ticker'])
    assert payload['n_strategies'] == 2
    assert payload['signal_dist'].get('Sell', 0) == 0
    assert set(payload['avoid']['ticker']) >= {'BAD'}
    html = RB._render_avoid(payload['avoid'], held={'BAD'})
    assert 'you hold it' in html and 'Deteriorating Quality' in html


# ── momentum construction ────────────────────────────────────────

def test_long_horizon_momentum_skips_the_reversal_month():
    assert ST.DEFAULT_STYLE_PROXIES['momentum'] == 'MOM_12_1'
    assert 'RETURN_1M' not in ST.get('quality_momentum').weights
    assert 'MOM_12_1' in ST.style_strategy().weights


# ── earnings reaction (PEAD) ─────────────────────────────────────

def _px():
    days = pd.DatetimeIndex([d for d in pd.date_range('2026-09-01', '2026-11-06')
                             if MC.is_trading_day(d.date())])
    a = pd.Series(100.0, index=days)
    filed = pd.Timestamp('2026-10-22')
    a[days >= filed] = 110.0                      # +10% on the results day
    a[days > filed + pd.Timedelta(days=1)] = 112.0
    spy = pd.Series(np.linspace(600, 606, len(days)), index=days)
    return pd.DataFrame({'AAA': a, 'SPY': spy}), filed


def test_reaction_is_market_adjusted_three_day_return():
    px, filed = _px()
    ev = pd.DataFrame({'ticker': ['AAA'], 'filed': [filed], 'item_code': ['2.02']})
    r = FA.earnings_reaction(px, ev, date(2026, 11, 6), 'SPY')
    i = px.index.searchsorted(filed)
    spy = px['SPY'].iloc[i + 1] / px['SPY'].iloc[i - 1] - 1
    assert r['AAA'] == pytest.approx(0.10 - spy)


def test_reaction_never_uses_bars_after_as_of():
    px, filed = _px()
    ev = pd.DataFrame({'ticker': ['AAA'], 'filed': [filed], 'item_code': ['2.02']})
    cut = px.loc[:filed]                           # as_of = the filing day
    r = FA.earnings_reaction(cut, ev, filed.date(), 'SPY')
    assert r['AAA'] == pytest.approx(0.10 - (cut['SPY'].iloc[-1] / cut['SPY'].iloc[-2] - 1))


def test_old_or_future_results_are_ignored():
    px, filed = _px()
    old = pd.DataFrame({'ticker': ['AAA'], 'filed': [filed - timedelta(days=90)],
                        'item_code': ['2.02']})
    assert FA.earnings_reaction(px, old, date(2026, 11, 6), 'SPY').empty
    later = pd.DataFrame({'ticker': ['AAA'], 'filed': [pd.Timestamp('2026-11-20')],
                          'item_code': ['2.02']})
    assert FA.earnings_reaction(px, later, date(2026, 11, 6), 'SPY').empty


def test_drift_strategy_is_driven_by_the_reaction():
    w = ST.get('earnings_drift').weights
    assert max(w, key=w.get) == 'EARNINGS_REACTION'
    assert 'EVENT_RESULTS' not in w


def test_track_record_grades_avoid_lists_inverted():
    from quant import track as TRACK
    cohorts = pd.DataFrame({
        'strategy': ['quality_breakdown'] * 3 + ['buffett'] * 3,
        'run_date': [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)] * 2,
        'horizon': '1M', 'sessions': 21, 'n_picks': 10,
        'pick_return': [-0.05, -0.04, 0.01, 0.03, 0.02, -0.01],
        'benchmark': 0.0, 'excess': [-0.05, -0.04, 0.01, 0.03, 0.02, -0.01],
        'ic': [-0.2, -0.1, -0.15, 0.1, 0.2, 0.0]})
    s = TRACK.summary(cohorts).set_index('strategy')
    assert s.loc['quality_breakdown', 'hit_rate'] == pytest.approx(2 / 3)
    assert s.loc['quality_breakdown', 'mean_ic'] == pytest.approx(0.15)
    assert 'avoid list' in s.loc['quality_breakdown', 'name']
    assert s.loc['buffett', 'hit_rate'] == pytest.approx(2 / 3)
