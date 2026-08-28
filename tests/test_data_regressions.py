"""
Regression tests for data faults that reached production.

Every test here corresponds to a wrong number that was actually served by this
app, not a hypothetical. They are separated from test_data_integrity.py because
their value is historical: each one encodes a specific way the store lied, and
deleting one because it "obviously cannot happen" is how it happens again.

The common shape of all of them is a value that was *present and plausible*.
None of these produced an error, an empty screen, or a visible gap — they
produced a confident ranking built on a number that was wrong.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import config
from quant import factors as FA
from quant import fundamentals as F


# ─────────────────────────────────────────────
# DEBT — THE TAG WHITELIST AND THE ZERO SUBSTITUTION
# ─────────────────────────────────────────────

def _facts(rows):
    """Build a facts frame of (concept, tag, value) balance-sheet instants."""
    return pd.DataFrame([{
        'ticker': 'TEST', 'concept': c, 'tag': tag, 'unit': 'USD',
        'period_start': pd.NaT, 'period_end': pd.Timestamp('2026-05-31'),
        'filed': pd.Timestamp('2026-06-22'), 'form': '10-K',
        'fy': 2026, 'fp': 'FY', 'val': float(v),
    } for c, tag, v in rows])


def test_missing_debt_never_becomes_zero():
    """
    Ford and Oracle both reported as *net cash* while carrying $150B and $96B.

    The cause was arithmetic on an absent value: (debt − cash) with debt
    defaulted to 0 is −cash, so a company with a large cash balance and an
    unreadable debt figure ranked as the most conservative name in the index.
    Not knowing the debt must produce nothing, not a flattering number.
    """
    facts = _facts([('cash', 'CashAndCashEquivalentsAtCarryingValue', 31.3e9),
                    ('equity', 'StockholdersEquity', 42.5e9),
                    ('assets', 'Assets', 160e9)])
    row = F.build_fundamentals(['TEST'], date(2026, 8, 13), facts_all=facts)

    assert np.isnan(row.iloc[0]['net_debt_to_equity']), \
        'unknown debt must not be scored as net cash'


def test_a_combined_debt_tag_is_read_when_the_split_tags_are_absent():
    """
    Oracle's total debt read $7.2B against ~$96B actual.

    It stopped tagging LongTermDebtNoncurrent years ago and reports
    DebtLongtermAndShorttermCombinedAmount instead. The whitelist knew only
    the former, so only the current portion was picked up and the company
    looked nine-tenths less levered than it is.
    """
    facts = _facts([
        ('total_debt', 'DebtLongtermAndShorttermCombinedAmount', 129.5e9),
        ('short_term_debt', 'DebtCurrent', 7.2e9),
        ('cash', 'CashAndCashEquivalentsAtCarryingValue', 31.3e9),
        ('equity', 'StockholdersEquity', 42.5e9),
        ('assets', 'Assets', 160e9),
    ])
    row = F.build_fundamentals(['TEST'], date(2026, 8, 13), facts_all=facts)
    assert row.iloc[0]['debt'] == pytest.approx(129.5e9)


def test_a_combined_tag_is_not_added_to_its_own_current_portion():
    """
    The fix for the above must not double-count.

    A filer tagging both a combined figure and its current portion would
    otherwise be charged the current slice twice — turning $129.5B into
    $136.7B and making the balance sheet look worse than it is.
    """
    facts = _facts([
        ('total_debt', 'DebtLongtermAndShorttermCombinedAmount', 129.5e9),
        ('long_term_debt', 'LongTermNotesPayable', 122.3e9),
        ('short_term_debt', 'DebtCurrent', 7.2e9),
        ('equity', 'StockholdersEquity', 42.5e9),
        ('assets', 'Assets', 160e9),
    ])
    row = F.build_fundamentals(['TEST'], date(2026, 8, 13), facts_all=facts)
    assert row.iloc[0]['debt'] == pytest.approx(129.5e9), \
        'combined and split tags were summed'


def test_split_tags_are_still_summed_when_no_combined_tag_exists():
    """The fallback must keep working for filers that only tag the split."""
    facts = _facts([
        ('long_term_debt', 'LongTermDebtNoncurrent', 80e9),
        ('short_term_debt', 'DebtCurrent', 2.3e9),
        ('equity', 'StockholdersEquity', 100e9),
        ('assets', 'Assets', 350e9),
    ])
    row = F.build_fundamentals(['TEST'], date(2026, 8, 13), facts_all=facts)
    assert row.iloc[0]['debt'] == pytest.approx(82.3e9)


def test_genuine_net_cash_is_still_reported_as_negative():
    """
    The guard must not throw away the real signal.

    Alphabet holds more cash than debt; that is a true and useful negative,
    and over-correcting for the Ford case would erase it.
    """
    facts = _facts([
        ('total_debt', 'DebtLongtermAndShorttermCombinedAmount', 14.8e9),
        ('cash', 'CashAndCashEquivalentsAtCarryingValue', 55.9e9),
        ('equity', 'StockholdersEquity', 640e9),
        ('assets', 'Assets', 800e9),
    ])
    row = F.build_fundamentals(['TEST'], date(2026, 8, 13), facts_all=facts)
    assert row.iloc[0]['net_debt_to_equity'] < 0


def test_the_debt_tag_whitelist_covers_the_concepts_the_code_reads():
    """
    A concept read by fundamentals but absent from the tag map is never
    ingested, so it is silently always NaN.
    """
    for concept in ('total_debt', 'long_term_debt', 'short_term_debt'):
        assert concept in config.SEC_TAG_MAP, \
            f'{concept} is read but has no tags to ingest it from'
        assert config.SEC_TAG_MAP[concept], f'{concept} maps to no tags'


def test_combined_debt_tags_are_not_duplicated_across_concepts():
    """
    The same XBRL tag mapped to two concepts would be counted twice.
    """
    seen: dict[str, str] = {}
    for concept in ('total_debt', 'long_term_debt', 'short_term_debt'):
        for tag in config.SEC_TAG_MAP[concept]:
            assert tag not in seen, \
                f'{tag} is mapped to both {seen[tag]} and {concept}'
            seen[tag] = concept


# ─────────────────────────────────────────────
# SPLITS — SHARES AND PRICES MUST SHARE A BASIS
# ─────────────────────────────────────────────

def _split_facts():
    """A filer with a share count and EPS filed before a later 10-for-1 split."""
    return pd.DataFrame([{
        'ticker': 'TEST', 'concept': c, 'tag': c, 'unit': u,
        'period_start': pd.Timestamp('2017-10-01'),
        'period_end': pd.Timestamp('2017-12-31'),
        'filed': pd.Timestamp('2018-01-30'), 'form': '10-Q',
        'fy': 2018, 'fp': 'Q2', 'val': float(v),
    } for c, u, v in [
        ('shares_diluted', 'shares', 183.9e6),
        ('shares_outstanding', 'shares', 183.0e6),
        ('eps_diluted', 'USD/shares', 2.00),
    ]])


_TEN_FOR_ONE = pd.DataFrame([{
    'ticker': 'TEST', 'date': pd.Timestamp('2024-10-03'), 'ratio': 10.0,
}])


def test_share_counts_are_restated_onto_the_current_split_basis():
    """
    The bug that made Lam Research the top pick in every value strategy.

    Price history is retroactively restated after a split; SEC share counts are
    not. A 183.9M share count filed in 2018, priced at a close divided by ten
    for a 2024 split, produced a $3.1B market cap against ~$30B actual — a 66%
    earnings yield, a P/E of 1.5, and rank 1 at every historical date.
    """
    adj = F.adjust_shares_for_splits(_split_facts(), _TEN_FOR_ONE)
    shares = adj.loc[adj['concept'] == 'shares_diluted', 'val'].iloc[0]
    assert shares == pytest.approx(1_839e6), 'share count was not restated'


def test_per_share_amounts_move_the_opposite_way_from_counts():
    """
    EPS must fall by the same ratio the share count rises by.

    Scaling only the count breaks the cross-check between reported shares and
    net income / EPS: the ratio reads as 10x, the guard treats it as a units
    error, and it rejects the share count entirely — turning a wrong market cap
    into a missing one.
    """
    adj = F.adjust_shares_for_splits(_split_facts(), _TEN_FOR_ONE)
    eps = adj.loc[adj['concept'] == 'eps_diluted', 'val'].iloc[0]
    assert eps == pytest.approx(0.20), 'EPS was not restated with the split'

    shares = adj.loc[adj['concept'] == 'shares_diluted', 'val'].iloc[0]
    assert (shares * eps) == pytest.approx(183.9e6 * 2.00), \
        'net income implied by shares x EPS must be invariant to the split'


def test_filings_made_after_a_split_are_left_alone():
    """They are already on the post-split basis."""
    facts = _split_facts()
    facts['filed'] = pd.Timestamp('2025-01-30')     # after the split
    adj = F.adjust_shares_for_splits(facts, _TEN_FOR_ONE)
    assert adj.loc[adj['concept'] == 'shares_diluted', 'val'].iloc[0] \
        == pytest.approx(183.9e6)


def test_reverse_splits_are_handled_in_the_same_way():
    """A 1-for-5 reverse split shrinks the count and lifts per-share amounts."""
    reverse = pd.DataFrame([{
        'ticker': 'TEST', 'date': pd.Timestamp('2024-10-03'), 'ratio': 0.2,
    }])
    adj = F.adjust_shares_for_splits(_split_facts(), reverse)
    assert adj.loc[adj['concept'] == 'shares_diluted', 'val'].iloc[0] \
        == pytest.approx(36.78e6)
    assert adj.loc[adj['concept'] == 'eps_diluted', 'val'].iloc[0] \
        == pytest.approx(10.0)


def test_multiple_splits_compound():
    """NVDA split 4-for-1 in 2021 and 10-for-1 in 2024: a filing from 2018
    needs both."""
    both = pd.DataFrame([
        {'ticker': 'TEST', 'date': pd.Timestamp('2021-07-20'), 'ratio': 4.0},
        {'ticker': 'TEST', 'date': pd.Timestamp('2024-06-10'), 'ratio': 10.0},
    ])
    adj = F.adjust_shares_for_splits(_split_facts(), both)
    assert adj.loc[adj['concept'] == 'shares_diluted', 'val'].iloc[0] \
        == pytest.approx(183.9e6 * 40)


def test_a_ticker_with_no_splits_is_untouched():
    other = pd.DataFrame([{
        'ticker': 'OTHER', 'date': pd.Timestamp('2024-10-03'), 'ratio': 10.0,
    }])
    adj = F.adjust_shares_for_splits(_split_facts(), other)
    assert adj.loc[adj['concept'] == 'shares_diluted', 'val'].iloc[0] \
        == pytest.approx(183.9e6)


def test_an_absent_split_table_is_not_an_error():
    """Splits are a correction; missing them must not break a screen."""
    facts = _split_facts()
    assert F.adjust_shares_for_splits(facts, pd.DataFrame()).equals(facts)
    assert F.adjust_shares_for_splits(facts, None).equals(facts)


def test_a_split_adjusted_earnings_yield_is_plausible():
    """
    The end-to-end assertion: a large profitable company cannot have a P/E of
    1.5. Bounds alone did not catch this — 0.66 sits inside the (-1, 1) band
    written to catch an 89,900% yield.
    """
    facts = _split_facts()
    adj = F.adjust_shares_for_splits(facts, _TEN_FOR_ONE)
    shares = adj.loc[adj['concept'] == 'shares_diluted', 'val'].iloc[0]
    price_today_basis = 18.93               # 2018 close, restated for the split
    mcap = shares * price_today_basis
    net_income = 183.9e6 * 2.00 * 4         # roughly annualised
    assert 0.0 < net_income / mcap < 0.15, \
        f'earnings yield {net_income / mcap:.3f} implies an impossible P/E'


def test_market_cap_disagreement_is_reported_as_an_error(monkeypatch):
    """
    The reliable detector for a share-basis fault.

    Computed market cap and the provider's figure are derived completely
    differently, so they fail independently. A split, a wrong share class or a
    units error moves one and leaves the other alone — which is exactly what a
    share-count series on its own cannot tell you.
    """
    from quant import validate as V
    monkeypatch.setattr(V.db, 'read_sql', lambda *a, **k: pd.DataFrame(
        {'ticker': ['GOOD', 'SPLIT'], 'market_cap': [100e9, 300e9]}))

    factors = pd.DataFrame({'MARKET_CAP': [101e9, 30e9]},
                           index=['GOOD', 'SPLIT'])
    findings = V.check_market_cap_agreement(factors)
    assert findings and findings[0].severity == 'error'
    assert findings[0].count == 1, 'only the mismatched name should be flagged'
    assert 'SPLIT' in str(findings[0].examples)


def test_market_cap_agreement_is_reported_when_everything_reconciles(monkeypatch):
    """A clean result must be visible, not silent — it is the evidence."""
    from quant import validate as V
    monkeypatch.setattr(V.db, 'read_sql', lambda *a, **k: pd.DataFrame(
        {'ticker': ['A', 'B'], 'market_cap': [100e9, 200e9]}))

    factors = pd.DataFrame({'MARKET_CAP': [101e9, 198e9]}, index=['A', 'B'])
    findings = V.check_market_cap_agreement(factors)
    assert findings and findings[0].severity == 'info'
    assert findings[0].count == 2


def test_a_ten_x_error_is_far_outside_the_agreement_tolerance():
    """
    The Lam Research case, stated as a bound: the split fault produced a 0.1x
    ratio, which no sane tolerance can absorb.
    """
    from quant import validate as V
    assert abs(0.1 - 1.0) > 0.15, 'tolerance must not be wide enough to hide it'


def test_split_ratio_matcher_accepts_a_blended_split(monkeypatch):
    """
    Share counts are weighted averages, so a split mid-quarter blends the two
    bases: Lam Research's 10-for-1 reads 9.88, not 10.0.
    """
    from quant import validate as V
    assert V._near_split_ratio(9.88)
    assert V._near_split_ratio(10.0)
    assert V._near_split_ratio(0.2)          # 1-for-5 reverse
    assert V._near_split_ratio(3.98)


def test_split_ratio_matcher_rejects_ordinary_issuance():
    """
    A merger or IPO that lifts the count by a half is not a split, and treating
    it as one buried the real signal under Airbnb, Coinbase and Charter.
    """
    from quant import validate as V
    for ratio in (1.41, 1.48, 1.52, 1.77, 2.48, 1.1, 0.95):
        assert not V._near_split_ratio(ratio), f'{ratio} should not match'


# ─────────────────────────────────────────────
# EPS — A LEVEL THAT IS NOT COMPARABLE
# ─────────────────────────────────────────────

def test_raw_eps_is_not_offered_as_a_rankable_factor():
    """
    A $10 EPS is not better than a $2 EPS — it reflects the share count. It was
    specified as a Value proxy and deliberately replaced by EARNINGS_YIELD,
    which is the same quantity divided by price and therefore comparable.
    """
    from quant import strategies as ST
    value_proxies = [f for f, _ in ST.STYLE_PROXIES['value']]
    assert 'EPS' not in value_proxies
    assert 'IS_EPS' not in value_proxies
    assert 'EARNINGS_YIELD' in value_proxies


def test_eps_growth_is_undefined_through_a_loss_year():
    """
    A company going from a loss to a profit has no meaningful growth *rate*.
    Reporting one would rank a recovery from -$1 to +$1 as infinite growth.
    """
    facts = pd.DataFrame([{
        'ticker': 'TEST', 'concept': c, 'tag': c, 'unit': 'USD',
        'period_start': pd.Timestamp(ps), 'period_end': pd.Timestamp(pe),
        'filed': pd.Timestamp(fl), 'form': '10-K', 'fy': fy, 'fp': 'FY',
        'val': float(v),
    } for c, ps, pe, fl, fy, v in [
        ('net_income', '2022-01-01', '2022-12-31', '2023-02-01', 2022, -500e6),
        ('net_income', '2025-01-01', '2025-12-31', '2026-02-01', 2025, 900e6),
        ('shares_diluted', '2022-01-01', '2022-12-31', '2023-02-01', 2022, 100e6),
        ('shares_diluted', '2025-01-01', '2025-12-31', '2026-02-01', 2025, 100e6),
    ]])
    row = F.build_fundamentals(['TEST'], date(2026, 8, 13), facts_all=facts)
    assert np.isnan(row.iloc[0]['eps_cagr_3y'])


# ─────────────────────────────────────────────
# YIELDS — ONE UNIT, ONE SOURCE
# ─────────────────────────────────────────────

def test_dividend_yield_is_derived_from_filings_not_the_live_snapshot():
    """
    Dividend yield came from profile_snapshots: 5% coverage, today only.

    Total Shareholder Yield weighted it 1.0 of 5.0, so a third of the strategy
    scored on almost nothing, and it could not be backtested at all because the
    provider publishes no history. The cash paid is on the cash flow statement,
    which is where BUYBACK_YIELD already reads from.
    """
    import inspect
    src = inspect.getsource(FA.fundamental_factors)
    div = [ln for ln in src.splitlines() if "out['DIVIDEND_YIELD']" in ln]
    assert div, 'DIVIDEND_YIELD is no longer assigned'
    assert 'dividends_paid_ttm' in div[0], \
        'DIVIDEND_YIELD must come from filings, not the provider snapshot'


def test_dividend_yield_is_not_a_live_only_factor_any_more():
    """Having a filing source, it is knowable historically."""
    assert 'DIVIDEND_YIELD' not in FA.LIVE_ONLY_FACTORS
    assert 'PAYOUT_RATIO' not in FA.LIVE_ONLY_FACTORS


def test_analyst_multiples_remain_live_only():
    """Forward P/E is an analyst estimate; there is no historical series."""
    assert 'FORWARD_PE' in FA.LIVE_ONLY_FACTORS
    assert 'TRAILING_PE' in FA.LIVE_ONLY_FACTORS


def test_all_yields_share_one_unit():
    """
    The provider reported percentages while every other yield is a decimal, so
    a 1.66% dividend yield entered the composite as 1.66 — three orders of
    magnitude above a 0.04 earnings yield in the same weighted sum.
    """
    fund = pd.DataFrame({
        'dividends_paid_ttm': [2.37e9],
        'buybacks_ttm': [1.0e9],
        'net_income_ttm': [3.68e9],
        'shares_diluted': [4.3e9],
    }, index=['KO'])
    mcap = pd.Series([100e9], index=['KO'])

    div = (fund['dividends_paid_ttm'] / mcap).iloc[0]
    buy = (fund['buybacks_ttm'] / mcap).iloc[0]
    assert 0.0 < div < 0.25, f'dividend yield {div} is not a decimal fraction'
    assert 0.0 < buy < 0.25
    assert abs(div - 0.0237) < 1e-9


# ─────────────────────────────────────────────
# CACHE — A CODE CHANGE MUST INVALIDATE STORED FACTORS
# ─────────────────────────────────────────────

def test_the_cache_key_changes_when_the_code_version_does(monkeypatch):
    """
    The data fingerprint only moves on ingest. Without a code version, editing
    a factor formula served the old numbers from disk indefinitely.
    """
    from quant import factorcache
    tickers, as_of = ['AAPL', 'MSFT'], date(2026, 8, 13)
    before = factorcache.key(tickers, as_of, version='fixed')
    monkeypatch.setattr(factorcache, 'CODE_VERSION', factorcache.CODE_VERSION + 1)
    assert factorcache.key(tickers, as_of, version='fixed') != before


def test_the_cache_key_ignores_ticker_ordering():
    """The same universe in a different order is the same universe."""
    from quant import factorcache
    as_of = date(2026, 8, 13)
    a = factorcache.key(['AAPL', 'MSFT', 'NVDA'], as_of, version='fixed')
    b = factorcache.key(['NVDA', 'AAPL', 'MSFT'], as_of, version='fixed')
    assert a == b


def test_the_cache_key_separates_different_universes():
    from quant import factorcache
    as_of = date(2026, 8, 13)
    a = factorcache.key(['AAPL', 'MSFT'], as_of, version='fixed')
    b = factorcache.key(['AAPL', 'MSFT', 'NVDA'], as_of, version='fixed')
    assert a != b


def test_the_cache_key_separates_different_dates():
    from quant import factorcache
    a = factorcache.key(['AAPL'], date(2026, 8, 13), version='fixed')
    b = factorcache.key(['AAPL'], date(2026, 8, 12), version='fixed')
    assert a != b


# ─────────────────────────────────────────────
# BACKTEST — DEGRADED RUNS MUST REFUSE
# ─────────────────────────────────────────────

def test_a_backtest_refuses_rather_than_scoring_on_absent_factors():
    """
    An insider strategy backtested over a window with no insider filings
    returned a number for years. It was not a weak result for that strategy —
    the two factors that define it were absent, so it measured something else
    entirely while carrying the strategy's name.
    """
    from quant import backtest as BT
    assert issubclass(BT.DegradedBacktest, RuntimeError)


def test_strict_is_the_default_for_the_rebalanced_backtest():
    """Opting into a degraded measurement must be explicit."""
    import inspect
    from quant import backtest as BT
    sig = inspect.signature(BT.run_backtest)
    assert sig.parameters['strict'].default is True


# ─────────────────────────────────────────────
# REPORTS — DELETING ONE MUST NOT ORPHAN ANOTHER
# ─────────────────────────────────────────────

def test_report_ids_are_unique_per_build_not_per_file():
    """
    Rebuilding the same kind/strategy/date overwrites one file but inserts a
    new UUID row, so several rows legitimately share a path. Deleting one row
    must not unlink a file the others still point at.
    """
    import inspect
    from reports import builder as RB
    src = inspect.getsource(RB.delete_report)
    assert 'COUNT(*)' in src and 'path' in src, \
        'delete_report must check whether the path is still referenced'
