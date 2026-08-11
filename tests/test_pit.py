"""
Point-in-time correctness — the tests that keep the research honest.

If any of these fail, every backtest number in the app is worthless, because
the strategy would be using information it could not have had. They are
therefore the most important tests in the suite.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quant import fundamentals as F


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

def _fact(concept, start, end, filed, val, form='10-Q', fp='Q1', fy=2026):
    return {
        'ticker': 'TEST', 'concept': concept, 'tag': concept, 'unit': 'USD',
        'period_start': pd.Timestamp(start), 'period_end': pd.Timestamp(end),
        'filed': pd.Timestamp(filed), 'form': form, 'fy': fy, 'fp': fp,
        'val': float(val),
    }


@pytest.fixture
def duplicate_duration_facts():
    """
    The real AAPL FY2026 Q3 case.

    Two NetIncomeLoss facts share an end date of 2026-06-27 and a filing date
    of 2026-07-31: 29,789M covers the quarter, 101,464M covers the nine months
    to date. Only `period_start` distinguishes them.
    """
    return pd.DataFrame([
        _fact('net_income', '2025-09-28', '2025-12-27', '2026-01-30', 42_097e6),
        _fact('net_income', '2025-12-28', '2026-03-28', '2026-05-01', 29_578e6),
        _fact('net_income', '2026-03-28', '2026-06-27', '2026-07-31', 29_789e6),
        # …and the nine-month cumulative filed the same day.
        _fact('net_income', '2025-09-28', '2026-06-27', '2026-07-31', 101_464e6),
        _fact('net_income', '2025-06-29', '2025-09-27', '2025-10-30', 27_466e6),
    ])


@pytest.fixture
def restated_facts():
    """
    Four quarters, one of which is later restated downward.

    A full year is needed because TTM deliberately refuses to annualize from a
    single quarter — see test_ttm_refuses_to_extrapolate_from_one_quarter.
    """
    return pd.DataFrame([
        _fact('revenue', '2025-04-01', '2025-06-30', '2025-07-25', 800e6),
        _fact('revenue', '2025-07-01', '2025-09-30', '2025-10-25', 850e6),
        _fact('revenue', '2025-10-01', '2025-12-31', '2026-01-25', 900e6),
        # Q1 2026 first reported at 1,000M…
        _fact('revenue', '2026-01-01', '2026-03-31', '2026-04-25', 1_000e6),
        # …then restated to 900M in August.
        _fact('revenue', '2026-01-01', '2026-03-31', '2026-08-01', 900e6),
    ])


# ─────────────────────────────────────────────
# DURATION DISAMBIGUATION
# ─────────────────────────────────────────────

def test_quarterly_flows_ignores_cumulative_row(duplicate_duration_facts):
    q = F.quarterly_flows(duplicate_duration_facts, 'net_income')
    ends = set(q['period_end'].dt.date)
    assert date(2026, 6, 27) in ends

    row = q[q['period_end'] == pd.Timestamp('2026-06-27')].iloc[0]
    assert row['val'] == pytest.approx(29_789e6), \
        'must take the quarterly figure, not the nine-month cumulative'


def test_ttm_sums_four_quarters_not_the_cumulative(duplicate_duration_facts):
    """
    Naively summing every row that ends on the period date inflates TTM by
    roughly 3.4x, which silently corrupts every derived ratio.
    """
    ttm = F.ttm(duplicate_duration_facts, 'net_income')
    expected = (42_097 + 29_578 + 29_789 + 27_466) * 1e6
    assert ttm == pytest.approx(expected, rel=1e-6)

    naive = duplicate_duration_facts['val'].sum()
    assert naive > ttm * 1.5, 'precondition: the naive sum is badly wrong'


def test_duration_classifier_separates_quarter_from_cumulative():
    assert F._classify_duration(91) == 'Q'
    assert F._classify_duration(273) == '9M'
    assert F._classify_duration(365) == 'A'
    assert F._classify_duration(182) == 'H'


# ─────────────────────────────────────────────
# RESTATEMENTS
# ─────────────────────────────────────────────

def test_restatement_uses_value_known_at_the_time(restated_facts):
    """
    Before the revision was filed, only the original figure existed. A
    backtest dated between the two filings must use 1,000M, not the 900M that
    nobody could yet see.
    """
    visible_early = restated_facts[
        restated_facts['filed'] <= pd.Timestamp('2026-06-01')]
    early_ttm = F.ttm(visible_early, 'revenue')
    assert early_ttm == pytest.approx((800 + 850 + 900 + 1_000) * 1e6)

    late_ttm = F.ttm(restated_facts, 'revenue')
    assert late_ttm == pytest.approx((800 + 850 + 900 + 900) * 1e6)
    assert late_ttm < early_ttm, 'the downward restatement must lower TTM'


def test_ttm_refuses_to_extrapolate_from_one_quarter():
    """
    A single quarter is not a trailing-twelve-month figure. Returning NaN is
    correct; annualizing one quarter would silently invent three.
    """
    one_quarter = pd.DataFrame([
        _fact('revenue', '2026-01-01', '2026-03-31', '2026-04-25', 1_000e6)])
    assert np.isnan(F.ttm(one_quarter, 'revenue'))


def test_latest_filed_collapses_restatements_per_period(restated_facts):
    collapsed = F._latest_filed(restated_facts.copy(), ['period_end'])
    assert len(collapsed) == 4, 'one row per period, restatements resolved'
    q1 = collapsed[collapsed['period_end'] == pd.Timestamp('2026-03-31')]
    assert q1.iloc[0]['val'] == pytest.approx(900e6), 'newest filing wins'


# ─────────────────────────────────────────────
# LOOK-AHEAD GATE
# ─────────────────────────────────────────────

def test_facts_asof_never_returns_a_future_filing():
    """
    The core invariant: a factor computed as of T must reference no row filed
    after T. Asserted against real stored data when present.
    """
    from data import sec

    tickers = ['AAPL']
    as_of = date.today() - timedelta(days=365)
    facts = sec.facts_asof(tickers, as_of)

    if facts.empty:
        pytest.skip('no ingested facts; run cli.py ingest first')

    latest = facts['filed'].max().date()
    assert latest <= as_of, (
        f'look-ahead leak: as_of={as_of} but a fact filed {latest} was visible')


def test_pit_gate_changes_the_answer_over_time():
    """A point-in-time read must actually differ between two dates."""
    from data import sec

    early = sec.facts_asof(['AAPL'], date.today() - timedelta(days=400))
    late = sec.facts_asof(['AAPL'], date.today())
    if early.empty or late.empty:
        pytest.skip('no ingested facts; run cli.py ingest first')

    assert len(late) > len(early), \
        'more filings should be visible today than a year ago'


def test_membership_is_point_in_time():
    """
    Index membership must reflect the constituents of the date, not today's.

    The S&P 500 on 2021-06-01 contained ABMD and ATVI; both were acquired
    since. A screen dated 2021 that silently excluded them would be loaded
    with survivors.
    """
    from data import members

    historical = members.members_asof('SPY', date(2021, 6, 1))
    if not historical:
        pytest.skip('no membership backfill; run cli.py ingest --members-from')

    current = members.members_asof('SPY', date.today())
    assert set(historical) != set(current), \
        'historical membership must differ from current membership'


# ─────────────────────────────────────────────
# DERIVED QUANTITIES
# ─────────────────────────────────────────────

def test_growth_rejects_sign_flips():
    """A loss turning into a profit is not '-250% growth'."""
    assert np.isnan(F._growth(100, -40))
    assert np.isnan(F._growth(-40, 100))
    assert F._growth(110, 100) == pytest.approx(0.10)


def test_cagr_requires_positive_endpoints():
    assert np.isnan(F._cagr(100, -50, 3))
    assert F._cagr(133.1, 100, 3) == pytest.approx(0.10, rel=1e-3)


def test_safe_div_handles_zero_and_nan():
    assert np.isnan(F._safe_div(1, 0))
    assert np.isnan(F._safe_div(np.nan, 5))
    assert F._safe_div(10, 4) == pytest.approx(2.5)
