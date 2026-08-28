"""
Storage integrity — what is written must be exactly what comes back.

The bugs this file exists to catch are the quiet ones. A factor that is simply
absent costs breadth and the coverage floor handles it. A number that survives
a round trip *changed* — truncated to an integer, rounded through a float, or
silently overwritten by a partial write — is far worse: it is present,
plausible, and false, and every ranking built on it is wrong in a way nothing
downstream can detect.

Four failures already found in this codebase motivate the cases below:

  * a partial upsert nulled every column it did not name, because the update
    clause is built from all non-PK columns (`reports.emailed_at`);
  * total debt read $7.2B for a company carrying $96B, because the tag
    whitelist missed the concept the filer actually uses;
  * a missing debt figure was treated as zero, reporting the most leveraged
    names in the index as the most cash-rich;
  * a factor with no data anywhere silently redistributed its weight, so a
    backtest of an insider strategy ran over a period with no insider filings
    and returned a number anyway.

These are storage-level tests: they use a real SQLite engine on a temp file
rather than mocks, because the failures above all lived in the dialect-specific
upsert and would pass against a mock.
"""

from __future__ import annotations

import importlib
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

import config
from quant import factors as FA
from quant import strategies as ST
from quant import validate as V


# ─────────────────────────────────────────────
# AN ISOLATED STORE
# ─────────────────────────────────────────────

@pytest.fixture
def store(tmp_path, monkeypatch):
    """
    A real SQLite database on disk, rebuilt per test.

    Deliberately not an in-memory mock: every bug listed in the module
    docstring lived in the SQLAlchemy upsert or the schema, so a fake would
    have reported success on all of them.
    """
    monkeypatch.setattr(config, 'DATABASE_URL', f'sqlite:///{tmp_path}/t.sqlite')
    from core import db as _db
    importlib.reload(_db)
    _db.init_db()
    yield _db
    _db.get_engine().dispose()


# ─────────────────────────────────────────────
# ROUND TRIP — VALUES MUST SURVIVE STORAGE
# ─────────────────────────────────────────────

@pytest.mark.parametrize('val', [
    1.0,
    0.0,
    -1.0,
    1e-9,                    # small enough to vanish under a float32 cast
    1.2345678901234567,      # more digits than float32 can hold
    129_541_000_000.0,       # Oracle's real total debt
    1_839_045.0,
    -0.4725218,              # a net-cash ratio
])
def test_float_survives_a_round_trip_exactly(store, val):
    """
    A stored float must come back bit-identical.

    Market caps run to 1e12 and yields to 1e-4 in the same column family. A
    float32 column would silently round both, and the resulting factor would
    still look entirely reasonable.
    """
    store.upsert(store.sec_facts, [{
        'cik': '0000000001', 'ticker': 'TEST', 'concept': 'net_income', 'tag': 'NetIncomeLoss',
        'unit': 'USD', 'period_start': date(2026, 1, 1),
        'period_end': date(2026, 3, 31), 'filed': date(2026, 4, 30),
        'form': '10-Q', 'fy': 2026, 'fp': 'Q1', 'val': val,
    }])
    got = store.read_sql('SELECT val FROM sec_facts').iloc[0]['val']
    assert got == val, f'{val!r} came back as {got!r}'


def test_large_and_small_magnitudes_coexist(store):
    """Both ends of the range in one column, read back together."""
    rows = [{
        'cik': f'{i:010d}', 'ticker': f'T{i}', 'concept': 'assets', 'tag': 'Assets', 'unit': 'USD',
        'period_start': date(2026, 1, 1), 'period_end': date(2026, 3, 31),
        'filed': date(2026, 4, 30), 'form': '10-Q', 'fy': 2026, 'fp': 'Q1',
        'val': v,
    } for i, v in enumerate([1e-6, 1.0, 1e6, 1e12, 4.36e10])]
    store.upsert(store.sec_facts, rows)
    got = store.read_sql('SELECT ticker, val FROM sec_facts ORDER BY ticker')
    assert list(got['val']) == [1e-6, 1.0, 1e6, 1e12, 4.36e10]


def test_dates_survive_a_round_trip(store):
    """A filing date shifted by a day silently breaks the PIT gate."""
    store.upsert(store.sec_facts, [{
        'cik': '0000000001', 'ticker': 'TEST', 'concept': 'revenue', 'tag': 'Revenues',
        'unit': 'USD', 'period_start': date(2025, 12, 28),
        'period_end': date(2026, 3, 28), 'filed': date(2026, 5, 1),
        'form': '10-Q', 'fy': 2026, 'fp': 'Q2', 'val': 1.0,
    }])
    got = store.read_sql(
        'SELECT period_start, period_end, filed FROM sec_facts',
        parse_dates=['period_start', 'period_end', 'filed']).iloc[0]
    assert got['period_start'].date() == date(2025, 12, 28)
    assert got['period_end'].date() == date(2026, 3, 28)
    assert got['filed'].date() == date(2026, 5, 1)


def test_none_stays_none_and_does_not_become_zero(store):
    """
    A missing value must not read back as 0.

    This is the failure that reported Ford and Oracle as net-cash: an absent
    debt figure treated as zero makes (debt − cash) a large negative number,
    and the most leveraged names in the index rank as the most conservative.
    """
    store.upsert(store.prices, [{
        'ticker': 'TEST', 'date': date(2026, 8, 12), 'open': 1.0, 'high': 1.0,
        'low': 1.0, 'close': 1.0, 'adj_close': 1.0, 'volume': None,
    }])
    got = store.read_sql('SELECT volume FROM prices').iloc[0]['volume']
    assert got is None or (isinstance(got, float) and np.isnan(got))
    assert got != 0


# ─────────────────────────────────────────────
# UPSERT SEMANTICS
# ─────────────────────────────────────────────

def test_upsert_replaces_a_row_rather_than_duplicating_it(store):
    """A restated fact must overwrite, not accumulate."""
    row = {
        'cik': '0000000001', 'ticker': 'TEST', 'concept': 'net_income', 'tag': 'NetIncomeLoss',
        'unit': 'USD', 'period_start': date(2026, 1, 1),
        'period_end': date(2026, 3, 31), 'filed': date(2026, 4, 30),
        'form': '10-Q', 'fy': 2026, 'fp': 'Q1', 'val': 100.0,
    }
    store.upsert(store.sec_facts, [row])
    store.upsert(store.sec_facts, [{**row, 'val': 250.0}])

    got = store.read_sql('SELECT val FROM sec_facts')
    assert len(got) == 1, 'same primary key must not create a second row'
    assert got.iloc[0]['val'] == 250.0, 'restatement must win'


def test_partial_upsert_does_not_null_the_columns_it_omits(store):
    """
    Writing one column must not erase the rest of the row.

    `upsert` builds its update clause from every non-PK column, so a caller
    passing only {report_id, emailed_at} sets path, kind and created_at to
    NULL. `reports.email` does exactly this, which is why the history table
    could show a report with no file and no date after it was emailed.
    """
    store.upsert(store.reports, [{
        'report_id': 'r1', 'kind': 'daily',
        'period_start': date(2026, 8, 12), 'period_end': date(2026, 8, 13),
        'path': '/tmp/daily.html', 'emailed_at': None,
        'created_at': datetime(2026, 8, 13, 9, 0),
    }])
    store.upsert(store.reports, [{
        'report_id': 'r1', 'emailed_at': datetime(2026, 8, 13, 9, 5),
    }])

    got = store.read_sql('SELECT * FROM reports').iloc[0]
    assert got['emailed_at'] is not None
    assert got['path'] == '/tmp/daily.html', 'path was erased by a partial write'
    assert got['kind'] == 'daily', 'kind was erased by a partial write'


def test_upsert_of_an_empty_batch_is_a_no_op(store):
    assert store.upsert(store.prices, []) == 0


def test_upsert_is_chunked_without_losing_rows(store):
    """Batches larger than the chunk size must still land in full."""
    rows = [{
        'ticker': 'TEST', 'date': date(2020, 1, 1) + timedelta(days=i),
        'open': 1.0, 'high': 1.0, 'low': 1.0,
        'close': float(i), 'adj_close': float(i), 'volume': 100,
    } for i in range(2500)]                       # chunk defaults to 2000
    store.upsert(store.prices, rows)
    got = store.read_sql('SELECT COUNT(*) n, MAX(close) mx FROM prices').iloc[0]
    assert got['n'] == 2500
    assert got['mx'] == 2499.0


# ─────────────────────────────────────────────
# POINT IN TIME — READS MUST NOT SEE THE FUTURE
# ─────────────────────────────────────────────

def test_facts_asof_hides_filings_made_after_the_date(store):
    """The gate is `filed`, never `period_end`."""
    from data import sec
    base = {
        'cik': '0000000001', 'ticker': 'TEST', 'concept': 'revenue', 'tag': 'Revenues',
        'unit': 'USD', 'period_start': date(2026, 1, 1),
        'form': '10-Q', 'fy': 2026, 'fp': 'Q1',
    }
    store.upsert(store.sec_facts, [
        {**base, 'period_end': date(2026, 3, 31),
         'filed': date(2026, 4, 30), 'val': 10.0},
        {**base, 'period_end': date(2026, 6, 30),
         'filed': date(2026, 7, 31), 'val': 20.0},
    ])
    seen = sec.facts_asof(['TEST'], date(2026, 5, 15))
    assert list(seen['val']) == [10.0], 'a later filing leaked into the past'


def test_a_quarter_is_invisible_between_period_end_and_filing(store):
    """
    The gap between period end and filing is real and must be respected.

    A quarter ending 2026-03-31 and filed 2026-04-30 was not knowable on
    2026-04-15, even though the period had closed.
    """
    from data import sec
    store.upsert(store.sec_facts, [{
        'cik': '0000000001', 'ticker': 'TEST', 'concept': 'revenue', 'tag': 'Revenues',
        'unit': 'USD', 'period_start': date(2026, 1, 1),
        'period_end': date(2026, 3, 31), 'filed': date(2026, 4, 30),
        'form': '10-Q', 'fy': 2026, 'fp': 'Q1', 'val': 10.0,
    }])
    assert sec.facts_asof(['TEST'], date(2026, 4, 15)).empty
    assert not sec.facts_asof(['TEST'], date(2026, 4, 30)).empty


# ─────────────────────────────────────────────
# FACTOR AVAILABILITY — DEGRADED RUNS MUST ANNOUNCE THEMSELVES
# ─────────────────────────────────────────────

def _frame(**cols):
    idx = [f'T{i}' for i in range(10)]
    return pd.DataFrame(cols, index=idx)


def test_a_factor_absent_everywhere_is_reported_not_ignored():
    """
    The failure that produced a meaningless backtest result.

    An insider strategy scored over a window with no insider filings drops
    those factors and ranks on what is left. The number it returns is not a
    weak result for that strategy — it is a result for a different one.
    """
    raw = _frame(ROIC=np.linspace(0.1, 0.3, 10),
                 INSIDER_NET_BUY=[np.nan] * 10)
    avail = FA.factor_availability(
        raw, {'ROIC': 1.0, 'INSIDER_NET_BUY': 1.2})
    by = avail.set_index('factor')['status']
    assert by['ROIC'] == 'ok'
    assert by['INSIDER_NET_BUY'] == 'empty'


def test_a_factor_the_frame_never_produced_is_reported_as_missing():
    avail = FA.factor_availability(_frame(ROIC=[1.0] * 10), {'NO_SUCH': 1.0})
    assert avail.iloc[0]['status'] == 'missing'
    assert avail.iloc[0]['coverage'] == 0.0


def test_live_only_factors_are_flagged_on_historical_dates():
    """
    An analyst forward multiple has no historical series — profile_snapshots
    holds only today. Scoring one in the past reads nothing while appearing to
    read something, so the reason is named rather than reported as a bare gap.
    """
    raw = _frame(FORWARD_PE=[np.nan] * 10)
    hist = FA.factor_availability(raw, {'FORWARD_PE': 1.0}, '2020-01-02')
    assert hist.iloc[0]['status'] == 'live_only'

    today = FA.factor_availability(raw, {'FORWARD_PE': 1.0}, date.today())
    assert today.iloc[0]['status'] == 'empty', 'only the past gets the excuse'


def test_every_live_only_factor_actually_exists():
    """Guard against a rename leaving the exclusion list pointing at nothing."""
    for name in FA.LIVE_ONLY_FACTORS:
        assert name in FA.FACTOR_DIRECTION, f'{name} is not a real factor'


def test_scoring_reports_the_share_of_weight_it_had_to_drop():
    from quant import engine as EN
    strat = ST.Strategy(
        key='t', name='T', horizon='long', description='',
        weights={'ROIC': 3.0, 'INSIDER_NET_BUY': 1.0})
    raw = _frame(ROIC=np.linspace(0.1, 0.3, 10),
                 INSIDER_NET_BUY=[np.nan] * 10,
                 sector=['Tech'] * 10)
    _scores, _detail, warnings = EN.score_universe(raw, strat)
    assert warnings, 'a dropped factor must be reported'
    assert any('INSIDER_NET_BUY' in w for w in warnings)
    assert any('25%' in w for w in warnings), 'must quantify the weight lost'


def test_a_healthy_run_reports_nothing():
    from quant import engine as EN
    strat = ST.Strategy(key='t', name='T', horizon='long', description='',
                        weights={'ROIC': 1.0})
    raw = _frame(ROIC=np.linspace(0.1, 0.3, 10), sector=['Tech'] * 10)
    _s, _d, warnings = EN.score_universe(raw, strat)
    assert warnings == []


# ─────────────────────────────────────────────
# FACTOR CONSTRUCTION INVARIANTS
# ─────────────────────────────────────────────

def test_reversal_is_exactly_the_negation_of_the_1m_return():
    """
    They are one series with two signs. Computing them separately let them
    drift; deriving one from the other makes disagreement impossible.
    """
    px = pd.DataFrame(
        {'A': np.linspace(100, 120, 300), 'B': np.linspace(100, 80, 300)},
        index=pd.bdate_range('2025-01-01', periods=300))
    out = FA.price_factors(['A', 'B'], date.today(), prices=px) \
        if 'prices' in FA.price_factors.__code__.co_varnames else None
    if out is None:
        pytest.skip('price_factors does not accept an injected frame')
    pair = out[['RETURN_1M', 'REVERSAL_1M']].dropna()
    assert np.allclose(pair['RETURN_1M'], -pair['REVERSAL_1M'])


def test_every_weighted_factor_has_a_declared_direction():
    """
    A factor with no entry in FACTOR_DIRECTION defaults to higher-is-better,
    which silently inverts anything where lower is better.
    """
    for key, strat in ST.ALL_STRATEGIES.items():
        for factor in strat.weights:
            assert factor in FA.FACTOR_DIRECTION, \
                f'{key} weights {factor}, which has no declared direction'


def test_lower_is_better_factors_are_declared_as_such():
    """A sign error here inverts the ranking without changing any value."""
    for name in ('ACCRUALS', 'DEBT_TO_EQUITY', 'NET_DEBT_TO_EQUITY',
                 'VOL_1Y', 'BETA', 'PAYOUT_RATIO'):
        assert FA.FACTOR_DIRECTION[name] is False, f'{name} has the wrong sign'


def test_every_strategy_factor_is_documented():
    from quant import factor_docs as FD
    for key, strat in ST.ALL_STRATEGIES.items():
        for factor in strat.weights:
            assert FD.doc_for(factor) is not None, \
                f'{key} weights {factor}, which has no FactorDoc'


# ─────────────────────────────────────────────
# VALUE BOUNDS
# ─────────────────────────────────────────────

def test_out_of_range_values_are_caught_not_winsorized_away():
    """
    Erie Indemnity carried an earnings yield of 899 — 89,900% — from a 2,542
    share partial class. Winsorization clips that to merely excellent, which
    is worse than leaving it obviously broken.
    """
    findings = V.check_factor_bounds(
        pd.DataFrame({'EARNINGS_YIELD': [0.05, 0.08, 899.0]},
                     index=['A', 'B', 'ERIE']))
    assert findings and findings[0].severity == 'error'
    assert findings[0].count == 1


def test_net_cash_is_allowed_but_absurd_leverage_is_not():
    """Negative net debt is a real state; 500x equity is a broken input."""
    ok = V.check_factor_bounds(
        pd.DataFrame({'NET_DEBT_TO_EQUITY': [-0.47, -0.01, 3.05]},
                     index=list('ABC')))
    assert ok == []

    bad = V.check_factor_bounds(
        pd.DataFrame({'NET_DEBT_TO_EQUITY': [561.0]}, index=['X']))
    assert bad and bad[0].severity == 'error'


def test_infinities_are_caught():
    findings = V.check_infinities(
        pd.DataFrame({'ROIC': [0.1, np.inf, 0.2]}, index=list('ABC')))
    assert findings and findings[0].severity == 'error'


def test_every_bounded_factor_is_a_real_factor():
    for name in V.BOUNDS:
        assert name in FA.FACTOR_DIRECTION or name == 'MARKET_CAP', \
            f'{name} has bounds but is not a factor'
