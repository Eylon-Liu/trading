"""
End-to-end integrity checks on a synthetic store.

These run the real engine and backtests against a small SQLite database built
here, so every rule below is enforced by the code path that produces numbers,
not by a unit test of one helper:

  * future-perturbation invariance — a score dated D is identical whether or
    not anything dated after D exists. Prices, filings, restatements, splits,
    insider trades, 8-K events and pageviews after D are all rewritten in a
    second store, and every built-in strategy must rank the same way. Any
    look-ahead anywhere in the pipeline breaks this.
  * live / backtest equivalence — the backtest's preloaded, incremental path
    gives exactly what a fresh live screen gives on the same date.
  * execution rules — fills strictly after the scan, no double-holding, no
    leverage, avoid lists never traded long.
"""

from __future__ import annotations

import contextlib
import importlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import config
from quant import factorcache

CUTOFF = date(2024, 6, 14)                   # a Friday, mid-month
TICKERS = [f'T{i:02d}' for i in range(23)] + ['STALE']
SECTORS = ['Industrials', 'Energy', 'Health Care', 'Financials']
SHARES = {t: 1e8 * (1 + i % 5) for i, t in enumerate(TICKERS)}
# T07 reports a cover-page share count half the EPS-implied one, so the
# crosscheck rejects it — unless something switches the crosscheck off.
MISREPORTED = 'T07'
# STALE stops filing in mid-2022. Its share count goes stale 501 days after
# the last filing, with no new filing to trigger a rebuild.
STALE_LAST_PERIOD = pd.Timestamp('2022-06-30')


class Spec:
    """Minimal universe spec: a fixed list, no size bounds."""
    preset = None
    min_market_cap = None
    max_market_cap = None

    def resolve(self, as_of):
        return list(TICKERS)

    def describe(self):
        return 'synthetic'


# ─────────────────────────────────────────────
# SYNTHETIC STORE
# ─────────────────────────────────────────────

def _rows(perturb_after: date | None):
    """
    Every table's rows. With `perturb_after`, everything dated after it is
    regenerated from a different seed and extra future-only rows are added;
    rows dated on or before it are identical to the unperturbed store.
    """
    cut = pd.Timestamp(perturb_after) if perturb_after else pd.Timestamp.max
    base, alt = np.random.default_rng(1), np.random.default_rng(99)

    def rng_for(i: int, d, after: bool):
        # Keyed per row, so changing a post-cutoff row cannot shift the
        # random draws of any pre-cutoff row.
        return np.random.default_rng([99 if after else 1, i, pd.Timestamp(d).toordinal()])

    days = pd.bdate_range('2020-01-01', '2025-12-31')
    prices, split_rows = [], []
    for i, t in enumerate(TICKERS + ['SPY']):
        r = base.normal(0.0003 + 0.00005 * (i % 7), 0.015, len(days))
        r_alt = alt.normal(-0.002, 0.04, len(days))
        r = np.where(days > cut, r_alt, r)
        close = 40 * np.exp(np.cumsum(r))
        for d, c in zip(days, close):
            prices.append({'ticker': t, 'date': d.date(), 'open': c,
                           'high': c * 1.012, 'low': c * 0.988, 'close': c,
                           'adj_close': c, 'volume': 2e6})

    if perturb_after:
        # A 2-for-1 split after the cutoff for the misreported filer. Prices
        # before it are restated, exactly as the provider would.
        split_day = pd.Timestamp(perturb_after) + pd.Timedelta(days=60)
        split_rows.append({'ticker': MISREPORTED, 'date': split_day.date(),
                           'ratio': 2.0})
        for row in prices:
            if row['ticker'] == MISREPORTED and pd.Timestamp(row['date']) < split_day:
                for k in ('open', 'high', 'low', 'close', 'adj_close'):
                    row[k] /= 2.0

    facts = []
    q_ends = pd.date_range('2018-03-31', '2025-09-30', freq='QE')
    for i, t in enumerate(TICKERS):
        rev0, margin = 1e9 * (1 + i % 6), 0.05 + 0.01 * (i % 9)
        for k, end in enumerate(q_ends):
            if t == 'STALE' and end > STALE_LAST_PERIOD:
                break
            q4 = end.month == 12
            filed = end + pd.Timedelta(days=55 if q4 else 35)
            rng = rng_for(i, filed, filed > cut)
            rev = rev0 * (1.02 ** k) * (1 + rng.normal(0, 0.03))
            ni = rev * margin * (1 + rng.normal(0, 0.4))
            start = end - pd.offsets.QuarterBegin(startingMonth=1)
            shares = SHARES[t]
            reported = shares * (0.5 if t == MISREPORTED else 1.0)
            flows = {'revenue': rev, 'net_income': ni,
                     'operating_income': ni * 1.3, 'gross_profit': rev * 0.4,
                     'operating_cash_flow': ni * 1.2, 'capex': -rev * 0.05,
                     'eps_diluted': ni / shares,
                     'dividends_paid': -abs(ni) * 0.03 * (i % 4),
                     'buybacks': -abs(ni) * 0.02 * (i % 3 + 1)}
            stocks = {'assets': rev * 6, 'equity': rev * 2.5, 'cash': rev * 0.4,
                      'long_term_debt': rev * 1.5, 'current_assets': rev * 1.2,
                      'current_liabilities': rev * 0.8,
                      'shares_outstanding': reported, 'shares_diluted': reported}
            common = {'cik': f'{i:010d}', 'unit': 'USD', 'period_end': end.date(),
                      'filed': filed.date(), 'form': '10-K' if q4 else '10-Q',
                      'ticker': t, 'fy': end.year, 'fp': 'FY' if q4 else f'Q{end.quarter}'}
            for c, v in flows.items():
                facts.append({**common, 'tag': c, 'concept': c, 'val': float(v),
                              'period_start': start.date()})
            for c, v in stocks.items():
                facts.append({**common, 'tag': c, 'concept': c, 'val': float(v),
                              'period_start': end.date()})
    if perturb_after:
        # A restatement of a quarter that ended before the cutoff, filed after
        # it. The as-of view must keep the original figure.
        for f in facts:
            if (f['ticker'] == 'T01' and f['concept'] == 'net_income'
                    and f['period_end'] == date(2024, 3, 31)):
                facts.append({**f, 'filed': (CUTOFF + timedelta(days=20)),
                              'val': f['val'] * -5, 'form': '10-Q/A'})
                break

    insiders, events, attention = [], [], []
    for i, t in enumerate(TICKERS):
        for d in pd.date_range('2022-01-03', '2025-12-31', freq='17D'):
            rng = rng_for(i, d, d > cut)
            if rng.random() < 0.5:
                continue
            buy = rng.random() < 0.4
            n = float(rng.integers(500, 20000))
            insiders.append({
                'accession': f'{t}-{d.date()}', 'ticker': t,
                'insider': f'Person {int(rng.integers(0, 4))}',
                'txn_date': (d - pd.Timedelta(days=2)).date(),
                'txn_code': 'P' if buy else 'S', 'cik': f'{i:010d}',
                'filed': d.date(), 'role': 'Director',
                'shares': n if buy else -n, 'price': 50.0, 'value': n * 50,
                'post_txn_shares': float(rng.integers(10000, 200000))})
        for end in q_ends:
            filed = end + pd.Timedelta(days=30)
            events.append({'ticker': t, 'filed': filed.date(),
                           'accession': f'{t}-r-{end.date()}', 'item_code': '2.02',
                           'category': 'results', 'subtype': None, 'detail': None})
            rng = rng_for(i + 1000, filed, filed > cut)
            if rng.random() < 0.15:
                events.append({'ticker': t, 'filed': filed.date(),
                               'accession': f'{t}-m-{end.date()}', 'item_code': '5.02',
                               'category': 'management', 'subtype': None, 'detail': None})
            if rng.random() > 0.9:
                events.append({'ticker': t, 'filed': filed.date(),
                               'accession': f'{t}-x-{end.date()}', 'item_code': '2.05',
                               'category': 'restructuring', 'subtype': None, 'detail': None})
        for d in pd.date_range('2023-01-01', '2025-12-31'):
            # The as-of day itself is perturbed too: its count is published
            # only after it ends.
            rng = rng_for(i + 2000, d, d >= cut)
            attention.append({'ticker': t, 'date': d.date(),
                              'wiki_views': float(1000 + rng.normal(0, 100) * (1 + i % 3))})

    securities = [{'ticker': t, 'name': t, 'sector': SECTORS[i % 4]}
                  for i, t in enumerate(TICKERS)]
    profiles = []
    if perturb_after:
        profiles = [{'ticker': t, 'snapshot_date': (CUTOFF + timedelta(days=5)),
                     'market_cap': 1e6, 'forward_pe': 1.0, 'trailing_pe': 1.0}
                    for t in TICKERS]
    return {'prices': prices, 'splits': split_rows, 'sec_facts': facts,
            'insider_txns': insiders, 'corporate_events': events,
            'attention': attention, 'securities': securities,
            'profile_snapshots': profiles}


# Which scoring path is running. The factor cache is keyed on it as well, so
# the live and backtest paths can never be served each other's frames - that
# would make the equivalence tests pass by construction.
_MODE = ['live']


@contextlib.contextmanager
def _use_store(path, cache: bool = True):
    """
    Point the whole app at `path`. The disk cache is replaced by a per-store,
    per-path in-memory one (or disabled), so nothing leaks between stores.
    """
    memo: dict = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, 'DATABASE_URL', f'sqlite:///{path}')
        if cache:
            mp.setattr(factorcache, 'load', lambda key: memo.get((_MODE[0], key)))
            mp.setattr(factorcache, 'save',
                       lambda key, df: memo.__setitem__((_MODE[0], key), df))
        else:
            mp.setattr(factorcache, 'load', lambda key: None)
            mp.setattr(factorcache, 'save', lambda key, df: None)
        from core import db
        importlib.reload(db)
        try:
            yield db
        finally:
            db.get_engine().dispose()
            mp.undo()
            importlib.reload(db)


def _build(path, perturb_after=None):
    with _use_store(path) as db:
        db.init_db()
        for name, rows in _rows(perturb_after).items():
            if rows:
                db.upsert(db.metadata.tables[name], rows)


@pytest.fixture(scope='module')
def stores(tmp_path_factory):
    d = tmp_path_factory.mktemp('stores')
    base, perturbed = d / 'base.sqlite', d / 'perturbed.sqlite'
    _build(base)
    _build(perturbed, perturb_after=CUTOFF)
    return base, perturbed


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

COMPARE = ['composite', 'rank', 'signal']


def _live(strategy, as_of):
    from quant import engine as EN
    _MODE[0] = 'live'
    return EN.run(Spec(), strategy, as_of=as_of, top_n=10, persist=False)


def _bt(strategy, dates, end):
    """Walk the backtest scoring path over `dates`, returning each result."""
    from quant import backtest as BT
    _MODE[0] = 'bt'
    try:
        idx = pd.DatetimeIndex(dates)
        scorer = BT._Scorer(Spec(), strategy, BT._preload(Spec(), idx[0].date(), end))
        return [scorer.run(d.date(), 10) for d in idx]
    finally:
        _MODE[0] = 'live'


def _same(a: pd.DataFrame, b: pd.DataFrame, cols=COMPARE, what=''):
    assert not a.empty, f'{what}: produced no scores'
    cols = [c for c in cols if c in a.columns]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=False,
                                  rtol=1e-9, obj=what)


def _strategies():
    from quant import strategies as ST
    return sorted(ST.ALL_STRATEGIES)


# ─────────────────────────────────────────────
# LOOK-AHEAD: FUTURE-PERTURBATION INVARIANCE
# ─────────────────────────────────────────────

@pytest.fixture(scope='module')
def scored_at_cutoff(stores):
    """Every strategy at CUTOFF, live and backtest path, in both stores."""
    out = {}
    for label, path in zip(('base', 'perturbed'), stores):
        with _use_store(path):
            for key in _strategies():
                live = _live(key, CUTOFF)
                bt = _bt(key, [CUTOFF], CUTOFF + timedelta(days=200))[0]
                out[(label, 'live', key)] = live
                out[(label, 'bt', key)] = bt
    return out


@pytest.mark.parametrize('key', _strategies())
def test_live_screen_ignores_everything_after_as_of(scored_at_cutoff, key):
    _same(scored_at_cutoff[('base', 'live', key)].scores,
          scored_at_cutoff[('perturbed', 'live', key)].scores,
          what=f'{key} live: data after {CUTOFF} changed the score')


@pytest.mark.parametrize('key', _strategies())
def test_backtest_path_ignores_preloaded_future_rows(scored_at_cutoff, key):
    """The preload holds rows past the scan date in memory; none may leak."""
    _same(scored_at_cutoff[('base', 'bt', key)].scores,
          scored_at_cutoff[('perturbed', 'bt', key)].scores,
          what=f'{key} backtest: preloaded future rows changed the score')


@pytest.mark.parametrize('key', _strategies())
def test_backtest_scores_equal_live_scores(scored_at_cutoff, key):
    """'The backtest calls the same scoring path' has to be literally true."""
    live = scored_at_cutoff[('base', 'live', key)]
    bt = scored_at_cutoff[('base', 'bt', key)]
    _same(live.scores, bt.scores,
          cols=COMPARE + ['MARKET_CAP', 'RSI_14', 'ATR_PCT', 'VOL_1Y'],
          what=f'{key}: backtest path differs from live screen')
    if not live.plans.empty:
        _same(live.plans, bt.plans, cols=['entry', 'stop', 'target', 'weight_pct'],
              what=f'{key}: plans differ')


def test_canary_store_really_differs_after_the_cutoff(stores):
    """Guard the guard: the perturbation must change later answers."""
    later = CUTOFF + timedelta(days=120)
    results = []
    for path in stores:
        with _use_store(path):
            results.append(_live('momentum_trend', later).scores)
    assert not results[0]['composite'].equals(results[1]['composite'])


def test_misreported_share_count_stays_rejected_despite_future_split(scored_at_cutoff):
    """A split after as_of must not switch off the share-count crosscheck."""
    for label in ('base', 'perturbed'):
        s = scored_at_cutoff[(label, 'live', 'buffett')].scores
        if MISREPORTED in s.index:
            assert pd.isna(s.loc[MISREPORTED, 'MARKET_CAP']), label


# ─────────────────────────────────────────────
# INCREMENTAL CARRY == FULL REBUILD
# ─────────────────────────────────────────────

def test_incremental_backtest_matches_fresh_rebuild_across_dates(stores):
    """
    The backtest reuses fundamentals for tickers with no new filing. Rows
    whose answer depends on the date — a share count going stale, a split
    entering or leaving the crosscheck window — must still be rebuilt, or the
    backtest and the live screen disagree. STALE's share count expires on
    2023-12-24 with no filing to trigger anything.
    """
    dates = pd.date_range('2023-06-30', '2024-06-30', freq='QE')
    with _use_store(stores[0], cache=False):      # a hit would skip the carry
        walked = _bt('buffett', dates, date(2024, 6, 30))
        for d, bt in zip(dates, walked):
            live = _live('buffett', d.date())
            _same(live.scores, bt.scores, cols=COMPARE + ['MARKET_CAP'],
                  what=f'incremental carry differs at {d.date()}')
        stale = [bt.scores.loc['STALE', 'MARKET_CAP'] if 'STALE' in bt.scores.index
                 else np.nan for bt in walked]
    assert np.isfinite(stale[0]) and not np.isfinite(stale[-1]), \
        'fixture precondition: STALE must lose its market cap inside the window'


# ─────────────────────────────────────────────
# EXECUTION RULES ON A REAL WALK
# ─────────────────────────────────────────────

@pytest.fixture(scope='module', params=['momentum_trend', 'breakout'])
def mid_run(stores, request):
    """One limit-entry strategy and one market-entry strategy."""
    from quant import backtest as BT
    with _use_store(stores[0]):
        return BT.run_trade_backtest(Spec(), request.param, '2023-01-01',
                                     '2025-06-30', n_positions=6, strict=False)


def test_every_fill_is_after_its_scan(mid_run):
    t = mid_run.trades
    assert len(t) > 10, 'fixture should generate trades'
    assert (pd.to_datetime(t['entry_date']) > pd.to_datetime(t['signal_date'])).all()
    assert (pd.to_datetime(t['exit_date']) >= pd.to_datetime(t['entry_date'])).all()


def test_book_never_exceeds_full_exposure_or_double_holds(mid_run):
    t = mid_run.trades
    for day in pd.to_datetime(t['entry_date']).unique():
        live = t[(pd.to_datetime(t['entry_date']) <= day)
                 & (pd.to_datetime(t['exit_date']) > day)]
        assert live['weight'].sum() <= 1.0 + 1e-9, f'leverage on {day.date()}'
        assert not live['ticker'].duplicated().any(), f'double hold on {day.date()}'


def test_long_backtest_trades_after_each_rebalance_date(stores):
    from quant import backtest as BT
    with _use_store(stores[0]):
        r = BT.run_backtest(Spec(), 'quality_momentum', '2023-01-01', '2024-12-31',
                            n_holdings=6, rebalance='Q', strict=False)
    assert r.equity.index[0] > pd.Timestamp(r.rebalance_dates[0])
    assert r.equity.index.is_monotonic_increasing
    assert 'mean_ic' in r.metrics


def test_preload_carries_the_benchmark(stores):
    """Without it BETA and IDIO_VOL silently vanish from every backtest."""
    from quant import backtest as BT
    with _use_store(stores[0]):
        pre = BT._preload(Spec(), date(2024, 1, 1), date(2024, 3, 1))
        assert config.BENCHMARK_TICKER in pre.ohlc[0].columns
        r = BT._Scorer(Spec(), 'relative_strength', pre).run(date(2024, 1, 31), 10)
    assert not any('IDIO_VOL' in w for w in r.warnings)


def test_avoid_list_cannot_be_trade_backtested():
    from quant import backtest as BT
    with pytest.raises(ValueError, match='avoid list'):
        BT.run_trade_backtest(Spec(), 'quality_breakdown', '2023-01-01', '2023-06-30')


# ─────────────────────────────────────────────
# STORE-LEVEL DATA CHECKS
# ─────────────────────────────────────────────

def test_insider_reads_carry_holdings_for_sell_weighting(stores):
    """Without post_txn_shares the sell-conviction weighting never runs."""
    from data import sec
    with _use_store(stores[0]):
        ins = sec.insiders_asof(TICKERS, CUTOFF)
    assert 'post_txn_shares' in ins.columns and ins['post_txn_shares'].notna().any()


def test_data_version_moves_on_every_factor_source(tmp_path):
    """A cache key blind to a source serves stale factors after its ingest."""
    with _use_store(tmp_path / 'v.sqlite') as db:
        db.init_db()
        seen = {factorcache.data_version()}
        inserts = [
            (db.corporate_events, {'ticker': 'A', 'filed': date(2024, 1, 2),
                                   'accession': 'x', 'item_code': '2.02',
                                   'category': 'results'}),
            (db.attention, {'ticker': 'A', 'date': date(2024, 1, 2), 'wiki_views': 5.0}),
            (db.insider_txns, {'accession': 'y', 'ticker': 'A', 'insider': 'P',
                               'txn_date': date(2024, 1, 2), 'txn_code': 'P'}),
            (db.splits, {'ticker': 'A', 'date': date(2024, 1, 2), 'ratio': 2.0}),
        ]
        for table, row in inserts:
            db.upsert(table, [row])
            v = factorcache.data_version()
            assert v not in seen, f'inserting into {table.name} did not move the version'
            seen.add(v)


def test_every_strategy_factor_varies_across_the_universe(stores):
    """
    A weighted factor that is the same for every name ranks nothing and only
    dilutes the others. EVENT_RESULTS (one results filing a quarter, for
    everyone) was exactly that while it anchored the PEAD strategy.
    """
    from quant import factors as FA
    from quant import strategies as ST
    with _use_store(stores[0]):
        raw = FA.build_all(TICKERS, CUTOFF, use_cache=False)
    for key, strat in ST.ALL_STRATEGIES.items():
        for f in strat.weights:
            if f not in raw or raw[f].notna().sum() < 5:
                continue                     # absent data is reported elsewhere
            assert raw[f].dropna().nunique() > 1, f'{key}: {f} is constant'
