"""Run persistence, store clean-ups, backups and news identity."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from core import maintenance as M
from data import news as NEWS
from data.universe import UniverseSpec
from quant import engine as EN
from quant import strategies as ST


def _scores(tickers):
    return pd.DataFrame({
        'composite': [1.0 - i * 0.1 for i in range(len(tickers))],
        'rank': range(1, len(tickers) + 1),
        'signal': '🟢 Buy', 'signal_score': 3.0, 'reasons': 'x',
        'coverage': 1.0, 'sector': 'Information Technology',
    }, index=tickers)


def _detail(tickers):
    return pd.DataFrame([{'ticker': t, 'factor': 'ROIC', 'raw': 0.1,
                          'winsorized': 0.1, 'z': 0.5, 'sector_z': 0.5,
                          'pct_rank': 0.5} for t in tickers])


def _persist(run_id, as_of, spec=None, strategy='buffett'):
    spec = spec or UniverseSpec(preset='DIA')
    EN._persist(run_id, as_of, spec, ST.get(strategy), ['AAPL', 'MSFT'],
                _scores(['AAPL', 'MSFT']), _detail(['AAPL', 'MSFT']))


def _count(db, table):
    return int(db.read_sql(f'SELECT COUNT(*) n FROM {table}')['n'].iloc[0])


# ── persistence ──────────────────────────────────────────────────

def test_rerun_on_the_same_day_replaces_rather_than_accumulates(fresh_db):
    d = date(2026, 11, 6)
    _persist('run-a', d)
    _persist('run-b', d)
    runs = fresh_db.read_sql('SELECT run_id FROM runs')
    assert runs['run_id'].tolist() == ['run-b']
    assert _count(fresh_db, 'scores') == 2
    assert _count(fresh_db, 'factor_scores') == 2


def test_different_dates_and_universes_are_kept(fresh_db):
    _persist('a', date(2026, 11, 5))
    _persist('b', date(2026, 11, 6))
    _persist('c', date(2026, 11, 6), spec=UniverseSpec(preset='SPY'))
    _persist('d', date(2026, 11, 6), strategy='momentum_trend')
    assert _count(fresh_db, 'runs') == 4


def test_failed_persist_leaves_no_partial_run(fresh_db, monkeypatch):
    _persist('good', date(2026, 11, 6))
    real = fresh_db.upsert

    def boom(table, rows, *a, **kw):
        if table is fresh_db.factor_scores:
            raise RuntimeError('disk full')
        return real(table, rows, *a, **kw)

    monkeypatch.setattr(fresh_db, 'upsert', boom)
    with pytest.raises(RuntimeError):
        _persist('bad', date(2026, 11, 6))
    # Rolled back: the earlier run survives, nothing of the new one exists.
    assert fresh_db.read_sql('SELECT run_id FROM runs')['run_id'].tolist() == ['good']
    assert _count(fresh_db, 'scores') == 2


# ── clean-ups ────────────────────────────────────────────────────

def _legacy_duplicate_runs(db, n=3):
    """What the store looked like before runs were replaced on write."""
    spec = UniverseSpec(preset='DIA').to_json()
    for i in range(n):
        rid = f'old-{i}'
        db.upsert(db.runs, [{'run_id': rid, 'as_of': date(2026, 11, 6),
                             'strategy': 'buffett',
                             'universe_spec_json': spec,
                             'created_at': datetime(2026, 11, 6, 7 + i)}])
        db.upsert(db.scores, [{'run_id': rid, 'ticker': 'AAPL', 'rank': 1,
                               'composite': 1.0}])


def test_dedupe_keeps_only_the_newest_copy(fresh_db):
    _legacy_duplicate_runs(fresh_db, 3)
    assert M.consistency_report()['duplicate_runs'] == 2
    assert M.dedupe_runs() == 2
    left = fresh_db.read_sql('SELECT run_id FROM runs')['run_id'].tolist()
    assert left == ['old-2']
    assert M.consistency_report()['orphan_scores'] == 0


def test_orphans_and_empty_runs_are_removed(fresh_db):
    fresh_db.upsert(fresh_db.scores, [{'run_id': 'ghost', 'ticker': 'X', 'rank': 1}])
    fresh_db.upsert(fresh_db.runs, [{'run_id': 'empty', 'strategy': 's',
                                     'as_of': date(2026, 1, 2)}])
    out = M.delete_orphans()
    assert out['scores'] == 1 and out['empty_runs'] == 1


def test_news_dedupe_keeps_earliest_and_rekeys(fresh_db):
    t0 = datetime(2026, 11, 6, 12)
    rows = [{'id': f'legacy-{i}', 'ticker': 'AAPL',
             'title': 'Apple  raises guidance' if i else 'apple raises guidance',
             'published': t0 + timedelta(hours=3 * i)} for i in range(3)]
    fresh_db.upsert(fresh_db.news, rows)
    out = M.dedupe_news()
    assert out == {'deleted': 2, 'rekeyed': 1}
    left = fresh_db.read_sql('SELECT id, published FROM news', parse_dates=['published'])
    assert left['id'].tolist() == [NEWS._mk_id('AAPL', 'Apple raises guidance')]
    assert left['published'].iloc[0] == pd.Timestamp(t0)


def test_clean_is_idempotent(fresh_db):
    _legacy_duplicate_runs(fresh_db, 2)
    first = M.clean()
    assert first['duplicate_runs'] == 1
    second = M.clean()
    assert all(v == 0 for v in second.values())


def test_backup_produces_a_readable_copy(fresh_db, tmp_path):
    _persist('r1', date(2026, 11, 6))
    out = M.backup(dest_dir=tmp_path / 'bk')
    assert out is not None and out.exists()
    import sqlite3
    with sqlite3.connect(out) as conn:
        assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 1


def test_integrity_check_on_a_healthy_store(fresh_db):
    assert M.integrity_check() == ['ok']


def test_new_index_is_added_to_an_existing_database(fresh_db):
    fresh_db.execute('DROP INDEX ix_runs_lookup')
    fresh_db.init_db()
    names = fresh_db.read_sql(
        "SELECT name FROM sqlite_master WHERE type = 'index'")['name']
    assert 'ix_runs_lookup' in set(names)


# ── news identity ────────────────────────────────────────────────

def test_news_id_ignores_publish_time_and_spacing():
    a = NEWS._mk_id('AAPL', 'Apple Raises Guidance', datetime(2026, 1, 1))
    b = NEWS._mk_id('AAPL', '  apple raises   guidance ', datetime(2026, 1, 2))
    assert a == b
    assert a != NEWS._mk_id('MSFT', 'Apple Raises Guidance')


def test_refetched_article_keeps_its_original_timestamp(fresh_db, monkeypatch):
    first = datetime(2026, 11, 6, 9)
    row = lambda when: {'ticker': 'AAPL', 'title': 'Undated item', 'url': '',   # noqa: E731
                        'published': when, 'source': 'RSS', 'body': ''}
    monkeypatch.setattr(NEWS, '_from_yfinance', lambda t: [])
    monkeypatch.setattr(NEWS, '_from_rss', lambda *a, **k: [])

    monkeypatch.setattr(NEWS, 'fetch_for_ticker',
                        lambda t, n=None: [_article(row(first))])
    NEWS.update_news(['AAPL'])
    monkeypatch.setattr(NEWS, 'fetch_for_ticker',
                        lambda t, n=None: [_article(row(first + timedelta(hours=3)))])
    NEWS.update_news(['AAPL'])

    df = fresh_db.read_sql('SELECT published, extracted_json FROM news',
                           parse_dates=['published'])
    assert len(df) == 1
    assert df['published'].iloc[0] == pd.Timestamp(first)


def _article(it):
    return {'id': NEWS._mk_id(it['ticker'], it['title']), 'ticker': it['ticker'],
            'published': it['published'], 'source': it['source'],
            'title': it['title'], 'url': '', 'body_summary': '',
            'sentiment': 0.0, 'sentiment_label': 'neutral', 'events': '',
            'extracted_json': '{}', 'fetched_at': datetime.utcnow()}


def test_extraction_is_stored_as_json(monkeypatch):
    import json
    monkeypatch.setattr(NEWS, '_from_yfinance', lambda t: [{
        'ticker': 'AAPL', 'title': 'Apple raises guidance and announces buyback',
        'url': '', 'published': datetime.utcnow(), 'source': 'Yahoo',
        'body': 'Management raised full-year guidance.'}])
    monkeypatch.setattr(NEWS, '_from_rss', lambda *a, **k: [])
    rows = NEWS.fetch_for_ticker('AAPL')
    assert rows
    assert isinstance(json.loads(rows[0]['extracted_json']), dict)


def test_vacuum_leaves_pooled_connections_transactional(fresh_db):
    """After VACUUM, a failed multi-statement write must still roll back."""
    _legacy_duplicate_runs(fresh_db, 2)
    M.clean(vacuum=True)
    with pytest.raises(RuntimeError):
        with fresh_db.connect() as conn:
            conn.exec_driver_sql("DELETE FROM runs")
            raise RuntimeError('abort mid-transaction')
    assert _count(fresh_db, 'runs') == 1
