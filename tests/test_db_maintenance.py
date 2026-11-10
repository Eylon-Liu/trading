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


def test_backup_is_compressed_and_restores_to_a_readable_copy(fresh_db, tmp_path):
    _persist('r1', date(2026, 11, 6))
    out = M.backup(dest_dir=tmp_path / 'bk')
    assert out is not None and out.exists() and out.name.endswith('.sqlite.gz')
    assert not list((tmp_path / 'bk').glob('*.partial'))
    restored = M.restore_to(out, tmp_path / 'restored.sqlite')
    import sqlite3
    with sqlite3.connect(restored) as conn:
        assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 1


def test_only_the_newest_backups_are_kept(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setattr(M, 'BACKUP_DIR', tmp_path / 'backups')
    (tmp_path / 'backups').mkdir()
    for stamp in ('20260101-0700', '20260102-0700', '20260103-0700'):
        (tmp_path / 'backups' / f'quant-{stamp}.sqlite').write_bytes(b'old plain copy')
    newest = M.backup(keep=2)
    names = [p.name for p in M.list_backups()]
    assert names[0] == newest.name and len(names) == 2
    assert 'quant-20260103-0700.sqlite' in names           # plain copies still count


# ── retention: disk use levels off ───────────────────────────────

def test_factor_detail_is_pruned_but_scores_are_kept(fresh_db):
    old = date.today() - timedelta(days=45)
    _persist('old', old)
    _persist('new', date.today())
    assert M.prune_factor_detail(30) == 2
    assert _count(fresh_db, 'factor_scores') == 2           # today's detail
    assert _count(fresh_db, 'scores') == 4                  # both rankings


def test_old_daily_runs_thin_to_one_per_week(fresh_db):
    # Two full weeks of daily runs, 200+ days ago, and one recent run.
    monday = date.today() - timedelta(days=210)
    monday -= timedelta(days=monday.weekday())
    for i in range(10):
        d = monday + timedelta(days=i + 2 * (i // 5))       # Mon-Fri, Mon-Fri
        _persist(f'old-{i}', d)
    _persist('other-strategy', monday, strategy='momentum_trend')
    _persist('recent', date.today())
    removed = M.thin_old_runs(120)
    assert removed == 8
    left = fresh_db.read_sql('SELECT run_id, as_of FROM runs ORDER BY as_of')
    assert set(left['run_id']) == {'old-4', 'old-9', 'other-strategy', 'recent'}
    assert _count(fresh_db, 'scores') == 8                  # whole runs went, none partial
    assert not any(M.delete_orphans().values())
    assert M.thin_old_runs(120) == 0                        # idempotent


def test_old_news_is_pruned(fresh_db):
    now = datetime.utcnow()
    fresh_db.upsert(fresh_db.news, [
        {'id': 'old', 'ticker': 'AAPL', 'published': now - timedelta(days=200), 'title': 'old'},
        {'id': 'new', 'ticker': 'AAPL', 'published': now - timedelta(days=3), 'title': 'new'}])
    assert M.prune_news(180) == 1
    assert fresh_db.read_sql('SELECT id FROM news')['id'].tolist() == ['new']


def test_cache_is_compressed_and_stale_entries_swept(tmp_path, monkeypatch):
    import gzip
    import os
    import time
    import config
    from core import http
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path)
    payload = b'{"facts": ' + b'"x" ' * 20000 + b'}'
    path = http._cache_path('https://example.test/a', None, 'sec_facts')
    http._cache_write(path, payload)
    assert path.stat().st_size < len(payload) / 5           # stored compressed
    assert http._cache_read(path, ttl=60) == payload         # read back intact

    legacy = tmp_path / 'sec_facts' / 'legacy.cache'         # written before compression
    legacy.write_bytes(payload)
    stale = tmp_path / 'prices' / 'stale.cache'
    stale.parent.mkdir()
    stale.write_bytes(b'x' * 5000)
    old = time.time() - 40 * 86400
    os.utime(stale, (old, old))
    in_flight = tmp_path / 'prices' / 'other.tmp'            # another process, mid-write
    in_flight.write_bytes(b'partial')
    mtime_before = legacy.stat().st_mtime

    res = http.sweep_cache(max_age_days=30)
    assert res['removed'] == 1 and not stale.exists()
    assert res['compressed'] == 1 and res['bytes_freed'] > len(payload) / 2
    assert gzip.decompress(legacy.read_bytes()) == payload
    assert http._cache_read(legacy, ttl=-1) == payload
    assert abs(legacy.stat().st_mtime - mtime_before) < 1    # its age is unchanged
    assert in_flight.exists()
    assert http.sweep_cache(max_age_days=30)['compressed'] == 0


def test_old_reports_and_logs_are_pruned(fresh_db, tmp_path, monkeypatch):
    import os
    import time
    import config
    monkeypatch.setattr(config, 'REPORT_DIR', tmp_path / 'out')
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    (tmp_path / 'out').mkdir()
    (tmp_path / 'logs' / 'jobs').mkdir(parents=True)
    old_day = date.today() - timedelta(days=200)
    rows = []
    for rid, kind, day in (('d-old', 'daily', old_day), ('m-old', 'monthly', old_day),
                           ('d-new', 'daily', date.today())):
        f = tmp_path / 'out' / f'{rid}.html'
        f.write_text('<html></html>')
        rows.append({'report_id': rid, 'kind': kind, 'period_start': day,
                     'period_end': day, 'path': str(f)})
    fresh_db.upsert(fresh_db.reports, rows)
    assert M.prune_reports(180) == 1
    assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == ['d-new.html', 'm-old.html']
    assert set(fresh_db.read_sql('SELECT report_id FROM reports')['report_id']) == {'m-old', 'd-new'}

    old_log = tmp_path / 'logs' / 'routine_20260101_0700_morning.log'
    old_log.write_text('x')
    past = time.time() - 40 * 86400
    os.utime(old_log, (past, past))
    new_log = tmp_path / 'logs' / 'jobs' / 'today.log'
    new_log.write_text('x')
    big = tmp_path / 'logs' / 'daily.log'
    big.write_bytes(b'line\n' * 1_400_000)                   # 7 MB of cron appends
    assert M.prune_logs(30) == 2
    assert not old_log.exists() and new_log.exists()
    assert 0.9e6 < big.stat().st_size <= 1e6 and big.read_bytes().startswith(b'line\n')


def test_housekeeping_runs_every_step_and_survives_one_failing(fresh_db, tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'CACHE_DIR', tmp_path / 'cache')
    monkeypatch.setattr(M, 'BACKUP_DIR', tmp_path / 'backups')
    monkeypatch.setattr(M, 'prune_news', lambda days: 1 / 0)
    _persist('old', date.today() - timedelta(days=45))
    said = []
    out = M.housekeeping(say=said.append)
    assert out['factor_detail_rows'] == 2
    assert str(out['news_rows']).startswith('failed')
    assert 'usage' in out and said and said[0].startswith('🧹 storage')


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


def test_events_from_8k_amendments_are_removed(fresh_db):
    fresh_db.upsert(fresh_db.sec_filings, [
        {'accession': 'orig', 'ticker': 'AAA', 'form': '8-K', 'filing_date': date(2026, 10, 22)},
        {'accession': 'amend', 'ticker': 'AAA', 'form': '8-K/A', 'filing_date': date(2026, 11, 20)}])
    fresh_db.upsert(fresh_db.corporate_events, [
        {'ticker': 'AAA', 'filed': date(2026, 10, 22), 'accession': 'orig',
         'item_code': '2.02', 'category': 'results'},
        {'ticker': 'AAA', 'filed': date(2026, 11, 20), 'accession': 'amend',
         'item_code': '2.02', 'category': 'results'}])
    assert M.delete_amended_events() == 1
    left = fresh_db.read_sql('SELECT accession FROM corporate_events')
    assert left['accession'].tolist() == ['orig']
