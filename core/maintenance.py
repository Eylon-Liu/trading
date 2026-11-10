"""
Database upkeep: backups, integrity checks and one-off clean-ups.

The store is the one thing in this project that cannot be rebuilt for free —
index-membership history, profile snapshots and news accumulate forward and
can never be backfilled — so it is worth a nightly copy and an occasional
check. Everything here is safe to run repeatedly; nothing deletes a row that
any read path could still return.

    python cli.py db status
    python cli.py db backup
    python cli.py db check
    python cli.py db clean            # dedupe + orphans; --vacuum to shrink
    python cli.py db tidy             # apply retention now (config.RETENTION)
    python cli.py db restore          # unpack the newest backup next to the live file

`housekeeping()` runs at the end of every routine so the footprint levels
off: per-factor detail, old daily runs, old news, stale cache entries, old
logs and report files are pruned on the schedule in config.RETENTION.
Prices, filings, fundamentals and index membership are never touched.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text

import config
from core import db

log = logging.getLogger(__name__)

BACKUP_DIR = config.DB_DIR / 'backups'


# ─────────────────────────────────────────────
# LOCATION
# ─────────────────────────────────────────────

def sqlite_path() -> Path | None:
    """The database file, or None when the store is not a local SQLite file."""
    url = config.DATABASE_URL
    if not url.startswith('sqlite:///'):
        return None
    raw = url[len('sqlite:///'):].split('?', 1)[0]
    if not raw or raw == ':memory:':
        return None
    return Path(raw)


def status() -> dict:
    """File size, WAL size, row counts and the newest backup."""
    path = sqlite_path()
    info: dict = {'url': config.DATABASE_URL, 'path': str(path) if path else None}
    if path and path.exists():
        info['size_mb'] = round(path.stat().st_size / 1e6, 1)
        wal = path.with_name(path.name + '-wal')
        info['wal_mb'] = round(wal.stat().st_size / 1e6, 1) if wal.exists() else 0.0
    backups = list_backups()
    info['backups'] = len(backups)
    info['latest_backup'] = str(backups[0]) if backups else None
    info['tables'] = db.table_counts()
    return info


# ─────────────────────────────────────────────
# BACKUP
# ─────────────────────────────────────────────

def list_backups() -> list[Path]:
    """Newest first; both compressed (.sqlite.gz) and older plain copies."""
    if not BACKUP_DIR.exists():
        return []
    found = list(BACKUP_DIR.glob('quant-*.sqlite.gz')) + list(BACKUP_DIR.glob('quant-*.sqlite'))
    return sorted(found, key=lambda p: p.name.split('.sqlite')[0], reverse=True)


def restore_to(backup_path: Path, dest: Path) -> Path:
    """Unpack a backup (compressed or plain) to `dest`. Does not touch the live file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.suffix == '.gz':
        with gzip.open(backup_path, 'rb') as src, open(dest, 'wb') as out:
            shutil.copyfileobj(src, out, 1 << 20)
    else:
        shutil.copyfile(backup_path, dest)
    return dest


def backup(keep: int | None = None, dest_dir: Path | None = None,
           compress: bool = True) -> Path | None:
    """
    Copy the live database to db/backups/quant-YYYYmmdd-HHMM.sqlite.gz.

    Uses SQLite's online backup API, which takes a consistent snapshot even
    while the UI or an ingest is writing — a plain file copy of a WAL-mode
    database can capture a half-applied transaction. The copy is gzipped
    (about a quarter of the size) and the newest `keep` are kept: seven
    uncompressed copies of a 0.8 GB store had been the largest thing on disk.
    Restore with `python cli.py db restore`.
    """
    keep = config.RETENTION['backups'] if keep is None else keep
    src = sqlite_path()
    if src is None or not src.exists():
        log.warning('backup skipped: %s is not a local SQLite file',
                    config.DATABASE_URL)
        return None

    out_dir = dest_dir or BACKUP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f'quant-{datetime.now():%Y%m%d-%H%M}.sqlite'
    tmp = dest.with_suffix('.partial')

    s = sqlite3.connect(str(src), timeout=60)
    d = sqlite3.connect(str(tmp))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    if compress:
        packed = dest.with_name(dest.name + '.gz')
        part = packed.with_suffix('.partial')
        with open(tmp, 'rb') as raw, gzip.open(part, 'wb', compresslevel=3) as out:
            shutil.copyfileobj(raw, out, 1 << 20)
        tmp.unlink()
        part.replace(packed)
        dest = packed
    else:
        tmp.replace(dest)

    if keep and dest_dir is None:
        for old in list_backups()[keep:]:
            try:
                old.unlink()
            except OSError as exc:
                log.warning('could not remove old backup %s: %s', old, exc)
    log.info('backup written: %s', dest)
    return dest


# ─────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────

def integrity_check(full: bool = False) -> list[str]:
    """['ok'] when healthy; otherwise SQLite's list of problems."""
    if db.get_engine().dialect.name != 'sqlite':
        return ['ok']
    pragma = 'integrity_check' if full else 'quick_check'
    with db.get_engine().connect() as conn:
        return [r[0] for r in conn.execute(text(f'PRAGMA {pragma}'))]


def consistency_report() -> dict[str, int]:
    """
    Counts of rows that should not exist. All zeros on a healthy store.

      duplicate_runs       older copies of a (strategy, date, universe) run
      orphan_scores        score / factor rows whose run is gone
      empty_runs           runs that stored no scores at all
      duplicate_news       the same headline stored more than once per ticker
      duplicate_reports    more than one history row for the same report file
    """
    q = lambda sql: int(db.read_sql(sql).iloc[0, 0] or 0)   # noqa: E731
    return {
        'duplicate_runs': q("""
            SELECT COALESCE(SUM(n - 1), 0) FROM (
              SELECT COUNT(*) AS n FROM runs
              GROUP BY strategy, as_of, universe_spec_json HAVING n > 1)"""),
        'orphan_scores': q("""
            SELECT COUNT(*) FROM scores s
            WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = s.run_id)"""),
        'orphan_factor_scores': q("""
            SELECT COUNT(*) FROM factor_scores f
            WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = f.run_id)"""),
        'empty_runs': q("""
            SELECT COUNT(*) FROM runs r
            WHERE NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = r.run_id)"""),
        'duplicate_news': len(_news_plan()[0]),
        'duplicate_reports': q("""
            SELECT COALESCE(SUM(n - 1), 0) FROM (
              SELECT COUNT(*) AS n FROM reports WHERE path IS NOT NULL
              GROUP BY path HAVING n > 1)"""),
    }


# ─────────────────────────────────────────────
# CLEAN-UPS
# ─────────────────────────────────────────────

def dedupe_runs() -> int:
    """
    Keep only the newest run per (strategy, as-of date, universe).

    Before runs were replaced on write, every scheduled screen added another
    full copy. Only the newest copy was ever read, so the older ones are dead
    weight — usually most of the database by size.
    """
    from quant.engine import delete_runs

    runs = db.read_sql("""
        SELECT run_id, strategy, as_of, universe_spec_json, created_at
        FROM runs ORDER BY created_at DESC
    """)
    if runs.empty:
        return 0
    dup = runs.duplicated(subset=['strategy', 'as_of', 'universe_spec_json'],
                          keep='first')
    victims = runs.loc[dup, 'run_id'].tolist()
    with db.connect() as conn:
        for i in range(0, len(victims), 500):
            delete_runs(conn, victims[i:i + 500])
    return len(victims)


def delete_orphans() -> dict[str, int]:
    """Scores and factor rows whose run no longer exists; runs with no scores."""
    out = {}
    out['factor_scores'] = db.execute("""
        DELETE FROM factor_scores
        WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = factor_scores.run_id)""")
    out['scores'] = db.execute("""
        DELETE FROM scores
        WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.run_id = scores.run_id)""")
    out['empty_runs'] = db.execute("""
        DELETE FROM runs
        WHERE NOT EXISTS (SELECT 1 FROM scores s WHERE s.run_id = runs.run_id)""")
    return out


def _news_plan() -> tuple[list[str], list[tuple[str, str]]]:
    """
    (ids to delete, (old_id, new_id) pairs to re-key) so that each headline
    is stored once per ticker, under the stable id scheme.
    """
    from data.news import _mk_id

    df = db.read_sql('SELECT id, ticker, title, published FROM news',
                     parse_dates=['published'])
    if df.empty:
        return [], []
    df['new_id'] = [_mk_id(t, ti) for t, ti in zip(df['ticker'], df['title'])]
    # Keep the earliest sighting: that is when the article was first public.
    df = df.sort_values(['new_id', 'published'], kind='stable')
    dup = df.duplicated('new_id', keep='first')
    keep = df[~dup]
    rekey = keep[keep['id'] != keep['new_id']]
    return df.loc[dup, 'id'].tolist(), list(zip(rekey['id'], rekey['new_id']))


def dedupe_news() -> dict[str, int]:
    """
    Collapse repeated headlines and move survivors onto the stable id scheme.

    Article ids used to include the publish time, and undated items were
    stamped with the fetch time, so the same article was stored again on
    every refresh. Survivors are re-keyed so future fetches recognise them.
    """
    drop, rekey = _news_plan()
    with db.connect() as conn:
        for i in range(0, len(drop), 500):
            conn.execute(text('DELETE FROM news WHERE id IN :ids')
                         .bindparams(bindparam('ids', expanding=True)),
                         {'ids': drop[i:i + 500]})
        for old, new in rekey:
            conn.execute(text('UPDATE news SET id = :new WHERE id = :old'),
                         {'new': new, 'old': old})
    return {'deleted': len(drop), 'rekeyed': len(rekey)}


def delete_amended_events() -> int:
    """Corporate events derived from 8-K/A amendments (see data.sec)."""
    return db.execute("""
        DELETE FROM corporate_events WHERE accession IN (
            SELECT accession FROM sec_filings WHERE form != '8-K')""")


def dedupe_reports() -> int:
    """One history row per report file, keeping the newest and its email stamp."""
    rows = db.read_sql("""
        SELECT report_id, path, emailed_at, created_at FROM reports
        WHERE path IS NOT NULL ORDER BY created_at DESC
    """, parse_dates=['emailed_at', 'created_at'])
    if rows.empty:
        return 0
    removed = 0
    with db.connect() as conn:
        for _path, grp in rows.groupby('path'):
            if len(grp) < 2:
                continue
            keeper = grp.iloc[0]['report_id']
            emailed = grp['emailed_at'].max()
            if pd.notna(emailed):
                conn.execute(text('UPDATE reports SET emailed_at = :e '
                                  'WHERE report_id = :r'),
                             {'e': emailed.to_pydatetime(), 'r': keeper})
            others = grp.iloc[1:]['report_id'].tolist()
            conn.execute(text('DELETE FROM reports WHERE report_id IN :ids')
                         .bindparams(bindparam('ids', expanding=True)),
                         {'ids': others})
            removed += len(others)
    return removed


def prune_factor_detail(older_than_days: int) -> int:
    """
    Drop per-factor detail for runs older than N days.

    Runs and their final scores are kept — those are what Compare and the
    reports read. The factor breakdown behind an old run is ~10x larger and
    only shown when inspecting that run. Opt-in; never run automatically.
    """
    cutoff = date.today() - timedelta(days=older_than_days)
    return db.execute("""
        DELETE FROM factor_scores WHERE run_id IN (
            SELECT run_id FROM runs WHERE as_of < :c)""", {'c': str(cutoff)})


def thin_old_runs(older_than_days: int) -> int:
    """
    Beyond `older_than_days`, keep one run per strategy per week.

    A screen is stored every trading day so that yesterday can be compared
    with today. A year on, five near-identical daily cohorts a week add
    nothing for a 3-6 month holder: the last run of each ISO week keeps the
    track record and the rank-IC history intact at a fifth of the rows.
    Whole runs are removed, never part of one, so nothing left is partial.
    """
    cutoff = date.today() - timedelta(days=older_than_days)
    runs = db.read_sql('SELECT run_id, as_of, strategy, universe_spec_json, created_at '
                       'FROM runs WHERE as_of < :c', {'c': str(cutoff)},
                       parse_dates=['as_of'])
    if runs.empty:
        return 0
    iso = runs['as_of'].dt.isocalendar()
    runs['week'] = iso['year'].astype(str) + '-' + iso['week'].astype(str)
    runs = runs.sort_values(['as_of', 'created_at'])
    keep = runs.groupby(['strategy', 'universe_spec_json', 'week'], dropna=False)['run_id'].last()
    drop = sorted(set(runs['run_id']) - set(keep))
    if not drop:
        return 0
    with db.get_engine().begin() as conn:
        for i in range(0, len(drop), 400):
            chunk = drop[i:i + 400]
            for table in ('factor_scores', 'scores', 'runs'):
                conn.execute(text(f'DELETE FROM {table} WHERE run_id IN :ids')
                             .bindparams(bindparam('ids', expanding=True)), {'ids': chunk})
    return len(drop)


def prune_news(older_than_days: int) -> int:
    cutoff = datetime.utcnow() - timedelta(days=older_than_days)
    return db.execute('DELETE FROM news WHERE published < :c', {'c': cutoff})


def prune_attention(older_than_days: int) -> int:
    cutoff = date.today() - timedelta(days=older_than_days)
    return db.execute('DELETE FROM attention WHERE date < :c', {'c': str(cutoff)})


def prune_logs(older_than_days: int, max_append_mb: float = 5.0) -> int:
    """Old per-run logs go; an append-only log (cron's daily.log) keeps its tail."""
    root = config.ROOT / 'logs'
    if not root.exists():
        return 0
    cutoff = time.time() - older_than_days * 86400
    removed = 0
    for path in root.rglob('*.log'):
        try:
            st = path.stat()
            if path.name in ('daily.log', 'web.log', 'scheduler.log'):
                if st.st_size > max_append_mb * 1e6:
                    with open(path, 'rb') as fh:
                        fh.seek(-int(1e6), 2)
                        tail = fh.read()
                    path.write_bytes(tail[tail.find(b'\n') + 1:])
                    removed += 1
            elif st.st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            log.debug('log prune skipped %s: %s', path, exc)
    return removed


def prune_reports(older_than_days: int) -> int:
    """Remove old daily / news report files and their rows; monthly ones stay."""
    cutoff = date.today() - timedelta(days=older_than_days)
    old = db.read_sql("SELECT report_id, path FROM reports "
                      "WHERE period_end < :c AND kind IN ('daily', 'news')",
                      {'c': str(cutoff)})
    root = config.REPORT_DIR.resolve()
    for path in old['path'].dropna():
        target = Path(path)
        try:
            # Only ever a file inside the report directory.
            if target.is_file() and root in target.resolve().parents:
                target.unlink()
        except OSError as exc:
            log.debug('report prune skipped %s: %s', path, exc)
    if not old.empty:
        with db.get_engine().begin() as conn:
            conn.execute(text('DELETE FROM reports WHERE report_id IN :ids')
                         .bindparams(bindparam('ids', expanding=True)),
                         {'ids': old['report_id'].tolist()})
    return len(old)


def _dir_mb(path: Path, pattern: str = '*') -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob(pattern):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return round(total / 1e6, 1)


def usage() -> dict[str, float]:
    """Megabytes on disk by what they hold — the answer to "why is it big?"."""
    path = sqlite_path()
    out = {'database': 0.0}
    if path and path.exists():
        out['database'] = round(sum(
            f.stat().st_size for f in (path, path.with_name(path.name + '-wal'))
            if f.exists()) / 1e6, 1)
    out['backups'] = _dir_mb(BACKUP_DIR)
    out['cache'] = _dir_mb(config.CACHE_DIR)
    out['reports'] = _dir_mb(config.REPORT_DIR)
    out['logs'] = _dir_mb(config.ROOT / 'logs')
    out['total'] = round(sum(out.values()), 1)
    return out


def reclaimable_mb() -> float:
    """Free pages inside the SQLite file — what a VACUUM would give back."""
    eng = db.get_engine()
    if eng.dialect.name != 'sqlite':
        return 0.0
    with eng.connect() as conn:
        free = conn.exec_driver_sql('PRAGMA freelist_count').scalar() or 0
        size = conn.exec_driver_sql('PRAGMA page_size').scalar() or 4096
    return round(free * size / 1e6, 1)


def housekeeping(vacuum: bool = False, say=None) -> dict:
    """
    Keep disk use level. Run at the end of every routine; cheap when there
    is nothing to do. Each step is independent: one failing (a locked table,
    a read-only folder) is reported and the rest still run.

    `vacuum=True` (the evening run) also rewrites the database file when a
    worthwhile share of it is free pages, which is the only way SQLite gives
    space back to the disk; deleted rows are otherwise reused in place.
    """
    from core import http
    keep = config.RETENTION
    out: dict = {}
    before = usage()

    def step(name, fn):
        try:
            out[name] = fn()
        except Exception as exc:                   # noqa: BLE001
            log.warning('housekeeping: %s failed: %s', name, exc)
            out[name] = f'failed: {type(exc).__name__}'

    step('factor_detail_rows', lambda: prune_factor_detail(keep['factor_detail_days']))
    step('old_daily_runs', lambda: thin_old_runs(keep['daily_runs_days']))
    step('news_rows', lambda: prune_news(keep['news_days']))
    step('attention_rows', lambda: prune_attention(keep['attention_days']))
    step('reports', lambda: prune_reports(keep['report_days']))
    step('logs', lambda: prune_logs(keep['log_days']))
    step('cache', lambda: http.sweep_cache(keep['cache_days']))

    def old_backups():
        gone = 0
        for old in list_backups()[keep['backups']:]:
            old.unlink()
            gone += 1
        return gone
    step('backups', old_backups)

    def compact():
        free, size = reclaimable_mb(), before['database']
        if vacuum and free > 200 and free > 0.2 * max(size, 1):
            optimize(vacuum=True)
            return free
        optimize(vacuum=False)
        return 0.0
    step('vacuumed_mb', compact)

    after = usage()
    out['freed_mb'] = round(before['total'] - after['total'], 1)
    out['usage'] = after
    if say:
        parts = ', '.join(f'{k} {v:,.0f} MB' for k, v in after.items() if k != 'total')
        say(f'🧹 storage {after["total"]:,.0f} MB ({parts})'
            + (f' — freed {out["freed_mb"]:,.0f} MB' if out['freed_mb'] >= 1 else ''))
    return out


def optimize(vacuum: bool = False) -> None:
    """Refresh planner statistics; optionally VACUUM to return space to disk."""
    eng = db.get_engine()
    if eng.dialect.name != 'sqlite':
        return
    with eng.connect() as conn:
        conn.exec_driver_sql('PRAGMA optimize')
        conn.exec_driver_sql('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.commit()
    if vacuum:
        # VACUUM cannot run inside a transaction, so it needs the driver in
        # autocommit. The connection goes back to the pool afterwards, so
        # the previous mode must be restored — left in autocommit, a later
        # "atomic" multi-statement write on it would commit piecemeal.
        raw = eng.raw_connection()
        dc = raw.driver_connection
        previous = dc.isolation_level
        try:
            dc.isolation_level = None
            dc.execute('VACUUM')
        finally:
            dc.isolation_level = previous
            raw.close()


def clean(vacuum: bool = False, factor_detail_days: int | None = None) -> dict:
    """Every safe clean-up in order. Returns what each step removed."""
    out: dict = {'duplicate_runs': dedupe_runs()}
    out.update({f'orphan_{k}': v for k, v in delete_orphans().items()})
    news = dedupe_news()
    out['duplicate_news'] = news['deleted']
    out['news_rekeyed'] = news['rekeyed']
    out['duplicate_reports'] = dedupe_reports()
    out['amended_8k_events'] = delete_amended_events()
    if factor_detail_days:
        out['pruned_factor_rows'] = prune_factor_detail(factor_detail_days)
    optimize(vacuum=vacuum)
    return out
