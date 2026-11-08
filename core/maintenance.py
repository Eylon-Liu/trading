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
"""

from __future__ import annotations

import logging
import sqlite3
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
    if not BACKUP_DIR.exists():
        return []
    return sorted(BACKUP_DIR.glob('quant-*.sqlite'), reverse=True)


def backup(keep: int = 7, dest_dir: Path | None = None) -> Path | None:
    """
    Copy the live database to db/backups/quant-YYYYmmdd-HHMM.sqlite.

    Uses SQLite's online backup API, which takes a consistent snapshot even
    while the UI or an ingest is writing — a plain file copy of a WAL-mode
    database can capture a half-applied transaction. Keeps the newest `keep`.
    """
    src = sqlite_path()
    if src is None or not src.exists():
        log.warning('backup skipped: %s is not a local SQLite file',
                    config.DATABASE_URL)
        return None

    out_dir = dest_dir or BACKUP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f'quant-{datetime.now():%Y%m%d-%H%M}.sqlite'
    tmp = dest.with_suffix('.partial')

    with sqlite3.connect(str(src), timeout=60) as s, \
            sqlite3.connect(str(tmp)) as d:
        s.backup(d)
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
