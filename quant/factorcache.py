"""
Disk cache for built factor frames.

Building factors for the S&P 500 reads ~120k SEC facts and derives quarterly
flows for every ticker. That is tens of seconds of pure CPU, and it produces
exactly the same answer every time until new data is ingested — so it is
cached rather than recomputed on each click.

Correctness rests on the key, which folds in:

  * the ticker set and the as-of date, because both change the answer;
  * a *data version* — the newest `filed` in sec_facts, the newest price bar,
    and row counts. Ingesting anything moves the version, so a stale entry can
    never be served. This is what makes caching safe for a point-in-time tool:
    the cache is keyed by what was knowable, not by wall-clock time.

A corrupt or unreadable entry is treated as a miss, never as an error — the
cache is an optimisation and must not be able to break a screen.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import time
from datetime import date

import pandas as pd

import config
from core import db

log = logging.getLogger(__name__)

CACHE_DIR = config.CACHE_DIR / 'factors'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Bump when factor computation logic changes (new factors, formula fixes,
# direction changes). The data_version fingerprint only moves when new data
# is ingested; without this, a code change serves stale cached results.
CODE_VERSION = 3

# Entries are self-invalidating via the data version, so this only bounds
# unbounded growth from many one-off as-of dates.
MAX_AGE_SECONDS = 14 * 24 * 3600
MAX_ENTRIES = 400


def data_version() -> str:
    """A fingerprint of the stored data that any ingest will change."""
    try:
        row = db.read_sql("""
            SELECT (SELECT COUNT(*) FROM sec_facts)            AS nf,
                   (SELECT MAX(filed) FROM sec_facts)          AS mf,
                   (SELECT COUNT(*) FROM prices)               AS np,
                   (SELECT MAX(date) FROM prices)              AS mp,
                   (SELECT COUNT(*) FROM insider_txns)         AS ni,
                   (SELECT COUNT(*) FROM securities)           AS ns,
                   (SELECT COUNT(*) FROM profile_snapshots)    AS nps,
                   (SELECT MAX(snapshot_date) FROM profile_snapshots) AS mps
        """)
        if row.empty:
            return 'unknown'
        r = row.iloc[0]
        return (f"{r['nf']}:{r['mf']}:{r['np']}:{r['mp']}"
                f":{r['ni']}:{r['ns']}:{r['nps']}:{r['mps']}")
    except Exception as exc:                       # noqa: BLE001
        # Without a version we cannot prove freshness, so make the key unique
        # and effectively bypass the cache rather than risk a stale hit.
        log.debug('data_version failed (%s) — bypassing factor cache', exc)
        return f'nover-{time.time()}'


def key(tickers: list[str], as_of: date | str, version: str | None = None) -> str:
    """Stable hash over the universe, the as-of date and the data version."""
    payload = '|'.join([
        str(CODE_VERSION),
        str(pd.to_datetime(as_of).date()),
        version if version is not None else data_version(),
        ','.join(sorted(tickers)),
    ])
    return hashlib.sha1(payload.encode()).hexdigest()[:24]


def load(cache_key: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f'{cache_key}.pkl'
    if not path.exists():
        return None
    try:
        with path.open('rb') as fh:
            df = pickle.load(fh)
        if isinstance(df, pd.DataFrame):
            return df
        log.debug('factor cache %s held a %s, ignoring', cache_key, type(df))
    except Exception as exc:                       # noqa: BLE001
        log.debug('unreadable factor cache %s (%s) — recomputing', cache_key, exc)
        try:
            path.unlink()
        except OSError:
            pass
    return None


def save(cache_key: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    path = CACHE_DIR / f'{cache_key}.pkl'
    tmp = path.with_suffix('.tmp')
    try:
        with tmp.open('wb') as fh:
            pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # Atomic: a reader can never observe a half-written entry.
        tmp.replace(path)
    except Exception as exc:                       # noqa: BLE001
        log.debug('could not write factor cache %s: %s', cache_key, exc)
        tmp.unlink(missing_ok=True)
        return
    _prune()


def _prune() -> None:
    """Drop entries that are too old or too many, oldest first."""
    try:
        entries = sorted(CACHE_DIR.glob('*.pkl'), key=lambda p: p.stat().st_mtime)
    except OSError:
        return

    now = time.time()
    stale = [p for p in entries if now - p.stat().st_mtime > MAX_AGE_SECONDS]
    excess = entries[:max(0, len(entries) - MAX_ENTRIES)]
    for p in set(stale) | set(excess):
        try:
            p.unlink()
        except OSError:
            pass


def clear() -> int:
    """Remove every entry. Returns how many were deleted."""
    n = 0
    for p in CACHE_DIR.glob('*.pkl'):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def stats() -> dict:
    """Entry count and total size, for the Data tab."""
    try:
        entries = list(CACHE_DIR.glob('*.pkl'))
        return {'entries': len(entries),
                'bytes': sum(p.stat().st_size for p in entries)}
    except OSError:
        return {'entries': 0, 'bytes': 0}
