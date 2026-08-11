"""
The pull phase.

Every run follows the same two steps: **sync, then read**. This module is the
sync. It decides what is stale, fetches only that, writes it to SQLite, and
returns a report. Nothing downstream touches the network — quant/ and ui/ read
the database and nothing else.

Why the split matters:

  * **Speed.** A source that was refreshed twenty minutes ago is not fetched
    again. On a quiet re-run nothing is fetched at all and the whole phase
    costs a handful of SQL queries.
  * **Cache correctness.** The factor cache is keyed on a data version derived
    from the store. If sync writes nothing the version does not move, so the
    cached factor frame is served — that is what makes a repeat screen
    near-instant rather than a rebuild.
  * **Reproducibility.** A read that cannot fetch cannot surprise you. Two
    reads of the same as-of date over an unchanged store give the same answer,
    which is the whole premise of a point-in-time tool.
  * **Honesty.** Freshness is measured and reported rather than assumed, so a
    stale source is visible in the UI instead of quietly ageing.

Staleness is per source and, where it makes sense, per ticker: a universe
where 490 of 500 names were updated this morning fetches the other ten.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

import config
from core import db

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# POLICY
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Source:
    """How often one source is worth re-fetching, and how to do it."""
    name: str
    label: str
    max_age_hours: float
    essential: bool          # required before a screen can score anything
    # How ingest_log is keyed for this source. The providers predate this
    # module and key themselves differently, so the mapping lives here rather
    # than churning every provider:
    #   'ticker'        one row per ticker
    #   'ticker_prefix' one row per ticker, stored as "TICKER:extra"
    #   'global'        one row for the whole source, keyed by date or name
    key_mode: str = 'ticker'
    log_source: str = ''     # ingest_log source name; defaults to `name`
    note: str = ''

    @property
    def log(self) -> str:
        return self.log_source or self.name

    @property
    def per_ticker(self) -> bool:
        return self.key_mode in ('ticker', 'ticker_prefix')


# Ages are chosen from how fast the underlying data actually changes. Daily
# bars settle once a day; a company files when it files; the Federal Register
# publishes on business mornings. Fetching faster than the source changes buys
# nothing and spends rate limit.
SOURCES: dict[str, Source] = {
    'members':    Source('members', 'Index membership', 24 * 7, True,
                         key_mode='global',
                         note='Constituents change on scheduled reviews.'),
    'securities': Source('securities', 'Security master', 24 * 30, True,
                         key_mode='global',
                         note='CIK and SIC sector; effectively static.'),
    'prices':     Source('prices', 'Daily bars', 12, True,
                         note='Also checked against the last session.'),
    'facts':      Source('facts', 'SEC XBRL facts', 12, True,
                         key_mode='ticker_prefix', log_source='sec_facts',
                         note='New facts appear only when a filing lands.'),
    'filings':    Source('filings', 'SEC filings & 8-K items', 12, False),
    'profiles':   Source('profiles', 'Market-cap snapshots', 24, False,
                         key_mode='global',
                         note='Yahoo enrichment; rate-limited, best effort.'),
    'insiders':   Source('insiders', 'Form 4 transactions', 24, False),
    'news':       Source('news', 'News articles', 3, False),
    'policy':     Source('policy', 'Federal Register', 12, False,
                         key_mode='global'),
    'attention':  Source('attention', 'Wikipedia pageviews', 24, False,
                         key_mode='global'),
}

ESSENTIAL = [k for k, s in SOURCES.items() if s.essential]
OPTIONAL = [k for k, s in SOURCES.items() if not s.essential]


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

@dataclass
class SourceResult:
    name: str
    action: str                      # 'fetched' | 'fresh' | 'skipped' | 'failed'
    rows: int = 0
    stale_keys: int = 0
    total_keys: int = 0
    seconds: float = 0.0
    detail: str = ''

    @property
    def changed(self) -> bool:
        return self.action == 'fetched' and self.rows > 0


@dataclass
class SyncReport:
    results: list[SourceResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def changed(self) -> bool:
        """True when anything was actually written."""
        return any(r.changed for r in self.results)

    @property
    def rows(self) -> int:
        return sum(r.rows for r in self.results)

    def summary(self) -> str:
        if not self.results:
            return 'nothing to sync'
        fetched = [r for r in self.results if r.action == 'fetched']
        failed = [r for r in self.results if r.action == 'failed']
        if not fetched and not failed:
            return f'all {len(self.results)} sources fresh — used stored data'
        bits = []
        if fetched:
            bits.append(', '.join(f'{r.name} +{r.rows}' for r in fetched if r.rows)
                        or f'{len(fetched)} checked')
        if failed:
            bits.append(f'{len(failed)} failed')
        return f"{'; '.join(bits)} in {self.seconds:.1f}s"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'source': SOURCES[r.name].label if r.name in SOURCES else r.name,
            'action': r.action,
            'stale': f'{r.stale_keys}/{r.total_keys}' if r.total_keys else '—',
            'rows': r.rows,
            'seconds': round(r.seconds, 1),
            'detail': r.detail,
        } for r in self.results])


# ─────────────────────────────────────────────
# FRESHNESS
# ─────────────────────────────────────────────

def _last_success(source: str) -> pd.DataFrame:
    """Rows from ingest_log for a source, under its own logged name."""
    src = SOURCES.get(source)
    return db.read_sql(
        'SELECT key, last_success FROM ingest_log WHERE source = :s',
        {'s': src.log if src else source}, parse_dates=['last_success'])


def _fresh_keys(source: str, now: datetime) -> set[str]:
    """Keys refreshed recently enough to skip, normalised to ticker form."""
    src = SOURCES[source]
    cutoff = now - timedelta(hours=src.max_age_hours)

    seen = _last_success(source)
    if seen.empty:
        return set()
    ok = seen[seen['last_success'].notna() & (seen['last_success'] >= cutoff)]
    keys = ok['key'].astype(str)
    if src.key_mode == 'ticker_prefix':
        # Stored as "AAPL:0000320193" — the ticker is what callers ask about.
        keys = keys.str.split(':').str[0]
    return set(keys)


def stale_keys(source: str, keys: list[str], now: datetime | None = None,
               force: bool = False) -> list[str]:
    """Which keys are due a refresh: never fetched, or fetched too long ago."""
    src = SOURCES[source]
    if force:
        return list(keys)

    now = now or datetime.utcnow()
    fresh = _fresh_keys(source, now)

    if src.key_mode == 'global':
        # One clock for the whole source: any recent row means all of it is
        # fresh. These providers refresh everything in a single call, so
        # tracking them per ticker would just force needless full refetches.
        return [] if fresh else list(keys)

    return [k for k in keys if k not in fresh]


# US equities close at 20:00 UTC (21:00 during DST); allow an hour for the
# daily bar to be published.
_CLOSE_UTC_HOUR = 21


def last_session(now: datetime | None = None) -> date:
    """
    The most recent trading session whose daily bar should exist by now.

    Today only counts once the close has been published — otherwise every
    check before 21:00 UTC declares the whole universe stale and refetches
    prices on every single run.

    Weekends only; holidays are not modelled, so on a market holiday this
    reports one stale day and prices are fetched once unnecessarily. A holiday
    calendar to save that is not worth the maintenance.
    """
    now = now or datetime.utcnow()
    d = now.date()
    if now.hour < _CLOSE_UTC_HOUR:
        d -= timedelta(days=1)
    while d.weekday() >= 5:                 # Saturday or Sunday
        d -= timedelta(days=1)
    return d


def price_gap(tickers: list[str]) -> list[str]:
    """Tickers whose newest stored bar predates the last session."""
    if not tickers:
        return []
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params = {f't{i}': t for i, t in enumerate(tickers)}
    have = db.read_sql(
        f'SELECT ticker, MAX(date) AS last FROM prices '
        f'WHERE ticker IN ({ph}) GROUP BY ticker', params, parse_dates=['last'])

    target = last_session()
    covered = set()
    if not have.empty:
        ok = have[have['last'].dt.date >= target]
        covered = set(ok['ticker'])
    return [t for t in tickers if t not in covered]


def freshness() -> pd.DataFrame:
    """Per-source age, for the Data tab."""
    rows = []
    now = datetime.utcnow()
    for name, src in SOURCES.items():
        seen = _last_success(name)
        newest = (seen['last_success'].max()
                  if not seen.empty and seen['last_success'].notna().any()
                  else pd.NaT)
        age = ((now - newest.to_pydatetime()).total_seconds() / 3600
               if pd.notna(newest) else None)
        rows.append({
            'source': src.label,
            'keys tracked': len(seen),
            'last refreshed': (newest.strftime('%Y-%m-%d %H:%M')
                               if pd.notna(newest) else 'never'),
            'age (h)': round(age, 1) if age is not None else None,
            'refresh after (h)': src.max_age_hours,
            'status': ('never fetched' if age is None
                       else 'fresh' if age < src.max_age_hours else 'stale'),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SYNC
# ─────────────────────────────────────────────

def sync(tickers: list[str], *, index: str | None = None,
         sources: list[str] | None = None, force: bool = False,
         include_optional: bool = True, progress=None) -> SyncReport:
    """
    Refresh whatever is stale, then return without touching anything else.

    Safe to call before every read: when everything is fresh this issues a few
    SQL queries and no network calls at all.
    """
    started = time.perf_counter()
    report = SyncReport()

    wanted = sources or (ESSENTIAL + OPTIONAL if include_optional else ESSENTIAL)

    def emit(msg: str) -> None:
        log.info('sync: %s', msg)
        if progress:
            progress(msg)

    for name in wanted:
        if name not in SOURCES:
            continue
        t0 = time.perf_counter()
        try:
            result = _sync_one(name, tickers, index=index, force=force,
                               emit=emit)
        except Exception as exc:                   # noqa: BLE001
            # A failed source must not abort the run: the read phase falls
            # back to whatever is already stored, which is the entire point of
            # persisting it.
            log.warning('sync %s failed: %s', name, exc)
            db.record_ingest(name, index or 'all', error=str(exc)[:300])
            result = SourceResult(name, 'failed', detail=str(exc)[:120])
        result.seconds = time.perf_counter() - t0
        report.results.append(result)

    report.seconds = time.perf_counter() - started
    log.info('sync complete: %s', report.summary())
    return report


def _sync_one(name: str, tickers: list[str], *, index: str | None,
              force: bool, emit) -> SourceResult:
    # Imported lazily so `import sync` stays cheap and a broken optional
    # provider cannot prevent the module from loading.
    from data import altdata, members, news, policy, sec, yahoo

    src = SOURCES[name]

    # ── index membership ─────────────────────────────────────────
    if name == 'members':
        if not index:
            return SourceResult(name, 'skipped', detail='no index given')
        due = stale_keys(name, [index], force=force)
        if not due:
            return SourceResult(name, 'fresh', total_keys=1)
        emit(f'refreshing {src.label} for {index}')
        rows = members.backfill(index, date.today() - timedelta(days=7))
        db.record_ingest(name, index, rows=rows)
        return SourceResult(name, 'fetched', rows=rows, stale_keys=1,
                            total_keys=1)

    # ── whole-market sources ─────────────────────────────────────
    if name == 'policy':
        due = stale_keys(name, ['federal_register'], force=force)
        if not due:
            return SourceResult(name, 'fresh', total_keys=1)
        emit(f'refreshing {src.label}')
        rows = policy.update_policy()
        db.record_ingest(name, 'federal_register', rows=rows)
        return SourceResult(name, 'fetched', rows=rows, stale_keys=1,
                            total_keys=1)

    if not tickers:
        return SourceResult(name, 'skipped', detail='empty universe')

    # ── per-ticker sources ───────────────────────────────────────
    if name == 'prices':
        # Two tests: the ingest clock, and whether the newest bar actually
        # reaches the last session. The second catches a ticker that was
        # "checked" recently but came back empty.
        due = sorted(set(stale_keys(name, tickers, force=force))
                     | set(price_gap(tickers)))
        if not due:
            return SourceResult(name, 'fresh', total_keys=len(tickers),
                                detail=f'bars current to {last_session()}')
        emit(f'{src.label}: {len(due)} of {len(tickers)} tickers stale')
        rows = yahoo.update_prices(due + [config.BENCHMARK_TICKER])
        for t in due:
            db.record_ingest(name, t, rows=0)
        return SourceResult(name, 'fetched', rows=rows, stale_keys=len(due),
                            total_keys=len(tickers))

    handlers = {
        'securities': lambda ts: sec.update_securities(ts),
        'facts': lambda ts: sec.update_facts(ts),
        'filings': lambda ts: sec.update_filings(ts),
        'profiles': lambda ts: yahoo.update_profiles(ts),
        'insiders': lambda ts: sec.update_insiders(ts),
        'news': lambda ts: news.update_news(ts),
        'attention': lambda ts: altdata.update_attention(ts, days_back=400),
    }
    fn = handlers.get(name)
    if fn is None:
        return SourceResult(name, 'skipped', detail='no handler')

    due = stale_keys(name, tickers, force=force)
    if not due:
        return SourceResult(name, 'fresh', total_keys=len(tickers))

    emit(f'{src.label}: {len(due)} of {len(tickers)} tickers stale')
    rows = fn(due)
    # Sources that log per ticker themselves (insiders, news) will overwrite
    # these with their own row counts; recording here guarantees that a ticker
    # which legitimately returned nothing still counts as checked, instead of
    # being retried on every single run.
    for t in due:
        db.record_ingest(name, t, rows=0)
    return SourceResult(name, 'fetched', rows=rows or 0, stale_keys=len(due),
                        total_keys=len(tickers))
