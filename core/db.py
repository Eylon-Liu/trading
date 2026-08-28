"""
Database layer — SQLAlchemy Core schema, engine, and upsert helpers.

Uses Core rather than the ORM: this is an analytical store, every read is a
bulk scan into pandas, and Core keeps the SQL legible.

The URL comes from config.DATABASE_URL, so pointing at a hosted libSQL or
Postgres instance is an env-var change and nothing else.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import (
    Column, Date, DateTime, Float, Index, Integer, MetaData, String, Table,
    Text, create_engine, event, text,
)
from sqlalchemy.engine import Engine

import config

log = logging.getLogger(__name__)

metadata = MetaData()


# ─────────────────────────────────────────────
# IDENTITY & UNIVERSE
# ─────────────────────────────────────────────

securities = Table(
    'securities', metadata,
    Column('ticker', String, primary_key=True),
    Column('cik', String, index=True),
    Column('name', String),
    Column('sector', String, index=True),      # normalized to GICS on write
    Column('industry', String),
    Column('exchange', String),
    Column('country', String, index=True),
    Column('is_adr', Integer, default=0),
    Column('first_seen', Date),
    Column('last_updated', DateTime),
)

# Point-in-time index membership. `as_of` is the date the membership was
# *knowable*, not the date the constituent list was effective — for N-PORT
# that means the filing date, which lags the period end by ~2 months.
index_members = Table(
    'index_members', metadata,
    Column('index_symbol', String, primary_key=True),
    Column('ticker', String, primary_key=True),
    Column('as_of', Date, primary_key=True),
    Column('source', String, primary_key=True),   # wiki_revision | nport | wiki_changes
    Column('confidence', Float, default=1.0),
    Column('weight', Float),                      # N-PORT gives us real weights
    Index('ix_members_lookup', 'index_symbol', 'as_of'),
)


# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────

# Stock splits, needed to put share counts on the same basis as prices.
#
# Price history from the provider is restated after every split, so a 2018 bar
# is quoted in today's shares. SEC share counts are not restated — they are
# whatever was filed at the time. Multiplying one by the other understates the
# market cap of any company that has since split, by exactly the split ratio,
# and every yield built on that market cap is overstated by the same factor.
splits = Table(
    'splits', metadata,
    Column('ticker', String, primary_key=True),
    Column('date', Date, primary_key=True),
    Column('ratio', Float),               # 10.0 for a 10-for-1 forward split
)

prices = Table(
    'prices', metadata,
    Column('ticker', String, primary_key=True),
    Column('date', Date, primary_key=True),
    Column('open', Float),
    Column('high', Float),
    Column('low', Float),
    Column('close', Float),
    Column('adj_close', Float),
    Column('volume', Float),
    Index('ix_prices_date', 'date'),
)


# ─────────────────────────────────────────────
# SEC — POINT-IN-TIME FUNDAMENTALS
# ─────────────────────────────────────────────
#
# One row per (concept, period, filing). Restatements produce multiple rows
# for the same period with different `filed` dates; readers always take the
# latest row with filed <= as_of. `period_start`/`period_end` together
# disambiguate a quarterly figure from a year-to-date cumulative one that
# shares the same end date.

sec_facts = Table(
    'sec_facts', metadata,
    Column('cik', String, primary_key=True),
    Column('tag', String, primary_key=True),
    Column('unit', String, primary_key=True),
    Column('period_start', Date, primary_key=True),
    Column('period_end', Date, primary_key=True),
    Column('filed', Date, primary_key=True),
    Column('form', String, primary_key=True),
    Column('ticker', String, index=True),
    Column('concept', String, index=True),     # our normalized name
    Column('fy', Integer),
    Column('fp', String),
    Column('val', Float),
    Index('ix_facts_pit', 'ticker', 'concept', 'filed'),
)

sec_filings = Table(
    'sec_filings', metadata,
    Column('accession', String, primary_key=True),
    Column('cik', String, index=True),
    Column('ticker', String, index=True),
    Column('form', String, index=True),
    Column('items', String),                   # raw 8-K item codes, comma separated
    Column('filing_date', Date, index=True),
    Column('report_date', Date),
    Column('primary_doc', String),
    Index('ix_filings_pit', 'ticker', 'filing_date'),
)

corporate_events = Table(
    'corporate_events', metadata,
    Column('ticker', String, primary_key=True),
    Column('filed', Date, primary_key=True),
    Column('accession', String, primary_key=True),
    Column('item_code', String, primary_key=True),
    Column('category', String, index=True),    # results | management | strategy | ...
    Column('subtype', String),
    Column('detail', Text),
    Index('ix_events_pit', 'ticker', 'filed'),
)

insider_txns = Table(
    'insider_txns', metadata,
    Column('accession', String, primary_key=True),
    Column('ticker', String, primary_key=True),
    Column('insider', String, primary_key=True),
    Column('txn_date', Date, primary_key=True),
    Column('txn_code', String, primary_key=True),
    Column('cik', String),
    Column('filed', Date, index=True),
    Column('role', String),
    Column('shares', Float),
    Column('price', Float),
    Column('value', Float),
    Index('ix_insider_pit', 'ticker', 'filed'),
)

risk_factor_diffs = Table(
    'risk_factor_diffs', metadata,
    Column('ticker', String, primary_key=True),
    Column('fy', Integer, primary_key=True),
    Column('filed', Date, index=True),
    Column('prev_filed', Date),
    Column('pct_changed', Float),
    Column('word_count', Integer),
    Column('added_topics', Text),
    Column('removed_topics', Text),
)

# Snapshot of provider-computed metrics that SEC does not carry (analyst
# forward PE, beta). Accumulates a forward point-in-time record from the day
# the app first runs — it can never be backfilled.
profile_snapshots = Table(
    'profile_snapshots', metadata,
    Column('ticker', String, primary_key=True),
    Column('snapshot_date', Date, primary_key=True),
    Column('market_cap', Float),
    Column('trailing_pe', Float),
    Column('forward_pe', Float),
    Column('price_to_book', Float),
    Column('trailing_eps', Float),
    Column('forward_eps', Float),
    Column('revenue_growth', Float),
    Column('earnings_growth', Float),
    Column('roe', Float),
    Column('roa', Float),
    Column('debt_to_equity', Float),
    Column('current_ratio', Float),
    Column('gross_margin', Float),
    Column('operating_margin', Float),
    Column('profit_margin', Float),
    Column('dividend_yield', Float),
    Column('payout_ratio', Float),
    Column('beta', Float),
    Column('shares_out', Float),
)


# ─────────────────────────────────────────────
# ALTERNATIVE DATA
# ─────────────────────────────────────────────

attention = Table(
    'attention', metadata,
    Column('ticker', String, primary_key=True),
    Column('date', Date, primary_key=True),
    Column('wiki_views', Float),
)

news = Table(
    'news', metadata,
    Column('id', String, primary_key=True),
    Column('ticker', String, index=True),
    Column('published', DateTime, index=True),
    Column('source', String),
    Column('title', Text),
    Column('url', Text),
    Column('body_summary', Text),
    Column('sentiment', Float),
    Column('sentiment_label', String),
    Column('events', String),
    Column('extracted_json', Text),
    Column('fetched_at', DateTime),
    Index('ix_news_pit', 'ticker', 'published'),
)

policy = Table(
    'policy', metadata,
    Column('doc_id', String, primary_key=True),
    Column('published', Date, index=True),
    Column('doc_type', String),
    Column('agencies', Text),
    Column('title', Text),
    Column('abstract', Text),
    Column('url', Text),
    Column('themes', String),
    Column('affected_sectors', String),
    Column('affected_tickers', String),
)

macro = Table(
    'macro', metadata,
    Column('series_id', String, primary_key=True),
    Column('date', Date, primary_key=True),
    Column('value', Float),
)


# ─────────────────────────────────────────────
# RUNS & SCORES
# ─────────────────────────────────────────────

runs = Table(
    'runs', metadata,
    Column('run_id', String, primary_key=True),
    Column('as_of', Date, index=True),
    Column('strategy', String, index=True),
    Column('universe_spec_json', Text),
    Column('universe_tickers', Text),   # snapshotted, so a re-run reproduces
    Column('params_json', Text),
    Column('universe_n', Integer),
    Column('created_at', DateTime),
)

factor_scores = Table(
    'factor_scores', metadata,
    Column('run_id', String, primary_key=True),
    Column('ticker', String, primary_key=True),
    Column('factor', String, primary_key=True),
    Column('raw', Float),
    Column('winsorized', Float),
    Column('z', Float),
    Column('sector_z', Float),
    Column('pct_rank', Float),
)

scores = Table(
    'scores', metadata,
    Column('run_id', String, primary_key=True),
    Column('ticker', String, primary_key=True),
    Column('composite', Float),
    Column('rank', Integer),
    Column('signal', String),
    Column('signal_score', Float),
    Column('reasons', Text),
    Column('coverage', Float),
    Column('sector', String),
)

reports = Table(
    'reports', metadata,
    Column('report_id', String, primary_key=True),
    Column('kind', String, index=True),
    Column('period_start', Date),
    Column('period_end', Date),
    Column('path', Text),
    Column('emailed_at', DateTime),
    Column('created_at', DateTime),
)

# User-defined strategies, stored as rows rather than code so they survive a
# restart. `based_on` records the preset a copy came from — provenance only;
# the copy holds its own weights and the two never track each other.
custom_strategies = Table(
    'custom_strategies', metadata,
    Column('key', String, primary_key=True),
    Column('name', String),
    Column('horizon', String, index=True),
    Column('description', Text),
    Column('thesis', Text),
    Column('based_on', String),
    Column('weights_json', Text),
    Column('filters_json', Text),
    Column('neutralize', String),
    Column('setup', String),
    Column('updated_at', DateTime),
)

ingest_log = Table(
    'ingest_log', metadata,
    Column('source', String, primary_key=True),
    Column('key', String, primary_key=True),
    Column('last_success', DateTime),
    Column('last_error', Text),
    Column('rows', Integer),
)


# ─────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────

_engine: Engine | None = None


def get_engine() -> Engine:
    """Process-wide engine. SQLite gets WAL so the UI can read during ingest."""
    global _engine
    if _engine is not None:
        return _engine

    url = config.DATABASE_URL
    kwargs: dict = {'future': True}
    if url.startswith('sqlite'):
        kwargs['connect_args'] = {'check_same_thread': False, 'timeout': 30}

    _engine = create_engine(url, **kwargs)

    if url.startswith('sqlite'):
        @event.listens_for(_engine, 'connect')
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute('PRAGMA journal_mode=WAL')
            cur.execute('PRAGMA synchronous=NORMAL')
            cur.execute('PRAGMA temp_store=MEMORY')
            cur.execute('PRAGMA cache_size=-64000')  # 64 MB
            cur.close()

    return _engine


def init_db() -> None:
    """Create every table and index if absent. Safe to run repeatedly."""
    metadata.create_all(get_engine())
    log.info('schema ready at %s', config.DATABASE_URL)


@contextmanager
def connect():
    """Transactional connection."""
    eng = get_engine()
    with eng.begin() as conn:
        yield conn


# ─────────────────────────────────────────────
# WRITE HELPERS
# ─────────────────────────────────────────────

def execute(sql: str, params: dict | None = None) -> int:
    """Run a statement and return the affected row count."""
    with connect() as conn:
        result = conn.execute(text(sql), params or {})
        return result.rowcount or 0


def upsert(table: Table, rows: Sequence[dict], chunk: int = 2000) -> int:
    """
    Insert-or-replace rows keyed on the table's primary key.

    Uses the dialect's native upsert. Restatement rows are *not* overwrites —
    they differ in `filed`, which is part of the sec_facts key — so history
    accumulates rather than being clobbered.

    On conflict, only the columns actually supplied are updated. Updating every
    non-key column instead meant a partial write erased the rest of the row:
    marking a report as emailed with {report_id, emailed_at} set its path, kind
    and created_at to NULL, and the report then showed in the history with no
    file and no date. A caller that names a column intends to change it; one
    that omits it does not intend to erase it.
    """
    rows = [r for r in rows if r]
    if not rows:
        return 0

    eng = get_engine()
    dialect = eng.dialect.name
    total = 0

    if dialect == 'sqlite':
        from sqlalchemy.dialects.sqlite import insert as _insert
    elif dialect == 'postgresql':
        from sqlalchemy.dialects.postgresql import insert as _insert
    else:
        _insert = None

    pk_cols = [c.name for c in table.primary_key.columns]

    with eng.begin() as conn:
        for i in range(0, len(rows), chunk):
            batch = rows[i:i + chunk]
            if _insert is None:
                conn.execute(table.insert(), batch)
            else:
                stmt = _insert(table).values(batch)
                # Only what this batch actually carries. A column absent from
                # every row in the batch is left alone rather than nulled.
                supplied = {k for r in batch for k in r}
                update_cols = {
                    c.name: stmt.excluded[c.name]
                    for c in table.columns
                    if c.name not in pk_cols and c.name in supplied
                }
                if update_cols:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=pk_cols, set_=update_cols
                    )
                else:
                    stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)
                conn.execute(stmt)
            total += len(batch)

    return total


def record_ingest(source: str, key: str, rows: int = 0, error: str | None = None) -> None:
    """Bookkeeping so the Data tab can show freshness and failures."""
    upsert(ingest_log, [{
        'source': source,
        'key': key,
        'last_success': pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None)
                        if error is None else None,
        'last_error': error,
        'rows': rows,
    }])


# ─────────────────────────────────────────────
# READ HELPERS
# ─────────────────────────────────────────────

def read_sql(sql: str, params: dict | None = None,
             parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Run a query and return a DataFrame."""
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {},
                           parse_dates=list(parse_dates) if parse_dates else None)


def table_counts() -> pd.DataFrame:
    """Row count per table — powers the Data tab."""
    out = []
    for name in metadata.tables:
        try:
            n = read_sql(f'SELECT COUNT(*) AS n FROM {name}')['n'].iloc[0]
        except Exception:
            n = 0
        out.append({'table': name, 'rows': int(n)})
    return pd.DataFrame(out).sort_values('rows', ascending=False)
