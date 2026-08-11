"""
Optional external market data, for what SEC filings cannot provide.

SEC EDGAR stays authoritative: it is the filing itself, it carries a `filed`
date, and it is free. This module exists only for the gap SEC structurally
cannot fill — multi-class filers such as Berkshire, Alphabet and Comcast
publish their per-class share counts as dimensioned XBRL facts that the public
API does not return, so no consolidated count exists to fetch. That is about
39 of the S&P 500, and without a market cap they drop out of all five
valuation factors.

Two rules keep this from undermining the rest of the pipeline:

  1. **Fill, never override.** A ticker with a filed share count keeps it. The
     external value is used only where SEC has nothing, so the audit trail
     from the Methodology tab holds for every name that has one.

  2. **A live quote is today's fact, and is stored as such.** Market cap from
     an API is a current snapshot with no history — it cannot be backfilled
     and must never be treated as if it could. Values are written to
     profile_snapshots stamped with today's date, and every read gates on
     `snapshot_date <= as_of`, so a backtest of 2021 cannot see a quote
     fetched in 2026. The record accumulates forward from the day you first
     run it, which is the only honest way to use a source with no past.

Without a key nothing here runs and nothing changes: the affected names keep
scoring on quality, growth and momentum, and simply carry no valuation score.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

import config
from core import db, http

log = logging.getLogger(__name__)

FINNHUB_PROFILE = 'https://finnhub.io/api/v1/stock/profile2'

# Finnhub allows 60 calls/minute on the free tier; this is one call per ticker
# and is only made for names SEC could not supply, so a run touches ~39.
BATCH_PAUSE_EVERY = 55


def available() -> bool:
    return bool(config.FINNHUB_API_KEY)


def status() -> dict:
    if not config.FINNHUB_API_KEY:
        return {'enabled': False,
                'reason': 'No FINNHUB_API_KEY in .env. Multi-class filers '
                          '(BRK-B, GOOGL, CMCSA…) carry no market cap and are '
                          'excluded from valuation factors.'}
    return {'enabled': True,
            'reason': 'Finnhub fills market cap where SEC has no share count.'}


def fetch_market_caps(tickers: list[str]) -> pd.DataFrame:
    """
    Current market cap per ticker. Returns (ticker, market_cap, shares_out).

    Finnhub reports market cap in millions of the listing currency; it is
    converted to absolute units here so callers never have to know that.
    """
    if not tickers or not available():
        return pd.DataFrame()

    rows = []
    for t in tickers:
        payload = http.fetch_json(
            FINNHUB_PROFILE, category='prices', ttl=6 * 3600,
            params={'symbol': t, 'token': config.FINNHUB_API_KEY})
        if not payload:
            continue
        mcap_m = payload.get('marketCapitalization')
        shares_m = payload.get('shareOutstanding')
        if not mcap_m:
            continue
        rows.append({
            'ticker': t,
            'market_cap': float(mcap_m) * 1e6,
            'shares_out': float(shares_m) * 1e6 if shares_m else None,
        })

    return pd.DataFrame(rows)


def gaps(tickers: list[str], as_of: date | str | None = None) -> list[str]:
    """
    Tickers with no usable filed share count.

    Computed from the store rather than guessed, so the external call is made
    only where it is actually needed.
    """
    if not tickers:
        return []
    as_of = pd.to_datetime(as_of or date.today()).date()
    cutoff = as_of - pd.Timedelta(days=500)

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params.update({'lo': str(cutoff), 'hi': str(as_of)})

    have = db.read_sql(f"""
        SELECT DISTINCT ticker FROM sec_facts
        WHERE ticker IN ({ph})
          AND concept IN ('shares_outstanding','shares_diluted','shares_basic')
          AND filed BETWEEN :lo AND :hi
    """, params)
    covered = set(have['ticker']) if not have.empty else set()
    return [t for t in tickers if t not in covered]


def update_market_caps(tickers: list[str],
                       as_of: date | str | None = None) -> int:
    """
    Fill market cap for names SEC cannot cover, stamped with today's date.

    Only the gaps are fetched — a ticker with a filed share count is left
    alone, so the filed figure always wins and the external source can never
    quietly replace an auditable number with an opaque one.
    """
    if not available():
        return 0

    as_of = pd.to_datetime(as_of or date.today()).date()
    missing = gaps(tickers, as_of)
    if not missing:
        log.info('market caps: no gaps to fill')
        return 0

    log.info('market caps: fetching %d name(s) SEC cannot supply', len(missing))
    df = fetch_market_caps(missing)
    if df.empty:
        return 0

    rows = [{
        'ticker': r['ticker'],
        # Today's snapshot. Reads gate on snapshot_date <= as_of, so this is
        # invisible to any historical run — a quote has no history and must
        # not be allowed to pretend otherwise.
        'snapshot_date': as_of,
        'market_cap': r['market_cap'],
        'shares_out': r.get('shares_out'),
    } for _i, r in df.iterrows()]

    written = db.upsert(db.profile_snapshots, rows)
    db.record_ingest('market_caps', str(as_of), rows=written)
    log.info('market caps: stored %d snapshot(s)', written)
    return written
