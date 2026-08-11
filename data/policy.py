"""
Government policy stream — Federal Register rules and executive orders.

Policy moves sectors, not single names, so this is kept as its own stream
rather than bolted onto company news. A BIS export-control rule is a semis
event; an EPA emissions rule is an energy and utilities event. Mapping runs
agency -> GICS sector, with keyword themes layered on top.

The Federal Register API is free, needs no key, and every document carries a
`publication_date` — so unlike company news this stream *is* point-in-time
correct and backtestable over its full history.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

import config
from core import db, http

log = logging.getLogger(__name__)

FR_API = 'https://www.federalregister.gov/api/v1/documents.json'

DOC_TYPES = ['RULE', 'PRORULE', 'PRESDOCU']
DOC_TYPE_LABEL = {
    'Rule': 'Final rule',
    'Proposed Rule': 'Proposed rule',
    'Presidential Document': 'Executive action',
    'Notice': 'Notice',
}


def _themes_for(text: str) -> list[str]:
    low = (text or '').lower()
    return [theme for theme, words in config.POLICY_THEMES.items()
            if any(w in low for w in words)]


def _sectors_for(agency_slugs: list[str], themes: list[str]) -> list[str]:
    sectors: list[str] = []
    for slug in agency_slugs:
        for s in config.AGENCY_SECTOR_MAP.get(slug, []):
            if s not in sectors:
                sectors.append(s)

    # Themes can imply exposure the issuing agency does not.
    theme_map = {
        'tariff': ['Industrials', 'Consumer Discretionary', 'Materials'],
        'export_control': ['Information Technology'],
        'antitrust': ['Information Technology', 'Communication Services'],
        'energy': ['Energy', 'Utilities'],
        'healthcare': ['Health Care'],
        'ai': ['Information Technology'],
    }
    for t in themes:
        for s in theme_map.get(t, []):
            if s not in sectors:
                sectors.append(s)
    return sectors


def fetch_policy(days_back: int = 30, per_page: int = 100,
                 max_pages: int = 3) -> list[dict]:
    """Recent rules, proposed rules and presidential documents."""
    start = date.today() - timedelta(days=days_back)
    rows: list[dict] = []

    for page in range(1, max_pages + 1):
        params = {
            'per_page': per_page, 'page': page, 'order': 'newest',
            'conditions[publication_date][gte]': start.isoformat(),
            'fields[]': ['document_number', 'title', 'abstract', 'type',
                         'publication_date', 'html_url', 'agencies'],
        }
        # requests encodes repeated keys from a list value
        params['conditions[type][]'] = DOC_TYPES

        payload = http.fetch_json(FR_API, params=params, category='policy')
        if not payload or not payload.get('results'):
            break

        for doc in payload['results']:
            agencies = doc.get('agencies') or []
            slugs = [a.get('slug') for a in agencies if a.get('slug')]
            names = [a.get('name') for a in agencies if a.get('name')]

            text = f"{doc.get('title', '')} {doc.get('abstract') or ''}"
            themes = _themes_for(text)
            sectors = _sectors_for(slugs, themes)

            rows.append({
                'doc_id': doc.get('document_number'),
                'published': pd.to_datetime(doc['publication_date']).date(),
                'doc_type': DOC_TYPE_LABEL.get(doc.get('type'), doc.get('type')),
                'agencies': '; '.join(names)[:300],
                'title': (doc.get('title') or '')[:500],
                'abstract': (doc.get('abstract') or '')[:2000],
                'url': doc.get('html_url', ''),
                'themes': ','.join(themes),
                'affected_sectors': ','.join(sectors),
                'affected_tickers': '',
            })

        if len(payload['results']) < per_page:
            break

    return rows


def update_policy(days_back: int = 30) -> int:
    """Fetch policy documents and map them to affected tickers."""
    rows = fetch_policy(days_back)
    if not rows:
        return 0

    # Map sector exposure through to individual names.
    secs = db.read_sql('SELECT ticker, sector, name FROM securities')
    by_sector: dict[str, list[str]] = {}
    if not secs.empty:
        for sector, grp in secs.groupby('sector'):
            by_sector[sector] = grp['ticker'].tolist()

    for r in rows:
        sectors = [s for s in r['affected_sectors'].split(',') if s]
        tickers: list[str] = []
        for s in sectors:
            tickers.extend(by_sector.get(s, [])[:40])
        r['affected_tickers'] = ','.join(sorted(set(tickers)))[:2000]

    n = db.upsert(db.policy, rows)
    db.record_ingest('policy', str(date.today()), rows=n)
    log.info('policy: %d documents', n)
    return n


# ─────────────────────────────────────────────
# READS
# ─────────────────────────────────────────────

def policy_asof(as_of: date | str, lookback_days: int = 30,
                sectors: list[str] | None = None,
                themes: list[str] | None = None) -> pd.DataFrame:
    """Policy documents published at or before `as_of`."""
    end = pd.to_datetime(as_of).date()
    start = end - timedelta(days=lookback_days)

    df = db.read_sql("""
        SELECT doc_id, published, doc_type, agencies, title, abstract,
               url, themes, affected_sectors, affected_tickers
        FROM policy WHERE published BETWEEN :lo AND :hi
        ORDER BY published DESC
    """, {'lo': str(start), 'hi': str(end)}, parse_dates=['published'])

    if df.empty:
        return df
    if sectors:
        df = df[df['affected_sectors'].fillna('').apply(
            lambda s: any(x in s for x in sectors))]
    if themes:
        df = df[df['themes'].fillna('').apply(
            lambda s: any(x in s for x in themes))]
    return df


def policy_exposure(tickers: list[str], as_of: date | str,
                    lookback_days: int = 30) -> pd.DataFrame:
    """
    How much recent policy activity touches each name's sector.

    A count, not a judgement: regulation can be a tailwind or a headwind, and
    the direction is not something a keyword match can honestly call.
    """
    df = policy_asof(as_of, lookback_days)
    if df.empty or not tickers:
        return pd.DataFrame()

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    secs = db.read_sql(
        f'SELECT ticker, sector FROM securities WHERE ticker IN ({ph})',
        {f't{i}': t for i, t in enumerate(tickers)})
    if secs.empty:
        return pd.DataFrame()

    rows = []
    for _i, s in secs.iterrows():
        hits = df[df['affected_sectors'].fillna('').str.contains(
            str(s['sector']), regex=False)]
        rows.append({
            'ticker': s['ticker'], 'sector': s['sector'],
            'policy_items_30d': len(hits),
            'top_themes': ','.join(
                sorted({t for th in hits['themes'].fillna('')
                        for t in th.split(',') if t})[:4]),
        })
    return pd.DataFrame(rows).set_index('ticker')


def sector_heat(as_of: date | str, lookback_days: int = 30) -> pd.DataFrame:
    """Policy document count per sector — drives the Intel tab heat panel."""
    df = policy_asof(as_of, lookback_days)
    if df.empty:
        return pd.DataFrame()
    counts: dict[str, int] = {}
    for _i, r in df.iterrows():
        for s in str(r['affected_sectors']).split(','):
            if s:
                counts[s] = counts.get(s, 0) + 1
    if not counts:
        return pd.DataFrame()
    return (pd.DataFrame({'sector': list(counts), 'documents': list(counts.values())})
            .sort_values('documents', ascending=False))
