"""
Alternative data — retail attention from Wikipedia pageviews.

Wikimedia's pageviews API is free, needs no key, and returns full daily
history, which makes it one of the few attention proxies that is genuinely
backtestable. A spike in pageviews for a company's article is a reasonable
stand-in for retail interest, and it leads or coincides with retail flow.

The other alt-data signals (insider transactions, 8-K event intensity) come
straight from EDGAR and live in data/sec.py, since they are filings.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd

import config
from core import db, http

log = logging.getLogger(__name__)

PAGEVIEWS = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'
             'en.wikipedia/all-access/user/{article}/daily/{start}/{end}')
WIKI_SEARCH = 'https://en.wikipedia.org/w/api.php'


def _article_for(ticker: str, company_name: str | None) -> str | None:
    """Resolve a company to its Wikipedia article title."""
    query = company_name or ticker
    payload = http.fetch_json(WIKI_SEARCH, category='members',
                              ttl=60 * 60 * 24 * 30,
                              params={'action': 'query', 'list': 'search',
                                      'srsearch': query, 'srlimit': 1,
                                      'format': 'json'})
    if not payload:
        return None
    try:
        hits = payload['query']['search']
        return hits[0]['title'].replace(' ', '_') if hits else None
    except (KeyError, IndexError):
        return None


def _fetch_views(ticker: str, article: str, start: date, end: date) -> list[dict]:
    url = PAGEVIEWS.format(article=article,
                           start=start.strftime('%Y%m%d'),
                           end=end.strftime('%Y%m%d'))
    # The URL embeds yesterday's date, so a cached copy is never read again.
    payload = http.fetch_json(url, category='_default', use_cache=False)
    if not payload or 'items' not in payload:
        return []

    rows = []
    for it in payload['items']:
        try:
            d = pd.to_datetime(it['timestamp'][:8]).date()
        except (KeyError, ValueError):
            continue
        rows.append({'ticker': ticker, 'date': d,
                     'wiki_views': float(it.get('views', 0))})
    return rows


def update_attention(tickers: list[str], days_back: int = 730,
                     overlap_days: int = 3) -> int:
    """Fetch and store daily pageviews, from each ticker's last stored day on."""
    if not tickers:
        return 0

    end = date.today() - timedelta(days=1)
    full_start = end - timedelta(days=days_back)

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params = {f't{i}': t for i, t in enumerate(tickers)}
    meta = db.read_sql(f'SELECT ticker, name FROM securities WHERE ticker IN ({ph})',
                       params)
    names = dict(zip(meta['ticker'], meta['name'])) if not meta.empty else {}
    have = db.read_sql(f'SELECT ticker, MAX(date) AS last FROM attention '
                       f'WHERE ticker IN ({ph}) GROUP BY ticker', params)
    last = dict(zip(have['ticker'], pd.to_datetime(have['last']).dt.date)) \
        if not have.empty else {}

    written = 0
    with ThreadPoolExecutor(max_workers=max(2, config.MAX_WORKERS // 2)) as pool:
        futures = {}
        for t in sorted(set(tickers)):
            start = max(full_start, last[t] - timedelta(days=overlap_days)) \
                if t in last else full_start
            if start > end:
                continue
            article = _article_for(t, names.get(t))
            if article:
                futures[pool.submit(_fetch_views, t, article, start, end)] = t

        for fut in as_completed(futures):
            t = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:               # noqa: BLE001
                log.debug('attention failed for %s: %s', t, exc)
                continue
            if rows:
                written += db.upsert(db.attention, rows)

    db.record_ingest('attention', str(date.today()), rows=written)
    log.info('attention: %d daily observations', written)
    return written
