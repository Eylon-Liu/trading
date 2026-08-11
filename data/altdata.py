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
import re
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
    payload = http.fetch_json(url, category='_default', ttl=60 * 60 * 24)
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


def update_attention(tickers: list[str], days_back: int = 730) -> int:
    """Fetch and store daily pageviews for each ticker's article."""
    if not tickers:
        return 0

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days_back)

    names = {}
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    meta = db.read_sql(f'SELECT ticker, name FROM securities WHERE ticker IN ({ph})',
                       {f't{i}': t for i, t in enumerate(tickers)})
    if not meta.empty:
        names = dict(zip(meta['ticker'], meta['name']))

    written = 0
    with ThreadPoolExecutor(max_workers=max(2, config.MAX_WORKERS // 2)) as pool:
        futures = {}
        for t in sorted(set(tickers)):
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


def attention_asof(tickers: list[str], as_of: date | str,
                   lookback_days: int = 90) -> pd.DataFrame:
    """Pageview series up to `as_of`."""
    if not tickers:
        return pd.DataFrame()
    end = pd.to_datetime(as_of).date()
    start = end - timedelta(days=lookback_days)

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params.update({'lo': str(start), 'hi': str(end)})

    return db.read_sql(
        f'SELECT ticker, date, wiki_views FROM attention '
        f'WHERE ticker IN ({ph}) AND date BETWEEN :lo AND :hi ORDER BY date',
        params, parse_dates=['date'])
