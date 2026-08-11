"""
News provider — company coverage with sentiment and direction extraction.

Three free sources, no API key: yfinance's `.news` (which carries real summary
text, not just headlines, and kept working while the price endpoints were
rate-limited), Yahoo's RSS feed, and Google News RSS.

The honest limitation, stated plainly because it governs how the output may be
used: free news APIs return only recent items. This stream cannot be
backfilled, so it powers the live screen from day one and becomes backtestable
only as history accumulates. Everything filing-derived (8-K events, Form 4,
risk-factor drift) carries a filing date and is backtestable immediately —
which is why the strategies lean on those instead.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import pandas as pd

import config
from core import db, http
from nlp import extract as EX
from nlp import sentiment as SENT
from nlp import summarize as SUM

log = logging.getLogger(__name__)

YAHOO_RSS = 'https://feeds.finance.yahoo.com/rss/2.0/headline'
GOOGLE_RSS = 'https://news.google.com/rss/search'


def _mk_id(ticker: str, title: str, published) -> str:
    raw = f'{ticker}|{(title or "").strip().lower()}|{published}'
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ─────────────────────────────────────────────
# SOURCES
# ─────────────────────────────────────────────

def _from_yfinance(ticker: str) -> list[dict]:
    """yfinance `.news` — richest of the three, includes a real summary."""
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception as exc:                       # noqa: BLE001
        log.debug('yfinance news failed for %s: %s', ticker, exc)
        return []

    out = []
    for it in items:
        content = it.get('content') or it
        title = content.get('title')
        if not title:
            continue
        pub = content.get('pubDate') or content.get('displayTime')
        try:
            published = pd.to_datetime(pub).tz_localize(None) if pub else datetime.utcnow()
        except (ValueError, TypeError):
            published = datetime.utcnow()

        provider = (content.get('provider') or {}).get('displayName', 'Yahoo')
        url = ((content.get('canonicalUrl') or {}).get('url')
               or (content.get('clickThroughUrl') or {}).get('url') or '')
        body = content.get('summary') or content.get('description') or ''

        out.append({'ticker': ticker, 'title': title, 'url': url,
                    'published': published, 'source': provider, 'body': body})
    return out


def _from_rss(url: str, params: dict, ticker: str, source: str) -> list[dict]:
    raw = http.fetch(url, params=params, category='news')
    if not raw:
        return []
    try:
        import feedparser
        feed = feedparser.parse(raw)
    except Exception as exc:                       # noqa: BLE001
        log.debug('rss parse failed (%s): %s', source, exc)
        return []

    out = []
    for entry in feed.entries[:config.MAX_NEWS_PER_TICKER]:
        title = getattr(entry, 'title', None)
        if not title:
            continue
        published = datetime.utcnow()
        if getattr(entry, 'published_parsed', None):
            try:
                published = datetime(*entry.published_parsed[:6])
            except (TypeError, ValueError):
                pass
        out.append({
            'ticker': ticker, 'title': title,
            'url': getattr(entry, 'link', ''), 'published': published,
            'source': source,
            'body': getattr(entry, 'summary', '') or getattr(entry, 'description', ''),
        })
    return out


def fetch_for_ticker(ticker: str, company_name: str | None = None) -> list[dict]:
    """Gather, deduplicate, score and structure coverage for one ticker."""
    items = _from_yfinance(ticker)
    items += _from_rss(YAHOO_RSS, {'s': ticker, 'region': 'US', 'lang': 'en-US'},
                       ticker, 'Yahoo RSS')
    query = f'{company_name} stock' if company_name else f'{ticker} stock'
    items += _from_rss(GOOGLE_RSS, {'q': query, 'hl': 'en-US', 'gl': 'US',
                                    'ceid': 'US:en'}, ticker, 'Google News')

    seen, unique = set(), []
    for it in items:
        key = (it['title'] or '').strip().lower()[:90]
        if key and key not in seen:
            seen.add(key)
            unique.append(it)

    cutoff = datetime.utcnow() - timedelta(days=config.NEWS_LOOKBACK_DAYS)
    unique = [i for i in unique if i['published'] >= cutoff]
    unique.sort(key=lambda x: x['published'], reverse=True)
    unique = unique[:config.MAX_NEWS_PER_TICKER]

    rows = []
    for it in unique:
        text = f"{it['title']}. {it['body']}"
        sent = SENT.score(text)
        ex = EX.extract(text)
        rows.append({
            'id': _mk_id(ticker, it['title'], it['published']),
            'ticker': ticker, 'published': it['published'],
            'source': it['source'], 'title': it['title'][:500],
            'url': (it['url'] or '')[:500],
            'body_summary': SUM.summarize(it['body'], max_sentences=2) if it['body'] else '',
            'sentiment': sent['polarity'], 'sentiment_label': sent['label'],
            'events': ','.join(ex.events)[:200],
            'extracted_json': pd.io.json.dumps(ex.to_dict())
            if hasattr(pd.io.json, 'dumps') else str(ex.to_dict()),
            'fetched_at': datetime.utcnow(),
        })
    return rows


def update_news(tickers: list[str]) -> int:
    """Fetch and store coverage for each ticker."""
    if not tickers:
        return 0

    names = {}
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    meta = db.read_sql(f'SELECT ticker, name FROM securities WHERE ticker IN ({ph})',
                       {f't{i}': t for i, t in enumerate(tickers)})
    if not meta.empty:
        names = dict(zip(meta['ticker'], meta['name']))

    written = 0
    with ThreadPoolExecutor(max_workers=max(2, config.MAX_WORKERS // 2)) as pool:
        futures = {pool.submit(fetch_for_ticker, t, names.get(t)): t
                   for t in sorted(set(tickers))}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:               # noqa: BLE001
                log.warning('news failed for %s: %s', t, exc)
                continue
            if rows:
                written += db.upsert(db.news, rows)
                db.record_ingest('news', t, rows=len(rows))

    log.info('news: %d articles across %d tickers', written, len(tickers))
    return written


# ─────────────────────────────────────────────
# READS  (point-in-time gated)
# ─────────────────────────────────────────────

def news_asof(tickers: list[str], as_of: date | str,
              lookback_days: int | None = None) -> pd.DataFrame:
    """Articles published at or before `as_of`."""
    if not tickers:
        return pd.DataFrame()
    lookback_days = lookback_days or config.NEWS_LOOKBACK_DAYS
    end = pd.to_datetime(as_of)
    start = end - timedelta(days=lookback_days)

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params.update({'lo': str(start), 'hi': str(end + timedelta(days=1))})

    return db.read_sql(f"""
        SELECT ticker, published, source, title, url, body_summary,
               sentiment, sentiment_label, events, extracted_json
        FROM news
        WHERE ticker IN ({ph}) AND published >= :lo AND published < :hi
        ORDER BY published DESC
    """, params, parse_dates=['published'])


def sentiment_scores(tickers: list[str], as_of: date | str,
                     lookback_days: int = 14) -> pd.DataFrame:
    """Per-ticker aggregate sentiment and coverage volume."""
    df = news_asof(tickers, as_of, lookback_days)
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby('ticker').agg(
        news_sentiment=('sentiment', 'mean'),
        news_count=('sentiment', 'size'),
        news_sentiment_std=('sentiment', 'std'),
    )
    # Weight recent coverage more heavily than three-week-old coverage.
    weighted = {}
    ref = pd.to_datetime(as_of)
    for t, grp in df.groupby('ticker'):
        age_days = (ref - grp['published']).dt.total_seconds() / 86400.0
        w = 0.5 ** (age_days / 7.0)            # one-week half-life
        if w.sum() > 0:
            weighted[t] = float((grp['sentiment'] * w).sum() / w.sum())
    agg['news_sentiment_weighted'] = pd.Series(weighted)
    return agg


def direction_summary(ticker: str, as_of: date | str,
                      lookback_days: int = 30) -> dict:
    """
    What recent coverage implies about where the company is heading.

    This is the part that matters more than the sentiment number: guidance
    changes, capital allocation, strategy shifts, management changes.
    """
    df = news_asof([ticker], as_of, lookback_days)
    if df.empty:
        return {'n_articles': 0, 'direction_notes': []}

    extractions = []
    for _i, row in df.iterrows():
        text = f"{row['title']}. {row.get('body_summary') or ''}"
        extractions.append(EX.extract(text))

    agg = EX.aggregate(extractions)
    agg['mean_sentiment'] = float(df['sentiment'].mean())
    agg['headlines'] = df.head(5)[['published', 'title', 'sentiment_label']] \
        .to_dict('records')
    return agg
