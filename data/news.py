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
import json
import logging
import threading
import time
import re
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


def normalize_title(title: str | None) -> str:
    """Case- and whitespace-insensitive form of a headline, for identity."""
    return re.sub(r'\s+', ' ', (title or '').strip().lower())


def _mk_id(ticker: str, title: str, published=None) -> str:
    """
    Stable article id: ticker plus normalised headline.

    The publish time used to be part of the id. Items with no timestamp were
    stamped with the fetch time, so each refresh (every three hours) minted a
    new id for the same article and stored it again — and the copy then read
    as fresh news in the next report. `published` is accepted and ignored so
    older callers keep working.
    """
    raw = f'{ticker}|{normalize_title(title)}'
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ─────────────────────────────────────────────
# SOURCES
# ─────────────────────────────────────────────

class NewsUnavailable(RuntimeError):
    """No news source answered — distinct from "answered, nothing new"."""


# What the last update_news() call saw, for the sync report.
LAST_RUN: dict = {'tickers': 0, 'failed': 0, 'written': 0}

# A source that fails this many times in a row is left alone for the rest of
# the run. Each failed request costs its retries and backoff; with a feed
# down, 500 tickers of that was most of an hour spent learning one fact.
_BREAKER_LIMIT = 5
_strikes: dict[str, int] = {}
_strikes_lock = threading.Lock()


def _source_open(name: str) -> bool:
    return _strikes.get(name, 0) < _BREAKER_LIMIT


def _note(name: str, ok: bool) -> None:
    with _strikes_lock:
        _strikes[name] = 0 if ok else _strikes.get(name, 0) + 1
        if not ok and _strikes[name] == _BREAKER_LIMIT:
            log.warning('news: %s failed %d times in a row — skipping it for '
                        'the rest of this run', name, _BREAKER_LIMIT)


def _from_yfinance(ticker: str) -> list[dict] | None:
    """yfinance `.news` — richest of the three, includes a real summary.

    Returns None when the source could not be reached, [] when it answered
    with nothing. The difference matters: the first is an outage to report,
    the second is a quiet day.
    """
    if not _source_open('yfinance'):
        return None
    try:
        import yfinance as yf
        items = yf.Ticker(ticker).news or []
    except Exception as exc:                       # noqa: BLE001
        log.debug('yfinance news failed for %s: %s', ticker, exc)
        _note('yfinance', False)
        return None
    _note('yfinance', True)

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


def _from_rss(url: str, params: dict, ticker: str, source: str) -> list[dict] | None:
    """None when the feed could not be fetched or parsed; [] when it is empty."""
    if not _source_open(source):
        return None
    raw = http.fetch(url, params=params, category='news', retries=2)
    if not raw:
        _note(source, False)
        return None
    try:
        import feedparser
        feed = feedparser.parse(raw)
    except Exception as exc:                       # noqa: BLE001
        log.debug('rss parse failed (%s): %s', source, exc)
        _note(source, False)
        return None
    _note(source, True)

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
    """Gather, deduplicate, score and structure coverage for one ticker.

    Raises NewsUnavailable when every source failed. One source answering —
    even with nothing — is a real answer, and the others are redundancy.
    """
    query = f'{company_name} stock' if company_name else f'{ticker} stock'
    answers = [
        _from_yfinance(ticker),
        _from_rss(YAHOO_RSS, {'s': ticker, 'region': 'US', 'lang': 'en-US'},
                  ticker, 'Yahoo RSS'),
        _from_rss(GOOGLE_RSS, {'q': query, 'hl': 'en-US', 'gl': 'US',
                               'ceid': 'US:en'}, ticker, 'Google News'),
    ]
    if all(a is None for a in answers):
        raise NewsUnavailable(f'no news source answered for {ticker}')
    items = [it for a in answers if a for it in a]

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
            # json, not repr: pandas 2 removed pd.io.json.dumps, and the
            # fallback stored a Python repr that no JSON reader can parse.
            'extracted_json': json.dumps(ex.to_dict(), default=str),
            'fetched_at': datetime.utcnow(),
        })
    return rows


def _fetch_order(tickers: list[str]) -> list[str]:
    """
    The names you hold or watch and the current top picks first, then the
    rest oldest-refresh-first — so when the time budget runs out, what is
    left unfetched is what matters least and waited least.
    """
    wanted = set(tickers)
    first: list[str] = []
    try:
        mine = db.read_sql("SELECT ticker FROM positions WHERE status = 'open' "
                           "UNION SELECT ticker FROM watchlist")['ticker'].tolist()
        picks = db.read_sql(
            "SELECT DISTINCT s.ticker FROM scores s JOIN runs r ON r.run_id = s.run_id "
            "WHERE r.as_of = (SELECT MAX(as_of) FROM runs) AND s.rank <= 20")['ticker'].tolist()
        first = [t for t in dict.fromkeys(mine + sorted(picks)) if t in wanted]
    except Exception as exc:                       # noqa: BLE001
        log.debug('news priority unavailable: %s', exc)
    seen = db.read_sql("SELECT key, last_success FROM ingest_log WHERE source = 'news'")
    # Never refreshed (or only ever failed) sorts first, as the oldest.
    age = {k: ('' if pd.isna(v) else str(v))
           for k, v in zip(seen['key'], seen['last_success'])}
    rest = sorted(wanted - set(first), key=lambda t: (age.get(t, ''), t))
    return first + rest


def update_news(tickers: list[str], budget_seconds: float | None = None) -> int:
    """
    Fetch and store coverage for each ticker, within a time budget.

    The budget (config.NEWS_BUDGET_SECONDS) keeps the morning email on time
    whatever the feeds are doing: names not reached stay stale and are first
    in line, after your own, at the next refresh.
    """
    if not tickers:
        return 0
    budget = config.NEWS_BUDGET_SECONDS if budget_seconds is None else budget_seconds
    deadline = time.monotonic() + budget
    with _strikes_lock:
        _strikes.clear()

    names = {}
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    meta = db.read_sql(f'SELECT ticker, name FROM securities WHERE ticker IN ({ph})',
                       {f't{i}': t for i, t in enumerate(tickers)})
    if not meta.empty:
        names = dict(zip(meta['ticker'], meta['name']))

    written = 0
    failed: list[str] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(2, config.MAX_WORKERS // 2)) as pool:
        futures = {pool.submit(fetch_for_ticker, t, names.get(t)): t
                   for t in _fetch_order(tickers)}
        for fut in as_completed(futures):
            t = futures[fut]
            if fut.cancelled():
                continue
            try:
                rows = fut.result()
            except Exception as exc:               # noqa: BLE001
                # Recorded as a failure, not as "0 articles": a feed that is
                # down used to be logged as a successful refresh, and a
                # weekend digest with no news looked like a quiet weekend.
                log.warning('news failed for %s: %s', t, exc)
                failed.append(t)
                db.record_ingest('news', t, error=str(exc)[:200])
                continue
            if rows:
                # First sighting wins: an article already stored keeps its
                # original publish time instead of being re-dated to now.
                written += db.upsert(db.news, rows, on_conflict='ignore')
            db.record_ingest('news', t, rows=len(rows) if rows else 0)
            if time.monotonic() > deadline and not skipped:
                skipped = sum(1 for f in futures if f.cancel())
                if skipped:
                    log.warning('news: time budget of %ds reached — %d tickers '
                                'left for the next refresh', int(budget), skipped)

    n = len(set(tickers)) - skipped
    LAST_RUN.update(tickers=n, failed=len(failed), written=written, skipped=skipped)
    log.info('news: %d articles across %d tickers (%d unreachable, %d left '
             'for next time)', written, n, len(failed), skipped)
    if failed and len(failed) == n:
        raise NewsUnavailable(
            f'no news source answered for any of {n} tickers — check the '
            f'network; Yahoo and Google News feeds were both unreachable')
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
