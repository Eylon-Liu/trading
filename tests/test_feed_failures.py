"""A feed that cannot be reached is a failure to report, not a quiet day."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from data import news as NEWS
from data import policy as POLICY
from reports import builder as RB


def _article(ticker='AAPL'):
    return [{'ticker': ticker, 'title': f'{ticker} raises guidance', 'url': 'https://x.test/a',
             'published': datetime.utcnow() - timedelta(hours=2), 'source': 'Yahoo',
             'body': 'Guidance raised for the year.'}]


@pytest.fixture
def feeds(monkeypatch):
    """Control each of the three news sources: None = unreachable."""
    state = {'yf': [], 'rss': []}
    monkeypatch.setattr(NEWS, '_from_yfinance', lambda t: state['yf'])
    monkeypatch.setattr(NEWS, '_from_rss', lambda url, params, t, src: state['rss'])
    return state


def test_every_source_down_raises_instead_of_returning_nothing(feeds):
    feeds.update(yf=None, rss=None)
    with pytest.raises(NEWS.NewsUnavailable):
        NEWS.fetch_for_ticker('AAPL')


def test_one_source_answering_is_enough(feeds):
    feeds.update(yf=None, rss=_article())
    rows = NEWS.fetch_for_ticker('AAPL')
    assert len(rows) == 1 and rows[0]['ticker'] == 'AAPL'


def test_an_empty_answer_is_a_quiet_day_not_a_failure(feeds):
    feeds.update(yf=[], rss=None)
    assert NEWS.fetch_for_ticker('AAPL') == []


def test_unreachable_feeds_are_logged_as_failures_and_warned_about(fresh_db, feeds):
    feeds.update(yf=None, rss=None)
    with pytest.raises(NEWS.NewsUnavailable, match='any of 2 tickers'):
        NEWS.update_news(['AAPL', 'MSFT'])
    log = fresh_db.read_sql("SELECT key, last_success, last_error FROM ingest_log "
                            "WHERE source = 'news'")
    assert len(log) == 2 and log['last_error'].notna().all()
    assert log['last_success'].isna().all()              # not a "successful refresh"
    warnings = RB.feed_warnings()
    assert any('News could not be fetched' in w for w in warnings)


def test_a_later_success_clears_the_warning(fresh_db, feeds):
    feeds.update(yf=None, rss=None)
    with pytest.raises(NEWS.NewsUnavailable):
        NEWS.update_news(['AAPL'])
    feeds.update(yf=_article(), rss=[])
    assert NEWS.update_news(['AAPL']) == 1
    assert NEWS.LAST_RUN['failed'] == 0
    assert RB.feed_warnings() == []


def test_partial_outage_is_reported_as_partial(fresh_db, feeds, monkeypatch):
    tickers = [f'T{i:02d}' for i in range(10)]
    monkeypatch.setattr(NEWS, '_from_yfinance',
                        lambda t: None if t < 'T04' else _article(t))
    feeds.update(rss=None)
    assert NEWS.update_news(tickers) == 6
    assert NEWS.LAST_RUN == {'tickers': 10, 'failed': 4, 'written': 6, 'skipped': 0}
    # Four of ten is under the "at least five" floor: one flaky ticker in a
    # small watchlist should not put a banner on every email.
    assert RB.feed_warnings() == []
    monkeypatch.setattr(NEWS, '_from_yfinance',
                        lambda t: None if t < 'T07' else _article(t))
    NEWS.update_news(tickers)
    assert any('7 of 10 tickers' in w for w in RB.feed_warnings())


def test_stale_articles_are_flagged_even_when_refreshes_report_success(fresh_db):
    old = datetime.utcnow() - timedelta(days=6)
    fresh_db.upsert(fresh_db.news, [{'id': 'a', 'ticker': 'AAPL', 'published': old,
                                     'title': 'old'}])
    fresh_db.record_ingest('news', 'AAPL', rows=0)
    warnings = RB.feed_warnings()
    assert len(warnings) == 1 and 'No article newer than' in warnings[0]


def test_weekend_digest_shows_the_warning(fresh_db, feeds):
    # The real path: a Saturday, feeds down, digest gathered from the store.
    from datetime import date
    feeds.update(yf=None, rss=None)
    with pytest.raises(NEWS.NewsUnavailable):
        NEWS.update_news(['AAPL'])
    digest = RB.gather_digest(date(2026, 11, 14), now=datetime(2026, 11, 14, 12, 0))
    digest['feed_warnings'] = RB.feed_warnings()
    html = RB.render_news_digest(digest)
    assert 'Weekend News Digest' in html
    assert 'News could not be fetched' in html


def test_federal_register_down_is_an_error_not_an_empty_month(fresh_db, monkeypatch):
    monkeypatch.setattr(POLICY.http, 'fetch_json', lambda *a, **k: None)
    with pytest.raises(RuntimeError, match='Federal Register'):
        POLICY.fetch_policy()
    fresh_db.record_ingest('policy', 'federal_register', error='Federal Register did not answer')
    assert any('Federal Register' in w for w in RB.feed_warnings())


# ── bounded time: the morning email must not wait on a dead feed ─

def test_a_dead_source_is_dropped_after_a_few_failures(monkeypatch):
    calls = []
    monkeypatch.setattr(NEWS.http, 'fetch', lambda url, **kw: calls.append(url))  # → None
    NEWS._strikes.clear()
    for i in range(20):
        assert NEWS._from_rss('https://feed.test/rss', {}, f'T{i}', 'Yahoo RSS') is None
    assert len(calls) == NEWS._BREAKER_LIMIT            # not 20 × retries


def test_news_refresh_stops_at_its_time_budget_and_keeps_priorities(fresh_db, monkeypatch):
    import time
    fresh_db.upsert(fresh_db.watchlist, [{'ticker': 'ZZZ'}])
    order = []

    def slow(ticker, name=None):
        order.append(ticker)
        time.sleep(0.05)
        return []

    monkeypatch.setattr(NEWS, 'fetch_for_ticker', slow)
    monkeypatch.setattr(NEWS.config, 'MAX_WORKERS', 2)       # → 2 workers
    tickers = [f'T{i:03d}' for i in range(60)] + ['ZZZ']
    NEWS.update_news(tickers, budget_seconds=0.2)
    assert order[0] == 'ZZZ'                              # the watched name went first
    assert NEWS.LAST_RUN['skipped'] > 20                  # stopped early
    done = set(fresh_db.read_sql("SELECT key FROM ingest_log WHERE source='news'")['key'])
    assert 'ZZZ' in done and len(done) < 40
    # Next refresh: the ones left out are tried before the ones just done.
    nxt = NEWS._fetch_order(tickers)
    left_out = [t for t in tickers if t not in done]
    assert nxt[0] == 'ZZZ' and set(nxt[1:1 + len(left_out)]) == set(left_out)
