"""
Sync-phase tests.

Two properties matter and both are load-bearing:

  * A fresh source is not re-fetched. That is the whole speed argument, and
    it is also what keeps the data version stable so the factor cache hits.
  * A stale source *is* re-fetched. A cache that never refreshes is worse
    than no cache.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from data import sync as SY


@pytest.fixture
def log_rows(monkeypatch):
    """Fake ingest_log contents for a source."""
    def _set(rows):
        df = pd.DataFrame(rows or [], columns=['key', 'last_success'])
        if not df.empty:
            df['last_success'] = pd.to_datetime(df['last_success'])
        monkeypatch.setattr(SY, '_last_success', lambda source: df)
    return _set


NOW = datetime(2026, 8, 11, 12, 0)


def _ago(hours):
    return NOW - timedelta(hours=hours)


# ── freshness windows ─────────────────────────────────────────────

def test_recently_fetched_ticker_is_skipped(log_rows):
    log_rows([{'key': 'AAPL', 'last_success': _ago(1)}])
    assert SY.stale_keys('news', ['AAPL'], now=NOW) == []


def test_ticker_past_its_window_is_refetched(log_rows):
    # news refreshes after 3h
    log_rows([{'key': 'AAPL', 'last_success': _ago(5)}])
    assert SY.stale_keys('news', ['AAPL'], now=NOW) == ['AAPL']


def test_never_fetched_ticker_is_stale(log_rows):
    log_rows([])
    assert SY.stale_keys('prices', ['AAPL', 'MSFT'], now=NOW) == ['AAPL', 'MSFT']


def test_only_the_stale_subset_is_returned(log_rows):
    """A universe where most names are current fetches only the remainder."""
    log_rows([{'key': 'AAPL', 'last_success': _ago(1)},
              {'key': 'MSFT', 'last_success': _ago(1)},
              {'key': 'NVDA', 'last_success': _ago(99)}])
    stale = SY.stale_keys('news', ['AAPL', 'MSFT', 'NVDA', 'AMZN'], now=NOW)
    assert stale == ['NVDA', 'AMZN']


def test_force_ignores_freshness(log_rows):
    log_rows([{'key': 'AAPL', 'last_success': _ago(0.1)}])
    assert SY.stale_keys('news', ['AAPL'], now=NOW, force=True) == ['AAPL']


def test_failed_rows_do_not_count_as_fresh(log_rows):
    """last_success is NULL after an error, so the next run retries."""
    log_rows([{'key': 'AAPL', 'last_success': None}])
    assert SY.stale_keys('prices', ['AAPL'], now=NOW) == ['AAPL']


# ── key-mode mapping ──────────────────────────────────────────────

def test_prefixed_keys_resolve_to_their_ticker(log_rows):
    """sec_facts logs 'TICKER:CIK'; callers ask about the ticker."""
    log_rows([{'key': 'AAPL:0000320193', 'last_success': _ago(1)}])
    assert SY.stale_keys('facts', ['AAPL'], now=NOW) == []


def test_prefixed_key_still_expires(log_rows):
    log_rows([{'key': 'AAPL:0000320193', 'last_success': _ago(48)}])
    assert SY.stale_keys('facts', ['AAPL'], now=NOW) == ['AAPL']


def test_global_source_is_fresh_for_every_ticker(log_rows):
    """Whole-market providers refresh everything in one call."""
    log_rows([{'key': '2026-08-11', 'last_success': _ago(1)}])
    assert SY.stale_keys('securities', ['AAPL', 'MSFT'], now=NOW) == []


def test_global_source_expires_as_a_whole(log_rows):
    log_rows([{'key': '2026-01-01', 'last_success': _ago(24 * 40)}])
    assert SY.stale_keys('securities', ['AAPL'], now=NOW) == ['AAPL']


def test_facts_reads_its_own_log_source_name():
    """The source is keyed 'sec_facts' in ingest_log, not 'facts'."""
    assert SY.SOURCES['facts'].log == 'sec_facts'
    assert SY.SOURCES['prices'].log == 'prices'


# ── session clock ─────────────────────────────────────────────────

def test_todays_bar_is_not_expected_before_the_close():
    """Otherwise every pre-close run declares the universe stale."""
    assert SY.last_session(datetime(2026, 8, 11, 9)) == pd.Timestamp('2026-08-10').date()


def test_todays_bar_is_expected_after_the_close():
    assert SY.last_session(datetime(2026, 8, 11, 22)) == pd.Timestamp('2026-08-11').date()


@pytest.mark.parametrize('when,expected', [
    (datetime(2026, 8, 15, 12), '2026-08-14'),   # Saturday -> Friday
    (datetime(2026, 8, 16, 12), '2026-08-14'),   # Sunday   -> Friday
    (datetime(2026, 8, 17, 9), '2026-08-14'),    # Monday pre-close -> Friday
])
def test_weekends_roll_back_to_friday(when, expected):
    assert SY.last_session(when) == pd.Timestamp(expected).date()


# ── report semantics ──────────────────────────────────────────────

def test_report_is_unchanged_when_nothing_was_written():
    rep = SY.SyncReport(results=[
        SY.SourceResult('prices', 'fresh'),
        SY.SourceResult('news', 'fresh'),
    ])
    assert rep.changed is False
    assert 'fresh' in rep.summary()


def test_report_is_changed_when_rows_were_written():
    rep = SY.SyncReport(results=[SY.SourceResult('news', 'fetched', rows=12)])
    assert rep.changed is True
    assert rep.rows == 12


def test_a_fetch_that_wrote_nothing_is_not_a_change():
    """No rows means the data version does not move, so caches stay valid."""
    rep = SY.SyncReport(results=[SY.SourceResult('news', 'fetched', rows=0)])
    assert rep.changed is False


def test_failure_is_not_a_change_and_does_not_raise():
    rep = SY.SyncReport(results=[SY.SourceResult('news', 'failed')])
    assert rep.changed is False


def test_one_source_failing_does_not_stop_the_rest(monkeypatch):
    """A dead provider must degrade to stored data, not abort the run."""
    calls = []

    def fake_sync_one(name, tickers, *, index, force, emit):
        calls.append(name)
        if name == 'news':
            raise RuntimeError('provider down')
        return SY.SourceResult(name, 'fresh')

    monkeypatch.setattr(SY, '_sync_one', fake_sync_one)
    monkeypatch.setattr(SY.db, 'record_ingest', lambda *a, **kw: None)

    rep = SY.sync(['AAPL'], sources=['prices', 'news', 'facts'])
    assert calls == ['prices', 'news', 'facts']
    actions = {r.name: r.action for r in rep.results}
    assert actions == {'prices': 'fresh', 'news': 'failed', 'facts': 'fresh'}


def test_every_source_has_a_positive_window():
    for name, src in SY.SOURCES.items():
        assert src.max_age_hours > 0, name
        assert src.key_mode in ('ticker', 'ticker_prefix', 'global'), name
