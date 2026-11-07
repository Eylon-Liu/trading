"""
Price-store consistency: completed sessions only, no skipped one-day gaps,
and a provider re-basing (split / dividend) triggers a full refetch.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from core import market_calendar as MC
from data import yahoo


def _bars(dates, close, adj=None, ticker='TEST'):
    close = list(close)
    adj = list(adj) if adj is not None else close
    return pd.DataFrame({
        'ticker': ticker,
        'date': pd.to_datetime(list(dates)),
        'open': close, 'high': [c * 1.01 for c in close],
        'low': [c * 0.99 for c in close], 'close': close,
        'adj_close': adj, 'volume': [1e6] * len(close),
    })


def _sessions(end: date, n: int) -> list[date]:
    out, d = [], end
    while len(out) < n:
        if MC.is_trading_day(d):
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


# ── period selection ─────────────────────────────────────────────

def test_one_day_gap_is_fetched(monkeypatch):
    """The old `gap <= 1: skip` meant the evening run never got today's bar."""
    monkeypatch.setattr(yahoo, '_today_et', lambda: date(2026, 11, 10))
    p = yahoo._period_for(date(2026, 11, 9), '10y', target=date(2026, 11, 10))
    assert p == '5d'


def test_current_store_needs_no_fetch():
    assert yahoo._period_for(date(2026, 11, 10), '10y',
                             target=date(2026, 11, 10)) is None


def test_long_weekend_is_still_one_small_fetch(monkeypatch):
    monkeypatch.setattr(yahoo, '_today_et', lambda: date(2026, 11, 30))
    assert yahoo._period_for(date(2026, 11, 25), '10y',
                             target=date(2026, 11, 27)) == '1mo'


def test_empty_store_fetches_the_full_period():
    assert yahoo._period_for(None, '10y') == '10y'


# ── unfinished bars ──────────────────────────────────────────────

def test_live_session_bar_is_dropped_from_payload():
    day = date(2026, 11, 10)
    df = _bars([day - timedelta(days=1), day], [100, 101])
    midday = datetime(2026, 11, 10, 12, 0, tzinfo=MC.ET).timestamp()
    out = yahoo._drop_unfinished_bar(df, {'regularMarketTime': int(midday)})
    assert list(out['date'].dt.date) == [day - timedelta(days=1)]


def test_closed_session_bar_is_kept():
    day = date(2026, 11, 10)
    df = _bars([day - timedelta(days=1), day], [100, 101])
    after = datetime(2026, 11, 10, 16, 0, 5, tzinfo=MC.ET).timestamp()
    out = yahoo._drop_unfinished_bar(df, {'regularMarketTime': int(after)})
    assert len(out) == 2


# ── basis changes ────────────────────────────────────────────────

def test_split_is_detected_as_a_basis_change():
    days = [date(2026, 11, 2), date(2026, 11, 3)]
    stored = _bars(days, [1000, 1010])
    fresh = _bars(days, [100, 101])           # 10-for-1, history restated
    assert yahoo._basis_changed(fresh, stored)


def test_dividend_readjustment_is_detected():
    days = [date(2026, 11, 2), date(2026, 11, 3)]
    stored = _bars(days, [100, 101], adj=[100, 101])
    fresh = _bars(days, [100, 101], adj=[99.5, 100.495])   # 0.5% dividend
    assert yahoo._basis_changed(fresh, stored)


def test_rounding_noise_is_not_a_basis_change():
    days = [date(2026, 11, 2), date(2026, 11, 3)]
    stored = _bars(days, [100, 101])
    fresh = _bars(days, [100.00001, 101.00001])
    assert not yahoo._basis_changed(fresh, stored)


# ── end to end against a real store ──────────────────────────────

@pytest.fixture
def fake_provider(monkeypatch):
    """Serve canned frames per (ticker, period); record what was asked."""
    responses: dict = {}
    calls: list = []

    def fetch(ticker, period='10y', ttl=None):
        calls.append((ticker, period))
        return responses.get((ticker, period), pd.DataFrame()).copy()

    monkeypatch.setattr(yahoo, '_fetch_chart', fetch)
    return responses, calls


def test_split_triggers_full_history_rewrite(fresh_db, fake_provider, monkeypatch):
    responses, calls = fake_provider
    target = MC.last_completed_session()
    days = _sessions(target, 6)
    old, new = days[:-1], days

    # Stored history on the pre-split basis, current to the day before target.
    fresh_db.upsert(fresh_db.prices, yahoo._to_rows(
        _bars(old, [1000.0] * len(old)), 'TEST'))

    restated = _bars(new, [100.0] * len(new))
    for _days, period in yahoo._RANGES:
        responses[('TEST', period)] = restated
    responses[('TEST', 'max')] = restated

    yahoo.update_prices(['TEST'])

    px = fresh_db.read_sql('SELECT date, close FROM prices ORDER BY date')
    assert len(px) == len(new)
    assert set(px['close']) == {100.0}, 'old bars must be rewritten, not mixed'
    assert len(calls) == 2                    # incremental, then full refetch


def test_failed_refetch_keeps_store_consistent(fresh_db, fake_provider):
    responses, _calls = fake_provider
    target = MC.last_completed_session()
    days = _sessions(target, 6)
    fresh_db.upsert(fresh_db.prices, yahoo._to_rows(
        _bars(days[:-1], [1000.0] * 5), 'TEST'))

    # Only the short incremental range answers; the full refetch is empty.
    responses[('TEST', yahoo._period_for(days[-2], '10y', target))] = \
        _bars(days, [100.0] * 6)

    yahoo.update_prices(['TEST'])
    px = fresh_db.read_sql('SELECT close FROM prices')
    assert set(px['close']) == {1000.0}, 'no post-split bars mixed in'


def test_bars_after_the_last_completed_session_are_never_stored(fresh_db, fake_provider):
    responses, _calls = fake_provider
    target = MC.last_completed_session()
    live = MC.next_trading_day(target)
    days = _sessions(target, 3) + [live]
    responses[('TEST', '10y')] = _bars(days, [10.0, 11.0, 12.0, 13.0])

    yahoo.update_prices(['TEST'])
    stored = pd.to_datetime(
        fresh_db.read_sql('SELECT date FROM prices')['date']).dt.date
    assert live not in set(stored)
    assert target in set(stored)
