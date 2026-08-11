"""
External market-cap fill.

The danger with a live quote is not that it is wrong, it is that it has no
history. Fetched today, it describes today, and letting it reach a historical
run would be look-ahead of the worst kind — invisible, and flattering. These
tests pin down that it cannot.

The second property is precedence: a filed share count always wins, so the
audit trail in the Methodology tab holds for every name that has one.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import config
from data import marketdata as MD


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(config, 'FINNHUB_API_KEY', None)


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(config, 'FINNHUB_API_KEY', 'test-key')


# ── availability ──────────────────────────────────────────────────

def test_disabled_without_a_key(no_key):
    assert MD.available() is False
    assert MD.status()['enabled'] is False
    assert MD.update_market_caps(['BRK-B']) == 0


def test_no_network_call_without_a_key(no_key, monkeypatch):
    def explode(*a, **kw):
        raise AssertionError('must not fetch without a key')

    monkeypatch.setattr(MD.http, 'fetch_json', explode)
    assert MD.fetch_market_caps(['BRK-B']).empty


# ── precedence: SEC wins ──────────────────────────────────────────

def test_only_names_without_a_filed_count_are_fetched(with_key, monkeypatch):
    """A filed share count is auditable; an API number is not. SEC wins."""
    monkeypatch.setattr(MD.db, 'read_sql',
                        lambda *a, **kw: pd.DataFrame({'ticker': ['AAPL', 'MSFT']}))
    out = MD.gaps(['AAPL', 'MSFT', 'BRK-B', 'GOOGL'], date(2026, 8, 11))
    assert out == ['BRK-B', 'GOOGL']


def test_no_gaps_means_no_calls(with_key, monkeypatch):
    monkeypatch.setattr(MD, 'gaps', lambda *a, **kw: [])

    def explode(*a, **kw):
        raise AssertionError('nothing to fill, so nothing should be fetched')

    monkeypatch.setattr(MD, 'fetch_market_caps', explode)
    assert MD.update_market_caps(['AAPL']) == 0


# ── point-in-time safety ──────────────────────────────────────────

def test_snapshot_is_stamped_with_the_run_date(with_key, monkeypatch):
    """
    A quote has no history. Stamping it with today's date, and reading through
    `snapshot_date <= as_of`, is what stops a 2026 fetch reaching a 2021 run.
    """
    monkeypatch.setattr(MD, 'gaps', lambda *a, **kw: ['BRK-B'])
    monkeypatch.setattr(MD, 'fetch_market_caps', lambda ts: pd.DataFrame(
        [{'ticker': 'BRK-B', 'market_cap': 1.1e12, 'shares_out': 2.16e9}]))
    monkeypatch.setattr(MD.db, 'record_ingest', lambda *a, **kw: None)

    captured = {}

    def fake_upsert(table, rows):
        captured['rows'] = rows
        return len(rows)

    monkeypatch.setattr(MD.db, 'upsert', fake_upsert)

    MD.update_market_caps(['BRK-B'], as_of=date(2026, 8, 11))
    row = captured['rows'][0]
    assert row['snapshot_date'] == date(2026, 8, 11)
    assert row['market_cap'] == 1.1e12


def test_a_todays_snapshot_is_invisible_to_a_historical_read():
    """The gate that makes the whole approach safe, asserted directly."""
    from data import yahoo

    sql_seen = {}

    class FakeDB:
        @staticmethod
        def read_sql(sql, params=None, **kw):
            sql_seen['sql'] = sql
            sql_seen['params'] = params
            return pd.DataFrame()

    import unittest.mock as mock
    with mock.patch.object(yahoo, 'db', FakeDB):
        yahoo.profile_asof(['BRK-B'], '2021-06-30')

    assert 'snapshot_date <= :ts' in sql_seen['sql'].replace('\n', ' ')
    assert sql_seen['params']['ts'] == '2021-06-30'


# ── units ─────────────────────────────────────────────────────────

def test_finnhub_millions_are_converted(with_key, monkeypatch):
    """Finnhub reports market cap in millions; callers expect absolute units."""
    monkeypatch.setattr(MD.http, 'fetch_json', lambda *a, **kw: {
        'marketCapitalization': 1_100_000.0, 'shareOutstanding': 2160.0})
    df = MD.fetch_market_caps(['BRK-B'])
    assert df.iloc[0]['market_cap'] == pytest.approx(1.1e12)
    assert df.iloc[0]['shares_out'] == pytest.approx(2.16e9)


def test_a_response_without_market_cap_is_skipped(with_key, monkeypatch):
    monkeypatch.setattr(MD.http, 'fetch_json', lambda *a, **kw: {'name': 'X'})
    assert MD.fetch_market_caps(['NOPE']).empty
