"""Macro, earnings calendar, portfolio and live track record."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import config
from core import market_calendar as MC
from data import earnings as ER
from data import macro as MACRO
from data.universe import UniverseSpec
from quant import engine as EN
from quant import portfolio as PF
from quant import strategies as ST
from quant import track as TRACK


def _sessions(start, end):
    return [d for d in pd.date_range(start, end).date if MC.is_trading_day(d)]


def _prices(db, series: dict[str, list[float]], days: list[date]):
    db.upsert(db.prices, [
        {'ticker': t, 'date': d, 'open': p, 'high': p * 1.01, 'low': p * 0.99,
         'close': p, 'adj_close': p, 'volume': 1e6}
        for t, px in series.items() for d, p in zip(days, px)])


# ── macro ────────────────────────────────────────────────────────

def test_fred_csv_parses_both_header_styles_and_missing_values():
    old = 'DATE,DGS3MO\n2026-11-02,4.21\n2026-11-03,.\n2026-11-04,4.19\n'
    new = 'observation_date,DGS3MO\n2026-11-02,4.21\n2026-11-03,\n'
    assert MACRO.parse_csv(old, 'DGS3MO')['value'].tolist() == [4.21, 4.19]
    assert MACRO.parse_csv(new, 'DGS3MO')['value'].tolist() == [4.21]


def test_fred_api_payload():
    df = MACRO.parse_api({'observations': [{'date': '2026-11-02', 'value': '17.1'},
                                           {'date': '2026-11-03', 'value': '.'}]})
    assert df['value'].tolist() == [17.1]


def test_risk_free_uses_stored_bills_else_config(fresh_db):
    assert MACRO.risk_free_rate(date(2020, 1, 1), date(2020, 12, 31)) == config.RISK_FREE_RATE
    days = _sessions('2020-01-01', '2020-12-31')
    fresh_db.upsert(fresh_db.macro, [{'series_id': 'DGS3MO', 'date': d, 'value': 0.1}
                                     for d in days])
    assert MACRO.risk_free_rate(date(2020, 1, 1), date(2020, 12, 31)) == pytest.approx(0.001)


def test_regime_flags(fresh_db):
    d = date(2026, 11, 6)
    fresh_db.upsert(fresh_db.macro, [
        {'series_id': 'T10Y2Y', 'date': d, 'value': -0.3},
        {'series_id': 'VIXCLS', 'date': d, 'value': 34.0},
        {'series_id': 'BAMLH0A0HYM2', 'date': d, 'value': 6.5}])
    levels = dict((t.split()[0], lvl) for lvl, t in MACRO.regime(d))
    assert levels['Yield'] == 'watch'
    assert levels['VIX'] == 'risk'
    assert levels['Credit'] == 'risk'


# ── earnings ─────────────────────────────────────────────────────

def test_nasdaq_rows_are_normalised():
    rows = ER.parse_nasdaq({'data': {'rows': [
        {'symbol': 'BRK.B', 'time': 'time-after-hours', 'epsForecast': '($0.12)',
         'fiscalQuarterEnding': 'Sep/2026'},
        {'symbol': 'AAPL', 'time': 'time-pre-market', 'epsForecast': '$1.50'},
    ]}}, date(2026, 11, 10))
    assert rows[0]['ticker'] == 'BRK-B' and rows[0]['time'] == 'amc'
    assert rows[0]['eps_estimate'] == pytest.approx(-0.12)
    assert rows[1]['time'] == 'bmo' and rows[1]['eps_estimate'] == 1.5


def test_finnhub_rows():
    rows = ER.parse_finnhub({'earningsCalendar': [
        {'symbol': 'NVDA', 'date': '2026-11-18', 'hour': 'amc', 'epsEstimate': 1.1,
         'quarter': 3, 'year': 2027}]})
    assert rows[0]['report_date'] == date(2026, 11, 18)
    assert rows[0]['fiscal_period'] == 'Q3 2027'


def test_estimate_uses_same_week_last_year():
    past = pd.Series(pd.to_datetime(['2025-11-19', '2026-02-25', '2026-05-27',
                                     '2026-08-26']))
    assert ER.estimate_next(past, date(2026, 11, 8)) == date(2026, 11, 18)


def test_rescheduled_report_moves(fresh_db, monkeypatch):
    monkeypatch.setattr(config, 'FINNHUB_API_KEY', 'k')
    payloads = iter([
        {'earningsCalendar': [{'symbol': 'AAPL', 'date': str(date.today() + timedelta(days=5))}]},
        {'earningsCalendar': [{'symbol': 'AAPL', 'date': str(date.today() + timedelta(days=9))}]},
    ])
    monkeypatch.setattr(ER.http, 'fetch_json', lambda *a, **k: next(payloads))
    ER.update_earnings(['AAPL'])
    ER.update_earnings(['AAPL'])
    up = ER.upcoming(['AAPL'], days=14)
    assert up['report_date'].dt.date.tolist() == [date.today() + timedelta(days=9)]


def test_failed_nasdaq_day_keeps_stored_dates(fresh_db, monkeypatch):
    monkeypatch.setattr(config, 'FINNHUB_API_KEY', None)
    d = next(x for x in (date.today() + timedelta(days=i) for i in range(1, 8))
             if MC.is_trading_day(x))
    fresh_db.upsert(fresh_db.earnings_calendar, [{
        'ticker': 'AAPL', 'report_date': d, 'time': 'amc', 'source': 'nasdaq'}])
    monkeypatch.setattr(ER.http, 'fetch_json',
                        lambda *a, **k: {'data': None, 'status': {'rCode': 400}})
    ER.update_earnings(['AAPL'], days_ahead=8)
    assert len(ER.upcoming(['AAPL'], days=10)) == 1


# ── portfolio ────────────────────────────────────────────────────

@pytest.fixture
def book(fresh_db, monkeypatch):
    monkeypatch.setattr(config, 'WATCHLIST', [])
    days = _sessions('2026-09-01', '2026-11-06')
    n = len(days)
    _prices(fresh_db, {'AAA': list(np.linspace(100, 120, n)),
                       'BBB': [50.0] * (n - 1) + [44.0],
                       'SPY': [600.0] * n}, days)
    return fresh_db, days


def test_pnl_against_cost(book):
    PF.add_lot('AAA', 10, 100, opened='2026-09-01')
    PF.add_lot('AAA', 10, 110, opened='2026-10-01')
    h = PF.holdings(date(2026, 11, 6))
    r = h.loc['AAA']
    assert r['shares'] == 20 and r['avg_cost'] == pytest.approx(105)
    assert r['pnl'] == pytest.approx(20 * 120 - 2100)


def test_split_after_purchase_is_not_a_loss(book):
    db, _days = book
    PF.add_lot('AAA', 10, 1000, opened='2026-08-01')     # pre 10-for-1 basis
    db.upsert(db.splits, [{'ticker': 'AAA', 'date': date(2026, 8, 20), 'ratio': 10.0}])
    r = PF.holdings(date(2026, 11, 6)).loc['AAA']
    assert r['shares'] == 100
    assert r['pnl'] == pytest.approx(100 * 120 - 10_000)


def test_stop_and_big_move_alerts(book):
    PF.add_lot('BBB', 100, 50, stop=45)
    PF.add_lot('AAA', 1, 100, target=119)
    al = PF.alerts(date(2026, 11, 6), watch=[])
    text = {(a['ticker'], a['level']): a['text'] for a in al}
    assert ('BBB', 'risk') in text and 'Stop 45.00 hit' in text[('BBB', 'risk')]
    assert any(t == 'AAA' and 'Target' in x for (t, _l), x in text.items())
    assert any(t == 'BBB' and 'Moved' in x for (t, _l), x in text.items())


def test_close_uses_last_close_and_realises(book):
    PF.add_lot('AAA', 10, 100)
    assert PF.close('AAA', closed='2026-11-06') == 1
    assert PF.holdings(date(2026, 11, 6)).empty
    r = PF.realized()
    assert r['pnl'].iloc[0] == pytest.approx(200)


def test_watchlist_merges_env(book, monkeypatch):
    PF.watch_add(['msft', 'brk.b'])
    monkeypatch.setattr(config, 'WATCHLIST', ['NVDA'])
    assert PF.watchlist() == ['BRK-B', 'MSFT', 'NVDA']


def test_csv_import(book, tmp_path):
    f = tmp_path / 'lots.csv'
    f.write_text('Symbol,Qty,Price,Date,Stop\nAAA,5,101.5,2026-09-15,90\nBBB,3,49,,\n')
    assert PF.import_csv(f) == 2
    assert set(PF.lots()['ticker']) == {'AAA', 'BBB'}


def test_bad_lots_are_rejected(book):
    with pytest.raises(ValueError):
        PF.add_lot('AAA', 0, 100)
    with pytest.raises(ValueError):
        PF.add_lot('AAA', 1, -5)


# ── live track record ────────────────────────────────────────────

def test_track_record_grades_stored_runs(fresh_db):
    days = _sessions('2026-06-01', '2026-11-06')
    n = len(days)
    winners = ['W%d' % i for i in range(10)]
    losers = ['L%d' % i for i in range(10)]
    order = winners + losers
    # Drift falls strictly with rank, so the ordering is perfectly right.
    drift = {t: 0.002 - 0.0002 * i for i, t in enumerate(order)}
    series = {t: list(100 * (1 + drift[t]) ** np.arange(n)) for t in order}
    series['SPY'] = [100.0] * n
    _prices(fresh_db, series, days)
    spec = UniverseSpec(preset='SPY')
    for d in days[:40]:
        scores = pd.DataFrame({
            'composite': np.linspace(2, -2, 20), 'rank': range(1, 21),
            'signal': '🟢 Buy', 'signal_score': 1.0, 'reasons': '',
            'coverage': 1.0, 'sector': 'X'}, index=order)
        EN._persist(f'r{d}', d, spec, ST.get('buffett'), order, scores, pd.DataFrame())

    cohorts = TRACK.cohort_returns(as_of=days[-1], spec_json=spec.to_json())
    first = cohorts[(cohorts['horizon'] == '1W')].iloc[0]
    # Entry is the close *after* the run date.
    assert first['entry_date'] == days[1]
    expect = np.mean([(1 + drift[t]) ** 5 - 1 for t in winners])
    assert first['pick_return'] == pytest.approx(expect)
    s = TRACK.summary(cohorts)
    row = s[s['horizon'] == '1M'].iloc[0]
    assert row['hit_rate'] == 1.0
    assert row['mean_ic'] == pytest.approx(1.0)
    assert row['independent'] < row['cohorts']


# ── earnings fallbacks ───────────────────────────────────────────

class _FakeTicker:
    calendars: dict = {}

    def __init__(self, t):
        self.t = t

    @property
    def calendar(self):
        return self.calendars.get(self.t, {})


def test_yahoo_fills_followed_names_when_nasdaq_is_down(fresh_db, monkeypatch):
    import sys
    import types
    monkeypatch.setattr(config, 'FINNHUB_API_KEY', None)
    calls = []
    monkeypatch.setattr(ER.http, 'fetch_json',
                        lambda *a, **k: calls.append(1) or None)
    d1 = date.today() + timedelta(days=6)
    _FakeTicker.calendars = {
        'AAPL': {'Earnings Date': [d1], 'Earnings Average': 1.6},
        'MSFT': {'Earnings Date': [d1, d1 + timedelta(days=4)]},
    }
    monkeypatch.setitem(sys.modules, 'yfinance',
                        types.SimpleNamespace(Ticker=_FakeTicker))
    ER.update_earnings(['AAPL', 'MSFT', 'XOM'], followed=['AAPL', 'MSFT'])

    assert len(calls) <= 3, 'a dead Nasdaq must be abandoned quickly'
    assert ER.LAST_STATUS['nasdaq'] == 'failed'
    up = ER.upcoming(['AAPL', 'MSFT'], days=14).set_index('ticker')
    assert bool(up.loc['AAPL', 'confirmed']) is True
    assert up.loc['AAPL', 'eps_estimate'] == pytest.approx(1.6)
    assert bool(up.loc['MSFT', 'confirmed']) is False      # Yahoo's date range


def test_yahoo_is_skipped_for_names_nasdaq_listed(fresh_db, monkeypatch):
    import sys
    import types
    monkeypatch.setattr(config, 'FINNHUB_API_KEY', None)
    d = next(x for x in (date.today() + timedelta(days=i) for i in range(1, 8))
             if MC.is_trading_day(x))
    monkeypatch.setattr(ER.http, 'fetch_json', lambda url, params=None, **k: (
        {'data': {'rows': [{'symbol': 'AAPL', 'time': 'time-after-hours'}]}}
        if params and params.get('date') == str(d) else {'data': {'rows': None}}))
    asked = []

    class T(_FakeTicker):
        def __init__(self, t):
            asked.append(t)
            super().__init__(t)
    monkeypatch.setitem(sys.modules, 'yfinance', types.SimpleNamespace(Ticker=T))
    ER.update_earnings(['AAPL'], followed=['AAPL'])
    assert asked == []
    assert ER.upcoming(['AAPL'], days=10)['time'].tolist() == ['amc']


def test_unresponsive_fred_is_abandoned_quickly(fresh_db, monkeypatch):
    calls = []
    monkeypatch.setattr(config, 'FRED_API_KEY', None)
    monkeypatch.setattr(MACRO.http, 'fetch_text',
                        lambda *a, **k: calls.append(k) or None)
    assert MACRO.update_macro() == 0
    assert len(calls) == 2
    assert all(k.get('timeout') and k.get('retries') for k in calls)
