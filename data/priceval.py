"""
Price cross-validation — spot-check stored bars against an independent source.

Yahoo's undocumented chart API is the sole price provider. If it silently
changes schema, starts returning wrong adjustments, or maps a ticker
differently, every momentum, volatility, beta, and market-cap calculation
is wrong. This module catches that by comparing a sample of recent bars
against Finnhub (when configured).

Called opportunistically after a price sync — not on every run, and not for
every ticker. A 5-bar sample per ticker across 20 names is enough to detect
a systematic basis mismatch without doubling API calls.
"""

from __future__ import annotations

import logging
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from core import db

log = logging.getLogger(__name__)

TOLERANCE = 0.02
SAMPLE_TICKERS = 20
SAMPLE_BARS = 5


def _finnhub_available() -> bool:
    return bool(config.FINNHUB_API_KEY)


def _fetch_finnhub_bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily candles from Finnhub for comparison."""
    from core import http

    url = 'https://finnhub.io/api/v1/stock/candle'
    params = {
        'symbol': ticker,
        'resolution': 'D',
        'from': int(pd.Timestamp(start).timestamp()),
        'to': int(pd.Timestamp(end).timestamp()),
        'token': config.FINNHUB_API_KEY,
    }
    payload = http.fetch_json(url, category='prices', params=params,
                              ttl=60 * 60 * 24)
    if not payload or payload.get('s') != 'ok':
        return pd.DataFrame()

    try:
        df = pd.DataFrame({
            'date': pd.to_datetime(payload['t'], unit='s').date,
            'close': payload['c'],
        })
        return df.drop_duplicates('date')
    except (KeyError, TypeError):
        return pd.DataFrame()


def validate_prices(tickers: list[str] | None = None,
                    n_tickers: int = SAMPLE_TICKERS,
                    n_bars: int = SAMPLE_BARS) -> pd.DataFrame:
    """
    Compare stored close prices against Finnhub for a random sample.

    Returns a DataFrame of disagreements (empty when everything agrees or
    when Finnhub is not configured).
    """
    if not _finnhub_available():
        log.debug('price validation skipped — no FINNHUB_API_KEY')
        return pd.DataFrame()

    if tickers is None:
        all_tickers = db.read_sql(
            'SELECT DISTINCT ticker FROM prices '
            'ORDER BY RANDOM() LIMIT :n', {'n': n_tickers})
        tickers = all_tickers['ticker'].tolist() if not all_tickers.empty else []
    else:
        tickers = random.sample(tickers, min(n_tickers, len(tickers)))

    if not tickers:
        return pd.DataFrame()

    end = date.today()
    start = end - timedelta(days=n_bars * 3)
    disagreements = []

    for ticker in tickers:
        stored = db.read_sql(
            'SELECT date, close FROM prices '
            'WHERE ticker = :t AND date >= :s AND date <= :e '
            'ORDER BY date DESC LIMIT :n',
            {'t': ticker, 's': str(start), 'e': str(end), 'n': n_bars},
            parse_dates=['date'])
        if stored.empty:
            continue

        try:
            ext = _fetch_finnhub_bars(ticker, start, end)
        except Exception as exc:                       # noqa: BLE001
            log.debug('finnhub fetch failed for %s: %s', ticker, exc)
            continue
        if ext.empty:
            continue

        ext_by_date = {d: c for d, c in zip(ext['date'], ext['close'])}
        for _, row in stored.iterrows():
            d = row['date'].date() if hasattr(row['date'], 'date') else row['date']
            ext_close = ext_by_date.get(d)
            if ext_close is None or ext_close <= 0:
                continue
            stored_close = float(row['close'])
            if stored_close <= 0:
                continue
            pct_diff = abs(stored_close - ext_close) / ext_close
            if pct_diff > TOLERANCE:
                disagreements.append({
                    'ticker': ticker, 'date': d,
                    'stored': round(stored_close, 4),
                    'finnhub': round(ext_close, 4),
                    'pct_diff': round(pct_diff * 100, 2),
                })

    if disagreements:
        log.warning('price validation: %d disagreement(s) across %d tickers '
                    '(>%.0f%% tolerance)',
                    len(disagreements), len(tickers), TOLERANCE * 100)
    else:
        log.info('price validation: %d tickers checked, all agree within %.0f%%',
                 len(tickers), TOLERANCE * 100)

    return pd.DataFrame(disagreements)
