"""
Yahoo Finance provider — prices and profile snapshots.

Prices go through the chart API directly rather than yfinance. During probing,
`yfinance.download` failed with YFRateLimitError while a direct call to
`query1.finance.yahoo.com/v8/finance/chart` returned 251 clean bars in the
same second, so the direct endpoint is primary and yfinance is the fallback.

Everything is written to SQLite and refreshed incrementally: a second run over
the same tickers issues no network calls at all.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import config
from core import db, http

log = logging.getLogger(__name__)

CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'


# ─────────────────────────────────────────────
# PRICES
# ─────────────────────────────────────────────

def _parse_chart(payload: dict, ticker: str) -> pd.DataFrame:
    """Turn a chart-API response into an OHLCV frame."""
    try:
        result = payload['chart']['result'][0]
    except (KeyError, IndexError, TypeError):
        return pd.DataFrame()

    stamps = result.get('timestamp') or []
    if not stamps:
        return pd.DataFrame()

    quote = (result.get('indicators', {}).get('quote') or [{}])[0]
    adj = (result.get('indicators', {}).get('adjclose') or [{}])
    adj_close = adj[0].get('adjclose') if adj else None

    df = pd.DataFrame({
        'date': pd.to_datetime(stamps, unit='s', utc=True).tz_convert(None).normalize(),
        'open': quote.get('open'),
        'high': quote.get('high'),
        'low': quote.get('low'),
        'close': quote.get('close'),
        'volume': quote.get('volume'),
    })
    df['adj_close'] = adj_close if adj_close is not None else df['close']
    df['ticker'] = ticker

    df = df.dropna(subset=['close']).drop_duplicates(subset=['date'], keep='last')
    return df.sort_values('date').reset_index(drop=True)


def _fetch_chart(ticker: str, period: str = '10y') -> pd.DataFrame:
    """Primary path: the chart endpoint, with a yfinance fallback."""
    payload = http.fetch_json(
        CHART_URL.format(ticker=ticker), category='prices',
        params={'range': period, 'interval': '1d', 'events': 'div,split'},
    )
    df = _parse_chart(payload, ticker) if payload else pd.DataFrame()
    if not df.empty:
        return df

    log.debug('chart API empty for %s, trying yfinance', ticker)
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        out = pd.DataFrame({
            'ticker': ticker,
            'date': pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize(),
            'open': hist.get('Open'), 'high': hist.get('High'),
            'low': hist.get('Low'), 'close': hist.get('Close'),
            'adj_close': hist.get('Adj Close', hist.get('Close')),
            'volume': hist.get('Volume'),
        })
        return out.dropna(subset=['close'])
    except Exception as exc:                      # noqa: BLE001 - provider is flaky by nature
        log.debug('yfinance fallback failed for %s: %s', ticker, exc)
        return pd.DataFrame()


def _existing_range(ticker: str) -> tuple[date | None, date | None]:
    row = db.read_sql(
        'SELECT MIN(date) AS lo, MAX(date) AS hi FROM prices WHERE ticker = :t',
        {'t': ticker})
    if row.empty or pd.isna(row.iloc[0]['hi']):
        return None, None
    return (pd.to_datetime(row.iloc[0]['lo']).date(),
            pd.to_datetime(row.iloc[0]['hi']).date())


def _period_for(last: date | None, full_period: str) -> str | None:
    """Smallest range that closes the gap, or None when already current."""
    if last is None:
        return full_period
    gap = (date.today() - last).days
    if gap <= 1:
        return None
    for days, period in ((5, '5d'), (28, '1mo'), (85, '3mo'),
                         (170, '6mo'), (350, '1y'), (720, '2y'), (1800, '5y')):
        if gap <= days:
            return period
    return full_period


def update_prices(tickers: list[str], period: str = '10y',
                  force: bool = False) -> int:
    """
    Fetch and store daily bars, skipping tickers already current.

    Returns the number of rows written.
    """
    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return 0

    jobs: list[tuple[str, str]] = []
    for t in tickers:
        _lo, hi = _existing_range(t)
        p = period if force else _period_for(hi, period)
        if p:
            jobs.append((t, p))

    skipped = len(tickers) - len(jobs)
    if skipped:
        log.info('prices: %d/%d tickers already current', skipped, len(tickers))
    if not jobs:
        return 0

    written = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_chart, t, p): t for t, p in jobs}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                df = fut.result()
            except Exception as exc:              # noqa: BLE001
                log.warning('price fetch failed for %s: %s', t, exc)
                db.record_ingest('prices', t, error=str(exc)[:200])
                continue
            if df.empty:
                continue
            rows = [{
                'ticker': r['ticker'], 'date': r['date'].date(),
                'open': _f(r['open']), 'high': _f(r['high']), 'low': _f(r['low']),
                'close': _f(r['close']), 'adj_close': _f(r['adj_close']),
                'volume': _f(r['volume']),
            } for _i, r in df.iterrows()]
            written += db.upsert(db.prices, rows)
            db.record_ingest('prices', t, rows=len(rows))

    log.info('prices: wrote %d rows across %d tickers', written, len(jobs))
    return written


def _f(v) -> float | None:
    """Coerce to a finite float, else None (SQL NULL)."""
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# PRICE READS  (all point-in-time gated)
# ─────────────────────────────────────────────

_PRICE_COLUMNS = frozenset({
    'open', 'high', 'low', 'close', 'adj_close', 'volume',
})


def price_history(tickers: list[str], start: date | str | None = None,
                  end: date | str | None = None,
                  field: str = 'adj_close') -> pd.DataFrame:
    """Wide frame of `field`, indexed by date with one column per ticker."""
    if field not in _PRICE_COLUMNS:
        raise ValueError(f'unknown price column {field!r}; '
                         f'expected one of {sorted(_PRICE_COLUMNS)}')
    if not tickers:
        return pd.DataFrame()

    placeholders = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    clauses = [f'ticker IN ({placeholders})']
    if start:
        clauses.append('date >= :start')
        params['start'] = str(pd.to_datetime(start).date())
    if end:
        clauses.append('date <= :end')
        params['end'] = str(pd.to_datetime(end).date())

    df = db.read_sql(
        f'SELECT ticker, date, {field} AS v FROM prices '
        f'WHERE {" AND ".join(clauses)} ORDER BY date',
        params, parse_dates=['date'])
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index='date', columns='ticker', values='v', aggfunc='last')


def adjusted_ohlc(tickers: list[str], start: date | str | None = None,
                  end: date | str | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    High, low and close all expressed on the adjusted-close basis.

    Yahoo returns `high`/`low` unadjusted while `adj_close` is back-adjusted
    for splits and dividends. Comparing a stop derived from the adjusted price
    against a raw high is meaningless — for CVX in early 2023 the raw high was
    $174.63 against an adjusted close of $150.55, so every level sat far below
    the bar and appeared to trigger instantly. Scaling by adj_close/close puts
    the whole bar on one basis.

    Returns (high, low, close), each a wide date x ticker frame.
    """
    close = price_history(tickers, start, end, field='close')
    adj = price_history(tickers, start, end, field='adj_close')
    high = price_history(tickers, start, end, field='high')
    low = price_history(tickers, start, end, field='low')

    if close.empty or adj.empty:
        return high, low, adj

    ratio = (adj / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.ffill().bfill().fillna(1.0)

    common = ratio.columns
    if not high.empty:
        high = high[high.columns.intersection(common)].mul(ratio, fill_value=np.nan)
    if not low.empty:
        low = low[low.columns.intersection(common)].mul(ratio, fill_value=np.nan)

    return high, low, adj


def last_close(prices: pd.DataFrame) -> pd.Series:
    """
    Pure: last observed close per ticker from a price frame.

    No I/O — hand it a frame and it returns a series. Separating this from the
    read makes it testable without a database and lets the caller decide how
    much history to load.
    """
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    return prices.ffill().iloc[-1]


# Enough sessions to survive a long weekend, a holiday, and a halted ticker.
_LAST_CLOSE_LOOKBACK_DAYS = 30


def latest_prices(tickers: list[str], as_of: date | str,
                  lookback_days: int = _LAST_CLOSE_LOOKBACK_DAYS,
                  field: str = 'adj_close') -> pd.Series:
    """
    Last close at or before `as_of` — never peeks past it.

    Bounded to a short window. Without a start date this read pulled every bar
    ever stored — at S&P 500 scale roughly 1.2 million rows — to use only the
    final one, which cost about 2.3 seconds on every universe resolution.

    Use `field='close'` for market-cap estimation from SEC share counts — those
    counts are the actual filing-date figure, not split-adjusted, so multiplying
    by adj_close produces a result that is wrong by the split ratio.
    """
    start = pd.to_datetime(as_of) - timedelta(days=lookback_days)
    px = price_history(tickers, start=start, end=as_of, field=field)
    if px.empty:
        px = price_history(tickers, end=as_of, field=field)
    return last_close(px)


def dollar_adv(tickers: list[str], as_of: date | str,
               window: int | None = None) -> pd.Series:
    """Average daily dollar volume over `window` sessions ending at `as_of`."""
    window = window or config.ADV_WINDOW
    start = pd.to_datetime(as_of) - timedelta(days=window * 3)
    close = price_history(tickers, start=start, end=as_of, field='close')
    vol = price_history(tickers, start=start, end=as_of, field='volume')
    if close.empty or vol.empty:
        return pd.Series(dtype=float)
    common = close.columns.intersection(vol.columns)
    return (close[common] * vol[common]).tail(window).mean()


# ─────────────────────────────────────────────
# PROFILE SNAPSHOTS
# ─────────────────────────────────────────────
#
# `.info` is a *current* snapshot with no history, so it can never be
# backfilled. It powers the live screen and accumulates a forward
# point-in-time record from the day the app first runs; historical factors
# come from SEC filings instead.

_INFO_FIELDS = {
    'market_cap': 'marketCap', 'trailing_pe': 'trailingPE',
    'forward_pe': 'forwardPE', 'price_to_book': 'priceToBook',
    'trailing_eps': 'trailingEps', 'forward_eps': 'forwardEps',
    'revenue_growth': 'revenueGrowth', 'earnings_growth': 'earningsGrowth',
    'roe': 'returnOnEquity', 'roa': 'returnOnAssets',
    'debt_to_equity': 'debtToEquity', 'current_ratio': 'currentRatio',
    'gross_margin': 'grossMargins', 'operating_margin': 'operatingMargins',
    'profit_margin': 'profitMargins', 'dividend_yield': 'dividendYield',
    'payout_ratio': 'payoutRatio', 'beta': 'beta',
    'shares_out': 'sharesOutstanding',
}


def _fetch_info(ticker: str) -> dict | None:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return info if info.get('symbol') or info.get('shortName') else None
    except Exception as exc:                      # noqa: BLE001
        log.debug('info failed for %s: %s', ticker, exc)
        return None


def update_profiles(tickers: list[str], snapshot_date: date | None = None) -> int:
    """Snapshot provider metrics and refresh the securities table."""
    snapshot_date = snapshot_date or date.today()
    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return 0

    have = db.read_sql(
        'SELECT DISTINCT ticker FROM profile_snapshots WHERE snapshot_date = :d',
        {'d': str(snapshot_date)})
    todo = [t for t in tickers if t not in set(have['ticker'])]
    if not todo:
        log.info('profiles: all %d already snapshotted for %s', len(tickers), snapshot_date)
        return 0

    snaps, secs = [], []
    with ThreadPoolExecutor(max_workers=max(2, config.MAX_WORKERS // 2)) as pool:
        futures = {pool.submit(_fetch_info, t): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            info = fut.result()
            if not info:
                continue

            row = {'ticker': t, 'snapshot_date': snapshot_date}
            for col, key in _INFO_FIELDS.items():
                row[col] = _f(info.get(key))
            snaps.append(row)

            # Enrich the securities row rather than replacing it. The SEC-derived
            # classification (data/sec.update_securities) is the authoritative
            # source; `.info` only fills gaps, because when Yahoo throttles it
            # returns nothing and would otherwise overwrite good sectors with
            # "Unknown".
            sector = config.normalize_sector(info.get('sector'))
            enriched = {'ticker': t}
            if info.get('longName') or info.get('shortName'):
                enriched['name'] = info.get('longName') or info.get('shortName')
            if sector != 'Unknown':
                enriched['sector'] = sector
            if info.get('industry'):
                enriched['industry'] = info['industry']
            if info.get('exchange'):
                enriched['exchange'] = info['exchange']
            if info.get('country'):
                enriched['country'] = info['country']
                enriched['is_adr'] = int(info['country'] != 'United States')
            if len(enriched) > 1:
                enriched['last_updated'] = datetime.utcnow()
                secs.append(enriched)

    n = db.upsert(db.profile_snapshots, snaps)
    if secs:
        _enrich_securities(secs)
    log.info('profiles: %d snapshots (%d requested)', n, len(todo))
    db.record_ingest('profiles', str(snapshot_date), rows=n)
    return n


def _enrich_securities(rows: list[dict]) -> None:
    """Merge partial updates into securities without nulling absent columns."""
    for r in rows:
        sets = [f'{k} = :{k}' for k in r if k != 'ticker']
        if not sets:
            continue
        try:
            with db.connect() as conn:
                from sqlalchemy import text
                conn.execute(
                    text(f'UPDATE securities SET {", ".join(sets)} '
                         f'WHERE ticker = :ticker'), r)
        except Exception as exc:                   # noqa: BLE001
            log.debug('securities enrich failed for %s: %s', r.get('ticker'), exc)


def profile_asof(tickers: list[str], as_of: date | str) -> pd.DataFrame:
    """Most recent snapshot at or before `as_of`, one row per ticker."""
    if not tickers:
        return pd.DataFrame()
    placeholders = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params['ts'] = str(pd.to_datetime(as_of).date())

    df = db.read_sql(f"""
        SELECT p.* FROM profile_snapshots p
        JOIN (
            SELECT ticker, MAX(snapshot_date) AS d FROM profile_snapshots
            WHERE ticker IN ({placeholders}) AND snapshot_date <= :ts
            GROUP BY ticker
        ) m ON p.ticker = m.ticker AND p.snapshot_date = m.d
    """, params, parse_dates=['snapshot_date'])
    return df.set_index('ticker') if not df.empty else pd.DataFrame()
