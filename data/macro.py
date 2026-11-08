"""
Macro series from FRED — rates, credit spreads, volatility.

Two uses:

  * the market-regime panel in the daily brief (curve shape, credit stress,
    volatility, and the S&P 500's own trend), because a long-only screen
    reads very differently in a 2008 tape than in a 2017 one;
  * the risk-free rate in backtest metrics. A fixed 5% was used for every
    window, which overstated the hurdle for 2010–2021 (bills near zero) and
    understated Sharpe ratios for those years.

No key needed: FRED serves every series as CSV at a public URL. With
FRED_API_KEY set the JSON API is used instead (higher limits, same data).
Everything is stored in the `macro` table and read point-in-time like the
rest of the store.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from core import db, http

log = logging.getLogger(__name__)

# series id -> (label, unit). Units: '%' level in percent, 'pp' a spread in
# percentage points, '' an index level.
SERIES: dict[str, tuple[str, str]] = {
    'DGS3MO': ('3-month T-bill', '%'),
    'DGS2': ('2-year Treasury', '%'),
    'DGS10': ('10-year Treasury', '%'),
    'T10Y2Y': ('10y − 2y curve', 'pp'),
    'BAMLH0A0HYM2': ('High-yield spread', 'pp'),
    'VIXCLS': ('VIX', ''),
    'DFF': ('Fed funds rate', '%'),
}

CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
API_URL = 'https://api.stlouisfed.org/fred/series/observations'
HISTORY_START = date(2000, 1, 1)


# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────

def parse_csv(text: str, series_id: str) -> pd.DataFrame:
    """FRED graph CSV -> (date, value). Missing observations are '.' or blank."""
    if not text or not text.strip():
        return pd.DataFrame(columns=['date', 'value'])
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        return pd.DataFrame(columns=['date', 'value'])
    date_col = df.columns[0]                 # 'DATE' or 'observation_date'
    val_col = series_id if series_id in df.columns else df.columns[1]
    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'value': pd.to_numeric(df[val_col], errors='coerce'),
    })
    return out.dropna()


def parse_api(payload: dict) -> pd.DataFrame:
    rows = (payload or {}).get('observations') or []
    out = pd.DataFrame({
        'date': pd.to_datetime([r.get('date') for r in rows], errors='coerce'),
        'value': pd.to_numeric([r.get('value') for r in rows], errors='coerce'),
    })
    return out.dropna()


def fetch_series(series_id: str, start: date) -> pd.DataFrame:
    if config.FRED_API_KEY:
        payload = http.fetch_json(API_URL, category='macro', params={
            'series_id': series_id, 'api_key': config.FRED_API_KEY,
            'file_type': 'json', 'observation_start': str(start)})
        return parse_api(payload) if payload else pd.DataFrame(columns=['date', 'value'])
    text = http.fetch_text(CSV_URL, category='macro',
                           params={'id': series_id, 'cosd': str(start)})
    return parse_csv(text or '', series_id)


def update_macro(series: list[str] | None = None) -> int:
    """Fetch new observations for each series; returns rows written."""
    series = series or list(SERIES)
    last = db.read_sql('SELECT series_id, MAX(date) AS d FROM macro GROUP BY series_id')
    newest = dict(zip(last['series_id'], pd.to_datetime(last['d']).dt.date)) \
        if not last.empty else {}
    written = 0
    for sid in series:
        start = newest.get(sid)
        # Re-read a couple of weeks: FRED revises recent prints.
        start = (start - timedelta(days=14)) if start else HISTORY_START
        try:
            df = fetch_series(sid, start)
        except Exception as exc:                   # noqa: BLE001
            log.warning('FRED %s failed: %s', sid, exc)
            continue
        if df.empty:
            continue
        written += db.upsert(db.macro, [
            {'series_id': sid, 'date': d.date(), 'value': float(v)}
            for d, v in zip(df['date'], df['value'])])
    db.record_ingest('macro', 'fred', rows=written)
    log.info('macro: %d observations across %d series', written, len(series))
    return written


# ─────────────────────────────────────────────
# READS
# ─────────────────────────────────────────────

def series(series_id: str, start: date | str | None = None,
           end: date | str | None = None) -> pd.Series:
    clauses, params = ['series_id = :s'], {'s': series_id}
    if start:
        clauses.append('date >= :lo')
        params['lo'] = str(pd.to_datetime(start).date())
    if end:
        clauses.append('date <= :hi')
        params['hi'] = str(pd.to_datetime(end).date())
    df = db.read_sql(f'SELECT date, value FROM macro WHERE {" AND ".join(clauses)} '
                     f'ORDER BY date', params, parse_dates=['date'])
    if df.empty:
        return pd.Series(dtype=float, name=series_id)
    return df.set_index('date')['value'].rename(series_id)


def risk_free_rate(start: date | str, end: date | str) -> float:
    """Average annual 3-month bill yield over the window, as a decimal."""
    s = series('DGS3MO', start, end)
    if len(s) < 5:
        return config.RISK_FREE_RATE
    return float(s.mean()) / 100.0


def _value_at(s: pd.Series, when: pd.Timestamp) -> float:
    before = s.loc[:when]
    return float(before.iloc[-1]) if len(before) else np.nan


def snapshot(as_of: date | str) -> pd.DataFrame:
    """Latest level per series with 1-week and 1-month changes."""
    as_of = pd.Timestamp(pd.to_datetime(as_of).date())
    rows = []
    for sid, (label, unit) in SERIES.items():
        s = series(sid, as_of - pd.Timedelta(days=70), as_of)
        if s.empty:
            continue
        now = _value_at(s, as_of)
        rows.append({
            'series_id': sid, 'label': label, 'unit': unit,
            'date': s.index[-1].date(), 'value': now,
            'chg_1w': now - _value_at(s, as_of - pd.Timedelta(days=7)),
            'chg_1m': now - _value_at(s, as_of - pd.Timedelta(days=30)),
        })
    return pd.DataFrame(rows)


def _spx_trend(as_of: date) -> tuple[float | None, float | None]:
    """(benchmark close / 200-day average − 1, close) from stored prices."""
    from data import yahoo
    px = yahoo.price_history([config.BENCHMARK_TICKER],
                             start=pd.Timestamp(as_of) - pd.Timedelta(days=320),
                             end=as_of, field='adj_close')
    if px.empty or len(px) < 200:
        return None, None
    col = px.iloc[:, 0].dropna()
    ma = float(col.tail(200).mean())
    return float(col.iloc[-1] / ma - 1.0), float(col.iloc[-1])


def regime(as_of: date | str, snap: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    """
    Plain-language regime flags as (level, text); level is ok | watch | risk.

    Deliberately few and blunt. These are context for reading a screen, not
    signals — nothing here changes a score.
    """
    as_of = pd.to_datetime(as_of).date()
    snap = snapshot(as_of) if snap is None else snap
    v = dict(zip(snap['series_id'], snap['value'])) if not snap.empty else {}
    m = dict(zip(snap['series_id'], snap['chg_1m'])) if not snap.empty else {}
    out: list[tuple[str, str]] = []

    trend, _close = _spx_trend(as_of)
    if trend is not None:
        if trend >= 0:
            out.append(('ok', f'S&P 500 {trend:+.1%} vs its 200-day average — uptrend'))
        else:
            out.append(('risk', f'S&P 500 {trend:+.1%} below its 200-day average — '
                                'downtrend; momentum and breakout setups fail more often'))

    curve = v.get('T10Y2Y')
    if curve is not None and np.isfinite(curve):
        if curve < 0:
            out.append(('watch', f'Yield curve inverted ({curve:+.2f}pp 10y−2y)'))
        elif m.get('T10Y2Y', 0) > 0.4 and curve < 0.75:
            out.append(('watch', f'Curve steepening fast from low levels '
                                 f'({m["T10Y2Y"]:+.2f}pp in a month) — often late-cycle'))

    hy = v.get('BAMLH0A0HYM2')
    if hy is not None and np.isfinite(hy):
        widening = m.get('BAMLH0A0HYM2', 0)
        if hy >= 6:
            out.append(('risk', f'Credit stress: high-yield spread {hy:.2f}pp'))
        elif widening >= 0.5:
            out.append(('watch', f'Credit spreads widening ({widening:+.2f}pp in a month, '
                                 f'now {hy:.2f}pp)'))
        else:
            out.append(('ok', f'Credit calm: high-yield spread {hy:.2f}pp'))

    vix = v.get('VIXCLS')
    if vix is not None and np.isfinite(vix):
        if vix >= 30:
            out.append(('risk', f'VIX {vix:.1f} — stressed; expect wide daily ranges, '
                                'size positions down'))
        elif vix >= 20:
            out.append(('watch', f'VIX {vix:.1f} — elevated'))
        else:
            out.append(('ok', f'VIX {vix:.1f} — calm'))
    return out
