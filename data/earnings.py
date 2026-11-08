"""
Earnings calendar — when each company reports next.

For a mid-term trade an earnings release is the single largest scheduled
risk: a stock can gap through any stop on the print. For a long-term holder
it is when the thesis gets tested. Either way the date belongs in the brief.

Two layers:

  confirmed  published dates from a calendar provider — Finnhub when
             FINNHUB_API_KEY is set, otherwise Nasdaq's public calendar (no
             key). Stored in `earnings_calendar`.
  estimated  for names the provider does not list, the next date implied by
             the company's own history of 8-K Item 2.02 ("results of
             operations") filings: the same week one year after a past
             report. Computed on read from stored filings, never stored, and
             always labelled as an estimate.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

import config
from core import db, http
from core import market_calendar as MC

log = logging.getLogger(__name__)

NASDAQ_URL = 'https://api.nasdaq.com/api/calendar/earnings'
FINNHUB_URL = 'https://finnhub.io/api/v1/calendar/earnings'
NASDAQ_HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'Origin': 'https://www.nasdaq.com',
    'Referer': 'https://www.nasdaq.com/',
}
DAYS_AHEAD = 21

_TIME = {'time-pre-market': 'bmo', 'time-after-hours': 'amc',
         'bmo': 'bmo', 'amc': 'amc', 'dmh': 'during'}


def _norm(sym: str | None) -> str:
    return str(sym or '').strip().upper().replace('.', '-').replace('/', '-')


def _money(v) -> float | None:
    """'$1.23' -> 1.23, '($0.12)' -> -0.12, '' -> None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if np.isfinite(v) else None
    s = str(v).strip().replace('$', '').replace(',', '')
    neg = s.startswith('(') and s.endswith(')')
    s = s.strip('()')
    try:
        f = float(s)
    except ValueError:
        return None
    return -f if neg else f


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────

def parse_nasdaq(payload: dict, day: date) -> list[dict]:
    rows = ((payload or {}).get('data') or {}).get('rows') or []
    out = []
    for r in rows:
        sym = _norm(r.get('symbol'))
        if not sym:
            continue
        out.append({'ticker': sym, 'report_date': day,
                    'time': _TIME.get(str(r.get('time') or '').lower(), 'unknown'),
                    'source': 'nasdaq',
                    'fiscal_period': r.get('fiscalQuarterEnding') or None,
                    'eps_estimate': _money(r.get('epsForecast'))})
    return out


def parse_finnhub(payload: dict) -> list[dict]:
    out = []
    for r in (payload or {}).get('earningsCalendar') or []:
        sym, d = _norm(r.get('symbol')), r.get('date')
        if not sym or not d:
            continue
        q, y = r.get('quarter'), r.get('year')
        out.append({'ticker': sym, 'report_date': pd.to_datetime(d).date(),
                    'time': _TIME.get(str(r.get('hour') or '').lower(), 'unknown'),
                    'source': 'finnhub',
                    'fiscal_period': f'Q{q} {y}' if q and y else None,
                    'eps_estimate': _money(r.get('epsEstimate'))})
    return out


# ─────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────

def _replace_window(rows: list[dict], lo: date, hi: date, wanted: set[str],
                    sources: tuple[str, ...]) -> int:
    """
    Swap the stored provider rows in [lo, hi] for a fresh set, atomically.

    Replacing rather than upserting is what makes a *rescheduled* report
    move: the old date simply is not in the new response.
    """
    keep = [dict(r, fetched_at=config.utc_now()) for r in rows if r['ticker'] in wanted]
    with db.connect() as conn:
        conn.execute(text(
            'DELETE FROM earnings_calendar WHERE report_date BETWEEN :lo AND :hi '
            f'AND source IN ({",".join(repr(s) for s in sources)})'),
            {'lo': str(lo), 'hi': str(hi)})
        if keep:
            db.upsert(db.earnings_calendar, keep, conn=conn)
    return len(keep)


def update_earnings(tickers: list[str], days_ahead: int = DAYS_AHEAD) -> int:
    """Refresh confirmed dates for `tickers` over the next `days_ahead` days."""
    wanted = {_norm(t) for t in tickers}
    today = date.today()
    hi = today + timedelta(days=days_ahead)
    written = 0

    if config.FINNHUB_API_KEY:
        payload = http.fetch_json(FINNHUB_URL, category='earnings', params={
            'from': str(today), 'to': str(hi), 'token': config.FINNHUB_API_KEY})
        if payload is not None:
            written = _replace_window(parse_finnhub(payload), today, hi, wanted,
                                      ('finnhub', 'nasdaq'))
            db.record_ingest('earnings', 'calendar', rows=written)
            return written
        log.warning('Finnhub earnings calendar unavailable — trying Nasdaq')

    d, failures = today, 0
    while d <= hi:
        if MC.is_trading_day(d):
            payload = http.fetch_json(NASDAQ_URL, category='earnings',
                                      params={'date': str(d)},
                                      headers=NASDAQ_HEADERS)
            # An error body still arrives as HTTP 200, with `data` null. Treat
            # it as a failure: read as "nobody reports that day" it would
            # delete the dates already stored for it.
            if payload is None or not isinstance(payload.get('data'), dict):
                failures += 1
            else:
                written += _replace_window(parse_nasdaq(payload, d), d, d,
                                           wanted, ('nasdaq',))
        d += timedelta(days=1)
    if failures:
        log.warning('earnings: %d calendar day(s) could not be fetched', failures)
    db.record_ingest('earnings', 'calendar', rows=written,
                     error=None if written or not failures else
                     f'{failures} day(s) failed')
    return written


# ─────────────────────────────────────────────
# ESTIMATES FROM FILING CADENCE
# ─────────────────────────────────────────────

def estimate_next(filed: pd.Series, as_of: date) -> date | None:
    """
    Next report date implied by past Item 2.02 filing dates.

    Companies report in the same week each year, so a filing 364 days ago
    (exactly 52 weeks: same weekday) is the best guess for the next one. Falls
    back to the last report plus a quarter.
    """
    dates = sorted({pd.to_datetime(x).date() for x in filed.dropna()})
    past = [d for d in dates if d <= as_of]
    if not past:
        return None
    for d in past:
        guess = d + timedelta(days=364)
        if guess >= as_of and (guess - past[-1]).days >= 45:
            return guess
    guess = past[-1] + timedelta(days=91)
    while guess < as_of:
        guess += timedelta(days=91)
    return guess


def upcoming(tickers: list[str], as_of: date | None = None,
             days: int = 14) -> pd.DataFrame:
    """
    Reports due in the next `days`: confirmed first, estimates for the rest.

    Columns: ticker, report_date, time, source, fiscal_period, eps_estimate,
    days_until, confirmed.
    """
    as_of = as_of or date.today()
    if not tickers:
        return pd.DataFrame()
    hi = as_of + timedelta(days=days)
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params = {f't{i}': t for i, t in enumerate(tickers)}
    confirmed = db.read_sql(f"""
        SELECT ticker, report_date, time, source, fiscal_period, eps_estimate
        FROM earnings_calendar
        WHERE ticker IN ({ph}) AND report_date >= :lo AND report_date <= :hi
        ORDER BY report_date
    """, {**params, 'lo': str(as_of), 'hi': str(hi)}, parse_dates=['report_date'])
    confirmed['confirmed'] = True

    have = set(confirmed['ticker'])
    rest = [t for t in tickers if t not in have]
    est_rows = []
    if rest:
        ph2 = ','.join(f':r{i}' for i in range(len(rest)))
        hist = db.read_sql(f"""
            SELECT ticker, filed FROM corporate_events
            WHERE item_code = '2.02' AND ticker IN ({ph2}) AND filed >= :since
        """, {**{f'r{i}': t for i, t in enumerate(rest)},
              'since': str(as_of - timedelta(days=800))}, parse_dates=['filed'])
        for t, grp in hist.groupby('ticker'):
            nxt = estimate_next(grp['filed'], as_of)
            if nxt and as_of <= nxt <= hi:
                est_rows.append({'ticker': t, 'report_date': pd.Timestamp(nxt),
                                 'time': 'unknown', 'source': 'estimated',
                                 'fiscal_period': None, 'eps_estimate': None,
                                 'confirmed': False})
    out = pd.concat([confirmed, pd.DataFrame(est_rows)], ignore_index=True) \
        if est_rows else confirmed
    if out.empty:
        return out
    out['days_until'] = (out['report_date'].dt.date - as_of).map(lambda x: x.days)
    return out.sort_values(['report_date', 'ticker']).reset_index(drop=True)
