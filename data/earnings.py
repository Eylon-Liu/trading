"""
Earnings calendar — when each company reports next.

For a mid-term trade an earnings release is the single largest scheduled
risk: a stock can gap through any stop on the print. For a long-term holder
it is when the thesis gets tested. Either way the date belongs in the brief.

Sources, in order — every one free, only the first needs a key:

  1. Finnhub        when FINNHUB_API_KEY is set: one call for the window.
  2. Nasdaq         public calendar, one call per trading day, no key.
  3. Yahoo          per-ticker calendar via yfinance, no key — only for the
                    names you follow (portfolio, watchlist) that 1–2 did not
                    cover, so it stays a few dozen calls. Yahoo marks dates it
                    has not confirmed with a two-day range; those are stored
                    as `yahoo-est` and labelled estimated.
  4. SEC cadence    for anything still missing, the next date implied by the
                    company's own 8-K Item 2.02 ("results of operations")
                    history: the same week one year after a past report.
                    Computed on read, never stored, always labelled estimated.

`LAST_STATUS` records what each source returned on the most recent refresh,
so `cli.py earnings --refresh` can say which one actually worked.
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
ESTIMATED_SOURCES = ('yahoo-est', 'estimated')
YAHOO_MAX_NAMES = 60
LAST_STATUS: dict = {}

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


def _followed() -> list[str]:
    """Portfolio and watchlist names — the ones worth a per-ticker lookup."""
    try:
        from quant import portfolio as PF
        return sorted(set(PF.watchlist()) | set(PF.lots('open')['ticker']))
    except Exception as exc:                       # noqa: BLE001
        log.debug('followed names unavailable: %s', exc)
        return list(config.WATCHLIST)


def parse_yahoo_calendar(ticker: str, cal: dict, today: date) -> dict | None:
    """yfinance `Ticker.calendar` -> one row, or None when nothing upcoming."""
    if not isinstance(cal, dict):           # older yfinance returned a frame
        try:
            cal = {k: list(v.values()) if isinstance(v, dict) else v
                   for k, v in cal.to_dict().items()}
        except Exception:                          # noqa: BLE001
            return None
    dates = [pd.to_datetime(d).date() for d in (cal or {}).get('Earnings Date') or []]
    dates = sorted(d for d in dates if d >= today)
    if not dates:
        return None
    # Two dates are Yahoo's estimated window; one date is a set date.
    estimated = len(dates) > 1 and dates[-1] != dates[0]
    return {'ticker': _norm(ticker), 'report_date': dates[0], 'time': 'unknown',
            'source': 'yahoo-est' if estimated else 'yahoo',
            'fiscal_period': None,
            'eps_estimate': _money(cal.get('Earnings Average'))}


def _from_yahoo(tickers: list[str], today: date) -> tuple[list[dict], int]:
    """Per-ticker calendar lookups through yfinance. Returns (rows, failures)."""
    try:
        import yfinance as yf
    except ImportError:
        return [], len(tickers)
    rows, failures = [], 0
    for t in tickers[:YAHOO_MAX_NAMES]:
        try:
            row = parse_yahoo_calendar(t, yf.Ticker(t).calendar or {}, today)
        except Exception as exc:                   # noqa: BLE001
            log.debug('yahoo calendar failed for %s: %s', t, exc)
            failures += 1
            if failures >= 5 and not rows:
                break                              # Yahoo is refusing; stop asking
            continue
        if row:
            rows.append(row)
    return rows, failures


def update_earnings(tickers: list[str], days_ahead: int = DAYS_AHEAD,
                    followed: list[str] | None = None) -> int:
    """
    Refresh confirmed dates for `tickers` over the next `days_ahead` days.

    Returns rows written. Never raises for a provider problem: each source
    that fails is recorded in LAST_STATUS and the next one is tried.
    """
    wanted = {_norm(t) for t in tickers}
    followed = [_norm(t) for t in (followed if followed is not None else _followed())]
    wanted |= set(followed)
    today = date.today()
    hi = today + timedelta(days=days_ahead)
    written = 0
    status: dict = {}

    provider_ok = False
    if config.FINNHUB_API_KEY:
        payload = http.fetch_json(FINNHUB_URL, category='earnings', params={
            'from': str(today), 'to': str(hi), 'token': config.FINNHUB_API_KEY},
            timeout=15, retries=2)
        if payload is not None:
            n = _replace_window(parse_finnhub(payload), today, hi, wanted,
                                ('finnhub', 'nasdaq'))
            written += n
            provider_ok = True
            status['finnhub'] = f'ok, {n} dates'
        else:
            status['finnhub'] = 'failed'

    if not provider_ok:
        d, ok_days, failures, streak = today, 0, 0, 0
        while d <= hi:
            if MC.is_trading_day(d):
                payload = http.fetch_json(NASDAQ_URL, category='earnings',
                                          params={'date': str(d)},
                                          headers=NASDAQ_HEADERS,
                                          timeout=12, retries=2)
                # An error body still arrives as HTTP 200, with `data` null.
                # Treat it as a failure: read as "nobody reports that day"
                # it would delete the dates already stored for it.
                if payload is None or not isinstance(payload.get('data'), dict):
                    failures += 1
                    streak += 1
                    if streak >= 3 and ok_days == 0:
                        log.warning('earnings: Nasdaq calendar not answering — '
                                    'skipping the rest of the window')
                        break
                else:
                    streak = 0
                    ok_days += 1
                    written += _replace_window(parse_nasdaq(payload, d), d, d,
                                               wanted, ('nasdaq',))
            d += timedelta(days=1)
        provider_ok = ok_days > 0
        status['nasdaq'] = (f'ok, {ok_days} day(s)' + (f', {failures} failed' if failures else '')
                            if ok_days else 'failed')

    # Yahoo for followed names the providers did not list in the window.
    have = set()
    if followed:
        listed = db.read_sql(
            'SELECT DISTINCT ticker FROM earnings_calendar '
            "WHERE report_date BETWEEN :lo AND :hi AND source IN ('finnhub', 'nasdaq')",
            {'lo': str(today), 'hi': str(hi)})
        have = set(listed['ticker'])
    gaps = [t for t in followed if t not in have]
    if gaps:
        rows, fails = _from_yahoo(gaps, today)
        if rows:
            with db.connect() as conn:
                for t in {r['ticker'] for r in rows}:
                    conn.execute(text(
                        "DELETE FROM earnings_calendar WHERE ticker = :t "
                        "AND report_date >= :lo AND source IN ('yahoo', 'yahoo-est')"),
                        {'t': t, 'lo': str(today)})
                db.upsert(db.earnings_calendar,
                          [dict(r, fetched_at=config.utc_now()) for r in rows], conn=conn)
            written += len(rows)
        status['yahoo'] = (f'ok, {len(rows)} of {len(gaps)} followed name(s)'
                           if rows else f'nothing for {len(gaps)} name(s)'
                           + (f' ({fails} failed)' if fails else ''))

    LAST_STATUS.clear()
    LAST_STATUS.update(status)
    any_ok = provider_ok or bool(gaps and 'ok' in status.get('yahoo', ''))
    db.record_ingest('earnings', 'calendar', rows=written,
                     error=None if any_ok else '; '.join(f'{k}: {v}' for k, v in status.items()))
    log.info('earnings: %s', '; '.join(f'{k} {v}' for k, v in status.items()))
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
    confirmed['confirmed'] = ~confirmed['source'].isin(ESTIMATED_SOURCES)

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
