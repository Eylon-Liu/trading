"""
NYSE trading calendar — which days the market is open, and when a session's
daily bar can be trusted.

Every rule here is computed, not looked up, so there is no table to update
each December. The holiday rules have been stable for decades; the one-off
closures (a hurricane, a national day of mourning) are listed explicitly in
`SPECIAL_CLOSURES` and are the only thing that ever needs editing.

Why this exists, rather than "Monday to Friday":

  * The daily email is a full brief on trading days and a news-only digest
    whenever the market is shut. A weekday holiday is shut too, and a full
    brief on Thanksgiving would repeat Wednesday's numbers as if they were new.
  * The price sync decides whether a stored bar is current by comparing it
    with the last completed session. Without holidays, every holiday made the
    whole universe look one day stale and triggered a pointless refetch.
  * "Completed" means after the close *in New York*. The previous check used a
    fixed UTC hour, which is an hour wrong for half the year because US
    daylight saving moves the close between 20:00 and 21:00 UTC.

All datetimes accepted here may be timezone-aware (any zone) or naive; naive
values are taken to be UTC, which is how the database stores timestamps.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')

OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

# How long after the closing bell a daily bar is treated as final. The closing
# auction prints a few minutes after 16:00 and providers republish the bar
# shortly after; a bar read inside this window can still move.
PUBLISH_DELAY = timedelta(minutes=45)

# Unscheduled full-day closures. Add to this when the exchange announces one.
SPECIAL_CLOSURES: dict[date, str] = {
    date(2001, 9, 11): 'September 11 attacks',
    date(2001, 9, 12): 'September 11 attacks',
    date(2001, 9, 13): 'September 11 attacks',
    date(2001, 9, 14): 'September 11 attacks',
    date(2004, 6, 11): 'National Day of Mourning (Reagan)',
    date(2007, 1, 2): 'National Day of Mourning (Ford)',
    date(2012, 10, 29): 'Hurricane Sandy',
    date(2012, 10, 30): 'Hurricane Sandy',
    date(2018, 12, 5): 'National Day of Mourning (G.H.W. Bush)',
    date(2025, 1, 9): 'National Day of Mourning (Carter)',
}


# ─────────────────────────────────────────────
# HOLIDAY RULES
# ─────────────────────────────────────────────

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th `weekday` (Mon=0) of a month; n=-1 for the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Western Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def holidays(year: int) -> dict[date, str]:
    """Full-day NYSE closures in `year`, each with its name."""
    out: dict[date, str] = {}

    # New Year's Day. A Sunday holiday moves to Monday, but a Saturday one is
    # *not* observed on the Friday before — that Friday is the last session of
    # the prior year and the exchange stays open for year-end.
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        out[ny + timedelta(days=1)] = "New Year's Day (observed)"
    elif ny.weekday() != 5:
        out[ny] = "New Year's Day"

    if year >= 1998:
        out[_nth_weekday(year, 1, 0, 3)] = 'Martin Luther King Jr. Day'
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    out[_easter(year) - timedelta(days=2)] = 'Good Friday'
    out[_nth_weekday(year, 5, 0, -1)] = 'Memorial Day'
    if year >= 2022:
        out[_observed(date(year, 6, 19))] = 'Juneteenth'
    out[_observed(date(year, 7, 4))] = 'Independence Day'
    out[_nth_weekday(year, 9, 0, 1)] = 'Labor Day'
    out[_nth_weekday(year, 11, 3, 4)] = 'Thanksgiving Day'
    out[_observed(date(year, 12, 25))] = 'Christmas Day'

    for d, name in SPECIAL_CLOSURES.items():
        if d.year == year:
            out[d] = name
    return out


def holiday_name(d: date) -> str | None:
    """The holiday closing the market on `d`, or None."""
    return holidays(d.year).get(d)


def is_trading_day(d: date) -> bool:
    """True when the NYSE holds a session on `d`."""
    d = _as_date(d)
    return d.weekday() < 5 and d not in holidays(d.year)


def closed_reason(d: date) -> str | None:
    """Why the market is shut on `d` ('Weekend', a holiday name), or None."""
    d = _as_date(d)
    if d.weekday() >= 5:
        return 'Weekend'
    return holiday_name(d)


def is_early_close(d: date) -> bool:
    """1 p.m. closes: July 3, the day after Thanksgiving, Christmas Eve."""
    d = _as_date(d)
    if not is_trading_day(d):
        return False
    if d.month == 7 and d.day == 3 and d.weekday() <= 3:
        return True
    if d.month == 12 and d.day == 24 and d.weekday() <= 3:
        return True
    thanksgiving = _nth_weekday(d.year, 11, 3, 4)
    return d == thanksgiving + timedelta(days=1)


# ─────────────────────────────────────────────
# SESSIONS
# ─────────────────────────────────────────────

def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    import pandas as pd                          # accept strings and Timestamps
    return pd.to_datetime(d).date()


def _to_et(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(ET)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(ET)


def previous_trading_day(d: date) -> date:
    """The last session strictly before `d`."""
    d = _as_date(d) - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def next_trading_day(d: date) -> date:
    """The first session strictly after `d`."""
    d = _as_date(d) + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def session_close(d: date) -> datetime:
    """The closing bell for session `d`, timezone-aware in New York time."""
    d = _as_date(d)
    t = EARLY_CLOSE if is_early_close(d) else REGULAR_CLOSE
    return datetime.combine(d, t, tzinfo=ET)


def session_published(d: date) -> datetime:
    """When session `d`'s daily bar can be treated as final."""
    return session_close(d) + PUBLISH_DELAY


def last_completed_session(now: datetime | None = None) -> date:
    """
    The most recent session whose daily bar is final as of `now`.

    Before today's close (plus the publish delay) that is the previous
    session; on a weekend or holiday it is the last day the market was open.
    """
    et = _to_et(now)
    d = et.date()
    if is_trading_day(d) and et >= session_published(d):
        return d
    return previous_trading_day(d)


def market_status(now: datetime | None = None) -> dict:
    """A small description of where the market stands right now."""
    et = _to_et(now)
    today = et.date()
    status = {
        'now_et': et,
        'date': today,
        'is_trading_day': is_trading_day(today),
        'closed_reason': closed_reason(today),
        'early_close': is_early_close(today),
        'last_completed_session': last_completed_session(et),
        'next_session': (today if is_trading_day(today)
                         and et < session_close(today)
                         else next_trading_day(today)),
    }
    if status['is_trading_day']:
        open_at = datetime.combine(today, OPEN, tzinfo=ET)
        close_at = session_close(today)
        status['phase'] = ('pre-market' if et < open_at
                           else 'open' if et < close_at else 'after-hours')
    else:
        status['phase'] = 'closed'
    return status
