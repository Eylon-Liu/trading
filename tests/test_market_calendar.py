"""NYSE calendar — holidays, early closes and the 'last completed session'."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from core import market_calendar as MC


# Published NYSE schedules. Checked against the exchange's own calendar.
@pytest.mark.parametrize('d,name', [
    (date(2025, 1, 1), "New Year's Day"),
    (date(2025, 1, 9), 'National Day of Mourning (Carter)'),
    (date(2025, 4, 18), 'Good Friday'),
    (date(2025, 6, 19), 'Juneteenth'),
    (date(2025, 11, 27), 'Thanksgiving Day'),
    (date(2026, 1, 19), 'Martin Luther King Jr. Day'),
    (date(2026, 2, 16), "Washington's Birthday"),
    (date(2026, 4, 3), 'Good Friday'),
    (date(2026, 5, 25), 'Memorial Day'),
    (date(2026, 7, 3), 'Independence Day'),        # July 4 is a Saturday
    (date(2026, 9, 7), 'Labor Day'),
    (date(2026, 11, 26), 'Thanksgiving Day'),
    (date(2026, 12, 25), 'Christmas Day'),
    (date(2027, 6, 18), 'Juneteenth'),             # June 19 is a Saturday
    (date(2027, 7, 5), 'Independence Day'),        # July 4 is a Sunday
    (date(2027, 12, 24), 'Christmas Day'),         # Dec 25 is a Saturday
])
def test_holidays_match_the_exchange_calendar(d, name):
    assert MC.holiday_name(d) == name
    assert not MC.is_trading_day(d)
    assert MC.closed_reason(d) == name


def test_saturday_new_year_is_not_observed_on_friday():
    """NYSE stays open on Dec 31 when Jan 1 falls on a Saturday."""
    assert MC.is_trading_day(date(2021, 12, 31))
    assert MC.is_trading_day(date(2027, 12, 31))


def test_juneteenth_only_from_2022():
    assert MC.is_trading_day(date(2021, 6, 18))
    assert not MC.is_trading_day(date(2022, 6, 20))


@pytest.mark.parametrize('d', [
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 11, 27), date(2026, 12, 24),
])
def test_early_closes(d):
    assert MC.is_early_close(d)
    assert MC.session_close(d).hour == 13


def test_no_early_close_when_july_3_is_the_holiday():
    assert not MC.is_early_close(date(2026, 7, 3))
    assert not MC.is_early_close(date(2026, 7, 2))


def test_weekends_are_closed():
    assert MC.closed_reason(date(2026, 11, 7)) == 'Weekend'
    assert MC.closed_reason(date(2026, 11, 8)) == 'Weekend'
    assert MC.closed_reason(date(2026, 11, 9)) is None


def test_previous_and_next_skip_holidays_and_weekends():
    assert MC.previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)
    assert MC.next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27)
    assert MC.next_trading_day(date(2026, 7, 2)) == date(2026, 7, 6)
    assert MC.previous_trading_day(date(2026, 1, 20)) == date(2026, 1, 16)


# ── last completed session ───────────────────────────────────────

def test_close_is_judged_in_new_york_time_across_dst():
    """16:00 ET is 20:00 UTC in summer and 21:00 UTC in winter."""
    # Summer (EDT): 20:50 UTC = 16:50 ET — bar published.
    assert MC.last_completed_session(datetime(2026, 8, 11, 20, 50)) == date(2026, 8, 11)
    # Winter (EST): 20:50 UTC = 15:50 ET — still trading.
    assert MC.last_completed_session(datetime(2026, 11, 10, 20, 50)) == date(2026, 11, 9)
    assert MC.last_completed_session(datetime(2026, 11, 10, 22, 0)) == date(2026, 11, 10)


def test_holiday_rolls_back_to_the_prior_session():
    thanksgiving_evening = datetime(2026, 11, 26, 23, 0)
    assert MC.last_completed_session(thanksgiving_evening) == date(2026, 11, 25)


def test_early_close_session_completes_in_the_afternoon():
    # 13:00 ET close + 45 min; 19:00 UTC on Nov 27 is 14:00 ET.
    assert MC.last_completed_session(datetime(2026, 11, 27, 19, 0)) == date(2026, 11, 27)


def test_aware_datetimes_in_any_zone_are_accepted():
    from zoneinfo import ZoneInfo
    tokyo = datetime(2026, 11, 10, 8, 0, tzinfo=ZoneInfo('Asia/Tokyo'))  # Nov 9 18:00 ET
    assert MC.last_completed_session(tokyo) == date(2026, 11, 9)
    utc = datetime(2026, 11, 10, 2, 0, tzinfo=timezone.utc)
    assert MC.last_completed_session(utc) == date(2026, 11, 9)


def test_market_status_on_a_weekend():
    st = MC.market_status(datetime(2026, 11, 7, 15, 0))
    assert st['phase'] == 'closed'
    assert st['closed_reason'] == 'Weekend'
    assert st['last_completed_session'] == date(2026, 11, 6)
    assert st['next_session'] == date(2026, 11, 9)
