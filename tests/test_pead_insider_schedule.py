"""
Earnings surprise (SUE), the insider-cluster gate, and the morning schedule.

Each test pins a decision that is easy to undo by accident: the drift screen
silently scoring names outside the drift window, a "cluster" strategy that no
longer counts buyers, an email that fires at the Mac's local 8:45 instead of
New York's, and a launchd file that launchd refuses to load.
"""

from __future__ import annotations

import plistlib
import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import config
from quant import engine as EN
from quant import factors as FA
from quant import fundamentals as F
from quant import strategies as ST

ROOT = config.ROOT


# ─────────────────────────────────────────────
# SUE ARITHMETIC
# ─────────────────────────────────────────────

VALS = [10, 11, 12, 13, 11, 12, 14, 13, 12, 13, 15, 14, 20]


def _ni_facts(values, filed_lag=35, start='2020-03-31'):
    ends = pd.date_range(start, periods=len(values), freq='QE')
    return pd.DataFrame([{
        'ticker': 'X', 'concept': 'net_income', 'tag': 'NetIncomeLoss',
        'unit': 'USD', 'period_start': e - pd.offsets.QuarterBegin(startingMonth=1),
        'period_end': e, 'filed': e + pd.Timedelta(days=filed_lag),
        'form': '10-Q', 'fy': e.year, 'fp': 'Q', 'val': float(v)}
        for e, v in zip(ends, values)])


def test_sue_is_latest_yoy_change_over_its_own_volatility():
    sue, filed = F.earnings_surprise(_ni_facts(VALS))
    changes = np.array(VALS[4:]) - np.array(VALS[:-4])
    assert sue == pytest.approx(changes[-1] / np.std(changes[-8:-1], ddof=1))
    assert filed == pd.Timestamp('2023-03-31') + pd.Timedelta(days=35)


def test_sue_needs_two_years_of_history():
    sue, filed = F.earnings_surprise(_ni_facts(VALS[:7]))
    assert np.isnan(sue) and filed is None


def test_sue_is_scale_free_and_signed():
    a, _ = F.earnings_surprise(_ni_facts(VALS))
    b, _ = F.earnings_surprise(_ni_facts([v * 1000 for v in VALS]))
    c, _ = F.earnings_surprise(_ni_facts(VALS[:-1] + [8]))
    assert a == pytest.approx(b) and a > 0 and c < 0


def test_sue_never_sees_a_quarter_filed_after_as_of():
    facts = _ni_facts(VALS[:-1] + [999])
    as_of = facts['filed'].iloc[-1] - pd.Timedelta(days=1)
    sue_then, filed = F.earnings_surprise(facts[facts['filed'] <= as_of])
    sue_full, _ = F.earnings_surprise(facts)
    assert filed < as_of and sue_then != pytest.approx(sue_full)


# ─────────────────────────────────────────────
# SUE AS A FACTOR — FRESHNESS GATE
# ─────────────────────────────────────────────

def _fund_factors(monkeypatch, as_of):
    facts = _ni_facts(VALS)
    monkeypatch.setattr(FA.yahoo, 'profile_asof', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(FA.yahoo, 'latest_prices',
                        lambda *a, **k: pd.Series({'X': 50.0}))
    gated = facts[facts['filed'] <= pd.Timestamp(as_of)]
    return FA.fundamental_factors(['X'], as_of, _facts_all=gated,
                                  _splits=pd.DataFrame(columns=['ticker', 'date', 'ratio']))


def test_surprise_is_present_while_the_quarter_is_fresh(monkeypatch):
    out = _fund_factors(monkeypatch, date(2023, 6, 1))     # filed 2023-05-05
    assert np.isfinite(out.loc['X', 'EARNINGS_SURPRISE'])


def test_surprise_expires_once_it_is_last_quarters_news(monkeypatch):
    out = _fund_factors(monkeypatch, date(2023, 9, 1))     # 119 days later
    assert np.isnan(out.loc['X', 'EARNINGS_SURPRISE'])


# ─────────────────────────────────────────────
# STRATEGY GATES
# ─────────────────────────────────────────────

def _strategy(**filters):
    return ST.Strategy('t', 't', 'mid', '', {'MOM_6_1': 1.0}, filters=filters)


def test_drift_window_is_an_explicit_filter():
    raw = pd.DataFrame({'EARNINGS_REACTION': [0.04, np.nan]}, index=['in', 'out'])
    kept = EN.apply_filters(raw, _strategy(require_earnings_reaction=1))
    assert list(kept) == ['in']
    absent = EN.apply_filters(raw.drop(columns='EARNINGS_REACTION'),
                              _strategy(require_earnings_reaction=1))
    assert list(absent) == []


def test_cluster_gate_counts_distinct_buyers():
    raw = pd.DataFrame({'INSIDER_CLUSTER': [3, 2, 1, np.nan]},
                       index=['three', 'two', 'one', 'none'])
    kept = EN.apply_filters(raw, _strategy(min_insider_buyers=2))
    assert list(kept) == ['three', 'two']


def test_earnings_drift_combines_reaction_and_surprise_inside_the_window():
    s = ST.ALL_STRATEGIES['earnings_drift']
    assert max(s.weights, key=s.weights.get) == 'EARNINGS_REACTION'
    assert 'EARNINGS_SURPRISE' in s.weights
    assert 'EVENT_RESULTS' not in s.weights           # ~1 for everyone
    assert s.filters.get('require_earnings_reaction')


def test_insider_cluster_requires_a_cluster_without_double_weighting():
    s = ST.ALL_STRATEGIES['insider_cluster']
    assert s.filters.get('min_insider_buyers', 0) >= 2
    # INSIDER_CLUSTER correlates 0.98 with INSIDER_NET_BUY: a gate, not a weight.
    assert 'INSIDER_CLUSTER' not in s.weights
    assert 'INSIDER_NET_BUY' in s.weights


# ─────────────────────────────────────────────
# SCHEDULE — NEW YORK TIME, WHATEVER THE MACHINE'S ZONE
# ─────────────────────────────────────────────

def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_morning_default_is_before_the_open_and_after_premarket_reports():
    hh, mm = (int(x) for x in config.QUANT_MORNING_TIME.split(':'))
    assert (8, 0) <= (hh, mm) < (9, 30), 'between pre-market releases and the open'
    assert config.QUANT_TZ == 'America/New_York' or 'QUANT_TZ' in __import__('os').environ


@pytest.mark.parametrize('utc_now,due', [
    ('2026-11-10T13:40:00', False),     # 08:40 EST
    ('2026-11-10T13:50:00', True),      # 08:50 EST
    ('2026-07-14T12:40:00', False),     # 08:40 EDT
    ('2026-07-14T12:50:00', True),      # 08:50 EDT
])
def test_morning_fires_at_845_new_york_across_dst(monkeypatch, utc_now, due):
    import cli
    monkeypatch.setattr(config, 'QUANT_MORNING_TIME', '08:45')
    got, _ = cli.scheduled_runs(_utc(utc_now), 'America/New_York')
    assert ('morning' in got) is due


def test_morning_run_pulls_this_mornings_8ks():
    """Pre-market releases land after the evening run; the brief must see them."""
    import cli
    assert 'filings' in cli._daily_sources('daily', 'quick')
    assert 'filings' in cli._daily_sources('news', 'quick')
    assert cli._daily_sources('daily', 'full') is None        # everything


def test_launchd_job_is_a_tick_not_a_local_clock_time():
    """
    launchd only knows the Mac's zone. A fixed Hour fires at the wrong New
    York time for anyone outside Eastern time, and drifts by an hour across
    DST wherever the local rules differ.
    """
    plists = sorted((ROOT / 'tools' / 'launchd').glob('*.plist'))
    names = {p.name for p in plists}
    assert 'com.quant.morning.plist' not in names
    assert 'com.quant.evening.plist' not in names
    for p in plists:
        data = plistlib.loads(p.read_bytes())       # also: valid XML
        assert 'StartCalendarInterval' not in data, f'{p.name} uses a local clock'
    sched = plistlib.loads((ROOT / 'tools/launchd/com.quant.scheduler.plist').read_bytes())
    assert sched['StartInterval'] <= 600
    assert sched['ProgramArguments'][-1] == 'tick'


def test_launchd_comments_contain_no_double_hyphen():
    """`--` inside an XML comment makes launchd reject the whole file."""
    for p in (ROOT / 'tools' / 'launchd').glob('*.plist'):
        for comment in re.findall(r'<!--(.*?)-->', p.read_text(), re.S):
            assert '--' not in comment, p.name


def test_tick_mode_and_legacy_cleanup_exist():
    routine = (ROOT / 'tools/daily_routine.sh').read_text()
    setup = (ROOT / 'tools/setup_schedule.sh').read_text()
    assert 'scheduler --once' in routine
    assert 'com.quant.morning' in setup and 'remove_legacy' in setup
