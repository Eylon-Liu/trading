"""
Backtest execution mechanics, on synthetic prices where the right answer is
known by construction.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core import market_calendar as MC
from quant import backtest as BT
from quant import research as RS
from quant import tradeplan as TP


def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    days = pd.date_range(start, end, freq='D')
    return pd.DatetimeIndex([d for d in days if MC.is_trading_day(d.date())])


# ── rebalance dates ──────────────────────────────────────────────

def test_rebalances_fall_on_sessions_not_calendar_month_ends():
    s = _sessions('2026-01-02', '2026-06-30')
    dates = [s[p].date() for p in BT.rebalance_positions(s, 'M')]
    # Jan 31, Feb 28 2026 are weekends: the last *session* is used instead.
    assert date(2026, 1, 30) in dates and date(2026, 2, 27) in dates
    assert all(MC.is_trading_day(d) for d in dates)
    assert dates[0] == date(2026, 1, 2)


def test_quarterly_and_annual_positions():
    s = _sessions('2025-01-02', '2026-12-31')
    q = [s[p].date() for p in BT.rebalance_positions(s, 'Q')]
    assert date(2025, 3, 31) in q and date(2025, 12, 31) in q
    a = [s[p].date() for p in BT.rebalance_positions(s, 'A')]
    assert a == [date(2025, 1, 2), date(2025, 12, 31), date(2026, 12, 31)]


# ── long-horizon: continuity and lag ─────────────────────────────

@pytest.fixture
def synthetic_long(monkeypatch):
    sessions = _sessions('2025-12-01', '2026-04-30')
    n = len(sessions)
    a = pd.Series(100 * 1.01 ** np.arange(n), index=sessions)     # +1% a day
    b = pd.Series(100.0, index=sessions)
    adj = pd.DataFrame({'A': a, 'B': b})
    pre = BT._Preload(['A', 'B'], pd.DataFrame(columns=['filed', 'ticker']), 'v',
                      pd.DataFrame(), (adj, adj, adj, adj),
                      pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(BT, '_preload', lambda *a, **k: pre)
    monkeypatch.setattr(BT, '_benchmark_curve',
                        lambda s, e, idx: pd.Series(1.0, index=idx))
    monkeypatch.setattr(BT, '_risk_free', lambda s, e: 0.0)
    calls = []

    def fake_run(self, d0, top_n, plan_params=None):
        calls.append(d0)
        scores = pd.DataFrame({'composite': [1.0, 0.0]}, index=['A', 'B'])
        return SimpleNamespace(scores=scores, degraded=False, warnings=[],
                               _fund_raw=None, plans=pd.DataFrame())
    monkeypatch.setattr(BT._Scorer, 'run', fake_run)
    return a, calls


def test_no_return_is_lost_between_rebalances(synthetic_long):
    a, _calls = synthetic_long
    res = BT.run_backtest(None, 'buffett', '2026-01-02', '2026-04-30',
                          n_holdings=1, rebalance='M', strict=False, cost_bps=0)
    rets = res.equity.pct_change().dropna()
    # Holding A the whole time with zero costs: every single session must
    # show A's +1%, including the ones that straddle a weekend month-end.
    assert np.allclose(rets.values, 0.01, atol=1e-12)
    assert res.equity.index.is_unique


def test_fill_is_one_session_after_the_signal(synthetic_long):
    _a, calls = synthetic_long
    res = BT.run_backtest(None, 'buffett', '2026-01-02', '2026-04-30',
                          n_holdings=1, rebalance='M', strict=False, cost_bps=0)
    assert calls[0] == date(2026, 1, 2)
    assert res.equity.index[0].date() == date(2026, 1, 5)
    assert res.metrics['execution_lag'] == 1


def test_initial_buy_is_charged_but_holding_is_free(synthetic_long, monkeypatch):
    import config
    monkeypatch.setattr(config, 'SLIPPAGE_BPS', 0.0)
    res = BT.run_backtest(None, 'buffett', '2026-01-02', '2026-04-30',
                          n_holdings=1, rebalance='M', strict=False, cost_bps=10)
    assert res.equity.iloc[0] == pytest.approx(1 - 0.001)
    assert res.metrics['turnover'] == pytest.approx(0.0)


# ── mid-horizon simulation ───────────────────────────────────────

def _bars(sessions, opens, highs, lows, closes, ticker='T'):
    def f(v):
        return pd.DataFrame({ticker: v}, index=sessions)
    return {'open': f(opens), 'high': f(highs), 'low': f(lows), 'close': f(closes)}


def _plan(entry, stop, target, kind='pullback', weight=10.0):
    return pd.DataFrame([{'entry': entry, 'entry_type': kind, 'stop': stop,
                          'target': target, 'weight_pct': weight}], index=['T'])


S = _sessions('2026-03-02', '2026-03-31')
P = TP.PlanParams(time_stop_days=60)


def test_pullback_that_is_never_reached_is_not_traded():
    n = len(S)
    bars = _bars(S, [100] * n, [101] * n, [99.5] * n, [100] * n)
    eq, trades, stats = BT.simulate_trades(bars, {S[0]: _plan(98, 95, 110)}, P, 0.0)
    assert trades.empty and stats['expired'] == 1
    assert eq.iloc[-1] == pytest.approx(1.0)


def test_signal_day_range_cannot_trigger_a_stop():
    n = len(S)
    lows = [90] + [99.5] * (n - 1)              # only the signal day dips to 90
    bars = _bars(S, [100] * n, [101] * n, lows, [100] * n)
    _eq, trades, _ = BT.simulate_trades(bars, {S[0]: _plan(100, 95, 120, 'market')},
                                        P, 0.0)
    assert trades['reason'].tolist() == ['end']


def test_gap_through_stop_fills_at_the_open():
    n = len(S)
    opens = [100, 100, 100, 88] + [88] * (n - 4)
    lows = [99, 99, 99, 87] + [87] * (n - 4)
    bars = _bars(S, opens, [101] * n, lows, opens)
    _eq, trades, _ = BT.simulate_trades(bars, {S[0]: _plan(100, 95, 120, 'market')},
                                        P, 0.0)
    t = trades.iloc[0]
    assert t['reason'] == 'stop' and t['exit'] == pytest.approx(88)


def test_limit_fill_day_does_not_count_a_target():
    n = len(S)
    # Day 1 trades from 103 down to 98 (fills the 99 limit) and up to 130.
    highs = [101, 130] + [101] * (n - 2)
    lows = [99.5, 98] + [99.5] * (n - 2)
    bars = _bars(S, [100, 103] + [100] * (n - 2), highs, lows, [100] * n)
    _eq, trades, _ = BT.simulate_trades(bars, {S[0]: _plan(99, 90, 120)}, P, 0.0)
    assert trades.iloc[0]['entry'] == pytest.approx(99)
    assert trades.iloc[0]['reason'] != 'target'


def test_gross_exposure_is_capped():
    n = len(S)
    two = ['T', 'U']
    frame = lambda v: pd.DataFrame({t: v for t in two}, index=S)   # noqa: E731
    bars = {'open': frame(100.0), 'high': frame(101.0), 'low': frame(99.5),
            'close': frame(100.0)}
    plans = pd.DataFrame([{'entry': 100, 'entry_type': 'market', 'stop': 90,
                           'target': 130, 'weight_pct': 80.0}] * 2, index=two)
    _eq, trades, stats = BT.simulate_trades(bars, {S[0]: plans}, P, 0.0)
    assert stats['avg_exposure'] <= 1.0 + 1e-9
    assert trades['weight'].sum() == pytest.approx(1.0)


def test_open_losses_show_in_the_drawdown():
    n = len(S)
    closes = [100, 100] + [92] * (n - 2)         # -8%, stop at 90 never hit
    bars = _bars(S, closes, [101] * n, [91] * n, closes)
    eq, _trades, _ = BT.simulate_trades(
        bars, {S[0]: _plan(100, 90, 150, 'market', weight=50)}, P, 0.0)
    m = BT.compute_metrics(eq, rf=0.0)
    assert m['max_drawdown'] == pytest.approx(-0.04, abs=1e-9)


def test_costs_are_charged_on_both_sides():
    n = len(S)
    bars = _bars(S, [100] * n, [101] * n, [99.5] * n, [100] * n)
    _eq, trades, _ = BT.simulate_trades(bars, {S[0]: _plan(100, 90, 150, 'market')},
                                        P, 0.001)
    assert trades.iloc[0]['net_return'] == pytest.approx(0.999 / 1.001 - 1)


# ── IC forward returns ───────────────────────────────────────────

def test_forward_returns_start_at_the_signal_close(monkeypatch):
    s = _sessions('2026-02-20', '2026-04-30')
    px = pd.DataFrame({'X': np.linspace(50, 120, len(s))}, index=s)
    monkeypatch.setattr(RS.yahoo, 'price_history',
                        lambda tickers, start=None, end=None, field=None:
                        px.loc[pd.Timestamp(start):pd.Timestamp(end)])
    as_of = date(2026, 3, 2)
    fwd = RS.forward_returns(['X'], as_of, 1)
    start_px = px.loc[:pd.Timestamp(as_of)].iloc[-1, 0]
    end_px = px.loc[:pd.Timestamp(as_of) + pd.Timedelta(days=30)].iloc[-1, 0]
    assert fwd['X'] == pytest.approx(end_px / start_px - 1)
