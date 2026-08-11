"""
Trade plans for mid-term positions.

A long-term screen answers "is this worth owning?" and needs no exit level —
you exit when the thesis breaks. A mid-term position held days-to-months is a
different object: it needs a defined entry, a stop that reflects the stock's
own volatility, a target expressed as a multiple of the risk taken, a time
stop for when nothing happens, and a position weight derived from the stop
distance rather than picked by feel.

Everything here is mechanical and derived from price. Weights are expressed as
a percentage of portfolio given a stated risk budget — never share counts —
because this is a research tool and knows nothing about anyone's account.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from data import yahoo

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────

@dataclass
class PlanParams:
    """Tunables for plan construction."""

    atr_stop_mult: float = 2.0       # stop distance in ATR units
    target_r: float = 2.5            # profit target as a multiple of risk
    time_stop_days: int = 60         # trading days before a stale trade is cut
    risk_budget_pct: float = 1.0     # % of portfolio risked per position
    max_weight_pct: float = 10.0     # cap on any single position
    min_weight_pct: float = 1.0
    swing_lookback: int = 20         # sessions used to find structural support
    max_stop_pct: float = 15.0       # refuse trades needing a wider stop
    entry_pullback_atr: float = 0.5  # how far below spot a limit entry sits


@dataclass
class TradePlan:
    """A complete mid-term plan for one name."""

    ticker: str
    as_of: date
    price: float
    entry: float
    entry_type: str                  # market | pullback | breakout
    stop: float
    stop_pct: float
    stop_basis: str                  # atr | swing_low
    target: float
    target_pct: float
    r_multiple: float
    weight_pct: float
    time_stop_date: date
    atr_pct: float
    setup: str
    invalidation: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d['as_of'] = str(self.as_of)
        d['time_stop_date'] = str(self.time_stop_date)
        return d


# ─────────────────────────────────────────────
# PLAN CONSTRUCTION
# ─────────────────────────────────────────────

def _swing_low(low: pd.Series, lookback: int) -> float:
    s = low.dropna().tail(lookback)
    return float(s.min()) if len(s) else np.nan


def build_plan(ticker: str, as_of: date, price: float, atr_value: float,
               ma50: float | None, ma200: float | None,
               low_series: pd.Series | None,
               setup: str = 'momentum',
               params: PlanParams | None = None) -> TradePlan | None:
    """
    Construct one trade plan, or None when the setup is not tradable.

    The stop is placed at whichever is *further* from spot: an ATR-based level
    or the recent swing low. Using the tighter of the two looks better on
    paper but gets stopped out by ordinary noise; the wider level respects
    both the stock's volatility and its market structure.
    """
    params = params or PlanParams()

    if not np.isfinite(price) or price <= 0 or not np.isfinite(atr_value) or atr_value <= 0:
        return None

    atr_stop = price - params.atr_stop_mult * atr_value
    stop_basis = 'atr'
    stop = atr_stop

    if low_series is not None:
        swing = _swing_low(low_series, params.swing_lookback)
        if np.isfinite(swing) and swing < price and swing < atr_stop:
            stop, stop_basis = swing * 0.995, 'swing_low'

    if not np.isfinite(stop) or stop <= 0 or stop >= price:
        return None

    stop_pct = (price - stop) / price * 100.0
    if stop_pct > params.max_stop_pct:
        # Too volatile to hold with a sane stop — sizing down far enough to
        # keep the risk budget would leave a position too small to matter.
        return None

    # Entry: pull the limit slightly below spot so the plan is not chasing.
    if setup == 'breakout':
        entry, entry_type = price, 'market'
    else:
        entry = price - params.entry_pullback_atr * atr_value
        entry_type = 'pullback'
        if entry <= stop:
            entry, entry_type = price, 'market'

    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None

    target = entry + params.target_r * risk_per_share
    entry_stop_pct = risk_per_share / entry * 100.0

    # Fixed-fractional sizing: risking `risk_budget_pct` of the portfolio with
    # a stop `entry_stop_pct` away implies this weight.
    weight = params.risk_budget_pct / (entry_stop_pct / 100.0)
    weight = float(np.clip(weight, params.min_weight_pct, params.max_weight_pct))

    invalidation = _invalidation_text(stop_basis, stop, ma50, ma200)

    return TradePlan(
        ticker=ticker, as_of=as_of, price=round(price, 2),
        entry=round(entry, 2), entry_type=entry_type,
        stop=round(stop, 2), stop_pct=round(entry_stop_pct, 2),
        stop_basis=stop_basis,
        target=round(target, 2),
        target_pct=round((target - entry) / entry * 100.0, 2),
        r_multiple=params.target_r,
        weight_pct=round(weight, 2),
        time_stop_date=as_of + timedelta(days=int(params.time_stop_days * 1.45)),
        atr_pct=round(atr_value / price * 100.0, 2),
        setup=setup,
        invalidation=invalidation,
    )


def _invalidation_text(basis: str, stop: float, ma50, ma200) -> str:
    bits = [f'close below {stop:.2f} ({basis.replace("_", " ")})']
    if ma50 is not None and np.isfinite(ma50):
        bits.append(f'or loss of 50-day MA ({ma50:.2f})')
    if ma200 is not None and np.isfinite(ma200):
        bits.append(f'trend gone below 200-day ({ma200:.2f})')
    return '; '.join(bits)


def build_plans(factors: pd.DataFrame, as_of: date | str,
                setup: str = 'momentum',
                params: PlanParams | None = None) -> pd.DataFrame:
    """Trade plans for every name in `factors` that supports one."""
    as_of = pd.to_datetime(as_of).date()
    params = params or PlanParams()
    if factors.empty:
        return pd.DataFrame()

    tickers = list(factors.index)
    # Adjusted basis: swing lows must be comparable to the adjusted PRICE.
    _hi, lows, _cl = yahoo.adjusted_ohlc(
        tickers, start=as_of - timedelta(days=120), end=as_of)

    plans = []
    for t in tickers:
        row = factors.loc[t]
        plan = build_plan(
            ticker=t, as_of=as_of,
            price=float(row.get('PRICE', np.nan)),
            atr_value=float(row.get('ATR_14', np.nan)),
            ma50=row.get('MA50'), ma200=row.get('MA200'),
            low_series=lows[t] if t in lows.columns else None,
            setup=setup, params=params,
        )
        if plan:
            plans.append(plan.to_dict())

    if not plans:
        return pd.DataFrame()
    return pd.DataFrame(plans).set_index('ticker')


# ─────────────────────────────────────────────
# EXIT MONITORING
# ─────────────────────────────────────────────

EXIT_REASONS = {
    'stop': '🔴 Stop hit',
    'target': '🟢 Target reached',
    'time': '🟠 Time stop — thesis has not played out',
    'trend': '🟠 Trend broken (below 200-day MA)',
    'signal': '🟡 Signal decayed — no longer ranks',
}


def check_exits(plans: pd.DataFrame, as_of: date | str,
                still_ranked: set[str] | None = None) -> pd.DataFrame:
    """
    Re-evaluate open plans against subsequent price action.

    Drives the "what changed" section of the daily report: a mid-term plan is
    only useful if something checks whether its levels have been hit.
    """
    if plans.empty:
        return pd.DataFrame()

    as_of = pd.to_datetime(as_of).date()
    tickers = list(plans.index)

    start = min(pd.to_datetime(plans['as_of']).min().date(),
                as_of - timedelta(days=5))
    highs, lows, closes = yahoo.adjusted_ohlc(tickers, start=start, end=as_of)

    rows = []
    for t in tickers:
        plan = plans.loc[t]
        opened = pd.to_datetime(plan['as_of']).date()
        if t not in closes.columns:
            continue

        window = slice(pd.Timestamp(opened), pd.Timestamp(as_of))
        hi = highs[t].loc[window].dropna() if t in highs.columns else pd.Series(dtype=float)
        lo = lows[t].loc[window].dropna() if t in lows.columns else pd.Series(dtype=float)
        cl = closes[t].loc[window].dropna()
        if cl.empty:
            continue

        status, reason, hit_date = 'open', None, None

        # Stop is checked first: on a day that traded through both levels we
        # cannot tell the order from daily bars, so assume the worse fill.
        if not lo.empty and (lo <= plan['stop']).any():
            status, reason = 'closed', 'stop'
            hit_date = lo[lo <= plan['stop']].index[0].date()
        elif not hi.empty and (hi >= plan['target']).any():
            status, reason = 'closed', 'target'
            hit_date = hi[hi >= plan['target']].index[0].date()
        elif as_of >= pd.to_datetime(plan['time_stop_date']).date():
            status, reason = 'closed', 'time'
            hit_date = pd.to_datetime(plan['time_stop_date']).date()
        elif still_ranked is not None and t not in still_ranked:
            status, reason = 'review', 'signal'

        last = float(cl.iloc[-1])
        entry = float(plan['entry'])
        rows.append({
            'ticker': t, 'status': status,
            'reason': EXIT_REASONS.get(reason) if reason else None,
            'opened': opened, 'hit_date': hit_date,
            'entry': entry, 'last': round(last, 2),
            'pnl_pct': round((last / entry - 1.0) * 100.0, 2),
            'r_realized': round((last - entry) / (entry - float(plan['stop'])), 2)
                          if entry > float(plan['stop']) else np.nan,
            'stop': plan['stop'], 'target': plan['target'],
        })

    return pd.DataFrame(rows).set_index('ticker') if rows else pd.DataFrame()


def portfolio_risk(plans: pd.DataFrame) -> dict:
    """Aggregate risk if every plan in the frame were taken."""
    if plans.empty:
        return {}
    total_weight = float(plans['weight_pct'].sum())
    risk = float((plans['weight_pct'] * plans['stop_pct'] / 100.0).sum())
    return {
        'n_positions': int(len(plans)),
        'total_weight_pct': round(total_weight, 1),
        'portfolio_risk_pct': round(risk, 2),
        'avg_stop_pct': round(float(plans['stop_pct'].mean()), 2),
        'avg_weight_pct': round(float(plans['weight_pct'].mean()), 2),
        'cash_pct': round(max(0.0, 100.0 - total_weight), 1),
    }
