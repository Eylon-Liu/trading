"""
Point-in-time research panel: every factor for every S&P 500 member at each
month-end, plus forward total returns over 1, 3 and 6 months.

Uses the same code path as a backtest (quant.backtest._preload + factors.build_all
with facts filtered to filed <= date), so nothing here can see the future.

    python build_panel.py 2020-01-31 2026-10-31 out.pkl
"""
import sys
import time
from datetime import timedelta

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import config
from core import market_calendar as MC
from data import yahoo
from data.universe import parse_spec
from quant import backtest as BT
from quant import engine as EN
from quant import factors as FA

start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
start, end = pd.to_datetime(start).date(), pd.to_datetime(end).date()
spec = parse_spec('SPY')

t0 = time.time()
pre = BT._preload(spec, start, end, 'M')
print(f'preloaded {len(pre.tickers)} names, {len(pre.facts):,} facts in {time.time()-t0:.0f}s', flush=True)

# Month-end sessions.
days = pd.date_range(start, end, freq='ME')
dates = []
for d in days:
    d = d.date()
    while not MC.is_trading_day(d):
        d -= timedelta(days=1)
    dates.append(d)

# Adjusted closes for forward returns (total return), all names + SPY.
px = yahoo.price_history(sorted(set(pre.tickers) | {config.BENCHMARK_TICKER}),
                         start=start - timedelta(days=10),
                         end=end + timedelta(days=200), field='adj_close').ffill(limit=5)


def fwd(d0, months):
    """Close at d0 → close `months` later; NaN where the name stopped trading."""
    a = px.loc[:pd.Timestamp(d0)]
    if a.empty:
        return pd.Series(dtype=float)
    target = pd.Timestamp(d0) + pd.DateOffset(months=months)
    b = px.loc[:target]
    if b.index[-1] < target - pd.Timedelta(days=7):       # horizon not elapsed yet
        return pd.Series(np.nan, index=px.columns)
    return b.iloc[-1] / a.iloc[-1] - 1.0


frames, prev_fund, prev_d0 = [], None, None
facts = pre.facts
for i, d0 in enumerate(dates):
    t1 = time.time()
    universe = spec.resolve(d0)
    facts_d0 = facts[facts['filed'] <= pd.Timestamp(d0)] if not facts.empty else None
    changed = None
    if prev_d0 is not None and not facts.empty:
        new = facts[(facts['filed'] > pd.Timestamp(prev_d0)) & (facts['filed'] <= pd.Timestamp(d0))]
        changed = set(new['ticker'].unique())
    stash = {}
    raw = FA.build_all(universe, d0, use_cache=False, _facts_all=facts_d0,
                       _data_version=pre.version, _prev_fund=prev_fund,
                       _changed_tickers=changed, _splits=pre.splits,
                       _splits_applied=True, _fund_stash=stash,
                       _bt_ohlc=pre.ohlc, _bt_insiders=pre.insiders,
                       _bt_events=pre.events)
    prev_fund, prev_d0 = stash.get('result'), d0
    raw = EN._apply_size_filters(raw, spec)
    raw = raw.copy()
    raw['n_universe'] = len(universe)
    for m in (1, 3, 6):
        r = fwd(d0, m)
        raw[f'fwd_{m}m'] = r.reindex(raw.index)
        raw[f'spy_{m}m'] = r.get(config.BENCHMARK_TICKER, np.nan)
    raw.insert(0, 'date', pd.Timestamp(d0))
    frames.append(raw.reset_index().rename(columns={'index': 'ticker'}))
    print(f'{i+1}/{len(dates)} {d0}: {len(universe)} members, {len(raw)} scored, '
          f'{raw["fwd_3m"].notna().sum()} with 3m return, {time.time()-t1:.0f}s', flush=True)
    if (i + 1) % 6 == 0 or i + 1 == len(dates):
        pd.concat(frames, ignore_index=True).to_pickle(out)

print(f'done in {(time.time()-t0)/60:.1f} min → {out}', flush=True)
