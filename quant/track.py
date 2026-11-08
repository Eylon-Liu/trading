"""
Live track record — what each strategy's picks actually did afterwards.

A backtest is a simulation over data the strategy was designed on. This is
the other half of the evidence: every screen the daily routine stores is a
dated, out-of-sample prediction, and here those predictions are graded
against what prices did next.

For every stored run (one per strategy per session) and each horizon:

  pick return    equal-weight return of the run's top-N names
  benchmark      SPY over the same window
  excess         pick return − benchmark
  rank IC        Spearman correlation between the composite score of *every*
                 ranked name and its forward return — does the whole
                 ordering carry information, not just the top

Returns start at the close of the session *after* the run date (the first
close at which you could have acted on the email), matching the backtest's
execution lag. A cohort counts for a horizon only once the horizon has fully
elapsed — nothing half-finished is averaged in.

Caveat printed with the numbers: daily cohorts with a 21-session horizon
overlap almost entirely, so 250 daily observations hold roughly a dozen
independent ones. `independent` reports that number; trust it, not `n`.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from core import db

log = logging.getLogger(__name__)

HORIZONS = {'1W': 5, '1M': 21, '3M': 63}


def _runs(since: date, spec_json: str | None) -> pd.DataFrame:
    clause = 'AND universe_spec_json = :u' if spec_json else ''
    params = {'d': str(since)}
    if spec_json:
        params['u'] = spec_json
    runs = db.read_sql(f"""
        SELECT run_id, as_of, strategy, created_at FROM runs
        WHERE as_of >= :d {clause}
        ORDER BY as_of, created_at
    """, params, parse_dates=['as_of', 'created_at'])
    # Older stores can hold several runs per day; the newest one stands.
    return runs.drop_duplicates(['strategy', 'as_of'], keep='last')


def cohort_returns(as_of: date | None = None, lookback_days: int = 400,
                   top_n: int = 10, spec_json: str | None = None,
                   horizons: dict[str, int] | None = None) -> pd.DataFrame:
    """One row per (strategy, run date, horizon) whose horizon has elapsed."""
    from data import yahoo

    as_of = as_of or date.today()
    horizons = horizons or HORIZONS
    runs = _runs(as_of - timedelta(days=lookback_days), spec_json)
    runs = runs[runs['as_of'].dt.date <= as_of]
    if runs.empty:
        return pd.DataFrame()

    ids = runs['run_id'].tolist()
    scores = []
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        ph = ','.join(f':r{j}' for j in range(len(chunk)))
        scores.append(db.read_sql(
            f'SELECT run_id, ticker, rank, composite FROM scores WHERE run_id IN ({ph})',
            {f'r{j}': r for j, r in enumerate(chunk)}))
    scores = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    if scores.empty:
        return pd.DataFrame()

    tickers = sorted(set(scores['ticker']) | {config.BENCHMARK_TICKER})
    first = runs['as_of'].min().date()
    px = yahoo.price_history(tickers, start=first - timedelta(days=7),
                             end=as_of, field='adj_close')
    if px.empty or config.BENCHMARK_TICKER not in px.columns:
        return pd.DataFrame()
    px = px.ffill()
    sessions = px.index
    bench = px[config.BENCHMARK_TICKER]

    by_run = {rid: g for rid, g in scores.groupby('run_id')}
    rows = []
    for _i, run in runs.iterrows():
        g = by_run.get(run['run_id'])
        if g is None or g.empty:
            continue
        # Entry: first close strictly after the run date.
        p_in = int(sessions.searchsorted(run['as_of'], side='right'))
        if p_in >= len(sessions):
            continue
        top = g.sort_values('rank').head(top_n)['ticker']
        top = [t for t in top if t in px.columns]
        for label, h in horizons.items():
            p_out = p_in + h
            if p_out >= len(sessions):
                continue
            entry, exit_ = px.iloc[p_in], px.iloc[p_out]
            fwd = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan)
            picks = fwd.reindex(top).dropna()
            if picks.empty:
                continue
            b = float(bench.iloc[p_out] / bench.iloc[p_in] - 1.0)
            pair = pd.concat([g.set_index('ticker')['composite'], fwd], axis=1).dropna()
            ic = (float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method='spearman'))
                  if len(pair) >= 8 else np.nan)
            rows.append({'strategy': run['strategy'], 'run_date': run['as_of'].date(),
                         'horizon': label, 'sessions': h, 'n_picks': len(picks),
                         'pick_return': float(picks.mean()), 'benchmark': b,
                         'excess': float(picks.mean()) - b, 'ic': ic,
                         'entry_date': sessions[p_in].date(),
                         'exit_date': sessions[p_out].date()})
    return pd.DataFrame(rows)


def summary(cohorts: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    """
    Per strategy and horizon: cohorts, independent periods, mean excess,
    hit rate (share of cohorts beating SPY) and mean live rank IC.
    """
    df = cohort_returns(**kw) if cohorts is None else cohorts
    if df is None or df.empty:
        return pd.DataFrame()
    from quant import strategies as ST

    def info(k):
        try:
            st = ST.get(k)
            return st.name, bool(st.invert)
        except Exception:                          # noqa: BLE001
            return k, False

    out = []
    for (strat, horizon), g in df.groupby(['strategy', 'horizon']):
        span = (pd.to_datetime(g['run_date']).max()
                - pd.to_datetime(g['run_date']).min()).days * 252 / 365
        h = int(g['sessions'].iloc[0])
        label, avoid = info(strat)
        # An avoid list works when its names *lag*: count those cohorts as
        # hits, and flip the IC sign so positive means right for every row.
        hits = (g['excess'] < 0) if avoid else (g['excess'] > 0)
        ic = float(g['ic'].mean()) if g['ic'].notna().any() else np.nan
        out.append({
            'strategy': strat,
            'name': f'{label} (avoid list)' if avoid else label,
            'horizon': horizon, 'avoid': avoid,
            'sessions': h, 'cohorts': len(g),
            'independent': max(1, int(round(span / h)) + 1),
            'mean_pick_return': float(g['pick_return'].mean()),
            'mean_excess': float(g['excess'].mean()),
            'hit_rate': float(hits.mean()),
            'mean_ic': -ic if avoid and np.isfinite(ic) else ic,
            'first': g['run_date'].min(), 'last': g['run_date'].max(),
        })
    order = {k: i for i, k in enumerate(HORIZONS)}
    res = pd.DataFrame(out)
    res['_o'] = res['horizon'].map(order).fillna(99)
    return (res.sort_values(['_o', 'mean_excess'], ascending=[True, False])
               .drop(columns='_o').reset_index(drop=True))
