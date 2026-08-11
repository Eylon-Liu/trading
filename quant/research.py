"""
Factor research — does the ranking actually predict anything?

A backtest can beat its benchmark on luck, especially in a small universe
where a couple of names dominate. These diagnostics separate "the strategy
made money" from "the score had predictive power":

  rank IC        Spearman correlation between score and subsequent return.
                 Near zero means the ranking is noise, whatever the equity
                 curve did.
  IC decay       How IC behaves as the horizon lengthens — tells you the
                 holding period the signal actually supports.
  quintile spread Return of the top fifth minus the bottom fifth. A monotonic
                 progression across quintiles is the sign of a real factor;
                 a lumpy one usually means a few outliers.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from data import yahoo
from data.universe import UniverseSpec
from quant import engine as EN
from quant import factors as FA
from quant import transforms as T

log = logging.getLogger(__name__)


def forward_returns(tickers: list[str], as_of: date, horizon_months: int
                    ) -> pd.Series:
    """Total return over the next `horizon_months`, for IC work."""
    end = as_of + timedelta(days=int(horizon_months * 30.44))
    px = yahoo.price_history(tickers, start=as_of - timedelta(days=7),
                             end=end, field='adj_close')
    if px.empty or len(px) < 2:
        return pd.Series(dtype=float)
    px = px.ffill()
    return (px.iloc[-1] / px.iloc[0] - 1.0).replace([np.inf, -np.inf], np.nan)


def information_coefficient(scores: pd.Series, fwd: pd.Series,
                            method: str = 'spearman') -> float:
    pair = pd.concat([scores, fwd], axis=1).dropna()
    if len(pair) < 8:
        return np.nan
    pair.columns = ['s', 'f']
    return float(pair['s'].corr(pair['f'], method=method))


def ic_decay(spec: UniverseSpec, strategy_key: str, start: date | str,
             end: date | str | None = None,
             horizons: list[int] | None = None,
             freq: str = 'QE') -> pd.DataFrame:
    """
    Mean IC at several forward horizons.

    Falling IC with horizon means a short-lived signal; rising means the
    factor needs time to work and should be held longer.
    """
    horizons = horizons or config.IC_HORIZONS
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    dates = pd.date_range(start, end, freq=freq)

    rows = []
    for ts in dates:
        d0 = ts.date()
        result = EN.run(spec, strategy_key, as_of=d0, persist=False)
        if result.scores.empty or len(result.scores) < 8:
            continue
        comp = result.scores['composite']
        for h in horizons:
            if d0 + timedelta(days=int(h * 30.44)) > end:
                continue
            fwd = forward_returns(list(comp.index), d0, h)
            ic = information_coefficient(comp, fwd)
            if np.isfinite(ic):
                rows.append({'date': d0, 'horizon_months': h, 'ic': ic})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = df.groupby('horizon_months')['ic'].agg(
        mean_ic='mean', std_ic='std', n='count',
        hit_rate=lambda s: float((s > 0).mean()))
    summary['ic_ir'] = summary['mean_ic'] / summary['std_ic'].replace(0, np.nan)
    return summary.reset_index()


def quintile_analysis(spec: UniverseSpec, strategy_key: str,
                      start: date | str, end: date | str | None = None,
                      horizon_months: int = 3, freq: str = 'QE',
                      n_buckets: int | None = None) -> pd.DataFrame:
    """
    Average forward return per score bucket.

    Monotonic buckets are the signature of a factor that works; a strong top
    bucket with a jumbled middle usually means a handful of names carried it.
    """
    n_buckets = n_buckets or config.QUINTILES
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()

    rows = []
    for ts in pd.date_range(start, end, freq=freq):
        d0 = ts.date()
        if d0 + timedelta(days=int(horizon_months * 30.44)) > end:
            continue
        result = EN.run(spec, strategy_key, as_of=d0, persist=False)
        if result.scores.empty or len(result.scores) < n_buckets * 2:
            continue

        comp = result.scores['composite'].dropna()
        fwd = forward_returns(list(comp.index), d0, horizon_months)
        pair = pd.concat([comp, fwd], axis=1).dropna()
        if len(pair) < n_buckets * 2:
            continue
        pair.columns = ['score', 'fwd']

        try:
            pair['bucket'] = pd.qcut(pair['score'], n_buckets,
                                     labels=range(1, n_buckets + 1),
                                     duplicates='drop')
        except ValueError:
            continue

        for b, grp in pair.groupby('bucket', observed=True):
            rows.append({'date': d0, 'bucket': int(b),
                         'mean_return': float(grp['fwd'].mean()),
                         'n': len(grp)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = df.groupby('bucket').agg(
        mean_return=('mean_return', 'mean'),
        std_return=('mean_return', 'std'),
        periods=('mean_return', 'count'),
        avg_names=('n', 'mean')).reset_index()
    summary['bucket_label'] = summary['bucket'].map(
        lambda b: f'Q{b}' + (' (worst)' if b == 1 else
                             ' (best)' if b == summary['bucket'].max() else ''))
    return summary


def factor_correlations(tickers: list[str], as_of: date | str,
                        factor_names: list[str] | None = None) -> pd.DataFrame:
    """
    Correlation matrix of standardized factor scores.

    Two factors correlated at 0.9 are one factor with two names, and blending
    them just double-weights the same bet.
    """
    raw = FA.build_all(tickers, as_of)
    if raw.empty:
        return pd.DataFrame()

    sectors = raw.get('sector', pd.Series('Unknown', index=raw.index))
    names = factor_names or [c for c in raw.columns
                             if c in FA.FACTOR_DIRECTION]

    cols = {}
    for n in names:
        if n not in raw.columns or raw[n].notna().sum() < 5:
            continue
        cols[n] = T.prepare_factor(
            raw[n], sectors=sectors,
            higher_is_better=FA.FACTOR_DIRECTION.get(n, True))['sector_z']

    if len(cols) < 2:
        return pd.DataFrame()
    return pd.DataFrame(cols).corr()


def compare_runs(run_id_a: str, run_id_b: str) -> pd.DataFrame:
    """
    Rank migration between two stored runs.

    This is what the -1M / -1Y comparison rests on: because every run is
    persisted, "what changed" is a join rather than a re-computation.
    """
    from core import db

    a = db.read_sql(
        'SELECT ticker, composite, rank, signal, sector FROM scores WHERE run_id = :r',
        {'r': run_id_a}).set_index('ticker')
    b = db.read_sql(
        'SELECT ticker, composite, rank, signal FROM scores WHERE run_id = :r',
        {'r': run_id_b}).set_index('ticker')
    if a.empty or b.empty:
        return pd.DataFrame()

    joined = a.join(b, how='outer', lsuffix='_then', rsuffix='_now')
    joined['rank_change'] = joined['rank_then'] - joined['rank_now']
    joined['score_change'] = joined['composite_now'] - joined['composite_then']
    joined['signal_changed'] = joined['signal_then'] != joined['signal_now']
    joined['status'] = np.where(
        joined['rank_then'].isna(), 'entered',
        np.where(joined['rank_now'].isna(), 'dropped', 'held'))
    return joined.sort_values('rank_change', ascending=False)
