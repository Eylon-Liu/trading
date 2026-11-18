"""Forward returns for signal evaluation, and rank migration between stored runs."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data import yahoo


def forward_returns(tickers: list[str], as_of: date, horizon_months: int
                    ) -> pd.Series:
    """
    Total return over the next `horizon_months`, for IC work.

    Measured from the last close at or before `as_of` — the moment the score
    was computed — to the last close inside the horizon.

    The start used to be the first bar of a window opened seven days *before*
    `as_of`, so up to a week of returns that had already happened when the
    score was computed counted as "forward". Every momentum and trend factor
    is built from exactly those returns, which inflated their IC and made
    the decay curve start from a number no strategy could have earned.
    """
    as_of = pd.to_datetime(as_of).date()
    end = as_of + timedelta(days=int(horizon_months * 30.44))
    px = yahoo.price_history(tickers, start=as_of - timedelta(days=10),
                             end=end, field='adj_close')
    if px.empty:
        return pd.Series(dtype=float)
    px = px.ffill()
    before = px.loc[:pd.Timestamp(as_of)]
    after = px.loc[pd.Timestamp(as_of) + pd.Timedelta(days=1):]
    if before.empty or after.empty:
        return pd.Series(dtype=float)
    return (after.iloc[-1] / before.iloc[-1] - 1.0).replace(
        [np.inf, -np.inf], np.nan)


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
