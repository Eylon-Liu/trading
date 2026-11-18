"""
Two comparisons a screen alone cannot answer.

The Compare tab used to diff two *stored* runs. That premise was wrong: runs
exist only when someone presses the button, so a day of clicking produced
fourteen identical runs of the same strategy on the same date, and diffing any
two of them showed nothing. Worse, "how has this changed since last month?"
required having owned the app last month.

The engine is point-in-time, so neither limitation is real — a screen can be
computed *as of* any past date from stored filings. Both comparisons below are
therefore computed on demand and need no run history at all:

  drift(strategy, t0, t1)
      One strategy, two dates. What entered, what left, who moved, and whether
      the move came from price or from fundamentals. Answers "is my thesis
      still intact, and how much would I be trading?"

  crossover(strategy_a, strategy_b, date)
      Two strategies, one date. Where they agree — a name that is top-decile
      on two independent methods is a different proposition from one that
      scrapes in on a single score — and where they flatly disagree, which is
      usually the more interesting list.

Both reuse the cached factor frame for the date, so the second view is nearly
free: only the weighting and ranking differ.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.universe import UniverseSpec
from quant import engine as EN
from quant import strategies as ST

log = logging.getLogger(__name__)

# Offsets a reader actually thinks in.
LOOKBACKS = {
    '1W': 7, '1M': 30, '3M': 91, '6M': 182, '1Y': 365,
}


# ─────────────────────────────────────────────
# TIME DRIFT
# ─────────────────────────────────────────────

@dataclass
class DriftResult:
    strategy: str
    then: date
    now: date
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    entered: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)
    held: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: float = 0.0
    rank_corr: float = float('nan')
    note: str = ''

    def summary(self) -> str:
        if self.table.empty:
            return self.note or 'no overlap between the two dates'
        return (f'{len(self.entered)} in, {len(self.dropped)} out, '
                f'{self.turnover*100:.0f}% turnover, '
                f'rank correlation {self.rank_corr:+.2f}')


def drift(spec: UniverseSpec, strategy_key: str, now: date | str,
          lookback: str = '1M', top_n: int = 20) -> DriftResult:
    """
    How one strategy's ranking changed between two dates.

    Both sides are computed with the point-in-time engine, so the earlier view
    uses only what had been filed by then — it is what the screen *would* have
    said, not today's data backdated.
    """
    now = pd.to_datetime(now).date()
    days = LOOKBACKS.get(lookback, 30)
    then = now - timedelta(days=days)

    a = EN.run(spec, strategy_key, as_of=then, top_n=top_n, persist=False)
    b = EN.run(spec, strategy_key, as_of=now, top_n=top_n, persist=False)
    name = ST.get(strategy_key).name

    if a.scores.empty or b.scores.empty:
        which = 'earlier' if a.scores.empty else 'later'
        return DriftResult(name, then, now,
                           note=f'The {which} date produced no scores — the '
                                f'store may not reach back to {then}.')

    left = _rank_frame(a.scores, top_n).add_suffix('_then')
    right = _rank_frame(b.scores, top_n).add_suffix('_now')
    merged = left.join(right, how='outer')

    merged['status'] = np.where(
        merged['rank_then'].isna(), 'entered',
        np.where(merged['rank_now'].isna(), 'dropped', 'held'))
    # Positive = climbed the table (rank 1 is best).
    merged['rank_change'] = merged['rank_then'] - merged['rank_now']
    merged['score_change'] = merged['composite_now'] - merged['composite_then']

    held = merged[merged['status'] == 'held']
    entered = merged[merged['status'] == 'entered']
    dropped = merged[merged['status'] == 'dropped']

    turnover = len(entered) / max(top_n, 1)
    corr = (held['rank_then'].corr(held['rank_now'], method='spearman')
            if len(held) > 2 else float('nan'))

    return DriftResult(name, then, now, merged, entered, dropped, held,
                       turnover, corr)


def _rank_frame(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = scores.head(top_n).copy()
    if 'ticker' in df.columns:
        df = df.set_index('ticker')
    keep = [c for c in ('rank', 'composite', 'signal', 'sector') if c in df.columns]
    return df[keep]


# ─────────────────────────────────────────────
# STRATEGY CROSSOVER
# ─────────────────────────────────────────────

@dataclass
class CrossoverResult:
    name_a: str
    name_b: str
    as_of: date
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    both: pd.DataFrame = field(default_factory=pd.DataFrame)
    only_a: pd.DataFrame = field(default_factory=pd.DataFrame)
    only_b: pd.DataFrame = field(default_factory=pd.DataFrame)
    conflict: pd.DataFrame = field(default_factory=pd.DataFrame)
    rank_corr: float = float('nan')
    note: str = ''

    def summary(self) -> str:
        if self.table.empty:
            return self.note or 'nothing to compare'
        return (f'{len(self.both)} names in both lists, '
                f'{len(self.conflict)} in direct conflict, '
                f'rank correlation {self.rank_corr:+.2f}')


def crossover(spec: UniverseSpec, strategy_a: str, strategy_b: str,
              as_of: date | str, top_n: int = 20) -> CrossoverResult:
    """
    Where two strategies agree, and where they contradict each other.

    Agreement is a conviction signal: two methods built on different evidence
    reaching the same name is stronger than one method ranking it first.
    Conflict is a research prompt — a name cheap enough to top a value screen
    while sitting at the bottom of a trend screen is either an opportunity or
    a value trap, and the screens cannot tell you which.
    """
    as_of = pd.to_datetime(as_of).date()
    a = EN.run(spec, strategy_a, as_of=as_of, top_n=top_n, persist=False)
    b = EN.run(spec, strategy_b, as_of=as_of, top_n=top_n, persist=False)
    name_a, name_b = ST.get(strategy_a).name, ST.get(strategy_b).name

    if a.scores.empty or b.scores.empty:
        empty = name_a if a.scores.empty else name_b
        return CrossoverResult(name_a, name_b, as_of,
                               note=f'{empty} produced no scores for {as_of}.')

    # Full ranking on both sides, so a name can be located even when it misses
    # one of the two top-N lists.
    full_a = _full_rank(a.scores).add_suffix('_a')
    full_b = _full_rank(b.scores).add_suffix('_b')
    merged = full_a.join(full_b, how='inner')
    if merged.empty:
        return CrossoverResult(name_a, name_b, as_of,
                               note='The two runs share no names.')

    merged['in_a'] = merged['rank_a'] <= top_n
    merged['in_b'] = merged['rank_b'] <= top_n
    merged['rank_gap'] = (merged['rank_a'] - merged['rank_b']).abs()
    merged['avg_rank'] = (merged['rank_a'] + merged['rank_b']) / 2

    both = merged[merged['in_a'] & merged['in_b']].sort_values('avg_rank')
    only_a = merged[merged['in_a'] & ~merged['in_b']].sort_values('rank_a')
    only_b = merged[merged['in_b'] & ~merged['in_a']].sort_values('rank_b')

    # Conflict: top quintile on one side, bottom quintile on the other.
    n = len(merged)
    hi, lo = max(int(n * 0.2), 1), int(n * 0.8)
    conflict = merged[((merged['rank_a'] <= hi) & (merged['rank_b'] >= lo)) |
                      ((merged['rank_b'] <= hi) & (merged['rank_a'] >= lo))]
    conflict = conflict.sort_values('rank_gap', ascending=False)

    corr = merged['rank_a'].corr(merged['rank_b'], method='spearman')

    return CrossoverResult(name_a, name_b, as_of, merged, both, only_a, only_b,
                           conflict, corr)


def _full_rank(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    if 'ticker' in df.columns:
        df = df.set_index('ticker')
    keep = [c for c in ('rank', 'composite', 'signal', 'sector') if c in df.columns]
    return df[keep]


def diversification_note(corr: float) -> tuple[str, str]:
    """Plain reading of a rank correlation between two strategies."""
    if not np.isfinite(corr):
        return 'unknown', 'Too few overlapping names to judge.'
    if corr > 0.7:
        return 'redundant', (
            'These two rank the universe almost identically. Running both '
            'adds work, not diversification — the second is telling you what '
            'the first already said.')
    if corr > 0.3:
        return 'related', (
            'Meaningful overlap. Names appearing high in both have genuine '
            'cross-method support, but the two are not independent evidence.')
    if corr > -0.1:
        return 'independent', (
            'Close to unrelated. Agreement between them is real corroboration, '
            'which is what makes the shared names worth a closer look.')
    return 'opposed', (
        'These rank the universe in broadly opposite orders. That is expected '
        'for, say, value against momentum, and it means the overlap list is '
        'very short but unusually interesting.')
