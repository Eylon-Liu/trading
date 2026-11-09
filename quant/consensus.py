"""
Multi-strategy consensus — how many *independent* views agree on a name.

The intuition: if a business screens well on quality and value, its price is
trending, insiders are buying and the latest results landed well, each of
those is separate evidence, and agreement between them should matter more
than any single score. Two things make that intuition hold up:

  1. **Count families, not strategies.** Momentum + Trend, Relative Strength
     and Quality Momentum rank names on nearly the same price history.
     Three of them agreeing is one opinion repeated, not three opinions.
     Strategies are grouped into families and a family casts one vote, when
     the average of its members' percentiles reaches `vote_pct` (top 20%).
     A member whose hard filter excludes the name (a momentum screen for a
     stock below its 200-day average) counts as a "no", not as silence.
  2. **Test it.** `agreement_study` measures, at every rebalance date, the
     forward return of names grouped by how many families agreed. If more
     agreement does not mean higher returns on your data, this strategy has
     no business in your process — the study says so either way.

An avoid list (Deteriorating Quality) flags its worst decile, and a flagged
name loses a vote. The AI earnings review votes on its own absolute scale: +1
at a view of +1 or better, −1 at −1 or worse, silent without a recent review.

Composite = votes + 0.5 × mean family percentile, so names rank by number
of agreeing families first and by how strongly they agree second. Names
scored by fewer than two families are not ranked: one view cannot agree with
itself.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_FAMILIES = 2
AI_FOR, AI_AGAINST = 1.0, -1.0
VETO_PCT = 0.90
MIN_BUCKET_NAMES = 50          # observations a vote bucket needs to be judged


def agreement_spread(by_date: pd.DataFrame) -> dict:
    """
    Per date: mean excess of names with 2+ family votes minus names with none.

    A paired comparison inside each date, so a strong or weak market moves
    both sides together and cancels. Returns {mean, wins, dates} or {}.
    """
    if by_date is None or by_date.empty:
        return {}
    hi = by_date[by_date['votes'] >= 2].groupby('date')['excess'].mean()
    lo = by_date[by_date['votes'] == 0].groupby('date')['excess'].mean()
    d = (hi - lo).dropna()
    if d.empty:
        return {}
    return {'mean': float(d.mean()), 'wins': int((d > 0).sum()), 'dates': int(len(d))}


def _member_percentiles(raw: pd.DataFrame, key: str, as_of
                        ) -> tuple[pd.Series, list[str]]:
    """
    One member strategy's composite percentile for every name in `raw`.

    Filter-excluded names get 0.0 (a "no"); names it could not score for lack
    of data stay NaN (no opinion).
    """
    from quant import engine as EN
    from quant import strategies as ST

    strat = ST.get(key)
    eligible = EN.apply_filters(raw, strat)
    out = pd.Series(0.0, index=raw.index)
    out.loc[eligible] = np.nan
    if len(eligible) == 0:
        return out, []
    scored, _detail, warnings = EN.score_universe(raw.loc[eligible], strat, as_of)
    if not scored.empty:
        pct = scored['composite'].rank(pct=True)
        out.loc[pct.dropna().index] = pct.dropna()
    return out, [f'{strat.name}: {w}' for w in warnings]


def score(raw: pd.DataFrame, strategy, as_of: date | str | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Family votes for every name in `raw` (already size-filtered).

    Returns (scores, tidy per-family detail, warnings) in the same shape the
    factor-weighted path returns, so the engine treats both alike.
    """
    from quant import strategies as ST

    warnings: list[str] = []
    fam_pct: dict[str, pd.Series] = {}
    for family, members in strategy.components.items():
        cols = []
        for key in members:
            try:
                pct, w = _member_percentiles(raw, key, as_of)
            except KeyError:
                warnings.append(f'{family}: unknown strategy {key!r}')
                continue
            warnings.extend(w)
            cols.append(pct)
        if cols:
            fam_pct[family] = pd.concat(cols, axis=1).mean(axis=1, skipna=True)

    pct_df = pd.DataFrame(fam_pct, index=raw.index)
    votes = (pct_df >= strategy.vote_pct).sum(axis=1).astype(float)
    scored_by = pct_df.notna().sum(axis=1)

    ai_cols = {}
    for family, factor in strategy.factor_votes.items():
        if factor not in raw.columns:
            continue
        v = pd.to_numeric(raw[factor], errors='coerce')
        ai_cols[family] = v
        votes += (v >= AI_FOR).astype(float)
        votes -= (v <= AI_AGAINST).astype(float)
        scored_by += v.notna().astype(int)

    vetoed = pd.Series(False, index=raw.index)
    for key in strategy.veto:
        try:
            pct, _w = _member_percentiles(raw, key, as_of)
        except KeyError:
            continue
        vetoed |= pct.fillna(0) >= VETO_PCT
    votes -= vetoed.astype(float)

    mean_pct = pct_df.mean(axis=1, skipna=True)
    composite = votes + 0.5 * mean_pct.fillna(0.0)
    composite[scored_by < MIN_FAMILIES] = np.nan

    n_fam = len(strategy.components) + len(strategy.factor_votes)
    out = pd.DataFrame({
        'composite': composite,
        'coverage': scored_by / max(n_fam, 1),
        'sector': raw['sector'] if 'sector' in raw else 'Unknown',
        'votes': votes,
        'families_scored': scored_by,
        'vetoed': vetoed,
    })
    for family, s in fam_pct.items():
        out[f'FAM::{family}'] = s
    for family, s in ai_cols.items():
        out[f'AI::{family}'] = s
    out = out.sort_values('composite', ascending=False, na_position='last')

    detail = []
    for family, s in fam_pct.items():
        d = s.dropna()
        detail.append(pd.DataFrame({
            'ticker': d.index, 'factor': f'family: {family}', 'raw': d.values,
            'winsorized': d.values, 'z': np.nan, 'sector_z': d.values,
            'pct_rank': d.values}))
    detail_df = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return out, detail_df, warnings


def signals(scores: pd.DataFrame, strategy) -> pd.DataFrame:
    """Labels and reasons read straight off the vote count."""
    if scores.empty:
        return scores
    n = len(strategy.components) + len(strategy.factor_votes)
    labels, reasons = [], []
    fam_cols = [c for c in scores.columns if c.startswith('FAM::')]
    ai_cols = [c for c in scores.columns if c.startswith('AI::')]
    for _t, r in scores.iterrows():
        v = int(r['votes']) if np.isfinite(r['votes']) else 0
        if r['vetoed'] and v <= 1:
            labels.append('🔴 Avoid — on the avoid list')
        elif v >= 3:
            labels.append(f'🟢 Strong Buy — {v} of {n} families agree')
        elif v == 2:
            labels.append(f'🟢 Buy — 2 of {n} families agree')
        elif v == 1:
            labels.append('🟡 Hold / watch — 1 family')
        else:
            labels.append('⚪ Neutral — no family agrees')
        bits = []
        for c in fam_cols:
            p = r[c]
            name = c.split('::', 1)[1]
            if not np.isfinite(p):
                bits.append(f'{name} n/a')
            else:
                mark = '✓' if p >= strategy.vote_pct else '·'
                bits.append(f'{name} {mark} {p:.0%}')
        for c in ai_cols:
            p = r[c]
            name = c.split('::', 1)[1]
            if np.isfinite(p):
                mark = '✓' if p >= AI_FOR else '✗' if p <= AI_AGAINST else '·'
                bits.append(f'{name} {mark} {p:+.1f}')
        if r['vetoed']:
            bits.append('on the avoid list (−1)')
        reasons.append(' · '.join(bits))
    out = scores.copy()
    out['signal'] = labels
    out['signal_score'] = out['votes']
    out['reasons'] = reasons
    return out


# ─────────────────────────────────────────────
# DOES AGREEMENT PREDICT RETURNS?
# ─────────────────────────────────────────────

def agreement_study(spec, start: date | str, end: date | str | None = None,
                    horizon_months: int = 3, freq: str = 'QE',
                    strategy_key: str = 'consensus',
                    progress=None) -> dict:
    """
    Forward return by number of agreeing families, over many dates.

    At each date the consensus is computed point-in-time; every ranked name's
    return over the next `horizon_months` (from that date's close, versus
    SPY over the same window) is grouped by its vote count. Returns
    {'buckets': per-vote table, 'by_date': per-date rows, 'agreement_ic':
    mean Spearman correlation between votes and forward excess return}.
    """
    import config
    from quant import engine as EN
    from quant import research as RS

    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    rows, ics = [], []
    for ts in pd.date_range(start, end, freq=freq):
        d0 = ts.date()
        if d0 + timedelta(days=int(horizon_months * 30.44)) > end:
            continue
        if progress:
            progress(f'agreement study at {d0}')
        res = EN.run(spec, strategy_key, as_of=d0, persist=False)
        if res.scores.empty or 'votes' not in res.scores:
            continue
        sc = res.scores
        names = list(sc.index)
        fwd = RS.forward_returns(names + [config.BENCHMARK_TICKER], d0, horizon_months)
        bench = fwd.get(config.BENCHMARK_TICKER, np.nan)
        if not np.isfinite(bench):
            continue
        ex = (fwd.reindex(names) - bench).dropna()
        if len(ex) < 8:
            continue
        votes = sc['votes'].reindex(ex.index)
        ics.append({'date': d0, 'ic': float(votes.corr(ex, method='spearman'))})
        for t in ex.index:
            rows.append({'date': d0, 'ticker': t, 'votes': int(votes[t]),
                         'excess': float(ex[t])})

    df = pd.DataFrame(rows)
    if df.empty:
        return {'buckets': pd.DataFrame(), 'by_date': df,
                'agreement_ic': np.nan, 'dates': 0}
    df['bucket'] = df['votes'].clip(lower=-1, upper=4)
    per_date = df.groupby(['date', 'bucket'])['excess'].mean().reset_index()
    buckets = per_date.groupby('bucket').agg(
        mean_excess=('excess', 'mean'), periods=('excess', 'count')).reset_index()
    names = df.groupby('bucket').agg(
        names=('excess', 'size'),
        hit_rate=('excess', lambda s: float((s > 0).mean())),
        median_excess=('excess', 'median')).reset_index()
    buckets = buckets.merge(names, on='bucket')
    buckets['label'] = buckets['bucket'].map(
        lambda b: 'vetoed / against' if b < 0 else
        f'{int(b)}{"+" if b == 4 else ""} famil{"y" if b == 1 else "ies"}')
    ic = pd.DataFrame(ics)['ic'].dropna() if ics else pd.Series(dtype=float)
    return {'buckets': buckets, 'by_date': df,
            'agreement_ic': float(ic.mean()) if len(ic) else np.nan,
            'agreement_ic_hit': float((ic > 0).mean()) if len(ic) else np.nan,
            'dates': int(df['date'].nunique()), 'horizon_months': horizon_months}
