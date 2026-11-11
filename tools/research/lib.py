"""Shared helpers for the 3–6 month strategy research (reads panel.pkl).

See docs/strategy-research.md. Build the panel first:

    python tools/research/build_panel.py 2020-01-31 2026-10-31 tools/research/panel.pkl
    python tools/research/evaluate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from quant import engine as EN
from quant import factors as FA
from quant import strategies as ST

PANEL = str(Path(__file__).resolve().parent / 'panel.pkl')

# In-sample: formation dates whose 6-month outcome ends by 2023-12.
# Out-of-sample: formation dates from 2024-01. Nothing in between is used
# to choose anything, so the two never share a return window.
IS_END = pd.Timestamp('2023-06-30')
OOS_START = pd.Timestamp('2024-01-01')


def load():
    df = pd.read_pickle(PANEL)
    df['date'] = pd.to_datetime(df['date'])
    for h in (1, 3, 6):
        df[f'ex_{h}m'] = df[f'fwd_{h}m'] - df[f'spy_{h}m']                 # vs SPY
        ew = df.groupby('date')[f'fwd_{h}m'].transform('mean')
        df[f'exew_{h}m'] = df[f'fwd_{h}m'] - ew                           # vs equal-weight universe
    return df


def split(df):
    return df[df['date'] <= IS_END], df[df['date'] >= OOS_START]


def factor_ic(df, factor, target='exew_3m', sector_neutral=True):
    """Mean Spearman rank IC per date, direction-adjusted (positive = works)."""
    sign = 1.0 if FA.FACTOR_DIRECTION.get(factor, True) else -1.0
    out = []
    for d, g in df.groupby('date'):
        x, y = g[factor], g[target]
        ok = x.notna() & y.notna()
        if ok.sum() < 50:
            continue
        x, y = x[ok] * sign, y[ok]
        if sector_neutral:
            x = x - x.groupby(g.loc[ok, 'sector']).transform('mean')
            y = y - y.groupby(g.loc[ok, 'sector']).transform('mean')
        out.append((d, x.rank().corr(y.rank())))
    s = pd.Series(dict(out))
    return s


def ic_summary(s, horizon_months):
    """Mean IC, share positive, and a t-stat deflated for overlapping windows."""
    if len(s) < 3:
        return dict(ic=np.nan, pos=np.nan, t=np.nan, n=len(s))
    n_eff = max(len(s) / horizon_months, 1.0)          # monthly samples of an h-month return
    t = s.mean() / (s.std(ddof=1) / np.sqrt(n_eff)) if s.std(ddof=1) > 0 else np.nan
    return dict(ic=s.mean(), pos=(s > 0).mean(), t=t, n=len(s))


def score(g, strategy):
    """Rank one date's cross-section with the live scoring path. → DataFrame(composite, rank)."""
    raw = g.set_index('ticker')
    strat = ST.get(strategy)
    scores, _detail, _w = EN.score_universe(raw, strat, g['date'].iloc[0].date())
    if scores.empty:
        return scores
    keep = EN.apply_filters(raw, strat)
    scores = scores[scores.index.isin(keep)]
    min_cov = 0.5
    if 'coverage' in scores:
        scores = scores[scores['coverage'] >= min_cov]
    return scores.dropna(subset=['composite']).sort_values('composite', ascending=False)


def picks_by_date(df, strategy, n=20):
    """{date: [tickers]} — the top n names of the strategy at each date."""
    out = {}
    for d, g in df.groupby('date'):
        sc = score(g, strategy)
        out[d] = list(sc.index[:n])
    return out


def portfolio_stats(df, picks, horizons=(3, 6), weight=None):
    """
    Forward return of holding each date's picks for h months, vs SPY and vs
    the equal-weight universe. Monthly cohorts overlap, so the per-date mean
    is a fair estimate of the average outcome but the hit rate and t-stat
    count overlapping windows (deflated t uses n/h effective samples).
    """
    rows = []
    idx = df.set_index(['date', 'ticker'])
    for d, names in picks.items():
        if not names:
            continue
        sub = idx.loc[d].reindex(names)
        row = {'date': d, 'n': len(names)}
        for h in horizons:
            r = sub[f'fwd_{h}m']
            if r.notna().sum() < max(3, len(names) // 2):
                continue
            if weight is not None:
                w = weight(sub).reindex(r.index).where(r.notna())
                port = float((r * w).sum() / w.sum())
            else:
                port = float(r.mean())
            row[f'ret_{h}m'] = port
            row[f'spy_{h}m'] = float(sub[f'spy_{h}m'].iloc[0])
            row[f'ew_{h}m'] = float(df.loc[df['date'] == d, f'fwd_{h}m'].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')


def summarize(stats, h):
    """Annualised-ish summary for horizon h from overlapping monthly cohorts."""
    if f'ret_{h}m' not in stats:
        return {}
    s = stats.dropna(subset=[f'ret_{h}m'])
    if s.empty:
        return {}
    ex = s[f'ret_{h}m'] - s[f'spy_{h}m']
    exew = s[f'ret_{h}m'] - s[f'ew_{h}m']
    n_eff = max(len(s) / h, 1.0)
    t = ex.mean() / (ex.std(ddof=1) / np.sqrt(n_eff)) if len(s) > 2 and ex.std(ddof=1) > 0 else np.nan
    ann = 12.0 / h
    return {
        'n': len(s),
        'ret_ann': s[f'ret_{h}m'].mean() * ann,
        'spy_ann': s[f'spy_{h}m'].mean() * ann,
        'ex_spy_ann': ex.mean() * ann,
        'ex_ew_ann': exew.mean() * ann,
        'hit_spy': (ex > 0).mean(),
        't_spy': t,
        'worst': s[f'ret_{h}m'].min(),
    }


def table(results, h, label=''):
    rows = []
    for name, stats in results.items():
        r = summarize(stats, h)
        if r:
            rows.append({'strategy': name, **r})
    out = pd.DataFrame(rows).set_index('strategy')
    fmt = out.copy()
    for c in ('ret_ann', 'spy_ann', 'ex_spy_ann', 'ex_ew_ann', 'hit_spy', 'worst'):
        fmt[c] = (out[c] * 100).round(1)
    fmt['t_spy'] = out['t_spy'].round(2)
    print(f'\n== {label} · hold {h} months (returns annualised, %)')
    print(fmt.to_string())
    return out
