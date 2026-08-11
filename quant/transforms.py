"""
Cross-sectional transforms — winsorize, standardize, neutralize.

Four defects in the original scoring path are fixed here:

  * No winsorization. A single P/B of 0.001 becomes a book-to-market of
    100,000%, which drags that sector's mean *and* inflates its standard
    deviation, corrupting the z-score of every other name in the sector. One
    bad print poisoned a whole bucket.
  * Sector z-scores computed on as few as two names. With 50 stocks spread
    across 11 sectors, most buckets held 2-3 names and their "z-scores" were
    noise. We now require a minimum and fall back to regression-based
    neutralization below it.
  * Composite scores averaged whatever factors happened to be present, so a
    stock with one valid factor was ranked against one with three, its single
    z-score treated as a full composite.
  * Row-wise `df.apply` for the composite — an O(n) Python loop where a
    vectorized reduction does.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# OUTLIER HANDLING
# ─────────────────────────────────────────────

def winsorize(s: pd.Series, pct: float | None = None) -> pd.Series:
    """Clip to the [pct, 1-pct] quantiles. Values are clipped, never dropped."""
    pct = config.WINSORIZE_PCT if pct is None else pct
    clean = s.dropna()
    if len(clean) < 3:
        return s
    lo, hi = clean.quantile(pct), clean.quantile(1 - pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return s
    return s.clip(lower=lo, upper=hi)


def winsorize_mad(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """
    Median-absolute-deviation clipping.

    More robust than quantile clipping when the tail is a handful of extreme
    prints rather than a smooth distribution — which is the usual shape of a
    bad fundamental datapoint.
    """
    clean = s.dropna()
    if len(clean) < 3:
        return s
    med = clean.median()
    mad = (clean - med).abs().median()
    if mad == 0 or not np.isfinite(mad):
        return winsorize(s)
    scaled = mad * config.MAD_SCALE
    return s.clip(lower=med - n_mad * scaled, upper=med + n_mad * scaled)


# ─────────────────────────────────────────────
# STANDARDIZATION
# ─────────────────────────────────────────────

def zscore(s: pd.Series, robust: bool = False) -> pd.Series:
    """Standardize to mean 0 / sd 1. Returns zeros when the series is constant."""
    clean = s.dropna()
    if len(clean) < 2:
        return pd.Series(np.nan, index=s.index, dtype=float)

    if robust:
        center = clean.median()
        spread = (clean - center).abs().median() * config.MAD_SCALE
    else:
        center = clean.mean()
        spread = clean.std(ddof=1)

    if spread == 0 or not np.isfinite(spread):
        return pd.Series(0.0, index=s.index).where(s.notna())
    return (s - center) / spread


def pct_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1]."""
    return s.rank(pct=True, na_option='keep')


def rank_to_normal(s: pd.Series) -> pd.Series:
    """
    Rank-based inverse-normal transform.

    Kills outlier influence entirely by using only ordering, which is what you
    want for factors whose raw scale is meaningless (accruals, sentiment).
    """
    from scipy.stats import norm
    r = s.rank(na_option='keep')
    n = r.notna().sum()
    if n < 2:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return pd.Series(norm.ppf((r - 0.5) / n), index=s.index)


# ─────────────────────────────────────────────
# NEUTRALIZATION
# ─────────────────────────────────────────────

def sector_zscore(values: pd.Series, sectors: pd.Series,
                  min_members: int | None = None,
                  robust: bool = False) -> pd.Series:
    """
    Standardize within sector, falling back to the whole cross-section when a
    sector is too thin to estimate a mean and spread from.

    The original guard was `len(vals) < 2`, which happily standardized a
    two-name sector and produced pure noise.
    """
    min_members = config.MIN_SECTOR_MEMBERS if min_members is None else min_members
    out = pd.Series(np.nan, index=values.index, dtype=float)
    sectors = sectors.reindex(values.index).fillna('Unknown')

    thin: list[str] = []
    for sector, idx in sectors.groupby(sectors).groups.items():
        idx = pd.Index(idx)
        sub = values.loc[idx]
        if sub.notna().sum() < min_members:
            thin.append(str(sector))
            continue
        out.loc[idx] = zscore(sub, robust=robust)

    if thin:
        # Thin sectors are scored against the full cross-section instead of
        # against two peers. Noisy, but honestly noisy.
        mask = sectors.isin(thin)
        out.loc[mask] = zscore(values, robust=robust).loc[mask]
        log.debug('sector_zscore: %d thin sectors scored cross-sectionally: %s',
                  len(thin), thin[:6])

    return out


def neutralize(values: pd.Series, sectors: pd.Series | None = None,
               size: pd.Series | None = None) -> pd.Series:
    """
    Residualize a factor on sector dummies and log market cap.

    This is what a real risk model does: rather than bucketing, regress the
    factor on its known exposures and keep the residual, so the score reflects
    what is specific to the name rather than to its sector or its size.
    """
    df = pd.DataFrame({'y': values})
    if sectors is not None:
        df['sector'] = sectors.reindex(values.index).fillna('Unknown')
    if size is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            df['logsize'] = np.log(size.reindex(values.index).replace(0, np.nan))

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['y'])
    if len(df) < 10:
        return zscore(values)

    designs = [pd.Series(1.0, index=df.index, name='const')]
    if 'sector' in df and df['sector'].nunique() > 1:
        designs.append(pd.get_dummies(df['sector'], prefix='s',
                                      drop_first=True, dtype=float))
    if 'logsize' in df and df['logsize'].notna().sum() > len(df) * 0.5:
        designs.append(df['logsize'].fillna(df['logsize'].median()))

    X = pd.concat(designs, axis=1).astype(float)
    y = df['y'].astype(float)

    try:
        beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        resid = y.values - X.values @ beta
    except np.linalg.LinAlgError:
        return zscore(values)

    out = pd.Series(np.nan, index=values.index, dtype=float)
    out.loc[df.index] = resid
    return zscore(out)


# ─────────────────────────────────────────────
# FACTOR PIPELINE
# ─────────────────────────────────────────────

def prepare_factor(values: pd.Series, sectors: pd.Series | None = None,
                   size: pd.Series | None = None, higher_is_better: bool = True,
                   method: str = 'sector_z', robust_outliers: bool = True
                   ) -> pd.DataFrame:
    """
    Raw factor -> comparable score, with every intermediate retained.

    Keeping raw/winsorized/z/sector_z/pct_rank means the UI can explain a
    ranking instead of just asserting one.
    """
    values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    wins = winsorize_mad(values) if robust_outliers else winsorize(values)

    if method == 'sector_z' and sectors is not None:
        score = sector_zscore(wins, sectors)
    elif method == 'neutralize':
        score = neutralize(wins, sectors, size)
    elif method == 'rank':
        score = rank_to_normal(wins)
    else:
        score = zscore(wins)

    if not higher_is_better:
        score = -score
        wins = -wins

    return pd.DataFrame({
        'raw': values,
        'winsorized': wins,
        'z': zscore(values) * (1 if higher_is_better else -1),
        'sector_z': score,
        'pct_rank': pct_rank(score),
    })


def composite_score(factor_scores: pd.DataFrame,
                    weights: dict[str, float] | None = None,
                    min_coverage: float | None = None
                    ) -> tuple[pd.Series, pd.Series]:
    """
    Weighted blend of factor scores, with an explicit coverage requirement.

    Weights are renormalized over the factors a name actually has, so a stock
    missing one input is not penalised on scale — but a stock missing most of
    them is excluded rather than ranked on a single number. Returns
    (composite, coverage).

    Vectorized: the original did this with a row-wise `apply`.
    """
    min_coverage = config.MIN_FACTOR_COVERAGE if min_coverage is None else min_coverage
    if factor_scores.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    cols = list(factor_scores.columns)
    w = pd.Series({c: (weights or {}).get(c, 1.0) for c in cols}, dtype=float)
    if w.sum() == 0:
        w = pd.Series(1.0, index=cols)
    w = w / w.sum()

    present = factor_scores.notna()
    coverage = (present * w).sum(axis=1)              # weighted coverage in [0,1]

    weighted = (factor_scores.fillna(0) * w).sum(axis=1)
    composite = weighted.divide(coverage.replace(0, np.nan))
    composite[coverage < min_coverage] = np.nan

    return composite, coverage
