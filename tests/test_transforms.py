"""Cross-sectional transform tests — the defects that broke the original scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import transforms as T


def test_zscore_matches_hand_computation():
    z = T.zscore(pd.Series([1, 2, 3, 4, 5]))
    expected = [-1.264911, -0.632456, 0.0, 0.632456, 1.264911]
    np.testing.assert_allclose(z.values, expected, rtol=1e-5)


def test_zscore_constant_series_is_zero_not_nan():
    z = T.zscore(pd.Series([7.0, 7.0, 7.0, 7.0]))
    assert (z == 0).all()


def test_zscore_too_few_points_returns_nan():
    assert T.zscore(pd.Series([1.0])).isna().all()


def test_winsorize_clips_rather_than_drops():
    s = pd.Series([1, 2, 3, 4, 1000])
    w = T.winsorize(s, pct=0.2)
    assert len(w) == len(s)
    assert w.max() < 1000


def test_outlier_no_longer_compresses_the_rest():
    """
    A single extreme print used to flatten every other name.

    With a P/B of 0.00001 the derived book-to-market is 100,000%, which drags
    the mean and inflates the standard deviation so much that the four healthy
    names land on essentially the same z-score.
    """
    s = pd.Series({'a': 1.0, 'b': 1.1, 'c': 0.9, 'd': 1.05, 'bad': 100_000.0})

    naive = (s - s.mean()) / s.std()
    good_spread_naive = naive[['a', 'b', 'c', 'd']].std()

    fixed = T.zscore(T.winsorize_mad(s))
    good_spread_fixed = fixed[['a', 'b', 'c', 'd']].std()

    assert good_spread_naive < 0.01, 'precondition: naive z-scores collapse'
    # The absolute level is not the point; recovering usable dispersion is.
    assert good_spread_fixed > good_spread_naive * 20, \
        'winsorizing must restore real dispersion among the healthy names'
    # b=1.10 > d=1.05 > a=1.00 > c=0.90
    assert fixed['b'] > fixed['d'] > fixed['a'] > fixed['c'], \
        'and must preserve the true ordering'


def test_sector_zscore_falls_back_when_sector_too_thin():
    """A two-name sector cannot support a mean and a standard deviation."""
    values = pd.Series(np.arange(20, dtype=float), index=[f'T{i}' for i in range(20)])
    sectors = pd.Series(['Tech'] * 15 + ['Energy'] * 2 + ['Utilities'] * 3,
                        index=values.index)

    z = T.sector_zscore(values, sectors, min_members=5)

    assert z.notna().all(), 'thin sectors must still be scored, not dropped'
    tech = z[sectors == 'Tech']
    assert abs(tech.mean()) < 1e-9, 'a full sector standardizes to mean zero'


def test_sector_zscore_uses_within_sector_stats_when_populated():
    values = pd.Series([1, 2, 3, 4, 5, 101, 102, 103, 104, 105], dtype=float,
                       index=list('abcdefghij'))
    sectors = pd.Series(['A'] * 5 + ['B'] * 5, index=values.index)
    z = T.sector_zscore(values, sectors, min_members=5)
    # Both sectors standardize independently, so their scores mirror.
    np.testing.assert_allclose(z[:5].values, z[5:].values, atol=1e-9)


def test_composite_excludes_names_below_coverage_floor():
    """One factor out of three is not a composite."""
    scores = pd.DataFrame(
        {'value': [1.5, np.nan, 0.8], 'quality': [1.2, np.nan, 1.0],
         'momentum': [0.9, 2.0, np.nan]},
        index=['full', 'one_only', 'two_of_three'])

    composite, coverage = T.composite_score(scores, min_coverage=0.5)

    assert np.isnan(composite['one_only']), \
        'a single high factor must not produce a rank'
    assert coverage['one_only'] == pytest.approx(1 / 3)
    assert not np.isnan(composite['two_of_three'])
    assert composite['full'] == pytest.approx(1.2)


def test_composite_renormalizes_over_available_factors():
    """A missing factor should not dilute the score toward zero."""
    scores = pd.DataFrame({'a': [2.0, 2.0], 'b': [2.0, np.nan]},
                          index=['both', 'one'])
    composite, _cov = T.composite_score(scores, min_coverage=0.4)
    assert composite['both'] == pytest.approx(2.0)
    assert composite['one'] == pytest.approx(2.0)


def test_composite_respects_weights():
    scores = pd.DataFrame({'a': [1.0], 'b': [0.0]}, index=['x'])
    composite, _ = T.composite_score(scores, weights={'a': 3.0, 'b': 1.0})
    assert composite['x'] == pytest.approx(0.75)


def test_prepare_factor_inverts_when_lower_is_better():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list('abcde'))
    sectors = pd.Series(['S'] * 5, index=values.index)

    higher = T.prepare_factor(values, sectors, higher_is_better=True)
    lower = T.prepare_factor(values, sectors, higher_is_better=False)

    np.testing.assert_allclose(higher['sector_z'].values,
                               -lower['sector_z'].values, atol=1e-9)


def test_neutralize_removes_sector_effect():
    """Residualizing on sector should strip a pure sector offset."""
    idx = [f'T{i}' for i in range(20)]
    sectors = pd.Series(['A'] * 10 + ['B'] * 10, index=idx)
    # Same within-sector shape, large between-sector offset.
    values = pd.Series(list(range(10)) + [x + 500 for x in range(10)],
                       index=idx, dtype=float)

    resid = T.neutralize(values, sectors=sectors)
    assert abs(resid[:10].mean() - resid[10:].mean()) < 0.2
