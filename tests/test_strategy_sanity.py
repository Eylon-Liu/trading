"""
Static checks on strategy definitions, factor plumbing and the codebase.

Each test here pins a mistake that was actually made once: a filter key the
engine never read, a momentum strategy rewarding the one-month reversal
window, an avoid list given long trade plans, a factor that was the same for
every company, and functions nobody called.
"""

from __future__ import annotations

import ast
import pathlib
import re
from datetime import date

import pandas as pd
import pytest

from quant import availability as AV
from quant import engine as EN
from quant import factor_docs as FD
from quant import factors as FA
from quant import fundamentals as F
from quant import strategies as ST

ROOT = pathlib.Path(__file__).resolve().parent.parent
STRATS = sorted(ST.ALL_STRATEGIES.items())


# ─────────────────────────────────────────────
# STRATEGY DEFINITIONS
# ─────────────────────────────────────────────

@pytest.mark.parametrize('key,strat', STRATS)
def test_filters_are_all_understood_by_the_engine(key, strat):
    """An unknown filter key is silently ignored — the gate never applies."""
    unknown = set(strat.filters) - EN.FILTER_KEYS
    assert not unknown, f'{key} uses filters the engine ignores: {unknown}'


@pytest.mark.parametrize('key,strat', STRATS)
def test_every_weighted_factor_is_fully_plumbed(key, strat):
    for f in strat.weights:
        assert f in FA.FACTOR_DIRECTION, f'{key}: {f} has no direction'
        assert f in FD.FACTOR_DOCS, f'{key}: {f} is undocumented'
        assert f in AV.FACTOR_SOURCE, f'{key}: {f} has no source for availability'
    assert all(w > 0 for w in strat.weights.values()), \
        f'{key}: use FACTOR_DIRECTION for sign, not negative weights'


@pytest.mark.parametrize('key,strat', STRATS)
def test_avoid_lists_get_no_long_trade_plan(key, strat):
    if strat.invert:
        assert not strat.needs_trade_plan, f'{key} would buy what it flags'


def test_one_month_return_is_not_used_as_positive_momentum():
    """
    The last month is the short-term reversal window. Rewarding it in a
    strategy that holds for a quarter or more bets against its own premise.
    """
    for key, strat in ST.ALL_STRATEGIES.items():
        if strat.horizon == 'long' and not strat.invert:
            assert 'RETURN_1M' not in strat.weights, key
    assert ST.DEFAULT_STYLE_PROXIES['momentum'] != 'RETURN_1M'


# ─────────────────────────────────────────────
# AS-OF-DEPENDENT FUNDAMENTALS
# ─────────────────────────────────────────────

def test_valid_until_marks_share_count_expiry_and_split_window():
    facts = pd.DataFrame([{
        'ticker': 'X', 'concept': 'shares_outstanding', 'tag': 'S', 'unit': 'shares',
        'period_start': pd.Timestamp('2023-12-31'), 'period_end': pd.Timestamp('2023-12-31'),
        'filed': pd.Timestamp('2024-02-01'), 'form': '10-K', 'fy': 2023, 'fp': 'FY',
        'val': 1e8}])
    as_of = pd.Timestamp('2024-03-01')
    expiry = pd.Timestamp('2024-02-01') + pd.Timedelta(days=F.MAX_SHARE_COUNT_AGE_DAYS + 1)
    assert F._valid_until(facts, {}, 'X', as_of, None) == expiry

    split = pd.DataFrame({'ticker': ['X'], 'date': [pd.Timestamp('2024-05-01')],
                          'ratio': [2.0]})
    assert F._valid_until(facts, {}, 'X', as_of, split) == pd.Timestamp('2024-05-01')


def test_stale_provider_snapshot_is_not_used_for_market_cap(monkeypatch):
    """A market cap from months ago is not the market cap on the as-of date."""
    fund = pd.DataFrame({c: [1.0] for c in (
        'shares_diluted', 'debt', 'cash', 'net_income_ttm', 'fcf_ttm',
        'revenue_ttm', 'equity', 'operating_income_ttm', 'roe', 'roic',
        'gross_profitability', 'net_margin', 'operating_margin', 'accruals',
        'debt_to_equity', 'net_debt_to_equity', 'fcf_conversion',
        'current_ratio', 'roa', 'asset_turnover', 'revenue_growth_1y',
        'revenue_cagr_3y', 'earnings_growth_1y', 'equity_cagr_3y',
        'eps_cagr_3y', 'buybacks_ttm', 'dividends_paid_ttm', 'piotroski_f',
        'earnings_surprise')}, index=['X'])
    fund['shares_diluted'] = 1e6
    fund['last_filed'] = pd.Timestamp('2024-01-01')
    fund['last_quarter_filed'] = pd.Timestamp('2024-01-01')
    as_of = date(2024, 6, 30)

    def profile(snapshot):
        return pd.DataFrame({'snapshot_date': [pd.Timestamp(snapshot)],
                             'market_cap': [5e12], 'forward_pe': [3.0]}, index=['X'])

    monkeypatch.setattr(FA.F, 'build_fundamentals', lambda *a, **k: fund)
    monkeypatch.setattr(FA.yahoo, 'latest_prices',
                        lambda *a, **k: pd.Series({'X': 50.0}))

    monkeypatch.setattr(FA.yahoo, 'profile_asof', lambda *a: profile('2024-01-15'))
    stale = FA.fundamental_factors(['X'], as_of, _facts_all=pd.DataFrame({'x': [1]}))
    assert stale.loc['X', 'MCAP_SOURCE'] == 'computed'
    assert stale.loc['X', 'MARKET_CAP'] == pytest.approx(5e7)

    monkeypatch.setattr(FA.yahoo, 'profile_asof', lambda *a: profile('2024-06-28'))
    fresh = FA.fundamental_factors(['X'], as_of, _facts_all=pd.DataFrame({'x': [1]}))
    assert fresh.loc['X', 'MCAP_SOURCE'] == 'provider'


# ─────────────────────────────────────────────
# DEAD CODE
# ─────────────────────────────────────────────

# Dash callbacks and Flask routes are called by the framework, by decorator.
_FRAMEWORK_DIRS = {'ui', 'tests'}
_FRAMEWORK_FILES = {'app.py'}


def _module_defs():
    for path in sorted(ROOT.rglob('*.py')):
        rel = path.relative_to(ROOT)
        if (rel.parts[0] in _FRAMEWORK_DIRS or str(rel) in _FRAMEWORK_FILES
                or '.git' in rel.parts):
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                yield rel, node.name
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        yield rel, t.id


def test_no_unreferenced_top_level_definitions():
    """
    A function, class or constant referenced nowhere outside its own
    definition — not even by a test — is dead. It still has to be read,
    maintained and kept consistent with everything around it.
    """
    sources = {p: p.read_text() for p in ROOT.rglob('*.py') if '.git' not in p.parts}
    corpus = '\n'.join(sources.values())
    dead = []
    for rel, name in _module_defs():
        if name.startswith('__') or name in ('log', 'main'):
            continue
        if len(re.findall(rf'\b{re.escape(name)}\b', corpus)) <= 1:
            dead.append(f'{rel}: {name}')
    assert not dead, 'unreferenced definitions:\n  ' + '\n  '.join(dead)
