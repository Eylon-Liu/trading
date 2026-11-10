"""
What the stored data can honestly support.

A backtest is a claim about the past, and it is only as good as the worst input
it depends on. This module answers one question — *from when is a given
strategy testable?* — so the answer can be "not yet" instead of a number.

Two kinds of limit apply, and they behave differently:

  *depth*         a source simply does not go back far enough. Insider filings
                  start in August 2024 here, so an insider strategy has no
                  history before that. Nothing about the result would be
                  slightly wrong — the factor is absent and its weight silently
                  redistributes, so the run measures a different strategy.

  *survivorship*  the universe is reachable but incomplete. 19% of the 2018
                  S&P 500 has no price history because those companies were
                  acquired or delisted, and the provider drops them. Every
                  strategy then picks from survivors and every return is
                  overstated — uniformly enough to look plausible, which is
                  what makes it dangerous.

The honest answer for most of this store is that long-horizon backtesting is
not available: two years of insider data and one month of news cannot support
a claim about a decade. Saying so is more useful than a number nobody should
act on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from core import db

log = logging.getLogger(__name__)


# Which stored source each factor family ultimately reads from. A factor is
# only as old as its source, whatever the price history says.
FACTOR_SOURCE: dict[str, str] = {}


def _register(source: str, *factors: str) -> None:
    for f in factors:
        FACTOR_SOURCE[f] = source


_register(
    'prices',
    'MOM_12_1', 'MOM_6_1', 'MOM_VOL_ADJ', 'RETURN_1M', 'RETURN_3M',
    'RETURN_6M', 'RETURN_1Y', 'REVERSAL_1M', 'PCT_FROM_52W_HIGH',
    'PCT_VS_MA200', 'PCT_VS_MA50', 'ABOVE_MA200', 'VOL_1Y', 'VOL_3M',
    'BETA', 'IDIO_VOL', 'MAX_DD_1Y', 'RSI_14', 'ATR_14', 'ATR_PCT',
    'MA50', 'MA200', 'PRICE',
)
_register(
    'sec_facts',
    'EARNINGS_YIELD', 'FCF_YIELD', 'SALES_YIELD', 'BOOK_TO_MARKET',
    'EBIT_TO_EV', 'ROE', 'ROA', 'ROIC', 'GROSS_PROFITABILITY', 'NET_MARGIN',
    'OPERATING_MARGIN', 'ASSET_TURNOVER', 'CURRENT_RATIO', 'PIOTROSKI_F',
    'ACCRUALS', 'DEBT_TO_EQUITY', 'NET_DEBT_TO_EQUITY', 'FCF_CONVERSION',
    'REVENUE_GROWTH_1Y', 'REVENUE_CAGR_3Y', 'EARNINGS_GROWTH_1Y',
    'EQUITY_CAGR_3Y', 'EPS_CAGR_3Y', 'BUYBACK_YIELD', 'DIVIDEND_YIELD',
    'PAYOUT_RATIO', 'MARKET_CAP',
)
_register('insider_txns', 'INSIDER_NET_BUY', 'INSIDER_CLUSTER')
_register('corporate_events', 'EVENT_INTENSITY_90D', 'EVENT_RESULTS',
          'EARNINGS_REACTION',
          'EVENT_MANAGEMENT', 'EVENT_RESTRUCTURING', 'EVENT_STRATEGY')
_register('attention', 'ATTENTION_Z')
_register('llm_reviews', 'LLM_EARNINGS_VIEW')
_register('profile_snapshots', 'FORWARD_PE', 'TRAILING_PE')

# Where each source's history lives, and how much lead time a factor needs
# before its first observation is usable. A 12-month momentum factor cannot be
# computed on the first day of price history.
SOURCE_DATE_COLUMN: dict[str, str] = {
    'prices': 'date',
    'sec_facts': 'filed',
    'insider_txns': 'filed',
    'corporate_events': 'filed',
    'attention': 'date',
    'profile_snapshots': 'snapshot_date',
    'llm_reviews': 'assessed_on',
}

# A source with less than this much history cannot support a backtest at all;
# it is a live signal that happens to be stored.
MIN_USABLE_HISTORY = timedelta(days=365)

# Above this share of an index missing from price history, results are
# survivorship-inflated enough that no reading should be trusted.
MAX_SURVIVORSHIP = 0.10


@dataclass
class Window:
    """When a strategy can be backtested, and why not earlier."""

    strategy: str
    earliest: date | None                 # None when no honest window exists
    latest: date
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def testable(self) -> bool:
        return self.earliest is not None and self.earliest < self.latest

    @property
    def years(self) -> float:
        if not self.testable:
            return 0.0
        return (self.latest - self.earliest).days / 365.25

    def explain(self) -> str:
        if not self.testable:
            return (f'{self.strategy}: not backtestable — '
                    + '; '.join(self.blockers))
        head = (f'{self.strategy}: testable from {self.earliest} '
                f'({self.years:.1f}y)')
        return head + ('  [' + '; '.join(self.notes) + ']' if self.notes else '')


def source_coverage() -> pd.DataFrame:
    """First and last observation per stored source."""
    rows = []
    for source, col in SOURCE_DATE_COLUMN.items():
        try:
            r = db.read_sql(
                f'SELECT MIN({col}) AS first, MAX({col}) AS last, '
                f'COUNT(*) AS rows FROM "{source}"').iloc[0]
        except Exception as exc:                   # noqa: BLE001
            log.debug('coverage lookup failed for %s: %s', source, exc)
            continue
        first = pd.to_datetime(r['first'], errors='coerce')
        last = pd.to_datetime(r['last'], errors='coerce')
        rows.append({
            'source': source,
            'first': first.date() if pd.notna(first) else None,
            'last': last.date() if pd.notna(last) else None,
            'rows': int(r['rows'] or 0),
            'span_days': (last - first).days if pd.notna(first) and pd.notna(last) else 0,
        })
    return pd.DataFrame(rows)


def survivorship_gap(preset: str, as_of: date) -> float:
    """Share of the index at `as_of` with no price history — 0.0 to 1.0."""
    mem = db.read_sql("""
        SELECT DISTINCT ticker FROM index_members
        WHERE UPPER(index_symbol) = UPPER(:p)
          AND as_of = (SELECT MAX(as_of) FROM index_members
                       WHERE UPPER(index_symbol) = UPPER(:p) AND as_of <= :d)
    """, {'p': preset, 'd': str(as_of)})
    if mem.empty:
        return 0.0

    from data.members import current_symbols
    tickers = current_symbols(mem['ticker'].tolist())
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params['a'] = str(as_of - timedelta(days=60))
    params['b'] = str(as_of + timedelta(days=30))
    priced = db.read_sql(
        f'SELECT DISTINCT ticker FROM prices WHERE ticker IN ({ph}) '
        f'AND date BETWEEN :a AND :b', params)
    missing = len(set(tickers) - set(priced['ticker']))
    return missing / len(tickers)


def earliest_survivorship_safe(preset: str = 'SPY',
                               threshold: float = MAX_SURVIVORSHIP) -> date | None:
    """
    The earliest year whose index is reachable enough to backtest.

    Walks back from today rather than forward: the gap only widens with age, so
    the first year that fails is the floor.
    """
    today = date.today()
    floor = None
    for years_back in range(0, 12):
        probe = date(today.year - years_back, 1, 2)
        if probe >= today:
            continue
        try:
            gap = survivorship_gap(preset, probe)
        except Exception as exc:                   # noqa: BLE001
            log.debug('survivorship probe failed at %s: %s', probe, exc)
            break
        if gap > threshold:
            break
        floor = probe
    return floor


def backtest_window(strategy, preset: str = 'SPY',
                    coverage: pd.DataFrame | None = None) -> Window:
    """
    The honest backtest window for a strategy, or none at all.

    Every weighted factor is traced to its source, and the window starts at the
    latest of those sources' first observations — plus the survivorship floor,
    which applies whatever the factors are.
    """
    from quant import strategies as ST

    strat = ST.get(strategy)
    cov = source_coverage() if coverage is None else coverage
    if strat.is_meta:
        return _meta_window(strat, preset, cov)
    by_source = cov.set_index('source') if not cov.empty else pd.DataFrame()

    today = date.today()
    starts: list[date] = []
    blockers: list[str] = []
    notes: list[str] = []

    for factor in strat.weights:
        source = FACTOR_SOURCE.get(factor)
        if source is None:
            notes.append(f'{factor} has no known source')
            continue
        if source not in by_source.index:
            blockers.append(f'{factor} needs {source}, which is empty')
            continue
        row = by_source.loc[source]
        first, last = row['first'], row['last']
        if first is None or last is None:
            blockers.append(f'{factor} needs {source}, which is empty')
            continue
        span = timedelta(days=int(row['span_days']))
        if span < MIN_USABLE_HISTORY:
            blockers.append(
                f'{factor} needs {source}, which holds only '
                f'{span.days} days from {first} — a live signal, not a history')
            continue
        starts.append(first)
        if first > today - timedelta(days=365 * 3):
            notes.append(f'{factor} only from {first}')

    floor = earliest_survivorship_safe(preset)
    if floor is None:
        blockers.append(
            f'no year of {preset} is within the {MAX_SURVIVORSHIP:.0%} '
            f'survivorship limit — delisted names are missing from prices')
    else:
        starts.append(floor)
        notes.append(f'survivorship floor {floor}')

    if blockers:
        return Window(strat.key, None, today, blockers, notes)
    if not starts:
        return Window(strat.key, None, today,
                      ['no factor in this strategy has a known source'], notes)
    return Window(strat.key, max(starts), today, [], notes)


def _family_windows(strat, preset: str, cov: pd.DataFrame
                    ) -> dict[str, dict[str, Window]]:
    return {family: {m: backtest_window(m, preset, coverage=cov) for m in members}
            for family, members in strat.components.items()}


def _meta_window(strat, preset: str, cov: pd.DataFrame) -> Window:
    """
    A consensus needs at least two families to agree, so it is testable from
    the date the second family becomes testable. Factor votes (the AI review)
    are forward-only and never part of a backtest.
    """
    today = date.today()
    firsts, notes = [], []
    for family, wins in _family_windows(strat, preset, cov).items():
        ok = [w.earliest for w in wins.values() if w.testable]
        if ok:
            firsts.append(min(ok))
            notes.append(f'{family} from {min(ok)}')
        else:
            notes.append(f'{family} not testable')
    if strat.factor_votes:
        notes.append(f'{", ".join(strat.factor_votes)}: forward-only, '
                     f'excluded from backtests')
    if len(firsts) < 2:
        return Window(strat.key, None, today,
                      ['fewer than two strategy families are backtestable'], notes)
    return Window(strat.key, sorted(firsts)[1], today, [], notes)


def meta_members_at(strategy, start: date, preset: str = 'SPY',
                    coverage: pd.DataFrame | None = None):
    """
    The consensus as it can honestly be tested from `start`: members whose
    data does not reach back that far are dropped (and families left empty
    with them) for the *whole* run, so a vote count means the same thing on
    every date of the backtest. Returns (strategy, notes).
    """
    import dataclasses

    from quant import strategies as ST

    strat = ST.get(strategy)
    cov = source_coverage() if coverage is None else coverage
    kept, notes = {}, []
    for family, wins in _family_windows(strat, preset, cov).items():
        members = tuple(m for m, w in wins.items()
                        if w.testable and w.earliest <= start)
        dropped = [m for m in wins if m not in members]
        if members:
            kept[family] = members
        if dropped:
            notes.append(f'{family}: {", ".join(dropped)} not testable from '
                         f'{start}' + ('' if members else ' — family left out'))
    if strat.factor_votes:
        notes.append(f'{", ".join(strat.factor_votes)} left out (forward-only)')
    # An avoid list that cannot be tested from `start` must not cast vetoes
    # either: degraded, it flags names on whatever factors happen to remain.
    veto = tuple(v for v in strat.veto
                 if (w := backtest_window(v, preset, coverage=cov)).testable
                 and w.earliest <= start)
    for v in strat.veto:
        if v not in veto:
            notes.append(f'veto {v} not testable from {start} — left out')
    return dataclasses.replace(strat, components=kept, factor_votes={},
                               veto=veto), notes


def report(preset: str = 'SPY') -> pd.DataFrame:
    """Backtest availability for every built-in strategy, for the UI and CLI."""
    from quant import strategies as ST

    cov = source_coverage()
    rows = []
    for key in ST.ALL_STRATEGIES:
        w = backtest_window(key, preset, coverage=cov)
        rows.append({
            'strategy': key,
            'testable': 'yes' if w.testable else 'NO',
            'earliest': str(w.earliest) if w.earliest else '—',
            'years': round(w.years, 1),
            'reason': '; '.join(w.blockers or w.notes)[:120],
        })
    return pd.DataFrame(rows).sort_values(['testable', 'years'],
                                          ascending=[True, False])
