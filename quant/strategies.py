"""
Strategy presets, split by holding horizon.

These are two genuinely different jobs, and collapsing them into one ranked
list — which the original app did — produces a screen that serves neither.

LONG-TERM (a year and up) is a *screen to accumulate*. It asks which
businesses are worth owning, so it leans on fundamentals that only mean
anything over years: returns on capital, balance-sheet strength, valuation,
earnings durability. There is no exit level, because the exit is a broken
thesis, not a price. Turnover should be low and rebalancing quarterly.

MID-TERM (days to months) is a *trade*. Fundamentals act as a quality filter
rather than the thesis; what drives it is trend, catalyst, and positioning.
Every candidate gets a full plan — entry, stop, target, time stop, size — from
quant/tradeplan.py, because a position with no exit rule is not a trade.

A strategy is a declarative dict, so adding one is configuration rather than
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ─────────────────────────────────────────────
# HORIZONS
# ─────────────────────────────────────────────

HORIZONS = {
    'long': {
        'label': 'Long-term (1 year+)',
        'blurb': 'Screen for businesses worth owning and accumulating.',
        'rebalance': 'Q',
        'needs_trade_plan': False,
        'default_holdings': 25,
    },
    'mid': {
        'label': 'Mid-term (1-6 months)',
        'blurb': 'Time-bound trades with a defined entry, stop and target.',
        'rebalance': 'M',
        'needs_trade_plan': True,
        'default_holdings': 10,
    },
}


@dataclass
class Strategy:
    """A declarative strategy definition."""

    key: str
    name: str
    horizon: str                                  # 'long' | 'mid'
    description: str
    weights: dict[str, float]                     # factor -> weight
    filters: dict[str, float] = field(default_factory=dict)
    neutralize: str = 'sector_z'                  # sector_z | neutralize | rank | z
    setup: str = 'momentum'                       # informs the trade plan
    thesis: str = ''

    @property
    def factors(self) -> list[str]:
        return list(self.weights)

    @property
    def needs_trade_plan(self) -> bool:
        return HORIZONS[self.horizon]['needs_trade_plan']


# ─────────────────────────────────────────────
# LONG-TERM — screen to own
# ─────────────────────────────────────────────

LONG_TERM = {
    'quality_value': Strategy(
        key='quality_value', name='Quality at a Reasonable Price', horizon='long',
        description='Profitable, well-capitalised businesses trading on modest yields.',
        thesis='Own durable earnings power bought below its worth; exit if returns '
               'on capital decay or leverage rises structurally.',
        weights={
            'EARNINGS_YIELD': 1.0, 'FCF_YIELD': 1.2, 'EBIT_TO_EV': 0.8,
            'ROIC': 1.2, 'GROSS_PROFITABILITY': 1.0, 'ROE': 0.8,
            'ACCRUALS': 0.6, 'DEBT_TO_EQUITY': 0.5,
        },
        filters={'min_market_cap': 2e9},
    ),
    'deep_value': Strategy(
        key='deep_value', name='Deep Value', horizon='long',
        description='Statistically cheap on assets and cash flow, screened for solvency.',
        thesis='Buy the discount to book and cash generation; exit on re-rating '
               'or if the balance sheet deteriorates.',
        weights={
            'BOOK_TO_MARKET': 1.5, 'EARNINGS_YIELD': 1.0, 'FCF_YIELD': 1.0,
            'SALES_YIELD': 0.5, 'PIOTROSKI_F': 1.0, 'DEBT_TO_EQUITY': 0.8,
            'CURRENT_RATIO': 0.5,
        },
    ),
    'piotroski': Strategy(
        key='piotroski', name='Piotroski F-Score', horizon='long',
        description='Nine fundamental health tests, all from filed statements.',
        thesis='Improving fundamentals in cheap names; exit when the score rolls over.',
        weights={
            'PIOTROSKI_F': 2.0, 'BOOK_TO_MARKET': 1.0,
            'ROA': 0.5, 'FCF_YIELD': 0.8, 'ACCRUALS': 0.5,
        },
    ),
    'compounder': Strategy(
        key='compounder', name='Quality Compounder', horizon='long',
        description='High and stable returns on capital with reinvestment runway.',
        thesis='Let capital compound internally; exit if ROIC or growth structurally breaks.',
        weights={
            'ROIC': 1.5, 'GROSS_PROFITABILITY': 1.2, 'OPERATING_MARGIN': 0.8,
            'REVENUE_CAGR_3Y': 1.0, 'EARNINGS_GROWTH_1Y': 0.6,
            'ACCRUALS': 0.5, 'DEBT_TO_EQUITY': 0.5,
        },
        filters={'min_market_cap': 5e9},
    ),
    'dividend_quality': Strategy(
        key='dividend_quality', name='Dividend Quality', horizon='long',
        description='Sustainable income: yield backed by cash flow, not by leverage.',
        thesis='Own the cash return; exit on payout stress or a cut.',
        weights={
            'DIVIDEND_YIELD': 1.2, 'FCF_YIELD': 1.2, 'PAYOUT_RATIO': 0.8,
            'ROE': 0.6, 'DEBT_TO_EQUITY': 0.8, 'NET_MARGIN': 0.5,
            'BUYBACK_YIELD': 0.4,
        },
    ),
    'low_volatility': Strategy(
        key='low_volatility', name='Low Volatility Defensive', horizon='long',
        description='The low-risk anomaly: low beta and low idiosyncratic vol, quality-screened.',
        thesis='Compound through drawdowns; exit if volatility regime-shifts higher.',
        weights={
            'VOL_1Y': 1.5, 'BETA': 1.0, 'IDIO_VOL': 0.8, 'MAX_DD_1Y': 0.8,
            'ROE': 0.6, 'DEBT_TO_EQUITY': 0.6,
        },
        neutralize='neutralize',
    ),
    'garp': Strategy(
        key='garp', name='Growth at a Reasonable Price', horizon='long',
        description='Growth that has not yet been fully paid for.',
        thesis='Own compounding growth bought at a sane multiple; exit if growth '
               'decelerates while the multiple stays rich.',
        weights={
            'REVENUE_CAGR_3Y': 1.2, 'EARNINGS_GROWTH_1Y': 1.0,
            'EARNINGS_YIELD': 1.0, 'ROIC': 0.8, 'GROSS_PROFITABILITY': 0.6,
            'DEBT_TO_EQUITY': 0.4,
        },
    ),
    'multifactor': Strategy(
        key='multifactor', name='Multi-Factor Composite', horizon='long',
        description='Balanced value, quality, growth and momentum blend.',
        thesis='Diversify across factor premia rather than betting on one.',
        weights={
            'EARNINGS_YIELD': 1.0, 'BOOK_TO_MARKET': 0.6, 'FCF_YIELD': 0.8,
            'ROIC': 1.0, 'GROSS_PROFITABILITY': 0.8,
            'REVENUE_CAGR_3Y': 0.8, 'MOM_12_1': 1.0, 'ACCRUALS': 0.4,
        },
    ),
    'quality_momentum': Strategy(
        key='quality_momentum', name='Quality Momentum', horizon='long',
        description='Profitable businesses whose prices are already trending up.',
        thesis='Quality and momentum are historically uncorrelated, so combining '
               'them diversifies the drawdowns of each; exit when profitability '
               'deteriorates or the long-term trend breaks.',
        weights={
            'GROSS_PROFITABILITY': 1.2, 'ROIC': 1.0, 'ACCRUALS': 0.6,
            'MOM_12_1': 1.2, 'MOM_VOL_ADJ': 0.8, 'PCT_VS_MA200': 0.6,
            'DEBT_TO_EQUITY': 0.4,
        },
        filters={'min_market_cap': 2e9},
    ),
    'shareholder_yield': Strategy(
        key='shareholder_yield', name='Total Shareholder Yield', horizon='long',
        description='All cash returned to owners — dividends plus buybacks.',
        thesis='Buybacks and dividends are the same act with different tax '
               'treatment; judging only the dividend misses half the return.',
        weights={
            'DIVIDEND_YIELD': 1.0, 'BUYBACK_YIELD': 1.0, 'FCF_YIELD': 1.2,
            'PAYOUT_RATIO': 0.6, 'ROIC': 0.6, 'DEBT_TO_EQUITY': 0.6,
        },
    ),
    'defensive_value': Strategy(
        key='defensive_value', name='Defensive Value', horizon='long',
        description='Cheap, profitable and low-volatility — value without the drama.',
        thesis='Cheapness alone selects distressed names; adding a quality and '
               'volatility screen keeps the discount and drops the value traps.',
        weights={
            'EARNINGS_YIELD': 1.0, 'FCF_YIELD': 1.0, 'BOOK_TO_MARKET': 0.6,
            'PIOTROSKI_F': 1.0, 'VOL_1Y': 0.8, 'MAX_DD_1Y': 0.6,
            'DEBT_TO_EQUITY': 0.8, 'ACCRUALS': 0.5,
        },
        neutralize='neutralize',
    ),
    'insider_conviction': Strategy(
        key='insider_conviction', name='Insider Conviction', horizon='long',
        description='Quality businesses whose own executives are buying on the '
                    'open market.',
        thesis='Open-market insider purchases are a costly, informed signal; '
               'pairing them with quality avoids buying a falling knife just '
               'because someone bought the dip.',
        weights={
            'INSIDER_NET_BUY': 1.2, 'INSIDER_CLUSTER': 1.0,
            'ROIC': 1.0, 'FCF_YIELD': 1.0, 'EARNINGS_YIELD': 0.8,
            'DEBT_TO_EQUITY': 0.5,
        },
    ),
}


# ─────────────────────────────────────────────
# MID-TERM — trade with a plan
# ─────────────────────────────────────────────

MID_TERM = {
    'momentum_trend': Strategy(
        key='momentum_trend', name='Momentum + Trend', horizon='mid',
        description='12-1 momentum in confirmed uptrends, scaled by volatility.',
        thesis='Ride an established trend while it holds; exit on the stop, the '
               'target, or loss of the 200-day.',
        weights={
            'MOM_12_1': 1.5, 'MOM_VOL_ADJ': 1.2, 'MOM_6_1': 0.8,
            'PCT_VS_MA200': 1.0, 'PCT_FROM_52W_HIGH': 0.8, 'VOL_1Y': 0.4,
        },
        filters={'min_above_ma200': 1, 'rsi_max': 80},
        setup='momentum',
    ),
    'breakout': Strategy(
        key='breakout', name='52-Week Breakout', horizon='mid',
        description='Names pressing against their yearly high on improving momentum.',
        thesis='Buy strength as it clears resistance; exit fast if it fails back through.',
        weights={
            'PCT_FROM_52W_HIGH': 2.0, 'MOM_6_1': 1.2, 'MOM_12_1': 0.8,
            'PCT_VS_MA50': 0.8, 'ATTENTION_Z': 0.4,
        },
        filters={'min_above_ma200': 1, 'rsi_max': 85},
        setup='breakout',
    ),
    'pullback': Strategy(
        key='pullback', name='Pullback in Uptrend', horizon='mid',
        description='Oversold entries inside intact long-term uptrends.',
        thesis='Buy temporary weakness in a strong name; exit if the pullback '
               'becomes a trend change.',
        weights={
            'PCT_VS_MA200': 1.2, 'MOM_12_1': 1.0,
            'REVERSAL_1M': 1.2, 'ROIC': 0.5,
        },
        filters={'min_above_ma200': 1, 'rsi_max': 45},
        setup='pullback',
    ),
    'earnings_drift': Strategy(
        key='earnings_drift', name='Post-Earnings Drift', horizon='mid',
        description='Recent results filings plus positive price reaction (PEAD).',
        thesis='Prices under-react to earnings news; exit at the target or when '
               'the drift window closes.',
        weights={
            'EVENT_RESULTS': 1.0, 'MOM_6_1': 1.0, 'RETURN_3M': 1.2,
            'EARNINGS_GROWTH_1Y': 1.0, 'PCT_VS_MA50': 0.8,
        },
        filters={'min_above_ma200': 1},
        setup='momentum',
    ),
    'insider_cluster': Strategy(
        key='insider_cluster', name='Insider Cluster Buying', horizon='mid',
        description='Multiple insiders buying on the open market, from Form 4 filings.',
        thesis='Insiders buying together is a costly signal; exit on the stop or '
               'when the cluster stops.',
        weights={
            'INSIDER_NET_BUY': 1.5, 'INSIDER_CLUSTER': 1.2,
            'EARNINGS_YIELD': 0.8, 'PCT_VS_MA200': 0.6, 'FCF_YIELD': 0.6,
        },
        setup='pullback',
    ),
    'mean_reversion': Strategy(
        key='mean_reversion', name='Quality Mean Reversion', horizon='mid',
        description='Deeply oversold but fundamentally sound — a contrarian bounce.',
        thesis='Fade the overshoot in a quality name; exit quickly if it keeps falling.',
        weights={
            'REVERSAL_1M': 1.5, 'ROIC': 1.0, 'FCF_YIELD': 0.8,
            'DEBT_TO_EQUITY': 0.8, 'PIOTROSKI_F': 0.6,
        },
        filters={'rsi_max': 35},
        setup='pullback',
    ),
    'relative_strength': Strategy(
        key='relative_strength', name='Low-Volatility Relative Strength', horizon='mid',
        description='Strong risk-adjusted trends in orderly, low-volatility names.',
        thesis='Trends in low-volatility names persist longer and allow tighter '
               'stops, which means a larger position for the same risk; exit on '
               'the stop or a volatility regime change.',
        weights={
            'MOM_VOL_ADJ': 1.5, 'MOM_6_1': 1.0, 'PCT_VS_MA200': 1.0,
            'VOL_1Y': 1.0, 'MAX_DD_1Y': 0.6, 'IDIO_VOL': 0.5,
        },
        filters={'min_above_ma200': 1, 'rsi_max': 75},
        setup='momentum',
    ),
    'quality_breakdown': Strategy(
        key='quality_breakdown', name='Deteriorating Quality (Avoid List)', horizon='mid',
        description='Names whose fundamentals and trend are both breaking down.',
        thesis='Ranks the *worst* candidates — a screen for what to trim or avoid '
               'rather than what to buy. Read the bottom of the list as the '
               'names to review in an existing portfolio.',
        weights={
            'ACCRUALS': 1.2, 'DEBT_TO_EQUITY': 1.0, 'PIOTROSKI_F': 1.2,
            'EVENT_MANAGEMENT': 0.8, 'EVENT_RESTRUCTURING': 0.8,
            'PCT_VS_MA200': 1.0, 'INSIDER_NET_BUY': 0.6,
        },
        setup='momentum',
    ),
}


ALL_STRATEGIES: dict[str, Strategy] = {**LONG_TERM, **MID_TERM}


# ─────────────────────────────────────────────
# LOOKUP
# ─────────────────────────────────────────────

def get(key: str) -> Strategy:
    """
    Resolve a strategy key, built-in or user-defined.

    Custom strategies live in the database, and quant.custom imports this
    module — so the import is deferred into the function body to break the
    cycle. Doing it here rather than at every call site means the engine, the
    backtester, the CLI and the report builder all gained custom-strategy
    support without changing a line.
    """
    if key in ALL_STRATEGIES:
        return ALL_STRATEGIES[key]

    try:
        from quant import custom
        extra = custom.load_all()
    except Exception as exc:                       # noqa: BLE001
        log.debug('custom strategies unavailable: %s', exc)
        extra = {}

    if key in extra:
        return extra[key]

    known = sorted(ALL_STRATEGIES) + sorted(extra)
    raise KeyError(f'unknown strategy {key!r}; have {known}')


def by_horizon(horizon: str) -> dict[str, Strategy]:
    return {k: s for k, s in ALL_STRATEGIES.items() if s.horizon == horizon}


def options(horizon: str | None = None) -> list[dict]:
    """Dropdown options for the UI."""
    pool = ALL_STRATEGIES if horizon is None else by_horizon(horizon)
    return [{'label': s.name, 'value': k} for k, s in sorted(pool.items())]


def summary_table() -> list[dict]:
    """Every strategy with its horizon and thesis — for docs and the UI."""
    return [{
        'key': s.key, 'name': s.name,
        'horizon': HORIZONS[s.horizon]['label'],
        'description': s.description, 'thesis': s.thesis,
        'factors': len(s.weights),
        'trade_plan': 'yes' if s.needs_trade_plan else 'no (thesis-based exit)',
    } for s in sorted(ALL_STRATEGIES.values(), key=lambda x: (x.horizon, x.key))]
