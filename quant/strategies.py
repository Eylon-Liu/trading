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

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

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
    invert: bool = False                          # rank worst-first (avoid lists)
    # A meta-strategy scores names by *agreement between other strategies*
    # instead of weighting factors (see quant/consensus.py). Each family is a
    # group of correlated strategies and casts one vote.
    components: dict[str, tuple[str, ...]] = field(default_factory=dict)
    factor_votes: dict[str, str] = field(default_factory=dict)  # family -> factor
    veto: tuple[str, ...] = ()                    # avoid lists that cost a vote
    vote_pct: float = 0.80                        # family percentile that counts as a vote

    @property
    def is_meta(self) -> bool:
        return bool(self.components)

    @property
    def factors(self) -> list[str]:
        return list(self.weights)

    @property
    def needs_trade_plan(self) -> bool:
        return HORIZONS[self.horizon]['needs_trade_plan']


# ─────────────────────────────────────────────
# STYLE PROXIES — the user-selectable factor per investment style
# ─────────────────────────────────────────────

# Each style can be represented by more than one factor, and which one is
# "right" is an empirical question rather than a settled one. Rather than pick
# silently, the options are named here and chosen on the Screen tab.
#
# Raw earnings per share is deliberately absent. A $10 EPS is not better than a
# $2 EPS — it reflects the share count, not the value — so ranking on it across
# companies is meaningless. EARNINGS_YIELD is the same quantity divided by
# price, which is what makes it comparable; the "is EPS rising?" question it was
# reaching for is answered by EPS_CAGR_3Y under Growth, where a rate of change
# is comparable even though a level is not.
STYLE_PROXIES: dict[str, list[tuple[str, str]]] = {
    'value': [
        ('EARNINGS_YIELD', 'Earnings yield (E/P)'),
        ('BOOK_TO_MARKET', 'Book to market (equity / market cap)'),
    ],
    'growth': [
        ('REVENUE_GROWTH_1Y', 'Sales growth (1Y)'),
        ('EQUITY_CAGR_3Y', 'Equity CAGR (3Y)'),
        ('EPS_CAGR_3Y', 'EPS CAGR (3Y)'),
    ],
    'momentum': [
        ('MOM_12_1', 'Momentum 12-1 (skips last month)'),
        ('MOM_6_1', 'Momentum 6-1 (skips last month)'),
        ('RETURN_6M', '6-month total return'),
        ('RETURN_1M', '1-month total return (reverses — see note)'),
    ],
}

# The momentum default was RETURN_1M. One-month returns are the textbook
# *reversal* horizon — last month's winners tend to give some of it back — which
# is exactly why 12-1 momentum skips the latest month. As the default momentum
# leg of a quarterly-rebalanced, long-horizon screen it pointed the wrong way.
DEFAULT_STYLE_PROXIES: dict[str, str] = {
    'value': 'EARNINGS_YIELD',
    'growth': 'REVENUE_GROWTH_1Y',
    'momentum': 'MOM_12_1',
}

STYLE_FACTORS_KEY = 'style_factors'


def style_strategy(value: str | None = None, growth: str | None = None,
                   momentum: str | None = None) -> 'Strategy':
    """
    Build the style-factor strategy from a choice of proxy per style.

    Returns a `Strategy` rather than registering one, so a screen, a backtest
    and a report can each run a different combination concurrently without any
    shared state. `get()` accepts the object directly.
    """
    chosen = {
        'value': value or DEFAULT_STYLE_PROXIES['value'],
        'growth': growth or DEFAULT_STYLE_PROXIES['growth'],
        'momentum': momentum or DEFAULT_STYLE_PROXIES['momentum'],
    }
    for style, factor in chosen.items():
        valid = [f for f, _ in STYLE_PROXIES[style]]
        if factor not in valid:
            raise ValueError(
                f'{factor!r} is not a {style} proxy; choose one of {valid}')

    base = LONG_TERM[STYLE_FACTORS_KEY]
    label = ' / '.join(chosen[s] for s in ('value', 'growth', 'momentum'))
    return Strategy(
        key=base.key, name=base.name, horizon=base.horizon,
        description=base.description,
        thesis=f'{base.thesis} Currently scoring on {label}.',
        # Equal weight across the three styles: the comparison being made is
        # between proxies, so a weighting scheme would confound it.
        weights={f: 1.0 for f in chosen.values()},
        filters=dict(base.filters), neutralize=base.neutralize,
    )


# ─────────────────────────────────────────────
# LONG-TERM — screen to own
# ─────────────────────────────────────────────

LONG_TERM = {
    'buffett': Strategy(
        key='buffett', name='Buffett Quality', horizon='long',
        description='A durable business, earnings backed by cash, a fortress '
                    'balance sheet, bought at a sensible price.',
        thesis='Underwrite the company the way an owner would: does it earn a '
               'high return on the capital it employs, does the reported profit '
               'actually arrive as cash, could it survive a closed credit '
               'market, and is it priced so that being right is rewarded? Exit '
               'when returns on capital decay structurally or leverage rises to '
               'fund the dividend.',
        # EBIT_TO_EV is gone: it correlates 0.80 with EARNINGS_YIELD on live
        # data, so carrying both spent weight twice on one idea. Gross
        # profitability is gone too — 51% coverage means half the index was
        # scored on eleven factors and half on twelve, which is a different
        # strategy for each half rather than a thin factor.
        weights={
            # margin of safety
            'BOOK_TO_MARKET': 1.2, 'EARNINGS_YIELD': 1.4,
            # cash flow reality — "cash is a fact, profit is an opinion"
            'FCF_YIELD': 1.0, 'FCF_CONVERSION': 1.0, 'ACCRUALS': 0.8,
            # fortress balance sheet
            'NET_DEBT_TO_EQUITY': 1.0, 'PIOTROSKI_F': 0.6, 'CURRENT_RATIO': 0.4,
            # moat / durable returns on capital
            'ROIC': 1.2, 'OPERATING_MARGIN': 0.6,
            # book value compounding — Buffett's own scorecard
            'EQUITY_CAGR_3Y': 0.6,
        },
        filters={'min_market_cap': 2e9},
    ),
    'quality_momentum': Strategy(
        key='quality_momentum', name='Quality Momentum', horizon='long',
        description='Profitable businesses whose prices are already trending up.',
        thesis='Quality and momentum are historically uncorrelated, so combining '
               'them diversifies the drawdowns of each; exit when profitability '
               'deteriorates or the long-term trend breaks.',
        # RETURN_6M, MOM_6_1 and PCT_VS_MA200 measure the same move — pairwise
        # rank correlations of 0.88 and 0.87 on live data. Weighting all three
        # put two-thirds of the strategy on one signal counted three times,
        # which is a concentrated momentum bet wearing a diversified label.
        # One medium-horizon measure, one short, and the quality overlay.
        #
        # RETURN_1M (+0.8) is gone as well. Over one month, returns *reverse*
        # rather than persist, so on a quarterly rebalance it rewarded the very
        # names most likely to give back their last month. 12-1 momentum is the
        # standard construction precisely because it skips that month.
        weights={
            'MOM_12_1': 1.5,
            'ROIC': 0.8, 'ACCRUALS': 0.6, 'NET_DEBT_TO_EQUITY': 0.4,
        },
        filters={'min_market_cap': 2e9},
    ),
    'style_factors': Strategy(
        key='style_factors', name='Style Factors (Value / Growth / Momentum)',
        horizon='long',
        description='Equal-weighted Value, Growth and Momentum, where you choose '
                    'the factor standing in for each style.',
        thesis='Rather than assert one definition of each style, expose the '
               'choice: the proxy for Value, Growth and Momentum is picked on '
               'the Screen tab and every combination is directly backtestable. '
               'The three styles are weighted equally so the comparison is '
               'between proxies, not between weightings.',
        weights={f: 1.0 for f in DEFAULT_STYLE_PROXIES.values()},
    ),
    'insider_conviction': Strategy(
        key='insider_conviction', name='Insider Conviction', horizon='long',
        description='Quality businesses whose own executives are buying on the '
                    'open market.',
        thesis='Open-market insider purchases are a costly, informed signal; '
               'pairing them with quality avoids buying a falling knife just '
               'because someone bought the dip.',
        # INSIDER_CLUSTER correlates 0.98 with INSIDER_NET_BUY — they are one
        # signal, and weighting both put 40% of the strategy on it twice. Only
        # the net-buy measure is kept, and at a weight the evidence supports:
        # this is the one screen whose defining data starts in August 2024, so
        # it has two years of history and cannot be tested further back.
        weights={
            'INSIDER_NET_BUY': 1.5,
            'ROIC': 1.0, 'FCF_YIELD': 1.0, 'EARNINGS_YIELD': 0.8,
            'NET_DEBT_TO_EQUITY': 0.5,
        },
    ),
    'ai_earnings': Strategy(
        key='ai_earnings', name='AI Earnings Review', horizon='long',
        description='Names whose latest results an LLM judged worth buying, '
                    'checked against quality and the market\'s own reaction.',
        thesis='A model reads each new earnings release — guidance, segment '
               'trends, margin commentary that no ratio captures — and grades '
               'it for a 6-18 month holder. The market reaction and two quality '
               'gauges keep a persuasive release from outranking the numbers. '
               'Only names reviewed in the last 120 days are scored. Forward-'
               'only: judge it by its live track record, not a backtest.',
        weights={
            'LLM_EARNINGS_VIEW': 2.0, 'EARNINGS_REACTION': 0.8,
            'ROIC': 0.5, 'FCF_YIELD': 0.5,
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
            'PAYOUT_RATIO': 0.6, 'ROIC': 0.6, 'NET_DEBT_TO_EQUITY': 0.6,
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
        description='Names whose latest results drew a strong positive reaction (PEAD).',
        thesis='Prices under-react to earnings news; exit at the target or when '
               'the drift window closes.',
        # This used to have no measure of the reaction at all. EVENT_RESULTS
        # only says a results 8-K was filed in the last 90 days — true of
        # nearly every company every quarter — and RETURN_3M / MOM_6_1 are
        # plain momentum. The drift literature sorts on the announcement
        # return itself, which EARNINGS_REACTION now measures; names without
        # a report in the last 63 days fall below the coverage floor and are
        # not scored, which is the point.
        weights={
            'EARNINGS_REACTION': 2.0, 'EARNINGS_GROWTH_1Y': 0.8,
            'MOM_6_1': 0.6, 'PCT_VS_MA50': 0.4,
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
            'INSIDER_NET_BUY': 2.0,
            'EARNINGS_YIELD': 1.0, 'PCT_VS_MA200': 0.8, 'FCF_YIELD': 0.8,
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
        key='quality_breakdown', name='Deteriorating Quality', horizon='mid',
        description='Names whose fundamentals and trend are both breaking down.',
        thesis='Surfaces the weakest candidates — rank 1 is the name with the '
               'worst combination of rising accruals, rising leverage, falling '
               'trend, insider selling and management churn.',
        weights={
            'ACCRUALS': 1.2, 'DEBT_TO_EQUITY': 1.0,
            'EVENT_MANAGEMENT': 0.8, 'EVENT_RESTRUCTURING': 0.8,
            'PIOTROSKI_F': 1.2, 'PCT_VS_MA200': 1.0, 'INSIDER_NET_BUY': 0.6,
        },
        setup='momentum',
        invert=True,
    ),
}


# ─────────────────────────────────────────────
# META — agreement between strategies
# ─────────────────────────────────────────────

META = {
    'consensus': Strategy(
        key='consensus', name='Multi-Strategy Consensus', horizon='long',
        description='Names that several independent strategy families rank '
                    'near the top at the same time.',
        thesis='Agreement between methods that look at different evidence — '
               'business quality and valuation, price trend, insider buying, '
               'the earnings reaction, and an AI read of the latest results — '
               'is better evidence than any single score. Correlated strategies '
               'are grouped into families and each family casts one vote, so '
               'three momentum variants agreeing count once, not three times. '
               'A name on the Deteriorating Quality avoid list loses a vote. '
               'Exit when agreement falls away.',
        weights={},
        components={
            'Quality & value': ('buffett', 'shareholder_yield'),
            'Momentum & trend': ('quality_momentum', 'momentum_trend',
                                 'relative_strength'),
            'Insider buying': ('insider_conviction', 'insider_cluster'),
            'Earnings reaction': ('earnings_drift',),
        },
        factor_votes={'AI earnings review': 'LLM_EARNINGS_VIEW'},
        veto=('quality_breakdown',),
    ),
}

ALL_STRATEGIES: dict[str, Strategy] = {**LONG_TERM, **MID_TERM, **META}


# ─────────────────────────────────────────────
# LOOKUP
# ─────────────────────────────────────────────

def get(key: str | Strategy) -> Strategy:
    """
    Resolve a strategy key, built-in or user-defined.

    Custom strategies live in the database, and quant.custom imports this
    module — so the import is deferred into the function body to break the
    cycle. Doing it here rather than at every call site means the engine, the
    backtester, the CLI and the report builder all gained custom-strategy
    support without changing a line.

    An already-built `Strategy` passes straight through, which is what lets a
    caller run a parameterised strategy — the style-factor screen, whose
    weights depend on dropdown choices — through the identical code path as a
    named one, backtest included.
    """
    if isinstance(key, Strategy):
        return key

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


def options(horizon: str | None = None, grouped: bool = False,
            include_meta: bool = True) -> list[dict]:
    """Dropdown options for the UI.

    With ``grouped=True``, returns option-groups keyed by horizon label so
    Dash renders them under section headers. `include_meta=False` leaves out
    strategies with no factor weights of their own (the consensus), for
    places that copy weights.
    """
    pool = ALL_STRATEGIES if horizon is None else by_horizon(horizon)
    if not include_meta:
        pool = {k: s for k, s in pool.items() if not s.is_meta}
    if not grouped or horizon is not None:
        return [{'label': s.name, 'value': k} for k, s in sorted(pool.items())]

    groups = []
    for hz_key in ('long', 'mid'):
        bucket = {k: s for k, s in pool.items() if s.horizon == hz_key}
        if bucket:
            groups.append({
                'label': HORIZONS[hz_key]['label'],
                'value': [{'label': s.name, 'value': k}
                          for k, s in sorted(bucket.items())],
            })
    return groups


def summary_table() -> list[dict]:
    """Every strategy with its horizon and thesis — for docs and the UI."""
    return [{
        'key': s.key, 'name': s.name,
        'horizon': HORIZONS[s.horizon]['label'],
        'description': s.description, 'thesis': s.thesis,
        'factors': len(s.weights) if not s.is_meta else 0,
        'trade_plan': ('no (exit when agreement falls away)' if s.is_meta
                       else 'yes' if s.needs_trade_plan else 'no (thesis-based exit)'),
    } for s in sorted(ALL_STRATEGIES.values(), key=lambda x: (x.horizon, x.key))]
