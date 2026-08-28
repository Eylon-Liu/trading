"""
What each strategy is good at, what it is bad at, and how it fails.

Kept apart from strategies.py for the same reason factor_docs.py is: that file
defines *what is computed*, this one explains *when to believe it*. A weight
vector cannot tell you that a value screen buys melting ice cubes in a
momentum regime, and a name like "Buffett Quality" does not distinguish itself
from "Quality Momentum" without help.

Every entry carries a `differentiator` for exactly that reason — several
strategies here are cousins, and the honest answer to "why would I pick this
one over that one" is usually a single sentence.

`risk` is deliberately the failure *mechanism*, not a disclaimer. "May lose
money" helps nobody; "screens cheapest on trailing earnings, so it concentrates
in cyclicals at their peak margins" tells you what to check.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyNote:
    differentiator: str      # how it differs from its nearest sibling
    benefit: str             # what it genuinely does well
    drawback: str            # the structural weakness, always present
    risk: str                # the mechanism by which it goes wrong
    best_when: str
    worst_when: str
    horizon_note: str = ''


NOTES: dict[str, StrategyNote] = {

    # ── LONG TERM ────────────────────────────────────────────────

    'buffett': StrategyNote(
        differentiator='The only screen that underwrites the whole business at '
                       'once — price, cash conversion, balance sheet and '
                       'returns on capital — rather than tilting at one of '
                       'them. Style Factors lets you pick a single dimension; '
                       'this one refuses to.',
        benefit='Insists that reported profit actually arrived as cash and that '
                'the balance sheet could survive a shut credit market, which '
                'together screen out most of the accounting-driven blowups a '
                'pure value or pure quality list walks into.',
        drawback='Twelve factors means no single one dominates, so it rarely '
                 'tops any individual dimension and will look slow next to a '
                 'concentrated screen in a trending market. Its size filter '
                 'also excludes the small end where mispricing is largest.',
        risk='Every input is trailing. A cyclical at peak margins scores well on '
             'returns, cash conversion and cheapness simultaneously — precisely '
             'at the top of its cycle, because peak earnings flatter the yield '
             'and the margin at the same moment. Net debt to equity is the only '
             'factor here that would flag the leverage such a name usually '
             'carries into the downturn.',
        best_when='Broad markets, rising real rates, and periods when balance '
                  'sheet strength is repriced.',
        worst_when='Late-stage momentum markets led by a few expensive names, '
                   'where owning cash-generative businesses at sane multiples '
                   'is exactly the wrong trade.',
    ),

    'style_factors': StrategyNote(
        differentiator='The only configurable strategy: you choose the factor '
                       'standing in for Value, Growth and Momentum, and the '
                       'three are weighted equally. Every other screen fixes '
                       'both the factors and their weights.',
        benefit='Makes the proxy choice explicit and testable. "Value" means '
                'something different depending on whether it is earnings yield '
                'or book to market, and here you can backtest the difference '
                'instead of inheriting someone else\'s definition.',
        drawback='Equal weighting across three styles is a decision, not a '
                 'neutral default — it dilutes whichever style is working. '
                 'Three factors is also thin cover: a name missing one of them '
                 'is scored on two.',
        risk='The freedom to choose is the freedom to overfit. Trying proxy '
             'combinations until the backtest looks good will find a winner by '
             'chance well before it finds one by signal — the growth proxies in '
             'particular have shown no reliable edge on this universe.',
        best_when='Investigating which definition of a style actually carries '
                  'the premium in a given regime.',
        worst_when='Used as a production screen without having backtested the '
                   'specific combination selected.',
    ),

    'shareholder_yield': StrategyNote(
        differentiator='Counts buybacks alongside dividends, so it sees the '
                       'whole of capital returned rather than only the taxable '
                       'half a dividend-only screen would see.',
        benefit='Captures companies returning cash through repurchase, which a '
                'dividend screen misses entirely despite it being economically '
                'the same act.',
        drawback='Buyback yield is gross of issuance. A company repurchasing '
                 'while issuing heavily to staff nets out far lower than the '
                 'factor suggests.',
        risk='Buybacks are pro-cyclical: managements repurchase most at peak '
             'prices with peak cash, and stop precisely when the shares get '
             'cheap. A high trailing buyback yield can mark the top.',
        best_when='Mature, cash-generative sectors with limited reinvestment.',
        worst_when='Credit stress, when buybacks are cut first.',
    ),

    'quality_momentum': StrategyNote(
        differentiator='The only long-term screen where price action leads. '
                       'Buffett Quality asks what a business is worth; this '
                       'one requires the market to already agree with you.',
        benefit='Momentum acts as a timing filter on a quality list, which '
                'historically avoids the long dead periods where a good '
                'business goes nowhere for years.',
        drawback='By construction it buys after the move. Entry prices are '
                 'worse than the pure quality screen, and it will miss the '
                 'first leg of every recovery.',
        risk='Momentum reverses hardest at turning points, so both factors can '
             'fail together: the crash takes the price, and the recession takes '
             'the earnings that made it quality.',
        best_when='Sustained trending markets.',
        worst_when='Sharp reversals and choppy, range-bound markets.',
    ),

    'insider_conviction': StrategyNote(
        differentiator='The only long strategy driven by an alternative '
                       'dataset — Form 4 filings — rather than by financial '
                       'statements or price.',
        benefit='Open-market purchases are costly and informed. Pairing them '
                'with a quality gate avoids buying a falling knife merely '
                'because an officer bought the dip.',
        drawback='Sparse. Many good businesses have no insider buying in any '
                 'given window, so coverage is thin and the list is short.',
        risk='Insiders are early and often wrong about timing; they also buy '
             'for signalling reasons. Sales are near-uninformative because '
             'they are dominated by scheduled diversification.',
        best_when='After broad selloffs, when insider buying clusters.',
        worst_when='Quiet markets, and during blackout windows around results.',
    ),

    # ── MID TERM ─────────────────────────────────────────────────

    'momentum_trend': StrategyNote(
        differentiator='The straightforward trend-follower: buy what is already '
                       'rising and above its moving averages.',
        benefit='Trend persistence is one of the most robust effects in markets, '
                'documented across asset classes and a century of data.',
        drawback='Whipsaws in range-bound markets, where it buys every false '
                 'breakout and stops out on each reversal.',
        risk='Momentum crashes are violent and happen at inflection points, '
             'so the worst losses arrive in a cluster rather than spread out.',
        best_when='Clear directional trends with expanding breadth.',
        worst_when='Choppy, mean-reverting markets; sharp regime turns.',
        horizon_note='Days to months, with a stop and a target.',
    ),

    'breakout': StrategyNote(
        differentiator='Requires a move to new highs specifically, rather than '
                       'general strength. A narrower and later entry than '
                       'Momentum + Trend.',
        benefit='New highs mean no overhead supply — nobody above is waiting to '
                'sell at breakeven, which is why breakouts can run.',
        drawback='Late by construction, so stops sit further away and positions '
                 'are correspondingly smaller.',
        risk='False breakouts are common, especially on thin volume. The '
             'failure mode is a fast reversal straight back into the range.',
        best_when='Expanding-breadth bull phases.',
        worst_when='Range-bound markets, where most breakouts fail.',
        horizon_note='Weeks to months.',
    ),

    'pullback': StrategyNote(
        differentiator='Buys temporary weakness *within* an uptrend — the '
                       'opposite entry timing to Breakout, on the same kind of '
                       'name.',
        benefit='Much better entry prices than breakout buying, so stops are '
                'tighter and position sizes larger for the same risk budget.',
        drawback='Distinguishing a pullback from the start of a real decline is '
                 'the whole problem, and no indicator does it reliably.',
        risk='Catching a falling knife: the trend filter uses trailing averages '
             'that keep saying "uptrend" for weeks after the top.',
        best_when='Established uptrends with orderly, shallow corrections.',
        worst_when='Trend exhaustion, where the first pullback becomes the '
                   'first leg down.',
        horizon_note='Days to weeks.',
    ),

    'earnings_drift': StrategyNote(
        differentiator='Event-driven rather than price-driven: it keys off a '
                       'recent results filing, not a chart pattern.',
        benefit='Post-earnings announcement drift is among the most persistent '
                'documented anomalies, and the 8-K item code identifies the '
                'event deterministically rather than by parsing text.',
        drawback='The window is short and crowded. The effect has weakened as '
                 'it has become widely known and traded.',
        risk='Concentrated event risk — one disappointing follow-up quarter '
             'reverses the entire move, often in a single gap.',
        best_when='Immediately following earnings season.',
        worst_when='Between reporting seasons, when there is nothing to drift '
                   'from.',
        horizon_note='Days to a few weeks after the filing.',
    ),

    'insider_cluster': StrategyNote(
        differentiator='Requires *several* insiders buying, not one. The mid-'
                       'horizon counterpart to Insider Conviction.',
        benefit='Multiple independent buyers is a far stronger signal than a '
                'single purchase, and clusters are rare enough to be noticed.',
        drawback='Very few names qualify at any moment, so the list is often '
                 'empty and cannot be relied on for continuous deployment.',
        risk='Insiders buy early. The cluster can be right about value and '
             'still be followed by months of further decline.',
        best_when='After sector-wide selloffs.',
        worst_when='Blackout periods; quiet markets.',
        horizon_note='Weeks to months.',
    ),

    'mean_reversion': StrategyNote(
        differentiator='Trades *against* short-term price moves, where every '
                       'other mid-horizon strategy here trades with them.',
        benefit='Short-horizon reversal is real and well documented, and the '
                'quality gate keeps it from buying genuinely broken companies.',
        drawback='Directly opposed to the momentum strategies, so running both '
                 'produces conflicting signals on the same names.',
        risk='Reversion assumes the move was noise. When it was information — '
             'a guidance cut, a lost contract — the position is on the wrong '
             'side of a repricing that will not revert.',
        best_when='Range-bound, high-volatility markets.',
        worst_when='Strong trends, where the cheap keeps getting cheaper.',
        horizon_note='Days to weeks.',
    ),

    'relative_strength': StrategyNote(
        differentiator='Ranks strength *per unit of volatility*, so it favours '
                       'steady climbers over violent movers — unlike plain '
                       'Momentum + Trend.',
        benefit='Volatility-scaling produces a smoother equity curve and more '
                'consistent position sizing than raw momentum.',
        drawback='Systematically avoids the highest-return names, which are '
                 'usually the most volatile ones.',
        risk='Low realised volatility understates risk in quiet periods that '
             'precede shocks — the calm is often the setup, not the safety.',
        best_when='Steady trending markets.',
        worst_when='Volatility regime shifts, where past volatility misleads.',
        horizon_note='Weeks to months.',
    ),

    'quality_breakdown': StrategyNote(
        differentiator='The only screen here built to produce a list of names '
                       'to *avoid* rather than to buy.',
        benefit='Useful as a veto layer over any other strategy, and as a '
                'review trigger for something already held.',
        drawback='Deteriorating fundamentals can persist for a long time '
                 'without the price responding, so it flags early and often.',
        risk='Deep-value opportunities look identical to deteriorating quality '
             'at the moment of maximum pessimism. Used mechanically it will '
             'veto the best entries.',
        best_when='Screening an existing portfolio for names to review.',
        worst_when='Used as a short list — being early is indistinguishable '
                   'from being wrong when you are paying borrow.',
        horizon_note='Review trigger, not a trade.',
    ),
}


def get(key: str) -> StrategyNote | None:
    return NOTES.get(key)


def coverage() -> dict:
    """Which strategies still lack notes — used by the test suite."""
    from quant import strategies as ST
    missing = sorted(set(ST.ALL_STRATEGIES) - set(NOTES))
    orphan = sorted(set(NOTES) - set(ST.ALL_STRATEGIES))
    return {'missing': missing, 'orphan': orphan}
