"""
What each strategy is good at, what it is bad at, and how it fails.

Kept apart from strategies.py for the same reason factor_docs.py is: that file
defines *what is computed*, this one explains *when to believe it*. A weight
vector cannot tell you that a value screen buys melting ice cubes in a
momentum regime, and a name like "Quality at a Reasonable Price" does not
distinguish itself from "Quality Compounder" without help.

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

    'quality_value': StrategyNote(
        differentiator='The balanced default: quality and cheapness weighted '
                       'roughly equally. Quality Compounder drops the value '
                       'constraint; Deep Value drops the quality one.',
        benefit='Avoids the two classic traps at once — paying any price for a '
                'good business, and buying a bad business because it is cheap. '
                'The most forgiving starting point if you only run one screen.',
        drawback='Compromise scoring means it rarely tops any single dimension. '
                 'In a strong momentum market it will look timid, and in a deep '
                 'value rally it will lag the pure value screens.',
        risk='Earnings yield uses trailing figures. A cyclical at peak margins '
             'looks cheap precisely when its earnings are about to fall, and '
             'the quality factors will not flag it because current returns are '
             'genuinely high.',
        best_when='Broad markets with no dominant factor; recovery phases.',
        worst_when='Narrow momentum-led markets where a handful of expensive '
                   'names drive the index.',
    ),

    'compounder': StrategyNote(
        differentiator='Quality only — valuation is deliberately ignored. If '
                       'you want the same businesses but bought on a yield, '
                       'use Quality at a Reasonable Price.',
        benefit='Selects for durable, capital-efficient businesses that can '
                'reinvest internally. Historically the lowest-turnover approach '
                'here, which makes it the cheapest to run and the easiest to '
                'hold through drawdowns.',
        drawback='No valuation discipline at all. It will happily rank a '
                 'superb business at fifty times earnings first, and your '
                 'return then depends on multiple expansion continuing.',
        risk='Quality is measured from the past. ROIC stays high right up until '
             'a moat breaks, and the screen has no mechanism to see the break '
             'coming — it will still be recommending the name while the thesis '
             'is dissolving.',
        best_when='Long holding periods; investors who will not trade often.',
        worst_when='Rising rates compressing the multiples of long-duration '
                   'growth assets.',
    ),

    'deep_value': StrategyNote(
        differentiator='Pure cheapness, minimal quality gate. The opposite '
                       'end of the spectrum from Quality Compounder.',
        benefit='Buys the largest discount to fundamentals available. When '
                'value works, this captures the most of it, and the entry '
                'price gives a genuine margin of safety.',
        drawback='Value traps are the norm, not the exception. Most statistically '
                 'cheap companies are cheap because their business is impaired, '
                 'and the screen cannot distinguish temporary from terminal.',
        risk='Concentrates by construction in whatever sector the market has '
             'given up on — often all at once. Sector-relative scoring softens '
             'this but does not remove it.',
        best_when='Early recovery from a broad drawdown; value factor rotations.',
        worst_when='Late-cycle momentum markets; secular decline in the cheap '
                   'sectors (the classic case being print media, then retail).',
    ),

    'defensive_value': StrategyNote(
        differentiator='Deep Value with a solvency and stability gate bolted '
                       'on. Gives up some discount to avoid the worst traps.',
        benefit='Keeps most of the value exposure while screening out the '
                'balance sheets most likely to fail. A reasonable middle for '
                'anyone who finds Deep Value uncomfortable.',
        drawback='The quality gate removes exactly the deepest discounts, which '
                 'is where a meaningful share of value returns historically sat.',
        risk='Low volatility and low leverage are correlated with expensive, '
             'bond-like defensives. In a rate shock those fall together and the '
             '"defensive" label is actively misleading.',
        best_when='Uncertain markets where balance-sheet strength is rewarded.',
        worst_when='Sharp risk-on rallies; rapid rate rises.',
    ),

    'dividend_quality': StrategyNote(
        differentiator='Income with a sustainability test. Total Shareholder '
                       'Yield counts buybacks too and is the better choice '
                       'unless you specifically need cash income.',
        benefit='Screens for dividends that are actually covered by cash flow, '
                'rather than for the highest headline yield — which is usually '
                'a cut waiting to happen.',
        drawback='Structurally excludes almost every high-growth business, '
                 'since the best reinvestment opportunities pay nothing out.',
        risk='A high yield is often the market pricing in a cut. Payout '
             'coverage helps, but coverage is computed from trailing cash flow '
             'and will not anticipate a collapse in it.',
        best_when='Income mandates; falling-rate environments.',
        worst_when='Rising rates, when bonds compete directly; growth-led markets.',
    ),

    'shareholder_yield': StrategyNote(
        differentiator='Counts buybacks alongside dividends, so it sees the '
                       'whole of capital returned rather than only the taxable '
                       'half that Dividend Quality looks at.',
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

    'piotroski': StrategyNote(
        differentiator='A single nine-point accounting checklist rather than a '
                       'weighted blend. The most mechanical and the easiest to '
                       'verify by hand.',
        benefit='Well documented in the literature, hard to overfit, and each '
                'of the nine tests is individually explicable to someone who '
                'does not trust the model.',
        drawback='Binary tests throw away magnitude. A company improving ROA by '
                 'a basis point scores exactly the same as one doubling it.',
        risk='Designed to be applied to already-cheap stocks. Run on its own '
             'across a broad universe it mostly identifies companies in a '
             'cyclical upswing, which is a different and weaker signal.',
        best_when='Paired with a value screen, which is its original use.',
        worst_when='Used alone on an expensive universe.',
    ),

    'garp': StrategyNote(
        differentiator='Growth with a valuation ceiling. Quality Compounder '
                       'cares about returns on capital; this cares about the '
                       'growth rate and what you pay for it.',
        benefit='Finds businesses compounding revenue and earnings without '
                'paying an unbounded multiple for the privilege.',
        drawback='Growth is measured from trailing filings and is heavily '
                 'mean-reverting. High past growth is a weak predictor of '
                 'future growth, which undercuts the premise.',
        risk='Vulnerable to the classic growth de-rating: the company keeps '
             'growing, the multiple halves, and the position still loses money.',
        best_when='Mid-cycle expansion with stable rates.',
        worst_when='Rate shocks; any regime where multiples compress faster '
                   'than earnings grow.',
    ),

    'quality_momentum': StrategyNote(
        differentiator='Quality Compounder plus a price-trend requirement — '
                       'the market must already agree with you.',
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

    'low_volatility': StrategyNote(
        differentiator='Selects on realised risk rather than on any business '
                       'characteristic. The only long screen here that does '
                       'not look at profitability.',
        benefit='Smaller drawdowns and a smoother path, which in practice is '
                'what determines whether an investor actually holds through a '
                'bad year.',
        drawback='Gives up meaningful upside in strong markets, and the whole '
                 'approach became crowded after a decade of popularity, which '
                 'compressed the premium.',
        risk='Low-volatility screens concentrate hard in utilities, staples and '
             'REITs — long-duration, bond-like assets that all fall together '
             'when rates rise. The diversification is an illusion.',
        best_when='High-volatility or falling markets.',
        worst_when='Rate rises; strong risk-on rallies.',
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

    'multifactor': StrategyNote(
        differentiator='Blends value, quality, momentum and low risk together '
                       'rather than committing to one. The most diversified '
                       'long screen here.',
        benefit='No single factor drawdown sinks it, and factor timing is a '
                'game few win — holding several removes the need to.',
        drawback='Blending dilutes. It will never top a leaderboard, and in a '
                 'year when one factor does all the work it will noticeably lag.',
        risk='Factors that look independent become correlated in a crisis, so '
             'the diversification is weakest exactly when it is needed. Wide '
             'coverage requirements also make it sensitive to missing data.',
        best_when='As a default when you have no view on factor regime.',
        worst_when='Strongly single-factor-led markets.',
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
