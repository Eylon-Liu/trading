"""
Factor and strategy documentation — the formulas behind every number.

A score you cannot reproduce by hand is a score you should not act on. This
module is the reference for exactly how each factor is computed, what the
inputs are, which direction is good, and where the construction is known to
break down. The UI renders it verbatim, so this file and the code in
quant/factors.py must stay in step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorDoc:
    name: str
    label: str
    family: str
    formula: str
    inputs: str
    direction: str            # 'higher' | 'lower'
    rationale: str
    caveat: str = ''


# ─────────────────────────────────────────────
# PIPELINE — applies to every factor
# ─────────────────────────────────────────────

PIPELINE = r"""
Every raw factor goes through the same four steps before it is scored. The
composite is a weighted blend of the results, never of the raw values —
raw units (dollars, ratios, percentages) are not comparable to each other.

1. WINSORIZE (median absolute deviation, 5 MAD)

       MAD   = median(|x − median(x)|)
       sigma = 1.4826 × MAD                (MAD → sd for a normal distribution)
       x'    = clip(x, median − 5·sigma, median + 5·sigma)

   Clipping, not dropping. One P/B of 0.00001 becomes a book-to-market of
   100,000%, which drags the sector mean and inflates its standard deviation
   until every other name in that sector scores nearly identically.

2. SECTOR-RELATIVE Z-SCORE

       z_i = (x'_i − mean(x'_sector)) / sd(x'_sector)

   A 12% ROE is unremarkable for software and excellent for a utility, so
   names are ranked against their own sector. Sectors with fewer than 5 valid
   observations fall back to the full cross-section: a mean and standard
   deviation estimated from two names is noise, not a benchmark.

   Some strategies instead use REGRESSION NEUTRALIZATION, which residualizes
   the factor on sector dummies and size before standardizing:

       x = α + Σ βₛ·sectorₛ + γ·log(market cap) + ε
       score = z(ε)

   This removes sector and size effects jointly rather than bucketing, which
   is what a risk model does.

3. DIRECTION

   Factors where lower is better (volatility, leverage, accruals) are negated
   so that a higher score is always better.

4. COMPOSITE with a coverage floor

       coverage_i  = Σ wⱼ · 1[factor j present for i]      (weights sum to 1)
       composite_i = Σ wⱼ · zⱼ,ᵢ  /  coverage_i            if coverage ≥ 0.5
                   = excluded                              otherwise

   Weights are renormalized over the factors a name actually has, so a
   missing input does not drag a score toward zero. But a name missing most
   of its inputs is excluded rather than ranked on whatever survived —
   otherwise a stock with one lucky factor outranks a stock with three good
   ones.
"""


# ─────────────────────────────────────────────
# FACTOR REFERENCE
# ─────────────────────────────────────────────

FACTOR_DOCS: dict[str, FactorDoc] = {

    # ── VALUE ────────────────────────────────────────────────────
    'EARNINGS_YIELD': FactorDoc(
        'EARNINGS_YIELD', 'Earnings Yield (E/P)', 'Value',
        'EARNINGS_YIELD = Net Income (TTM) / Market Cap',
        'SEC NetIncomeLoss (4 quarters, point-in-time) ÷ shares × price',
        'higher',
        'The inverse of P/E, and the reason this tool uses a yield rather than '
        'dollar EPS. Trailing EPS of $10 says nothing about whether a stock is '
        'cheap — it depends entirely on the price. A yield is comparable across '
        'the whole cross-section; a dollar figure is not.',
        'Negative for loss-makers, where the ranking becomes meaningless. '
        'Cyclicals look cheapest at peak earnings, right before the cycle turns.'),

    'FCF_YIELD': FactorDoc(
        'FCF_YIELD', 'Free Cash Flow Yield', 'Value',
        'FCF        = Operating Cash Flow (TTM) − Capital Expenditure (TTM)\n'
        'FCF_YIELD  = FCF / Market Cap',
        'SEC NetCashProvidedByUsedInOperatingActivities and '
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'higher',
        'Harder to manipulate than earnings, because accruals, depreciation '
        'schedules and revenue-recognition choices do not move cash. Usually '
        'the single most reliable value factor.',
        'Lumpy for capital-intensive businesses — a year of heavy capex depresses '
        'it without the business deteriorating. Banks report cash flow on a '
        'different basis, so this is often absent for financials.'),

    'BOOK_TO_MARKET': FactorDoc(
        'BOOK_TO_MARKET', 'Book-to-Market', 'Value',
        'BOOK_TO_MARKET = Shareholders Equity / Market Cap        (= 1 / P/B)',
        'SEC StockholdersEquity ÷ market cap',
        'higher',
        'The classic Fama-French value factor. Measures how much accounting '
        'net worth you get per dollar of price.',
        'Book value badly understates asset-light businesses — software, brands '
        'and R&D are expensed, not capitalized — so this factor systematically '
        'tilts away from technology and toward heavy industry.'),

    'EBIT_TO_EV': FactorDoc(
        'EBIT_TO_EV', 'EBIT / Enterprise Value', 'Value',
        'EV         = Market Cap + Total Debt − Cash\n'
        'EBIT_TO_EV = Operating Income (TTM) / EV',
        'SEC OperatingIncomeLoss, debt and cash balances',
        'higher',
        'Greenblatt\'s earnings yield. Capital-structure neutral: it compares '
        'operating profit to the price of the whole business, so a leveraged '
        'and an unleveraged company are judged on the same basis.',
        'EV is distorted for companies with very large cash balances or '
        'pension liabilities that do not appear as debt.'),

    'SALES_YIELD': FactorDoc(
        'SALES_YIELD', 'Sales Yield (S/P)', 'Value',
        'SALES_YIELD = Revenue (TTM) / Market Cap                 (= 1 / P/S)',
        'SEC revenue tags',
        'higher',
        'Survives when earnings are negative or distorted, which makes it the '
        'value factor of last resort for loss-making or turnaround names.',
        'Ignores profitability entirely — a company can have excellent sales '
        'yield and never make money.'),

    # ── QUALITY ──────────────────────────────────────────────────
    'ROIC': FactorDoc(
        'ROIC', 'Return on Invested Capital', 'Quality',
        'ROIC = Operating Income (TTM) / (Shareholders Equity + Total Debt)',
        'SEC OperatingIncomeLoss, StockholdersEquity, debt',
        'higher',
        'The core measure of whether a business creates value: it earns its '
        'cost of capital or it does not. Unlike ROE it is not flattered by '
        'leverage, because debt sits in the denominator.',
        'Uses book capital, which understates the true capital base of '
        'acquisitive companies carrying large goodwill balances.'),

    'ROE': FactorDoc(
        'ROE', 'Return on Equity', 'Quality',
        'ROE = Net Income (TTM) / Shareholders Equity',
        'SEC NetIncomeLoss ÷ StockholdersEquity',
        'higher',
        'Return to shareholders on their book capital. Familiar, and the '
        'standard profitability measure for financials.',
        'Rises mechanically with leverage, and becomes meaningless when equity '
        'is small or negative — Apple\'s ~120% ROE reflects buybacks shrinking '
        'the equity base, not extraordinary economics.'),

    'GROSS_PROFITABILITY': FactorDoc(
        'GROSS_PROFITABILITY', 'Gross Profitability', 'Quality',
        'GROSS_PROFITABILITY = Gross Profit (TTM) / Total Assets',
        'SEC GrossProfit ÷ Assets',
        'higher',
        'Novy-Marx\'s "other side of value". Gross profit sits above the many '
        'discretionary lines (R&D, marketing, depreciation policy) that '
        'management can flex, so it is a cleaner read on economics than net '
        'income and predicts returns about as well as book-to-market.',
        'Not reported by financials, and comparisons across industries with '
        'very different cost structures are weak.'),

    'ACCRUALS': FactorDoc(
        'ACCRUALS', 'Accruals', 'Quality',
        'ACCRUALS = (Net Income (TTM) − Operating Cash Flow (TTM)) / Total Assets',
        'SEC NetIncomeLoss, NetCashProvidedByUsedInOperatingActivities, Assets',
        'lower',
        'The gap between reported profit and cash collected. Sloan\'s accrual '
        'anomaly: firms whose earnings are mostly accrual rather than cash '
        'systematically underperform, because those earnings tend to reverse.',
        'Legitimately high for genuinely growing businesses building working '
        'capital — growth and aggressive accounting look similar here.'),

    'DEBT_TO_EQUITY': FactorDoc(
        'DEBT_TO_EQUITY', 'Debt to Equity', 'Quality',
        'DEBT_TO_EQUITY = (Long-Term Debt + Short-Term Debt) / Shareholders Equity',
        'SEC debt tags ÷ StockholdersEquity',
        'lower',
        'Balance-sheet risk. Leverage amplifies both returns and the '
        'probability of not surviving a downturn.',
        'Not comparable across sectors — banks and utilities are structurally '
        'leveraged. Sector-relative scoring handles most of this.'),

    'NET_DEBT_TO_EQUITY': FactorDoc(
        'NET_DEBT_TO_EQUITY', 'Net Debt to Equity', 'Quality',
        'NET_DEBT_TO_EQUITY = (Total Debt − Cash) / Shareholders Equity',
        'SEC debt tags, CashAndCashEquivalents ÷ StockholdersEquity',
        'lower',
        'The fortress-balance-sheet test. Gross debt-to-equity calls a company '
        'levered when it holds more cash than debt; netting the cash off shows '
        'who is actually a borrower. Negative values are the net-cash case — '
        'the company could repay every dollar of debt tomorrow.',
        'Treats all cash as available, which overstates flexibility where much '
        'of it is held overseas or committed. Same sector caveat as gross '
        'leverage.'),

    'FCF_CONVERSION': FactorDoc(
        'FCF_CONVERSION', 'FCF Conversion', 'Quality',
        'FCF_CONVERSION = Free Cash Flow (TTM) / Net Income (TTM)',
        'SEC OperatingCashFlow − CapEx ÷ NetIncomeLoss',
        'higher',
        'Cash is a fact, profit is an opinion. Reported earnings involve '
        'judgement — revenue recognition, depreciation schedules, provisions — '
        'while cash in the bank does not. A business converting close to or '
        'above 100% of its earnings into free cash flow is reporting profits '
        'it actually collected.',
        'Meaningless when net income is negative or near zero, where the ratio '
        'explodes or flips sign. Genuinely lumpy for capital-intensive names '
        'in a heavy investment year, which depresses it for a good reason.'),

    'PIOTROSKI_F': FactorDoc(
        'PIOTROSKI_F', 'Piotroski F-Score', 'Quality',
        'Sum of 9 binary tests (0-9), all from filed statements:\n'
        '  Profitability   1. ROA > 0\n'
        '                  2. Operating cash flow > 0\n'
        '                  3. ROA improving year over year\n'
        '                  4. Operating cash flow > net income (earnings quality)\n'
        '  Leverage        5. Long-term debt / assets falling\n'
        '                  6. Current ratio rising\n'
        '                  7. No meaningful share issuance\n'
        '  Efficiency      8. Gross margin rising\n'
        '                  9. Asset turnover rising',
        'SEC filings across two years',
        'higher',
        'Nine fundamental health checks. Designed to separate cheap-and-'
        'improving from cheap-and-dying, which is the central problem with '
        'value screens.',
        'Needs two full years of comparable filings; recent IPOs and companies '
        'that changed fiscal year score artificially low.'),

    'ROA': FactorDoc(
        'ROA', 'Return on Assets', 'Quality',
        'ROA = Net Income (TTM) / Total Assets',
        'SEC NetIncomeLoss ÷ Assets', 'higher',
        'Profit generated per dollar of assets, regardless of how those assets '
        'were financed. Unlike ROE it is not inflated by leverage.',
        'Structurally low for asset-heavy businesses (utilities, banks), so it '
        'is only meaningful within a sector.'),

    'OPERATING_MARGIN': FactorDoc(
        'OPERATING_MARGIN', 'Operating Margin', 'Quality',
        'OPERATING_MARGIN = Operating Income (TTM) / Revenue (TTM)',
        'SEC OperatingIncomeLoss ÷ revenue', 'higher',
        'Profitability of the core business before financing and tax. A proxy '
        'for pricing power and cost discipline.',
        'Varies enormously by industry — software runs 30%+, grocery retail '
        'runs low single digits. Only comparable within a sector.'),

    'NET_MARGIN': FactorDoc(
        'NET_MARGIN', 'Net Margin', 'Quality',
        'NET_MARGIN = Net Income (TTM) / Revenue (TTM)',
        'SEC NetIncomeLoss ÷ revenue', 'higher',
        'What survives to shareholders from each dollar of sales, after every '
        'cost including tax and interest.',
        'Distorted by one-off gains, tax settlements and impairments.'),

    'BUYBACK_YIELD': FactorDoc(
        'BUYBACK_YIELD', 'Buyback Yield', 'Income',
        'BUYBACK_YIELD = Payments for Repurchase of Common Stock (TTM) / Market Cap',
        'SEC PaymentsForRepurchaseOfCommonStock', 'higher',
        'Capital returned through share repurchase. Economically equivalent to '
        'a dividend but taxed differently, so a screen that looks only at '
        'dividend yield misses half the shareholder return.',
        'Buybacks executed at high valuations destroy value; the yield alone '
        'does not say whether the price paid was sensible. Gross of issuance — '
        'a company buying back while issuing to staff nets out lower.'),

    'PCT_VS_MA50': FactorDoc(
        'PCT_VS_MA50', 'Price vs 50-Day MA', 'Trend',
        'PCT_VS_MA50 = P(t) / mean(P over last 50 sessions) − 1',
        'Adjusted closes', 'higher',
        'Intermediate trend. Faster than the 200-day, used to confirm that a '
        'name is not merely above its long-term average but currently advancing.',
        'Whipsaws more than the 200-day; a cross is not a signal on its own.'),

    'RETURN_1M': FactorDoc(
        'RETURN_1M', '1-Month Total Return', 'Momentum',
        'RETURN_1M = P(t) / P(t − 21 sessions) − 1',
        'Adjusted closes (dividends reinvested)', 'higher',
        'The most recent month of total return, read as continuation. This is '
        'the same series as REVERSAL_1M with the opposite sign, and which one '
        'is right is an empirical question rather than a settled one: the '
        'academic prior is short-term reversal, but 1-month momentum has been '
        'the stronger of the two in recent testing on this universe.',
        'The classic literature says the last month mean-reverts, which is why '
        'MOM_12_1 skips it. Using RETURN_1M is a bet against that prior — check '
        'it on the Backtest tab before relying on it, and note it turns over '
        'the portfolio far faster than a 6- or 12-month signal.'),

    'RETURN_6M': FactorDoc(
        'RETURN_6M', '6-Month Total Return', 'Momentum',
        'RETURN_6M = P(t) / P(t − 126 sessions) − 1',
        'Adjusted closes (dividends reinvested)', 'higher',
        'Half a year of total return, including the most recent month. Long '
        'enough to express a real trend and short enough to turn before the '
        'trend is exhausted; computed on adjusted closes, so it is a total '
        'return rather than price-only.',
        'Includes the last month, so it carries the short-term reversal '
        'contamination MOM_6_1 deliberately skips. The two are otherwise the '
        'same window and correlate heavily — weighting both double-counts one '
        'signal.'),

    'RETURN_3M': FactorDoc(
        'RETURN_3M', '3-Month Total Return', 'Momentum',
        'RETURN_3M = P(t) / P(t − 63 sessions) − 1',
        'Adjusted closes (dividends reinvested)', 'higher',
        'Recent total return. Used by the post-earnings-drift strategy, where '
        'the relevant window is the quarter since the last results filing.',
        'Includes the most recent month, so it carries some short-term reversal '
        'contamination that MOM_12_1 deliberately excludes.'),

    'EVENT_RESULTS': FactorDoc(
        'EVENT_RESULTS', '8-K Results Filings', 'Alt Data',
        'EVENT_RESULTS = count of 8-K filings carrying item 2.02\n'
        '                (Results of Operations) in the last 90 days',
        'SEC submissions item codes', 'higher',
        'A recent earnings release, identified from the SEC item code rather '
        'than by parsing text. Marks the start of the post-earnings-announcement '
        'drift window.',
        'Presence only — the code says results were filed, not whether they '
        'beat or missed. The price reaction carries that information.'),

    'EVENT_MANAGEMENT': FactorDoc(
        'EVENT_MANAGEMENT', '8-K Management Changes', 'Alt Data',
        'EVENT_MANAGEMENT = count of 8-K filings carrying item 5.02\n'
        '                   (Departure/Election of Directors or Officers)',
        'SEC submissions item codes', 'lower',
        'Executive and board turnover, captured deterministically from the '
        'filing rather than inferred from news. Frequent senior churn is a '
        'recognised instability signal.',
        'Direction-blind: hiring a well-regarded CEO and losing one both file '
        'under 5.02. Routine board retirements inflate the count.'),

    'EVENT_RESTRUCTURING': FactorDoc(
        'EVENT_RESTRUCTURING', '8-K Restructuring Charges', 'Alt Data',
        'EVENT_RESTRUCTURING = count of 8-K filings carrying item 2.05\n'
        '                      (Costs Associated with Exit or Disposal Activities)',
        'SEC submissions item codes', 'lower',
        'Formal restructuring or exit charges — layoffs, plant closures, '
        'discontinued lines. A company booking these is under cost pressure.',
        'Can precede a genuine turnaround as easily as a decline; treat as a '
        'flag to investigate, not a verdict.'),

    'CURRENT_RATIO': FactorDoc(
        'CURRENT_RATIO', 'Current Ratio', 'Quality',
        'CURRENT_RATIO = Current Assets / Current Liabilities',
        'SEC AssetsCurrent ÷ LiabilitiesCurrent',
        'higher', 'Short-term solvency — can the company cover the next year of '
        'obligations from liquid assets.',
        'A very high ratio can mean idle cash rather than strength.'),

    # ── GROWTH ───────────────────────────────────────────────────
    'REVENUE_CAGR_3Y': FactorDoc(
        'REVENUE_CAGR_3Y', 'Revenue CAGR (3Y)', 'Growth',
        'REVENUE_CAGR_3Y = (Revenue_now / Revenue_3y_ago)^(1/3) − 1',
        'SEC revenue, TTM now versus TTM three years back',
        'higher',
        'Compound top-line growth. Smoother than a single-year rate and less '
        'sensitive to one weak comparison quarter.',
        'Undefined when either endpoint is non-positive. Flatters acquisitive '
        'companies, since bought revenue counts the same as organic.'),

    'EARNINGS_GROWTH_1Y': FactorDoc(
        'EARNINGS_GROWTH_1Y', 'Earnings Growth (1Y)', 'Growth',
        'EARNINGS_GROWTH_1Y = (NI_now − NI_1y_ago) / |NI_1y_ago|',
        'SEC NetIncomeLoss, TTM versus prior TTM',
        'higher', 'Bottom-line momentum over the past year.',
        'Deliberately returns nothing when the sign flips: a company going from '
        'a $40M loss to a $100M profit is not "−250% growth", and reporting a '
        'number there would be worse than reporting none.'),

    'REVENUE_GROWTH_1Y': FactorDoc(
        'REVENUE_GROWTH_1Y', 'Revenue Growth (1Y)', 'Growth',
        'REVENUE_GROWTH_1Y = (Revenue_TTM − Revenue_TTM_1y_ago) / |Revenue_1y_ago|',
        'SEC Revenues / RevenueFromContractWithCustomer', 'higher',
        'Top-line growth over the last year on a trailing-twelve-month basis, '
        'so it compares like with like rather than a quarter against a year. '
        'The Style Factors screen offers it as the Growth proxy.',
        'Revenue growth says nothing about whether the growth earns its cost '
        'of capital — a company can buy revenue with margin. Returns nothing '
        'across a sign flip, and is the noisiest of the growth measures here '
        'because a single acquisition moves it.'),

    'EQUITY_CAGR_3Y': FactorDoc(
        'EQUITY_CAGR_3Y', 'Equity CAGR (3Y)', 'Growth',
        'EQUITY_CAGR_3Y = (Equity_now / Equity_3y_ago)^(1/3) − 1',
        'SEC StockholdersEquity',
        'higher', 'Book-value compounding — retained earnings accumulating.',
        'Buybacks shrink equity, so a company returning heavy capital can show '
        'negative equity CAGR while performing well.'),

    'EPS_CAGR_3Y': FactorDoc(
        'EPS_CAGR_3Y', 'EPS CAGR (3Y)', 'Growth',
        'EPS       = Net income (TTM) / Diluted shares\n'
        'EPS_CAGR_3Y = (EPS_now / EPS_3y_ago)^(1/3) − 1',
        'SEC NetIncomeLoss, WeightedAverageNumberOfDilutedSharesOutstanding',
        'higher', 'Durable earning power — is the business earning more per '
        'share than it did three years ago? Because it is per *share*, it '
        'credits buybacks and penalises dilution, and because it is a *rate* '
        'it is comparable across companies where raw EPS is not.',
        'Undefined through a loss year — a growth rate across a sign flip is '
        'meaningless — so it is NaN for about a quarter of the S&P 500 and '
        'names scored on it skew profitable. It also rewards a steady 8% '
        'compounder and a name that halved then quadrupled identically; '
        'consistency is not measured here.'),

    # ── MOMENTUM ─────────────────────────────────────────────────
    'MOM_12_1': FactorDoc(
        'MOM_12_1', 'Momentum 12-1', 'Momentum',
        'MOM_12_1 = P(t − 21 sessions) / P(t − 252 sessions) − 1\n'
        '           (12-month return, skipping the most recent month)',
        'Split- and dividend-adjusted closes',
        'higher',
        'The standard academic momentum factor. The most recent month is '
        'deliberately skipped because short-horizon returns exhibit *reversal*, '
        'not continuation — including it contaminates the signal with the '
        'opposite effect.',
        'Crashes hard at sharp market turns, when yesterday\'s winners fall '
        'fastest. The 200-day trend filter in the mid-term strategies exists '
        'mainly to blunt this.'),

    'MOM_6_1': FactorDoc(
        'MOM_6_1', 'Momentum 6-1', 'Momentum',
        'MOM_6_1 = P(t − 21) / P(t − 126) − 1',
        'Adjusted closes', 'higher',
        'Shorter-horizon momentum. Reacts faster than 12-1, at the cost of more '
        'turnover.', 'Noisier; more sensitive to single events.'),

    'MOM_VOL_ADJ': FactorDoc(
        'MOM_VOL_ADJ', 'Volatility-Adjusted Momentum', 'Momentum',
        'MOM_VOL_ADJ = MOM_12_1 / annualized volatility (1Y)',
        'Adjusted closes', 'higher',
        'Return per unit of risk taken to earn it. Prevents the ranking from '
        'simply selecting the most volatile names, which raw momentum tends to do.',
        'Penalizes high-beta names in a strong bull market.'),

    'REVERSAL_1M': FactorDoc(
        'REVERSAL_1M', 'Short-Term Reversal (1M)', 'Momentum',
        'REVERSAL_1M = − ( P(t) / P(t − 21 sessions) − 1 )',
        'Adjusted closes', 'higher',
        'Deliberately sign-flipped: recent one-month losers tend to outperform '
        'over the following weeks. This is the factor the original tool used as '
        '"momentum" — with the wrong sign, and the wrong lookback (35 rows was '
        'called "one month" but spans about seven weeks).',
        'Fragile in trending markets, and the effect is largely arbitraged in '
        'large caps.'),

    'PCT_FROM_52W_HIGH': FactorDoc(
        'PCT_FROM_52W_HIGH', 'Distance from 52-Week High', 'Momentum',
        'PCT_FROM_52W_HIGH = P(t) / max(P over 252 sessions) − 1     (≤ 0)',
        'Adjusted closes', 'higher',
        'Proximity to the yearly high. George & Hwang showed this carries '
        'momentum information independent of past return — anchoring near the '
        'high predicts continuation.',
        'By construction never positive; a value near 0 means "at the high".'),

    'PCT_VS_MA200': FactorDoc(
        'PCT_VS_MA200', 'Price vs 200-Day MA', 'Trend',
        'PCT_VS_MA200 = P(t) / mean(P over last 200 sessions) − 1',
        'Adjusted closes', 'higher',
        'The primary trend filter. Above the 200-day is the conventional '
        'definition of an uptrend and is used as a hard gate by several '
        'mid-term strategies.',
        'Whipsaws in range-bound markets, generating repeated false signals.'),

    # ── RISK ─────────────────────────────────────────────────────
    'VOL_1Y': FactorDoc(
        'VOL_1Y', 'Annualized Volatility (1Y)', 'Risk',
        'VOL_1Y = sd(daily returns over 252 sessions) × √252',
        'Adjusted closes', 'lower',
        'Realized risk. The low-volatility anomaly is the persistent finding '
        'that low-risk stocks have delivered better risk-adjusted returns than '
        'high-risk ones — the opposite of what CAPM predicts.',
        'Backward-looking: volatility regimes shift, and a quiet past year does '
        'not guarantee a quiet next one.'),

    'BETA': FactorDoc(
        'BETA', 'Beta vs SPY', 'Risk',
        'BETA = Cov(r_stock, r_market) / Var(r_market)      over 252 sessions',
        'Daily returns of the stock and of SPY', 'lower',
        'Sensitivity to the market. Beta below 1 means the stock has historically '
        'moved less than the index.',
        'Unstable over time and estimated from a single year — treat as an '
        'approximation, not a constant.'),

    'IDIO_VOL': FactorDoc(
        'IDIO_VOL', 'Idiosyncratic Volatility', 'Risk',
        'residual = r_stock − β · r_market\n'
        'IDIO_VOL = sd(residual) × √252',
        'Daily returns versus SPY', 'lower',
        'Stock-specific risk, with market movement stripped out. High idio vol '
        'has historically predicted poor returns.',
        'Sensitive to the beta estimate feeding it.'),

    'MAX_DD_1Y': FactorDoc(
        'MAX_DD_1Y', 'Maximum Drawdown (1Y)', 'Risk',
        'MAX_DD_1Y = min( P(t) / running_max(P) − 1 )    over 252 sessions',
        'Adjusted closes', 'higher',
        'Worst peak-to-trough loss over the past year (less negative is better). '
        'Closer to the pain an investor actually experiences than volatility is.',
        'A single event can dominate it.'),

    # ── INCOME ───────────────────────────────────────────────────
    'DIVIDEND_YIELD': FactorDoc(
        'DIVIDEND_YIELD', 'Dividend Yield', 'Income',
        'DIVIDEND_YIELD = Annual Dividends per Share / Price',
        'Provider profile snapshot', 'higher',
        'Cash return to shareholders.',
        'A very high yield usually signals a price that has collapsed on an '
        'expectation the dividend will be cut — pair it with payout and FCF.'),

    'PAYOUT_RATIO': FactorDoc(
        'PAYOUT_RATIO', 'Payout Ratio', 'Income',
        'PAYOUT_RATIO = Dividends Paid / Net Income',
        'Provider profile snapshot', 'lower',
        'Dividend sustainability. A low ratio leaves room to keep paying '
        'through a bad year.',
        'Above 1 means the dividend exceeds earnings — funded from cash or debt, '
        'and rarely sustainable.'),

    # ── ALTERNATIVE DATA ─────────────────────────────────────────
    'INSIDER_NET_BUY': FactorDoc(
        'INSIDER_NET_BUY', 'Insider Net Buying', 'Alt Data',
        'INSIDER_NET_BUY = (Σ buy value − Σ sell value) / (Σ buy + Σ sell)\n'
        '                  over Form 4 filings in the last 180 days,\n'
        '                  open-market transactions only (codes P and S).\n'
        '                  Likely ESPP purchases are filtered out (recurring\n'
        '                  quarterly buys under $25k by the same insider).\n'
        '                  Sells are conviction-weighted by % of holdings\n'
        '                  disposed (via post_txn_shares from Form 4).',
        'SEC Form 4 filings', 'higher',
        'Insiders trade with better information than anyone else. Restricted to '
        'open-market purchases and sales — grants, option exercises and '
        'tax-withholding disposals (codes A, M, F) are compensation mechanics, '
        'not opinions. ESPP payroll purchases (code P but small, recurring, '
        'quarterly) are heuristically removed since they reflect a standing '
        'enrollment, not a conviction buy.',
        'Selling is far noisier than buying: executives diversify and pay tax '
        'bills for reasons unrelated to their view. To reduce this noise, sell '
        'signals are scaled by the fraction of total holdings disposed — a CEO '
        'selling 50% of their stake is a much stronger signal than one trimming '
        '2% for tax purposes.'),

    'INSIDER_CLUSTER': FactorDoc(
        'INSIDER_CLUSTER', 'Insider Cluster Buying', 'Alt Data',
        'INSIDER_CLUSTER = count of distinct insiders making open-market\n'
        '                  purchases in the last 180 days',
        'SEC Form 4 filings', 'higher',
        'Several executives buying independently is a much stronger signal than '
        'one — it is harder to explain away as a personal liquidity decision.',
        'Larger boards mechanically produce larger counts.'),

    'EVENT_INTENSITY_90D': FactorDoc(
        'EVENT_INTENSITY_90D', '8-K Event Intensity', 'Alt Data',
        'EVENT_INTENSITY_90D = count of 8-K filings in the last 90 days',
        'SEC submissions (item codes)', 'lower',
        'Filing velocity as a proxy for corporate turbulence. Elevated 8-K '
        'activity often accompanies instability.',
        'Direction-blind: a flurry of 8-Ks could be a transformative acquisition '
        'or an accounting problem.'),

    'ATTENTION_Z': FactorDoc(
        'ATTENTION_Z', 'Retail Attention (Wikipedia)', 'Alt Data',
        'ATTENTION_Z = (mean views last 14d − mean views prior period)\n'
        '              / sd(views over prior period)',
        'Wikimedia pageviews API', 'higher',
        'Retail attention proxy. Unlike news sentiment this has full daily '
        'history, so it is genuinely backtestable.',
        'Spikes on news of any kind, good or bad. Ambiguous in isolation.'),

    'RSI_14': FactorDoc(
        'RSI_14', 'Relative Strength Index (14)', 'Technical',
        'Wilder smoothing, α = 1/14:\n'
        '  avg_gain = EWM(max(ΔP, 0), α)\n'
        '  avg_loss = EWM(max(−ΔP, 0), α)\n'
        '  RS       = avg_gain / avg_loss\n'
        '  RSI      = 100 − 100 / (1 + RS)',
        'Adjusted closes', 'higher',
        'Overbought/oversold oscillator, used as a gate rather than a ranking '
        'factor. Wilder\'s exponential smoothing is what makes this match '
        'published RSI values — a simple rolling mean produces a different '
        'number and matches nothing.',
        'Can stay above 70 for a long time in a strong trend; overbought is not '
        'a sell signal on its own.'),

    'ATR_14': FactorDoc(
        'ATR_14', 'Average True Range (14)', 'Technical',
        'TR  = max(High − Low, |High − Close_prev|, |Low − Close_prev|)\n'
        'ATR = EWM(TR, α = 1/14)',
        'Adjusted high, low and close', 'lower',
        'The volatility unit for stop placement and position sizing. Computed '
        'on the adjusted price basis so a stop set N ATRs below an adjusted '
        'price is measured in the same units as that price.',
        'Widens after a shock, which mechanically loosens stops just when risk '
        'is highest.'),
}


# ─────────────────────────────────────────────
# TRADE-PLAN MATHEMATICS  (mid-term horizon)
# ─────────────────────────────────────────────

TRADE_PLAN_MATH = r"""
Mid-term strategies emit a full plan. Every level below is mechanical.

STOP — the wider of two candidates

    atr_stop   = entry − k · ATR(14),        k = 2.0 by default
    swing_stop = lowest low of the last 20 sessions × 0.995
    stop       = min(atr_stop, swing_stop)          i.e. the further one

  The tighter stop backtests better and gets stopped out by ordinary noise in
  practice. Taking the wider level respects both the stock's own volatility
  and its market structure.

ENTRY

    momentum / pullback setups:  entry = price − 0.5 · ATR   (limit, not chasing)
    breakout setups:             entry = price               (market, buying strength)

TARGET — a multiple of the risk taken

    risk_per_share = entry − stop
    target         = entry + R · risk_per_share,     R = 2.5 by default

POSITION SIZE — fixed fractional, derived from the stop

    stop_distance_% = (entry − stop) / entry × 100
    weight_%        = risk_budget_% / (stop_distance_% / 100)
    weight_%        = clip(weight_%, 1%, 10%)

  With a 1% risk budget, a name needing a 12.7% stop earns a 7.9% position,
  while one with a 4% stop is capped at 10%. Risk per position is equalized;
  position size is the output, not an input.

REJECTION

    if stop_distance_% > 15%:  no plan is emitted

  Sizing down far enough to keep the risk budget would leave a position too
  small to matter.

EXITS — whichever comes first

    stop hit          intraday low ≤ stop
    target hit        intraday high ≥ target
    time stop         ~60 trading days elapsed with neither hit
    trend break       close below the 200-day moving average

  When a single day trades through both stop and target, daily bars cannot
  reveal the order, so the backtest assumes the stop. That is the pessimistic
  reading, chosen so results are not flattered by an ambiguity.

PORTFOLIO RISK

    portfolio_risk_% = Σ (weight_i × stop_distance_i)

  What you lose if every open position stops out at once.
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def doc_for(factor: str) -> FactorDoc | None:
    return FACTOR_DOCS.get(factor)


def by_family() -> dict[str, list[FactorDoc]]:
    out: dict[str, list[FactorDoc]] = {}
    for d in FACTOR_DOCS.values():
        out.setdefault(d.family, []).append(d)
    return out


def strategy_breakdown(strategy) -> list[dict]:
    """
    Per-factor weight table for one strategy, with documentation attached.

    Weights are shown both raw and normalized, because the normalized figure
    is what actually determines influence on the composite.
    """
    total = sum(strategy.weights.values()) or 1.0
    rows = []
    for name, weight in sorted(strategy.weights.items(),
                               key=lambda kv: -kv[1]):
        d = FACTOR_DOCS.get(name)
        rows.append({
            'factor': name,
            'label': d.label if d else name,
            'family': d.family if d else '—',
            'weight': weight,
            'weight_pct': round(100 * weight / total, 1),
            'direction': ('higher is better' if not d else
                          f'{d.direction} is better'),
            'formula': d.formula if d else 'Not documented.',
            'inputs': d.inputs if d else '',
            'rationale': d.rationale if d else '',
            'caveat': d.caveat if d else '',
        })
    return rows
