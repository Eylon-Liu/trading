"""
Central configuration for the Quant Research Terminal.

Everything tunable lives here so strategy code never hardcodes a threshold.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()
DB_DIR = ROOT / 'db'
CACHE_DIR = ROOT / 'cache'
REPORT_DIR = ROOT / 'reports' / 'out'
DATA_DIR = ROOT / 'data' / 'static'

for _d in (DB_DIR, CACHE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# SQLAlchemy URL. Default is a local SQLite file; swap for a hosted libSQL/
# Postgres URL via the env var without touching any other code.
DATABASE_URL = os.environ.get('DATABASE_URL', f'sqlite:///{DB_DIR / "quant.sqlite"}')


# ─────────────────────────────────────────────
# HTTP / RATE LIMITING
# ─────────────────────────────────────────────

# SEC requires a descriptive User-Agent with contact info, and throttles at
# 10 req/s. We stay well under.
SEC_USER_AGENT = os.environ.get(
    'SEC_USER_AGENT', 'QuantResearchTerminal research@example.com'
)
BROWSER_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'

# Requests per second, per host. Yahoo is the fragile one — it rate-limited us
# during a handful of probe calls, so we keep the tap narrow and lean on cache.
RATE_LIMITS = {
    'data.sec.gov': 8.0,
    'www.sec.gov': 8.0,
    'efts.sec.gov': 5.0,
    'query1.finance.yahoo.com': 2.0,
    'query2.finance.yahoo.com': 2.0,
    'feeds.finance.yahoo.com': 2.0,
    'en.wikipedia.org': 5.0,
    'wikimedia.org': 5.0,
    'news.google.com': 2.0,
    'www.federalregister.gov': 5.0,
    'api.stlouisfed.org': 5.0,
    '_default': 3.0,
}

HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 4
HTTP_BACKOFF_BASE = 1.5  # seconds; exponential
CACHE_TTL_SECONDS = {
    'prices': 60 * 60 * 6,
    'sec_facts': 60 * 60 * 24 * 7,
    'submissions': 60 * 60 * 24,
    'profile': 60 * 60 * 12,
    'news': 60 * 30,
    'policy': 60 * 60 * 6,
    'members': 60 * 60 * 24 * 7,
    '_default': 60 * 60,
}

MAX_WORKERS = 8  # concurrent fetches; SEC tolerates more than Yahoo


# ─────────────────────────────────────────────
# SECTOR NORMALIZATION  (fixes the silent gray-bar bug)
# ─────────────────────────────────────────────
#
# yfinance returns its own sector vocabulary ("Financial Services",
# "Consumer Cyclical"), while GICS — and the old SECTOR_COLORS dict — used
# "Financials" and "Consumer Discretionary". Unmapped names fell through to
# gray, which is why most bars rendered the same color. Normalize everything
# to GICS on the way in so downstream code only ever sees one vocabulary.

SECTOR_ALIASES = {
    # yfinance vocabulary -> GICS
    'technology': 'Information Technology',
    'information technology': 'Information Technology',
    'communication services': 'Communication Services',
    'consumer cyclical': 'Consumer Discretionary',
    'consumer discretionary': 'Consumer Discretionary',
    'consumer defensive': 'Consumer Staples',
    'consumer staples': 'Consumer Staples',
    'financial services': 'Financials',
    'financial': 'Financials',
    'financials': 'Financials',
    'healthcare': 'Health Care',
    'health care': 'Health Care',
    'industrials': 'Industrials',
    'industrial goods': 'Industrials',
    'basic materials': 'Materials',
    'materials': 'Materials',
    'energy': 'Energy',
    'utilities': 'Utilities',
    'real estate': 'Real Estate',
}

GICS_SECTORS = [
    'Information Technology',
    'Health Care',
    'Financials',
    'Consumer Discretionary',
    'Communication Services',
    'Industrials',
    'Consumer Staples',
    'Energy',
    'Utilities',
    'Materials',
    'Real Estate',
    'Unknown',
]


def normalize_sector(raw: str | None) -> str:
    """Map any provider's sector string onto the GICS vocabulary."""
    if not raw:
        return 'Unknown'
    return SECTOR_ALIASES.get(str(raw).strip().lower(), 'Unknown')


# ─────────────────────────────────────────────
# UNIVERSE / LIQUIDITY SCREENS
# ─────────────────────────────────────────────

INDEX_OPTIONS = {
    'SPY': 'S&P 500',
    'QQQ': 'NASDAQ 100',
    'DIA': 'Dow Jones 30',
    'IWM': 'Russell 2000 (sample)',
}

# CIKs of the ETF trusts, for authoritative N-PORT holdings.
INDEX_ETF_CIK = {
    'SPY': '0000884394',  # SPDR S&P 500 ETF Trust
    'QQQ': '0001067839',  # Invesco QQQ Trust, Series 1
}

# Wikipedia pages carrying the constituent tables (used with the MediaWiki
# revision API to reconstruct point-in-time membership).
INDEX_WIKI_PAGE = {
    'SPY': 'List of S&P 500 companies',
    'QQQ': 'Nasdaq-100',
    'DIA': 'Dow Jones Industrial Average',
}

# Defaults for the tradability screen. A signal you cannot fill is not a signal.
MIN_MARKET_CAP = 300_000_000     # USD
MIN_DOLLAR_ADV = 1_000_000       # USD, 20-day average
MIN_PRICE = 3.0                  # USD, avoids sub-penny microstructure
ADV_WINDOW = 20


# ─────────────────────────────────────────────
# FACTOR CONSTRUCTION
# ─────────────────────────────────────────────

WINSORIZE_PCT = 0.01        # clip at 1st / 99th percentile before standardizing
MIN_SECTOR_MEMBERS = 5      # below this, fall back to size/sector regression
MIN_FACTOR_COVERAGE = 0.5   # a name needs >= half its factors to be scored
MAD_SCALE = 1.4826          # MAD -> sigma for a normal distribution

# Trading-day conventions. The old code used 35 rows for "1 month" (~7 weeks).
TRADING_DAYS = {
    '1W': 5, '1M': 21, '3M': 63, '6M': 126,
    '1Y': 252, '2Y': 504, '3Y': 756, '5Y': 1260,
}
LOOKBACK_OPTIONS = ['1M', '3M', '6M', '1Y', '3Y', '5Y']
REBALANCE_OPTIONS = {'M': 'Monthly', 'Q': 'Quarterly', 'SA': 'Semi-Annual', 'A': 'Annual'}


# ─────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────

BENCHMARK_TICKER = 'SPY'
RISK_FREE_SERIES = 'DGS3MO'      # FRED 3-month T-bill
TRANSACTION_COST_BPS = 10.0      # one-way, per trade
SLIPPAGE_BPS = 5.0
DEFAULT_PORTFOLIO_SIZE = 20
IC_HORIZONS = [1, 3, 6, 12]      # months, for IC-decay analysis
QUINTILES = 5


# ─────────────────────────────────────────────
# SEC / XBRL
# ─────────────────────────────────────────────

# The tags we extract from companyfacts. Each logical concept lists candidate
# us-gaap tags in priority order, because filers disagree about which to use.
SEC_TAG_MAP = {
    'revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet',
        # Banks and REITs report revenue under none of the tags above. Without
        # these, Regions Financial showed $0.10B of revenue against $2.23B of
        # net income — a net margin of 2,146% — and Camden Property $0.01B.
        'RevenuesNetOfInterestExpense',
        'InterestAndDividendIncomeOperating',
        'RealEstateRevenueNet',
        'OperatingLeasesIncomeStatementLeaseRevenue',
        # ASC 842. REITs moved rental income here in 2019, which is why Camden
        # Property's current contract-with-customer tag holds only $5M of
        # ancillary revenue against ~$390M of actual rent.
        'OperatingLeaseLeaseIncome',
    ],
    'net_income': ['NetIncomeLoss', 'ProfitLoss'],
    'operating_income': ['OperatingIncomeLoss'],
    'gross_profit': ['GrossProfit'],
    'assets': ['Assets'],
    'current_assets': ['AssetsCurrent'],
    'liabilities': ['Liabilities'],
    'current_liabilities': ['LiabilitiesCurrent'],
    'equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        'CommonStockholdersEquity',
    ],
    'cash': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
    ],
    'operating_cash_flow': ['NetCashProvidedByUsedInOperatingActivities'],
    'capex': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsToAcquireProductiveAssets',
    ],
    'long_term_debt': ['LongTermDebtNoncurrent', 'LongTermDebt'],
    'short_term_debt': ['ShortTermBorrowings', 'DebtCurrent'],
    'shares_diluted': ['WeightedAverageNumberOfDilutedSharesOutstanding'],
    'shares_basic': ['WeightedAverageNumberOfSharesOutstandingBasic'],
    'eps_diluted': ['EarningsPerShareDiluted'],
    'interest_expense': ['InterestExpense', 'InterestIncomeExpenseNet'],
    'rnd': ['ResearchAndDevelopmentExpense'],
    'dividends_paid': [
        'PaymentsOfDividendsCommonStock', 'PaymentsOfDividends',
    ],
    'buybacks': ['PaymentsForRepurchaseOfCommonStock'],
    'inventory': ['InventoryNet', 'InventoryGross',
                  'InventoryFinishedGoodsNetOfReserves'],
}

# The `dei` (Document & Entity Information) namespace, which arrives in the
# same companyfacts payload as us-gaap and used to be discarded.
#
# EntityCommonStockSharesOutstanding is the cover-page share count — the exact
# number of shares outstanding on the filing date, not the weighted average
# used as an EPS denominator. It is what market capitalisation actually means,
# and it carries a `filed` date like everything else, so it stays point-in-time.
SEC_DEI_TAG_MAP = {
    'shares_outstanding': ['EntityCommonStockSharesOutstanding'],
}

# 8-K item codes worth treating as structured corporate events. These need no
# NLP at all — the SEC already classified them.
EIGHTK_ITEMS = {
    '1.01': ('strategy', 'Material definitive agreement'),
    '1.02': ('strategy', 'Termination of material agreement'),
    '2.01': ('strategy', 'Completion of acquisition or disposition'),
    '2.02': ('results', 'Results of operations'),
    '2.05': ('restructuring', 'Costs associated with exit or disposal'),
    '2.06': ('impairment', 'Material impairment'),
    '3.01': ('listing', 'Delisting / listing-rule noncompliance'),
    '4.01': ('audit', 'Change in certifying accountant'),
    '4.02': ('audit', 'Non-reliance on prior financials'),
    '5.02': ('management', 'Director / officer departure or appointment'),
    '5.07': ('governance', 'Shareholder vote results'),
    '7.01': ('disclosure', 'Regulation FD disclosure'),
    '8.01': ('disclosure', 'Other events'),
}

# Form 4 transaction codes. P and S are the open-market ones that carry signal;
# A/M/G are grants and option mechanics and are mostly noise.
INSIDER_CODES = {
    'P': ('buy', 'Open-market purchase'),
    'S': ('sell', 'Open-market sale'),
    'A': ('grant', 'Grant or award'),
    'M': ('exercise', 'Option exercise'),
    'G': ('gift', 'Gift'),
    'F': ('tax', 'Shares withheld for tax'),
}
INSIDER_SIGNAL_CODES = {'P', 'S'}


# ─────────────────────────────────────────────
# POLICY  (Federal Register agency -> affected GICS sectors)
# ─────────────────────────────────────────────

AGENCY_SECTOR_MAP = {
    'environmental-protection-agency': ['Energy', 'Utilities', 'Materials'],
    'energy-department': ['Energy', 'Utilities'],
    'federal-energy-regulatory-commission': ['Utilities', 'Energy'],
    'food-and-drug-administration': ['Health Care'],
    'health-and-human-services-department': ['Health Care'],
    'centers-for-medicare-medicaid-services': ['Health Care'],
    'federal-communications-commission': ['Communication Services', 'Information Technology'],
    'securities-and-exchange-commission': ['Financials'],
    'federal-reserve-system': ['Financials'],
    'treasury-department': ['Financials'],
    'internal-revenue-service': ['Financials'],
    'comptroller-of-the-currency': ['Financials'],
    'federal-deposit-insurance-corporation': ['Financials'],
    'consumer-financial-protection-bureau': ['Financials'],
    'industry-and-security-bureau': ['Information Technology'],
    'commerce-department': ['Information Technology', 'Industrials'],
    'transportation-department': ['Industrials', 'Consumer Discretionary'],
    'national-highway-traffic-safety-administration': ['Consumer Discretionary'],
    'federal-aviation-administration': ['Industrials'],
    'agriculture-department': ['Consumer Staples', 'Materials'],
    'labor-department': ['Industrials', 'Consumer Discretionary'],
    'defense-department': ['Industrials'],
    'homeland-security-department': ['Industrials'],
    'federal-trade-commission': ['Information Technology', 'Health Care', 'Consumer Discretionary'],
    'justice-department': ['Information Technology', 'Health Care'],
    'housing-and-urban-development-department': ['Real Estate', 'Financials'],
    'interior-department': ['Energy', 'Materials'],
}

# Theme keywords scanned across policy titles and abstracts.
POLICY_THEMES = {
    'tariff': ['tariff', 'duty', 'duties', 'import restriction', 'trade remedy', 'antidumping'],
    'export_control': ['export control', 'entity list', 'export administration', 'sanction'],
    'antitrust': ['antitrust', 'merger review', 'competition', 'monopol'],
    'subsidy': ['subsidy', 'subsidies', 'tax credit', 'grant program', 'incentive'],
    'tax': ['tax', 'withholding', 'revenue procedure', 'internal revenue'],
    'energy': ['emission', 'renewable', 'clean energy', 'drilling', 'pipeline', 'greenhouse'],
    'healthcare': ['medicare', 'medicaid', 'drug pricing', 'clinical', 'reimbursement'],
    'labor': ['overtime', 'minimum wage', 'labor standard', 'union', 'workplace safety'],
    'privacy': ['privacy', 'data protection', 'consumer data'],
    'ai': ['artificial intelligence', 'machine learning', 'automated system'],
}


# ─────────────────────────────────────────────
# SIGNALS
# ─────────────────────────────────────────────

SIGNAL_LABELS = [
    (5, '🟢 Strong Buy'),
    (3, '🟢 Buy'),
    (1, '🟡 Hold / Accumulate'),
    (-1, '⚪ Neutral'),
    (-3, '🟠 Reduce'),
    (float('-inf'), '🔴 Sell / Avoid'),
]


def signal_label(score: float) -> str:
    """Map a raw signal score onto its display label."""
    for threshold, label in SIGNAL_LABELS:
        if score >= threshold:
            return label
    return SIGNAL_LABELS[-1][1]


# ─────────────────────────────────────────────
# NEWS / LLM
# ─────────────────────────────────────────────

NEWS_LOOKBACK_DAYS = 30
MAX_NEWS_PER_TICKER = 25
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
LLM_MODEL = os.environ.get('LLM_MODEL', 'claude-opus-5')

# Google AI Studio. GOOGLE_API_KEY is accepted as an alias because that is
# what the Google SDKs read by default.
GEMINI_API_KEY = (os.environ.get('GEMINI_API_KEY')
                  or os.environ.get('GOOGLE_API_KEY'))
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.5-flash')

# 'low' keeps latency and cost down; the reasoning here is interpretation of
# numbers that are already computed, not derivation.
GEMINI_THINKING_LEVEL = os.environ.get('GEMINI_THINKING_LEVEL', 'low')

# auto | gemini | anthropic | off
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'auto')

LLM_MAX_ARTICLES = int(os.environ.get('LLM_MAX_ARTICLES', '12'))


def llm_available() -> bool:
    """True when an implemented LLM backend is configured.

    Both backends live in nlp/providers.py. A key for any other vendor is not
    a substitute, so it is not accepted here — silently ignoring an unusable
    key is how you end up debugging a panel that never renders.
    """
    if (LLM_PROVIDER or 'auto').strip().lower() == 'off':
        return False
    return bool(ANTHROPIC_API_KEY or GEMINI_API_KEY)


# ─────────────────────────────────────────────
# EMAIL / REPORTS
# ─────────────────────────────────────────────

SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USER = (os.environ.get('SMTP_USER') or '').strip() or None

# Google presents an App Password as four groups of four ("abcd efgh ijkl
# mnop"). The spaces are display only — SMTP AUTH expects the 16 characters —
# so they are stripped here rather than leaving the user to discover that a
# copy-paste straight from Google fails authentication.
SMTP_PASS = (os.environ.get('SMTP_PASS') or '').replace(' ', '').strip() or None
FRED_API_KEY = os.environ.get('FRED_API_KEY')


# ─────────────────────────────────────────────
# UI THEME
# ─────────────────────────────────────────────

SECTOR_COLORS = {
    'Information Technology': '#00d4ff',
    'Health Care': '#38ef7d',
    'Financials': '#f7c59f',
    'Consumer Discretionary': '#ff6b9d',
    'Industrials': '#ffd93d',
    'Communication Services': '#c77dff',
    'Consumer Staples': '#06d6a0',
    'Energy': '#ff9f43',
    'Utilities': '#74b9ff',
    'Materials': '#fd79a8',
    'Real Estate': '#a29bfe',
    'Unknown': '#636e72',
}
