"""
XBRL tag mappings — which us-gaap and dei tags represent each financial concept.

Filers disagree about which tag to use. A concept like 'revenue' maps to a
priority-ordered list of candidate tags because one issuer files under
RevenueFromContractWithCustomerExcludingAssessedTax and another under
SalesRevenueNet. The selection logic in fundamentals._single_tag() picks
one per filer; this module defines the candidate set.

Extracted from config.py so that tag maps are independently testable and
reviewable — they change when FASB updates the taxonomy, not when a
threshold or path is tuned.
"""

from __future__ import annotations

# The tags we extract from companyfacts. Each logical concept lists candidate
# us-gaap tags in priority order, because filers disagree about which to use.
SEC_TAG_MAP: dict[str, list[str]] = {
    'revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet',
        'RevenuesNetOfInterestExpense',
        'InterestAndDividendIncomeOperating',
        'RealEstateRevenueNet',
        'OperatingLeasesIncomeStatementLeaseRevenue',
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
    'total_debt': [
        'DebtLongtermAndShorttermCombinedAmount',
        'DebtAndCapitalLeaseObligations',
        'LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities',
    ],
    'long_term_debt': [
        'LongTermDebtNoncurrent', 'LongTermDebt',
        'LongTermDebtAndCapitalLeaseObligations', 'LongTermNotesPayable',
    ],
    'short_term_debt': [
        'ShortTermBorrowings', 'DebtCurrent',
        'LongTermDebtAndCapitalLeaseObligationsCurrent',
    ],
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

# The `dei` (Document & Entity Information) namespace carries the cover-page
# share count — the exact number outstanding on the filing date.
SEC_DEI_TAG_MAP: dict[str, list[str]] = {
    'shares_outstanding': ['EntityCommonStockSharesOutstanding'],
}


def all_tags() -> set[str]:
    """Every XBRL tag referenced by the maps, for validation."""
    tags: set[str] = set()
    for lst in SEC_TAG_MAP.values():
        tags.update(lst)
    for lst in SEC_DEI_TAG_MAP.values():
        tags.update(lst)
    return tags


def concepts() -> list[str]:
    """All logical concept names, sorted."""
    return sorted(set(SEC_TAG_MAP) | set(SEC_DEI_TAG_MAP))
