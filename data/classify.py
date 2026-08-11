"""
SIC -> GICS sector classification.

The securities table originally depended on yfinance `.info` for sector and
market cap. That endpoint is the single most rate-limited thing in the whole
provider set — during this project's first ingest it returned nothing at all,
leaving every name classified "Unknown" and collapsing the sector-relative
z-scores into one meaningless bucket.

SEC submissions carry an SIC code for every filer, cost one request we already
make, and never rate-limit. SIC is coarser than GICS and the mapping is
imperfect at the margins, but a mostly-right sector beats a universally
"Unknown" one, and yfinance can still enrich on top when it is available.
"""

from __future__ import annotations

# Scanned in order, first match wins, so narrow ranges MUST precede the broad
# ones they sit inside. Getting this backwards put Procter & Gamble (SIC 2840,
# soap and detergents) and Nike (SIC 3021, rubber footwear) in Materials
# alongside chemicals and rubber producers.
SIC_RANGES: list[tuple[int, int, str]] = [
    # ── narrow overrides first ────────────────────────────────────
    (2840, 2844, 'Consumer Staples'),      # soap, detergents, cosmetics
    (3021, 3021, 'Consumer Discretionary'),  # rubber footwear
    (3140, 3149, 'Consumer Discretionary'),  # footwear
    (3011, 3011, 'Consumer Discretionary'),  # tyres
    # Energy
    (1300, 1399, 'Energy'), (2900, 2999, 'Energy'), (4610, 4619, 'Energy'),
    # Materials
    (1000, 1099, 'Materials'), (1400, 1499, 'Materials'),
    (2600, 2699, 'Materials'), (2800, 2829, 'Materials'),
    (2845, 2899, 'Materials'), (3000, 3099, 'Materials'),
    (3200, 3299, 'Materials'), (3300, 3399, 'Materials'),
    # Health Care
    (2830, 2836, 'Health Care'), (3826, 3829, 'Health Care'),
    (3840, 3851, 'Health Care'), (5047, 5047, 'Health Care'),
    (5122, 5122, 'Health Care'), (5912, 5912, 'Health Care'),
    (8000, 8099, 'Health Care'), (8731, 8731, 'Health Care'),
    # Information Technology
    (3570, 3579, 'Information Technology'),
    (3600, 3629, 'Information Technology'),
    (3670, 3679, 'Information Technology'),
    (3690, 3699, 'Information Technology'),
    (3825, 3825, 'Information Technology'),
    (3861, 3861, 'Information Technology'),
    (5045, 5045, 'Information Technology'),
    (7370, 7379, 'Information Technology'),
    (7389, 7389, 'Information Technology'),
    # Communication Services
    (2711, 2799, 'Communication Services'),
    (4810, 4899, 'Communication Services'),
    (7310, 7319, 'Communication Services'),
    (7812, 7841, 'Communication Services'),
    (7900, 7999, 'Communication Services'),
    # Consumer Staples
    (100, 999, 'Consumer Staples'), (2000, 2199, 'Consumer Staples'),
    (2840, 2844, 'Consumer Staples'), (5140, 5149, 'Consumer Staples'),
    (5400, 5499, 'Consumer Staples'), (5912, 5912, 'Consumer Staples'),
    # Consumer Discretionary
    (2300, 2399, 'Consumer Discretionary'),
    (2500, 2599, 'Consumer Discretionary'),
    (3630, 3669, 'Consumer Discretionary'),
    (3700, 3716, 'Consumer Discretionary'),
    (3900, 3999, 'Consumer Discretionary'),
    (5200, 5399, 'Consumer Discretionary'),
    (5500, 5799, 'Consumer Discretionary'),
    (5900, 5911, 'Consumer Discretionary'),
    (5913, 5999, 'Consumer Discretionary'),
    (7000, 7099, 'Consumer Discretionary'),
    (7500, 7599, 'Consumer Discretionary'),
    (8200, 8299, 'Consumer Discretionary'),
    # Industrials
    (1500, 1799, 'Industrials'), (2400, 2499, 'Industrials'),
    (3400, 3499, 'Industrials'), (3500, 3569, 'Industrials'),
    (3580, 3599, 'Industrials'), (3717, 3799, 'Industrials'),
    (3800, 3824, 'Industrials'), (4000, 4499, 'Industrials'),
    (4500, 4599, 'Industrials'), (4700, 4789, 'Industrials'),
    (5000, 5044, 'Industrials'), (5046, 5046, 'Industrials'),
    (5048, 5099, 'Industrials'), (7340, 7359, 'Industrials'),
    (8700, 8730, 'Industrials'), (8732, 8799, 'Industrials'),
    # Financials
    (6000, 6199, 'Financials'), (6200, 6299, 'Financials'),
    (6300, 6411, 'Financials'), (6700, 6799, 'Financials'),
    # Real Estate
    (6500, 6599, 'Financials'),          # overridden below for REITs
    (6798, 6798, 'Real Estate'),
    # Utilities
    (4900, 4999, 'Utilities'),
]

# REITs and property operators are Real Estate, not Financials.
REAL_ESTATE_CODES = {6500, 6510, 6512, 6513, 6519, 6531, 6532, 6552, 6770, 6798}

TWO_DIGIT_FALLBACK = {
    1: 'Energy', 10: 'Materials', 12: 'Energy', 13: 'Energy', 14: 'Materials',
    15: 'Industrials', 16: 'Industrials', 17: 'Industrials',
    20: 'Consumer Staples', 21: 'Consumer Staples', 22: 'Consumer Discretionary',
    23: 'Consumer Discretionary', 24: 'Industrials', 25: 'Consumer Discretionary',
    26: 'Materials', 27: 'Communication Services', 28: 'Health Care',
    29: 'Energy', 30: 'Materials', 31: 'Consumer Discretionary',
    32: 'Materials', 33: 'Materials', 34: 'Industrials', 35: 'Industrials',
    36: 'Information Technology', 37: 'Industrials', 38: 'Health Care',
    39: 'Consumer Discretionary', 40: 'Industrials', 41: 'Industrials',
    42: 'Industrials', 44: 'Industrials', 45: 'Industrials', 46: 'Energy',
    47: 'Industrials', 48: 'Communication Services', 49: 'Utilities',
    50: 'Industrials', 51: 'Consumer Staples', 52: 'Consumer Discretionary',
    53: 'Consumer Discretionary', 54: 'Consumer Staples',
    55: 'Consumer Discretionary', 56: 'Consumer Discretionary',
    57: 'Consumer Discretionary', 58: 'Consumer Discretionary',
    59: 'Consumer Discretionary', 60: 'Financials', 61: 'Financials',
    62: 'Financials', 63: 'Financials', 64: 'Financials', 65: 'Real Estate',
    67: 'Financials', 70: 'Consumer Discretionary',
    72: 'Consumer Discretionary', 73: 'Information Technology',
    75: 'Consumer Discretionary', 78: 'Communication Services',
    79: 'Communication Services', 80: 'Health Care',
    82: 'Consumer Discretionary', 83: 'Health Care', 87: 'Industrials',
}


def sector_from_sic(sic: str | int | None) -> str:
    """Map an SIC code onto a GICS sector name."""
    if sic in (None, '', 'None'):
        return 'Unknown'
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return 'Unknown'

    if code in REAL_ESTATE_CODES:
        return 'Real Estate'

    for lo, hi, sector in SIC_RANGES:
        if lo <= code <= hi:
            return sector

    return TWO_DIGIT_FALLBACK.get(code // 100, 'Unknown')


def industry_from_description(description: str | None) -> str:
    """SEC's own SIC description makes a serviceable industry label."""
    return (description or 'Unknown').strip().title()
