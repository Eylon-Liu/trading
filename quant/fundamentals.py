"""
Point-in-time fundamentals — SEC facts to TTM financials.

Two problems have to be solved before an XBRL fact is usable:

1. Duration ambiguity. Income-statement and cash-flow concepts are *flows*
   measured over a span. AAPL's FY2026 Q3 NetIncomeLoss returns two facts
   sharing an end date of 2026-06-27: 29,789M for the three months and
   101,464M for the nine months to date. They are distinguishable only by
   `period_start`. Summing them, or picking whichever appears first, silently
   corrupts every derived ratio.

2. Restatements. A period is re-reported as figures are revised, so the same
   (concept, period) has several rows with different `filed` dates. A backtest
   must use the number that was on file at the time, not the final revision.

Balance-sheet concepts are *stocks* measured at an instant and need neither
step, just the latest observation at or before the as-of date.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from data import sec

log = logging.getLogger(__name__)

# Flows accumulate over a span and must be duration-matched.
FLOW_CONCEPTS = {
    'revenue', 'net_income', 'operating_income', 'gross_profit',
    'operating_cash_flow', 'capex', 'interest_expense', 'rnd',
    'dividends_paid', 'buybacks', 'eps_diluted',
}
# Stocks are instantaneous balances.
STOCK_CONCEPTS = {
    'assets', 'current_assets', 'liabilities', 'current_liabilities',
    'equity', 'cash', 'long_term_debt', 'short_term_debt', 'inventory',
    'shares_diluted', 'shares_basic',
}

# Day-count windows for classifying a fact's span.
QUARTER_DAYS = (60, 115)
ANNUAL_DAYS = (330, 400)


def _classify_duration(days: float) -> str:
    if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
        return 'Q'
    if ANNUAL_DAYS[0] <= days <= ANNUAL_DAYS[1]:
        return 'A'
    if 150 <= days <= 220:
        return 'H'      # half year
    if 240 <= days <= 310:
        return '9M'     # three quarters cumulative
    return 'other'


# Reserved key under which a per-ticker memo caches the concept split. A real
# concept is never named this, so it cannot collide.
_GROUPS_KEY = '\x00concept_groups'


def _concept_frame(facts: pd.DataFrame, concept: str,
                   memo: dict | None = None) -> pd.DataFrame:
    """
    Rows for one concept.

    Without a memo this is a boolean scan of the ticker's whole fact frame,
    repeated for every concept asked for. Splitting once and reusing turns
    ~12 scans per ticker into one groupby.
    """
    if memo is None:
        return facts[facts['concept'] == concept]

    groups = memo.get(_GROUPS_KEY)
    if groups is None:
        groups = {c: _single_tag(g)
                  for c, g in facts.groupby('concept', sort=False)}
        memo[_GROUPS_KEY] = groups
    return groups.get(concept, facts.iloc[:0])


def _single_tag(group: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one XBRL tag per concept rather than pooling several.

    A concept maps to a list of tags because filers differ, but a single filer
    may report more than one of them at different scales. Camden Property
    files both `Revenues` — a small fragment — and `RealEstateRevenueNet`,
    its actual rental income. Pooling them let whichever was filed last win
    per period, mixing the two and producing a net margin of 2,565%.

    Two rules, in order, because neither alone is enough:

    1. Keep only tags the filer still uses — within a year of its newest
       filing for this concept. Ranking by count alone picked retired tags,
       giving Microsoft a 134% net margin from an obsolete `Revenues` line.
    2. Among those, take the largest. A filer that reports both a total and a
       component tags both; the component is smaller by construction. Camden
       Property files `Revenues` as a fragment alongside
       `RealEstateRevenueNet`, its actual rental income, and picking by
       recency alone chose the fragment — a net margin of 2,486%.
    """
    if 'tag' not in group.columns or group['tag'].nunique() <= 1:
        return group

    stats = group.groupby('tag').agg(newest=('filed', 'max'),
                                     scale=('val', lambda s: s.abs().median()))
    cutoff = stats['newest'].max() - pd.Timedelta(days=365)
    current = stats[stats['newest'] >= cutoff]
    if current.empty:
        current = stats

    best = current['scale'].idxmax()
    return group[group['tag'] == best]


def _latest_filed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Collapse restatements: one row per period, the newest filing available.

    `df` must already be gated to filed <= as_of, so "newest" means newest
    *knowable*, not newest ever.

    drop_duplicates rather than groupby().last(): the latter aggregates
    column-by-column skipping NaN, so a null field in the newest filing is
    silently backfilled from an older one, producing a row that was never
    actually filed. Keeping the last row keeps one coherent filing — and it is
    several times faster, which matters at S&P 500 scale.
    """
    if df.empty:
        return df
    return (df.sort_values('filed', kind='stable')
              .drop_duplicates(subset=keys, keep='last'))


def quarterly_flows(facts: pd.DataFrame, concept: str,
                    memo: dict | None = None) -> pd.DataFrame:
    """
    Extract genuine single-quarter observations for a flow concept.

    Where a quarter is only reported cumulatively (common for Q4, and for
    filers who report year-to-date throughout), it is derived by differencing
    consecutive cumulative figures within the same fiscal year.

    `memo` is an optional per-ticker dict. Deriving quarters costs a sort, a
    groupby and a row-wise differencing pass, and a single ticker asks for the
    same concept several times over (TTM, last year's TTM, the 3-year base).
    Threading a memo through turned a 503-name S&P screen from 13,418 of these
    calls into roughly 3,000.
    """
    if memo is not None and concept in memo:
        return memo[concept]

    result = _quarterly_flows_uncached(facts, concept, memo)
    if memo is not None:
        memo[concept] = result
    return result


def _quarterly_flows_uncached(facts: pd.DataFrame, concept: str,
                              memo: dict | None = None) -> pd.DataFrame:
    df = _concept_frame(facts, concept, memo).copy()
    if df.empty:
        return pd.DataFrame()

    df['days'] = (df['period_end'] - df['period_start']).dt.days
    df['dur'] = df['days'].map(_classify_duration)
    df = _latest_filed(df, ['period_start', 'period_end'])

    quarters = df[df['dur'] == 'Q'][
        ['period_start', 'period_end', 'filed', 'val', 'fy', 'fp']].copy()

    # Derive the missing quarters from cumulative reports.
    cumulative = df[df['dur'].isin(['H', '9M', 'A'])].sort_values('period_end')
    if not cumulative.empty:
        derived = []
        for _start, grp in cumulative.groupby('period_start'):
            grp = grp.sort_values('period_end')
            prev_end, prev_val, prev_filed = None, 0.0, None
            for _i, row in grp.iterrows():
                if prev_end is not None:
                    span = (row['period_end'] - prev_end).days
                    if QUARTER_DAYS[0] <= span <= QUARTER_DAYS[1]:
                        derived.append({
                            'period_start': prev_end,
                            'period_end': row['period_end'],
                            'filed': max(row['filed'], prev_filed),
                            'val': row['val'] - prev_val,
                            'fy': row['fy'], 'fp': row['fp'],
                        })
                prev_end, prev_val, prev_filed = (
                    row['period_end'], row['val'], row['filed'])
        if derived:
            quarters = pd.concat([quarters, pd.DataFrame(derived)],
                                 ignore_index=True)

    if quarters.empty:
        return quarters

    quarters = (quarters.sort_values(['period_end', 'filed'])
                        .drop_duplicates(subset=['period_end'], keep='last')
                        .sort_values('period_end')
                        .reset_index(drop=True))
    return quarters


def ttm(facts: pd.DataFrame, concept: str, min_quarters: int = 4,
        memo: dict | None = None) -> float:
    """Trailing-twelve-month total for a flow concept."""
    q = quarterly_flows(facts, concept, memo)
    if len(q) >= min_quarters:
        return float(q.tail(4)['val'].sum())

    # Fall back to the most recent annual figure when quarters are sparse.
    df = _concept_frame(facts, concept, memo).copy()
    if df.empty:
        return np.nan
    df['days'] = (df['period_end'] - df['period_start']).dt.days
    annual = df[df['days'].map(_classify_duration) == 'A']
    if annual.empty:
        return np.nan
    annual = _latest_filed(annual, ['period_start', 'period_end'])
    return float(annual.sort_values('period_end').iloc[-1]['val'])


def latest_stock(facts: pd.DataFrame, concept: str,
                 memo: dict | None = None) -> float:
    """Most recent balance-sheet value knowable at the as-of date."""
    df = _concept_frame(facts, concept, memo)
    if df.empty:
        return np.nan
    df = _latest_filed(df.copy(), ['period_end'])
    return float(df.sort_values('period_end').iloc[-1]['val'])


def _latest_filed_date(facts: pd.DataFrame, concept: str,
                       memo: dict | None = None):
    """When the newest fact for a concept was filed, or None."""
    df = _concept_frame(facts, concept, memo)
    if df.empty:
        return None
    return df['filed'].max()


# A share count more than this far out of date cannot describe the company
# today. Two annual reports' worth of slack.
MAX_SHARE_COUNT_AGE_DAYS = 500


def _fresh_share_value(facts: pd.DataFrame, concept: str, as_of,
                       memo: dict | None) -> float:
    """A share count, but only if it was filed recently enough to be current."""
    value = latest_stock(facts, concept, memo)
    if not np.isfinite(value) or value <= 0:
        return np.nan
    filed = _latest_filed_date(facts, concept, memo)
    if filed is None or as_of is None:
        return value
    age = (pd.Timestamp(as_of) - pd.Timestamp(filed)).days
    return value if age <= MAX_SHARE_COUNT_AGE_DAYS else np.nan


# A reported share count is accepted only if an independent figure derived
# from EPS agrees within this band. Measured agreement on healthy filers is
# 0.95-1.00x, so 25% is loose enough for rounding and share-class rounding
# while still catching an order-of-magnitude error.
SHARE_CROSSCHECK_BAND = (0.75, 1.33)


def implied_share_count(facts: pd.DataFrame, memo: dict | None = None) -> float:
    """
    Shares implied by net income / diluted EPS.

    An independent read on the same quantity, built from two us-gaap facts the
    filer had to keep consistent with each other. Useful precisely because it
    fails differently from the cover-page count: a wrong share class or a
    units error moves one and not the other.
    """
    ni = ttm(facts, 'net_income', memo=memo)
    eps = ttm(facts, 'eps_diluted', memo=memo)
    if not (np.isfinite(ni) and np.isfinite(eps)) or eps == 0:
        return np.nan
    implied = ni / eps
    return implied if np.isfinite(implied) and implied > 0 else np.nan


def share_count(facts: pd.DataFrame, memo: dict | None = None,
                as_of=None, crosscheck: bool = True) -> float:
    """
    Shares outstanding, in descending order of what the number actually means.

    1. `dei:EntityCommonStockSharesOutstanding` — the cover-page count, the
       exact shares outstanding on the filing date. This is what market
       capitalisation is defined on, so it is used whenever present.
    2. Diluted weighted average — an EPS denominator, not a point-in-time
       count. It averages over the period and includes dilutive securities, so
       it drifts from the true figure for anyone issuing or buying back.
    3. Basic weighted average — same caveat, used when the diluted tag is
       absent or has gone stale.

    The fallbacks matter because filers switch tags: Exxon stopped reporting
    the diluted tag after 2013, so taking it unconditionally pinned its share
    count — and through it market cap and every yield factor — to a twelve-year
    old figure. Between the two averages, whichever was filed most recently
    wins.
    """
    chosen = np.nan
    exact = _fresh_share_value(facts, 'shares_outstanding', as_of, memo)
    if np.isfinite(exact):
        chosen = exact
    else:
        diluted = _fresh_share_value(facts, 'shares_diluted', as_of, memo)
        basic = _fresh_share_value(facts, 'shares_basic', as_of, memo)
        if not np.isfinite(diluted):
            chosen = basic
        elif not np.isfinite(basic):
            chosen = diluted
        else:
            d_filed = _latest_filed_date(facts, 'shares_diluted', memo)
            b_filed = _latest_filed_date(facts, 'shares_basic', memo)
            chosen = (basic if (d_filed is not None and b_filed is not None
                                and b_filed > d_filed) else diluted)

    if not crosscheck or not np.isfinite(chosen):
        return chosen

    # Reject a count that an independent derivation contradicts. Berkshire's
    # cover page reports Class A only, which is roughly a thousandth of the
    # economic share base — multiplied by a Class B price it produced a market
    # cap of under a billion against a real trillion, and an earnings yield of
    # 9,894%. Better to score nothing than to score that.
    implied = implied_share_count(facts, memo)
    if np.isfinite(implied):
        ratio = chosen / implied
        lo, hi = SHARE_CROSSCHECK_BAND
        if not (lo <= ratio <= hi):
            log.debug('share count rejected: reported %.4g vs EPS-implied '
                      '%.4g (%.2fx)', chosen, implied, ratio)
            return np.nan
    return chosen


def value_n_periods_ago(facts: pd.DataFrame, concept: str, years: int,
                        memo: dict | None = None) -> float:
    """Value roughly `years` back — for growth and CAGR calculations."""
    is_flow = concept in FLOW_CONCEPTS
    if is_flow:
        q = quarterly_flows(facts, concept, memo)
        if len(q) < 4 * (years + 1):
            return np.nan
        window = q.iloc[-4 * (years + 1): -4 * years] if years else q.tail(4)
        return float(window['val'].sum()) if len(window) == 4 else np.nan

    df = _concept_frame(facts, concept, memo)
    if df.empty:
        return np.nan
    df = _latest_filed(df.copy(), ['period_end']).sort_values('period_end')
    target = df['period_end'].max() - pd.DateOffset(years=years)
    prior = df[df['period_end'] <= target]
    return float(prior.iloc[-1]['val']) if not prior.empty else np.nan


# Book equity below this share of assets makes any equity-denominated ratio
# meaningless rather than merely large.
MIN_EQUITY_TO_ASSETS = 0.01


def _invested_capital(equity: float, debt: float) -> float:
    """Equity + debt, or NaN when equity is not usable."""
    if not np.isfinite(equity):
        return np.nan
    return equity + (debt if np.isfinite(debt) else 0.0)


def _meaningful_equity(equity: float, assets: float) -> float:
    """
    Equity, or NaN when it is too small or negative to divide by.

    Sustained buybacks can drive book equity to nearly nothing without the
    business being distressed: GoDaddy carries $0.01B of equity on $0.91B of
    earnings, which produces an ROE of 13,587% and a debt-to-equity of 561.
    Those figures are arithmetically correct and tell a reader nothing, and
    once winsorized they look merely excellent — which is worse, because the
    name then ranks on a quality score it has not earned.

    NaN drops the factor for that name and the coverage floor decides whether
    enough remains to score it at all.
    """
    if not np.isfinite(equity) or equity <= 0:
        return np.nan
    if np.isfinite(assets) and assets > 0 and equity / assets < MIN_EQUITY_TO_ASSETS:
        return np.nan
    return equity


def build_fundamentals(tickers: list[str], as_of: date | str,
                       facts_all: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One row per ticker of point-in-time fundamentals.

    Every input passes through the `filed <= as_of` gate in sec.facts_asof, so
    nothing here can see a filing that had not happened yet.

    `facts_all` lets a caller that already fetched the facts hand them over
    rather than paying for the query twice — it is a several-second read at
    S&P 500 scale.
    """
    if facts_all is None:
        facts_all = sec.facts_asof(tickers, as_of)
    if facts_all.empty:
        return pd.DataFrame()

    rows = []
    for ticker, facts in facts_all.groupby('ticker'):
        memo: dict = {}
        rev = ttm(facts, 'revenue', memo=memo)
        ni = ttm(facts, 'net_income', memo=memo)
        op_inc = ttm(facts, 'operating_income', memo=memo)
        gp = ttm(facts, 'gross_profit', memo=memo)
        ocf = ttm(facts, 'operating_cash_flow', memo=memo)
        capex_raw = ttm(facts, 'capex', memo=memo)
        capex = abs(capex_raw) if np.isfinite(capex_raw) else np.nan
        buybacks = abs(ttm(facts, 'buybacks', memo=memo))
        divs = abs(ttm(facts, 'dividends_paid', memo=memo))

        assets = latest_stock(facts, 'assets', memo)
        equity = latest_stock(facts, 'equity', memo)
        cash = latest_stock(facts, 'cash', memo)
        ltd = latest_stock(facts, 'long_term_debt', memo)
        std = latest_stock(facts, 'short_term_debt', memo)
        cur_a = latest_stock(facts, 'current_assets', memo)
        cur_l = latest_stock(facts, 'current_liabilities', memo)
        shares = share_count(facts, memo, as_of=as_of)

        debt = np.nansum([ltd if np.isfinite(ltd) else 0,
                          std if np.isfinite(std) else 0])
        debt = debt if debt > 0 else np.nan
        fcf = (ocf - capex) if (np.isfinite(ocf) and np.isfinite(capex)) else np.nan

        rows.append({
            'ticker': ticker,
            # levels
            'revenue_ttm': rev, 'net_income_ttm': ni, 'operating_income_ttm': op_inc,
            'gross_profit_ttm': gp, 'ocf_ttm': ocf, 'capex_ttm': capex, 'fcf_ttm': fcf,
            # Cash returned to shareholders. Both are reported as outflows, so
            # they are stored as positive magnitudes for use as yields.
            'buybacks_ttm': buybacks, 'dividends_paid_ttm': divs,
            'assets': assets, 'equity': equity, 'cash': cash, 'debt': debt,
            'current_assets': cur_a, 'current_liabilities': cur_l,
            'shares_diluted': shares,
            # margins & returns
            'gross_margin': _safe_div(gp, rev),
            'operating_margin': _safe_div(op_inc, rev),
            'net_margin': _safe_div(ni, rev),
            'roe': _safe_div(ni, _meaningful_equity(equity, assets)),
            'roa': _safe_div(ni, assets),
            # Invested capital is equity plus debt. With equity unusable
            # the sum collapses to debt alone, which is not invested
            # capital and inflates the ratio — Marriott reached 185.
            'roic': _safe_div(op_inc, _invested_capital(
                _meaningful_equity(equity, assets), debt)),
            'gross_profitability': _safe_div(gp, assets),
            'asset_turnover': _safe_div(rev, assets),
            # balance-sheet quality
            'debt_to_equity': _safe_div(debt, _meaningful_equity(equity, assets)),
            'current_ratio': _safe_div(cur_a, cur_l),
            'accruals': _safe_div(ni - ocf if np.isfinite(ni) and np.isfinite(ocf)
                                  else np.nan, assets),
            # growth
            'revenue_growth_1y': _growth(
                rev, value_n_periods_ago(facts, 'revenue', 1, memo)),
            'revenue_cagr_3y': _cagr(
                rev, value_n_periods_ago(facts, 'revenue', 3, memo), 3),
            'earnings_growth_1y': _growth(
                ni, value_n_periods_ago(facts, 'net_income', 1, memo)),
            'equity_cagr_3y': _cagr(
                equity, value_n_periods_ago(facts, 'equity', 3, memo), 3),
            # provenance — lets the UI show how stale a name's data is
            'last_filed': facts['filed'].max(),
            'last_period_end': facts['period_end'].max(),
        })

    return pd.DataFrame(rows).set_index('ticker')


# ─────────────────────────────────────────────
# PIOTROSKI F-SCORE
# ─────────────────────────────────────────────

def piotroski_f(tickers: list[str], as_of: date | str,
                facts_all: pd.DataFrame | None = None) -> pd.Series:
    """
    Piotroski F-Score (0-9): nine binary fundamental health tests.

    Computed entirely from filed data, so it is honest at any historical date.

    Accepts a pre-fetched `facts_all` for the same reason build_fundamentals
    does: the two are always called together, and the fetch is not cheap.
    """
    if facts_all is None:
        facts_all = sec.facts_asof(tickers, as_of)
    if facts_all.empty:
        return pd.Series(dtype=float)

    scores = {}
    for ticker, facts in facts_all.groupby('ticker'):
        memo: dict = {}
        ni = ttm(facts, 'net_income', memo=memo)
        ocf = ttm(facts, 'operating_cash_flow', memo=memo)
        assets = latest_stock(facts, 'assets', memo)
        assets_prev = value_n_periods_ago(facts, 'assets', 1, memo)
        rev = ttm(facts, 'revenue', memo=memo)
        rev_prev = value_n_periods_ago(facts, 'revenue', 1, memo)
        gp = ttm(facts, 'gross_profit', memo=memo)
        gp_prev = value_n_periods_ago(facts, 'gross_profit', 1, memo)
        ni_prev = value_n_periods_ago(facts, 'net_income', 1, memo)
        ltd = latest_stock(facts, 'long_term_debt', memo)
        ltd_prev = value_n_periods_ago(facts, 'long_term_debt', 1, memo)
        ca, cl = latest_stock(facts, 'current_assets', memo), latest_stock(facts, 'current_liabilities', memo)
        ca_p = value_n_periods_ago(facts, 'current_assets', 1, memo)
        cl_p = value_n_periods_ago(facts, 'current_liabilities', 1, memo)
        sh = latest_stock(facts, 'shares_diluted', memo)
        sh_prev = value_n_periods_ago(facts, 'shares_diluted', 1, memo)

        roa = _safe_div(ni, assets)
        roa_prev = _safe_div(ni_prev, assets_prev)

        tests = [
            roa > 0,                                              # profitable
            ocf > 0,                                              # cash generative
            roa > roa_prev,                                       # improving returns
            (ocf > ni) if np.isfinite(ocf) and np.isfinite(ni) else False,   # quality of earnings
            _safe_div(ltd, assets) < _safe_div(ltd_prev, assets_prev),       # deleveraging
            _safe_div(ca, cl) > _safe_div(ca_p, cl_p),            # liquidity up
            (sh <= sh_prev * 1.02) if np.isfinite(sh) and np.isfinite(sh_prev) else False,
            _safe_div(gp, rev) > _safe_div(gp_prev, rev_prev),    # margin up
            _safe_div(rev, assets) > _safe_div(rev_prev, assets_prev),       # turnover up
        ]
        scores[ticker] = float(sum(1 for t in tests if t is True or t is np.True_))

    return pd.Series(scores, name='piotroski_f')


# ─────────────────────────────────────────────
# ARITHMETIC HELPERS  (NaN-safe)
# ─────────────────────────────────────────────

def _safe_div(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def _nan_sum(*vals):
    finite = [float(v) for v in vals
              if v is not None and np.isfinite(_as_float(v))]
    return sum(finite) if finite else np.nan


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _growth(now, before):
    if not np.isfinite(_as_float(now)) or not np.isfinite(_as_float(before)):
        return np.nan
    now, before = float(now), float(before)
    if before == 0:
        return np.nan
    # Sign flips make percentage growth meaningless (a loss turning to profit
    # is not "-250% growth"), so they are excluded rather than reported wrong.
    if before < 0 < now or now < 0 < before:
        return np.nan
    return (now - before) / abs(before)


def _cagr(now, before, years: int):
    if not np.isfinite(_as_float(now)) or not np.isfinite(_as_float(before)):
        return np.nan
    now, before = float(now), float(before)
    if before <= 0 or now <= 0 or years <= 0:
        return np.nan
    return (now / before) ** (1.0 / years) - 1.0
