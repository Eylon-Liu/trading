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
    'equity', 'cash', 'long_term_debt', 'short_term_debt', 'total_debt',
    'inventory', 'shares_diluted', 'shares_basic',
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


_TAG_AUDIT_KEY = '\x00tag_audit'


# ─────────────────────────────────────────────
# BATCH PRE-COMPUTATION
# ─────────────────────────────────────────────

def _batch_concept_groups(
    facts_all: pd.DataFrame,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Group facts by (ticker, concept) with single-tag selection, in one pass.

    Returns (groups_by_ticker, audits_by_ticker) where each value is a dict
    keyed by concept name.
    """
    has_tag = 'tag' in facts_all.columns

    # Pre-compute the best tag per (ticker, concept) in bulk rather than
    # calling _single_tag 10 000 times with per-group groupby('tag').agg().
    best_tags: dict[tuple, str] = {}
    if has_tag:
        tag_stats = (
            facts_all
            .assign(_abs_val=facts_all['val'].abs())
            .groupby(['ticker', 'concept', 'tag'], sort=False)
            .agg(newest=('filed', 'max'), scale=('_abs_val', 'median'))
            .reset_index()
        )
        for (ticker, concept), ts in tag_stats.groupby(
                ['ticker', 'concept'], sort=False):
            if len(ts) <= 1:
                best_tags[(ticker, concept)] = ts.iloc[0]['tag']
                continue
            cutoff = ts['newest'].max() - pd.Timedelta(days=365)
            current = ts[ts['newest'] >= cutoff]
            if current.empty:
                current = ts
            best_tags[(ticker, concept)] = current.loc[
                current['scale'].idxmax(), 'tag']

    groups: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    for (ticker, concept), g in facts_all.groupby(
            ['ticker', 'concept'], sort=False):
        if has_tag and (ticker, concept) in best_tags:
            best = best_tags[(ticker, concept)]
            if g['tag'].nunique() > 1:
                g = g[g['tag'] == best]
            selected = g
        else:
            selected = g
        groups.setdefault(ticker, {})[concept] = selected
        if has_tag and not selected.empty:
            audits.setdefault(ticker, {})[concept] = selected.iloc[0]['tag']
    return groups, audits


def _batch_flow_quarters(
    concept_groups: dict[str, dict],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute quarterly flows for all flow concepts across all tickers.

    Uses vectorized pandas operations on the concatenated per-concept data
    instead of calling ``_quarterly_flows_uncached`` 5 000 times.

    Returns ``{concept: {ticker: quarterly_df}}``.
    """
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for concept in FLOW_CONCEPTS:
        parts = []
        for ticker, cgroups in concept_groups.items():
            if concept in cgroups:
                df = cgroups[concept]
                if not df.empty:
                    parts.append(df)
        if not parts:
            result[concept] = {}
            continue

        df = pd.concat(parts, ignore_index=True)
        df['days'] = (df['period_end'] - df['period_start']).dt.days
        df['dur'] = df['days'].map(_classify_duration)
        df = (df.sort_values('filed', kind='stable')
                .drop_duplicates(
                    subset=['ticker', 'period_start', 'period_end'],
                    keep='last'))

        quarters = df[df['dur'] == 'Q'][
            ['ticker', 'period_start', 'period_end',
             'filed', 'val', 'fy', 'fp']].copy()

        cumulative = df[df['dur'].isin(['H', '9M', 'A'])].sort_values(
            ['ticker', 'period_start', 'period_end'])
        if not cumulative.empty:
            cumulative = cumulative.copy()
            g = cumulative.groupby(['ticker', 'period_start'], sort=False)
            cumulative['prev_end'] = g['period_end'].shift(1)
            cumulative['prev_val'] = g['val'].shift(1).fillna(0)
            cumulative['prev_filed'] = g['filed'].shift(1)
            cumulative['span'] = (
                cumulative['period_end'] - cumulative['prev_end']).dt.days

            mask = (cumulative['prev_end'].notna()
                    & cumulative['span'].between(
                        QUARTER_DAYS[0], QUARTER_DAYS[1]))
            if mask.any():
                derived = cumulative[mask].copy()
                derived['period_start'] = derived['prev_end']
                derived['val'] = derived['val'] - derived['prev_val']
                derived['filed'] = derived[['filed', 'prev_filed']].max(axis=1)
                quarters = pd.concat(
                    [quarters,
                     derived[['ticker', 'period_start', 'period_end',
                              'filed', 'val', 'fy', 'fp']]],
                    ignore_index=True)

        if quarters.empty:
            result[concept] = {}
            continue

        quarters = (
            quarters.sort_values(['ticker', 'period_end', 'filed'])
            .drop_duplicates(subset=['ticker', 'period_end'], keep='last')
            .sort_values(['ticker', 'period_end'])
            .reset_index(drop=True))

        # Suspect quarter detection: flag derived values that are implausibly
        # large relative to the same quarter a year ago.
        if len(quarters) >= 5:
            prev_val = quarters.groupby('ticker', sort=False)['val'].shift(4)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (quarters['val'] / prev_val).abs()
            sign_flip = (quarters['val'] > 0) != (prev_val > 0)
            suspect = (ratio > 10) & sign_flip & prev_val.notna() & (prev_val != 0)
            if suspect.any():
                quarters.loc[suspect, 'val'] = np.nan

        cols = ['period_start', 'period_end', 'filed', 'val', 'fy', 'fp']
        result[concept] = {
            t: g[cols].reset_index(drop=True)
            for t, g in quarters.groupby('ticker', sort=False)
        }

    return result


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
        audit = {}
        groups = {}
        for c, g in facts.groupby('concept', sort=False):
            selected = _single_tag(g)
            groups[c] = selected
            if 'tag' in selected.columns and not selected.empty:
                audit[c] = selected.iloc[0]['tag']
        memo[_GROUPS_KEY] = groups
        memo[_TAG_AUDIT_KEY] = audit
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

    # Flag derived quarters that are implausibly large relative to the same
    # quarter a year ago — a likely sign of misaligned restatements rather
    # than real economic change.
    if len(quarters) >= 5:
        vals = quarters['val'].to_numpy()
        for i in range(4, len(quarters)):
            prev = vals[i - 4]
            curr = vals[i]
            if prev == 0 or not np.isfinite(prev) or not np.isfinite(curr):
                continue
            ratio = abs(curr / prev)
            if ratio > 10 and (prev > 0) != (curr > 0):
                log.debug('suspect derived quarter at idx %d: '
                          'curr=%.2g prev=%.2g (%.1fx, sign flip) — '
                          'possible restatement artefact', i, curr, prev, ratio)
                quarters.loc[quarters.index[i], 'val'] = np.nan

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
SHARE_CROSSCHECK_BAND = (0.70, 1.45)


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

# Net income below this share of revenue is too small to divide by: the ratio
# it produces says more about how close the denominator got to zero than about
# the business.
MIN_INCOME_TO_REVENUE = 0.005


def _meaningful_income(net_income: float, revenue: float) -> float:
    """Net income, or NaN when it is too near zero to be a denominator."""
    if not np.isfinite(_as_float(net_income)):
        return np.nan
    if not np.isfinite(_as_float(revenue)) or revenue <= 0:
        return float(net_income)
    if abs(float(net_income)) < MIN_INCOME_TO_REVENUE * float(revenue):
        return np.nan
    return float(net_income)


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


# Counts scale up with a forward split; per-share amounts scale down by the
# same ratio. Both have to move together or the cross-check between them —
# reported shares against net income / EPS — reads the split as a units error
# and rejects a perfectly good share count.
SHARE_CONCEPTS = ('shares_diluted', 'shares_basic', 'shares_outstanding')
PER_SHARE_CONCEPTS = ('eps_diluted', 'eps_basic')


def adjust_shares_for_splits(facts: pd.DataFrame,
                             splits: pd.DataFrame) -> pd.DataFrame:
    """
    Restate filed share counts onto the current split basis.

    Price history is retroactively restated after every split — a 2018 bar is
    quoted in today's shares. SEC share counts are not: they are whatever was
    filed at the time. Multiplying the two together understates the market cap
    of any company that has since split, by exactly the split ratio.

    Lam Research showed a $3.1B market cap in January 2018 against ~$30B
    actual, because a 165M share count filed then was priced at a close that
    had been divided by ten for a 2024 split. That produced a 66% earnings
    yield — a P/E of 1.5 — and made it the top-ranked name in every
    valuation-driven strategy at every historical date.

    The correction has no look-ahead: shares and price are both moved onto the
    same basis, and their product — the market capitalisation — is unchanged by
    the choice of basis. What it removes is a bias *toward* companies that
    later split, which is a bias toward companies whose price later rose.
    """
    if facts.empty or splits is None or splits.empty:
        return facts

    is_share = facts['concept'].isin(SHARE_CONCEPTS)
    is_per_share = facts['concept'].isin(PER_SHARE_CONCEPTS)
    if not (is_share.any() or is_per_share.any()):
        return facts

    splits_clean = splits[
        splits['ratio'].apply(lambda r: np.isfinite(r) and r > 0)
    ].copy()
    if splits_clean.empty:
        return facts

    facts = facts.copy()
    need_adj = is_share | is_per_share
    adj_rows = facts.loc[need_adj, ['ticker', 'filed']].copy()
    adj_rows['filed'] = pd.to_datetime(adj_rows['filed'])
    adj_rows['_idx'] = adj_rows.index

    sp = splits_clean[['ticker', 'date', 'ratio']].copy()
    sp['date'] = pd.to_datetime(sp['date'])

    merged = adj_rows.merge(sp, on='ticker', how='inner')
    merged = merged[merged['filed'] < merged['date']]

    if merged.empty:
        return facts

    cum_factors = merged.groupby('_idx')['ratio'].prod()
    up_idx = cum_factors.index.intersection(facts.index[is_share])
    down_idx = cum_factors.index.intersection(facts.index[is_per_share])
    if not up_idx.empty:
        facts.loc[up_idx, 'val'] *= cum_factors[up_idx]
    if not down_idx.empty:
        facts.loc[down_idx, 'val'] /= cum_factors[down_idx]
    return facts


def build_fundamentals(tickers: list[str], as_of: date | str,
                       facts_all: pd.DataFrame | None = None,
                       splits: pd.DataFrame | None = None,
                       _prev_result: pd.DataFrame | None = None,
                       _changed_tickers: set[str] | None = None,
                       _splits_applied: bool = False) -> pd.DataFrame:
    """
    One row per ticker of point-in-time fundamentals, including Piotroski F.

    Piotroski is computed in the same per-ticker loop rather than in a separate
    pass. Both need the same TTM and balance-sheet values, so sharing the memo
    eliminates a second iteration over 500 tickers (~4s on S&P 500).

    Every input passes through the `filed <= as_of` gate in sec.facts_asof, so
    nothing here can see a filing that had not happened yet.
    """
    if facts_all is None:
        facts_all = sec.facts_asof(tickers, as_of)
    if facts_all.empty:
        return pd.DataFrame()

    if splits is None:
        from data import yahoo
        try:
            splits = yahoo.split_factors(sorted(facts_all['ticker'].unique()))
        except Exception as exc:                   # noqa: BLE001
            log.debug('split lookup failed (%s) — share counts left as filed', exc)
            splits = pd.DataFrame()
    if not _splits_applied:
        facts_all = adjust_shares_for_splits(facts_all, splits)

    # Proxy statements (DEF 14A, PRE 14A, etc.) sometimes report financial
    # figures in thousands while 10-K/10-Q use full-scale dollars. Because
    # _latest_filed keeps the most recently filed row per period, a proxy filed
    # after a 10-K silently replaces $3.5B with $3,511 — corrupting every
    # derived ratio. ANET, PCG, ED, SCHW, STZ were all affected.
    if 'form' in facts_all.columns:
        _FINANCIAL_FORMS = {'10-K', '10-Q', '10-K/A', '10-Q/A', '10-KT', '10-QT',
                            '20-F', '20-F/A'}
        before = len(facts_all)
        facts_all = facts_all[facts_all['form'].isin(_FINANCIAL_FORMS)]
        dropped = before - len(facts_all)
        if dropped:
            log.debug('excluded %d non-financial-statement facts (proxy/8-K/etc.)', dropped)

    # The share crosscheck (reported vs EPS-implied) breaks for tickers that
    # split within the TTM window: the quarterly EPS decomposition crosses a
    # split boundary, making the TTM EPS — and therefore the implied count —
    # unreliable. BKNG's 25:1 (Apr 2026) produced a 2.65x ratio and lost its
    # market cap; DELL's 1.8x (2018) landed at 1.39x, just outside the band.
    _recent_split_tickers: set[str] = set()
    if splits is not None and not splits.empty:
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=400)
        recent = splits[pd.to_datetime(splits['date']) >= cutoff]
        _recent_split_tickers = set(recent['ticker'].unique())

    # Incremental mode: only recompute tickers with new filings.
    carry = pd.DataFrame()
    if _prev_result is not None and _changed_tickers is not None:
        available = set(facts_all['ticker'].unique())
        recompute = {t for t in available
                     if t in _changed_tickers or t not in _prev_result.index}
        carry_idx = [t for t in available
                     if t not in recompute and t in _prev_result.index]
        if carry_idx:
            carry = _prev_result.loc[carry_idx]
        if not recompute:
            return carry if not carry.empty else pd.DataFrame()
        facts_all = facts_all[facts_all['ticker'].isin(recompute)]
        log.debug('incremental build_fundamentals: %d recompute, %d carry',
                  len(recompute), len(carry_idx))

    # Batch pre-computation: one pass over the data replaces thousands of
    # small per-ticker groupby + sort + dedup operations.
    all_groups, all_audits = _batch_concept_groups(facts_all)
    all_qflows = _batch_flow_quarters(all_groups)

    # A dummy facts frame that functions fall back to when a concept is not
    # in the memo. All concepts are pre-populated, so this is never actually
    # read — but the helper signatures require it.
    _empty = facts_all.iloc[:0]

    rows = []
    for ticker in sorted(all_groups):
        memo: dict = {
            _GROUPS_KEY: all_groups[ticker],
            _TAG_AUDIT_KEY: all_audits.get(ticker, {}),
        }
        for fc in FLOW_CONCEPTS:
            qf = all_qflows.get(fc, {}).get(ticker)
            if qf is not None:
                memo[fc] = qf

        f = _empty
        rev = ttm(f, 'revenue', memo=memo)
        ni = ttm(f, 'net_income', memo=memo)
        op_inc = ttm(f, 'operating_income', memo=memo)
        gp = ttm(f, 'gross_profit', memo=memo)
        ocf = ttm(f, 'operating_cash_flow', memo=memo)
        capex_raw = ttm(f, 'capex', memo=memo)
        capex = abs(capex_raw) if np.isfinite(capex_raw) else np.nan
        buybacks = abs(ttm(f, 'buybacks', memo=memo))
        divs = abs(ttm(f, 'dividends_paid', memo=memo))

        assets = latest_stock(f, 'assets', memo)
        equity = latest_stock(f, 'equity', memo)
        cash = latest_stock(f, 'cash', memo)
        ltd = latest_stock(f, 'long_term_debt', memo)
        std = latest_stock(f, 'short_term_debt', memo)
        cur_a = latest_stock(f, 'current_assets', memo)
        cur_l = latest_stock(f, 'current_liabilities', memo)
        shares = share_count(f, memo, as_of=as_of,
                             crosscheck=ticker not in _recent_split_tickers)

        tot = latest_stock(f, 'total_debt', memo)
        if np.isfinite(tot) and tot > 0:
            debt = tot
        else:
            debt = np.nansum([ltd if np.isfinite(ltd) else 0,
                              std if np.isfinite(std) else 0])
            debt = debt if debt > 0 else np.nan
        fcf = (ocf - capex) if (np.isfinite(ocf) and np.isfinite(capex)) else np.nan

        roa = _safe_div(ni, assets)
        eps = _safe_div(ni, shares)

        # ── Piotroski F-Score (shared memo avoids re-deriving TTMs) ──
        rev_prev = value_n_periods_ago(f, 'revenue', 1, memo)
        ni_prev = value_n_periods_ago(f, 'net_income', 1, memo)
        gp_prev = value_n_periods_ago(f, 'gross_profit', 1, memo)
        assets_prev = value_n_periods_ago(f, 'assets', 1, memo)
        ltd_prev = value_n_periods_ago(f, 'long_term_debt', 1, memo)
        ca_prev = value_n_periods_ago(f, 'current_assets', 1, memo)
        cl_prev = value_n_periods_ago(f, 'current_liabilities', 1, memo)
        sh = latest_stock(f, 'shares_diluted', memo)
        sh_prev = value_n_periods_ago(f, 'shares_diluted', 1, memo)
        roa_prev = _safe_div(ni_prev, assets_prev)

        pf_tests = [
            roa > 0,
            ocf > 0,
            roa > roa_prev,
            (ocf > ni) if np.isfinite(ocf) and np.isfinite(ni) else False,
            _safe_div(ltd, assets) < _safe_div(ltd_prev, assets_prev),
            _safe_div(cur_a, cur_l) > _safe_div(ca_prev, cl_prev),
            (sh <= sh_prev * 1.02) if np.isfinite(sh) and np.isfinite(sh_prev) else False,
            _safe_div(gp, rev) > _safe_div(gp_prev, rev_prev),
            _safe_div(rev, assets) > _safe_div(rev_prev, assets_prev),
        ]
        pf_score = float(sum(1 for t in pf_tests if t is True or t is np.True_))

        tag_audit = memo.get(_TAG_AUDIT_KEY, {})
        rev_tag = tag_audit.get('revenue', '')
        ni_tag = tag_audit.get('net_income', '')

        # last_filed / last_period_end from the concept groups
        cg = all_groups[ticker]
        all_filed = [g['filed'].max() for g in cg.values()
                     if not g.empty and 'filed' in g.columns]
        all_pe = [g['period_end'].max() for g in cg.values()
                  if not g.empty and 'period_end' in g.columns]

        rows.append({
            'ticker': ticker,
            'revenue_ttm': rev, 'net_income_ttm': ni, 'operating_income_ttm': op_inc,
            'gross_profit_ttm': gp, 'ocf_ttm': ocf, 'capex_ttm': capex, 'fcf_ttm': fcf,
            'buybacks_ttm': buybacks, 'dividends_paid_ttm': divs,
            'assets': assets, 'equity': equity, 'cash': cash, 'debt': debt,
            'current_assets': cur_a, 'current_liabilities': cur_l,
            'shares_diluted': shares,
            'revenue_tag': rev_tag, 'net_income_tag': ni_tag,
            'gross_margin': _safe_div(gp, rev),
            'operating_margin': _safe_div(op_inc, rev),
            'net_margin': _safe_div(ni, rev),
            'roe': _safe_div(ni, _meaningful_equity(equity, assets)),
            'roa': roa,
            'roic': _safe_div(op_inc, _invested_capital(
                _meaningful_equity(equity, assets), debt)),
            'gross_profitability': _safe_div(gp, assets),
            'asset_turnover': _safe_div(rev, assets),
            'debt_to_equity': _safe_div(debt, _meaningful_equity(equity, assets)),
            'net_debt_to_equity': _safe_div(
                debt - cash if np.isfinite(debt) and np.isfinite(cash)
                else np.nan,
                _meaningful_equity(equity, assets)),
            'current_ratio': _safe_div(cur_a, cur_l),
            'fcf_conversion': _safe_div(fcf, _meaningful_income(ni, rev)),
            'eps': eps,
            'accruals': _safe_div(ni - ocf if np.isfinite(ni) and np.isfinite(ocf)
                                  else np.nan, assets),
            'revenue_growth_1y': _growth(rev, rev_prev),
            'revenue_cagr_3y': _cagr(
                rev, value_n_periods_ago(f, 'revenue', 3, memo), 3),
            'earnings_growth_1y': _growth(ni, ni_prev),
            'equity_cagr_3y': _cagr(
                equity, value_n_periods_ago(f, 'equity', 3, memo), 3),
            'eps_cagr_3y': _cagr(
                eps,
                _safe_div(value_n_periods_ago(f, 'net_income', 3, memo),
                          value_n_periods_ago(f, 'shares_diluted', 3, memo)),
                3),
            'piotroski_f': pf_score,
            'last_filed': max(all_filed) if all_filed else pd.NaT,
            'last_period_end': max(all_pe) if all_pe else pd.NaT,
        })

    result = pd.DataFrame(rows).set_index('ticker') if rows else pd.DataFrame()
    if not carry.empty:
        parts = [p for p in (carry, result) if not p.empty]
        return pd.concat(parts) if parts else pd.DataFrame()
    return result


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
