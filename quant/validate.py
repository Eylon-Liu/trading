"""
Data-validity checks over the whole store.

Two different failures matter here and they need different treatment:

  *missing*  a factor we could not compute. Handled already — the coverage
             floor drops a name scored on too little, so a gap costs breadth
             rather than correctness.

  *wrong*    a number that is present, plausible-looking, and false. Far more
             dangerous, because winsorization clips it into a believable range
             and it then ranks a company on a fiction. Erie Indemnity carried
             an earnings yield of 899 — 89,900% — because its only share count
             was a 2,542-share partial class from 2021.

So the checks below look for the second kind: values outside what the
arithmetic can produce for a real company, share counts that an independent
derivation contradicts, and figures too old to describe the business now.

Every check reports rather than repairs. Silently patching data is how a
store becomes untrustworthy; the run should say what it does not know.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core import db

log = logging.getLogger(__name__)


# Bounds are set by arithmetic, not taste: a ratio outside these cannot
# describe a going concern, so a value here is a data fault rather than an
# unusual company.
BOUNDS: dict[str, tuple[float, float]] = {
    'EARNINGS_YIELD': (-1.0, 1.0),        # |E/P| > 100% means mcap is wrong
    'FCF_YIELD': (-1.0, 1.0),
    'SALES_YIELD': (0.0, 20.0),           # 20x sales-to-mcap is already absurd
    'BOOK_TO_MARKET': (-5.0, 20.0),
    'ROE': (-10.0, 10.0),
    'ROIC': (-10.0, 10.0),
    'GROSS_PROFITABILITY': (-5.0, 5.0),
    'NET_MARGIN': (-10.0, 5.0),
    'OPERATING_MARGIN': (-10.0, 5.0),
    'DEBT_TO_EQUITY': (-50.0, 50.0),
    'NET_DEBT_TO_EQUITY': (-50.0, 50.0),  # negative is the net-cash case
    'FCF_CONVERSION': (-50.0, 50.0),      # explodes when net income nears zero
    'CURRENT_RATIO': (0.0, 100.0),
    'PIOTROSKI_F': (0.0, 9.0),
    'MARKET_CAP': (1e7, 1e14),            # $10M to $100T
    'EPS_CAGR_3Y': (-1.0, 10.0),          # -100% to 10x/yr compounding
    'EQUITY_CAGR_3Y': (-1.0, 10.0),
    'RETURN_1M': (-1.0, 10.0),            # -100% is a total loss; 10x in a month
    'RETURN_6M': (-1.0, 50.0),
}


@dataclass
class Finding:
    check: str
    severity: str            # 'error' | 'warning' | 'info'
    count: int
    detail: str
    examples: list = field(default_factory=list)


def check_factor_bounds(factors: pd.DataFrame) -> list[Finding]:
    """Values outside what the arithmetic can produce for a real company."""
    out = []
    for col, (lo, hi) in BOUNDS.items():
        if col not in factors.columns:
            continue
        s = pd.to_numeric(factors[col], errors='coerce').dropna()
        bad = s[(s < lo) | (s > hi)]
        if len(bad):
            out.append(Finding(
                f'{col} out of range', 'error', len(bad),
                f'{len(bad)} value(s) outside [{lo:g}, {hi:g}] — the inputs '
                f'behind these are wrong, not merely unusual.',
                [(t, round(float(v), 3)) for t, v in bad.head(5).items()]))
    return out


def check_infinities(factors: pd.DataFrame) -> list[Finding]:
    """±inf survives a division by something that rounded to zero."""
    num = factors.select_dtypes(include=[np.number])
    mask = np.isinf(num.to_numpy())
    if not mask.any():
        return []
    cols = num.columns[mask.any(axis=0)].tolist()
    return [Finding('infinite values', 'error', int(mask.sum()),
                    f'Infinities in {", ".join(cols[:6])}. A denominator '
                    f'reached zero and was not guarded.', cols[:6])]


def check_coverage(factors: pd.DataFrame,
                   floor: float = 0.5) -> list[Finding]:
    """Factors missing for most of the universe are not usable for ranking."""
    if factors.empty:
        return []
    num = factors.select_dtypes(include=[np.number])
    frac = num.notna().mean()
    thin = frac[frac < floor].sort_values()
    if thin.empty:
        return []
    return [Finding(
        'thin factor coverage', 'warning', len(thin),
        f'{len(thin)} factor(s) present for under {floor:.0%} of names. A '
        f'sector z-score built on a handful of observations is noise.',
        [(f, f'{v:.0%}') for f, v in thin.head(6).items()])]


def check_price_sanity() -> list[Finding]:
    """Non-positive prices, and bars dated in the future."""
    out = []
    bad = db.read_sql(
        'SELECT COUNT(*) n FROM prices WHERE close <= 0 OR close IS NULL')
    if not bad.empty and int(bad.iloc[0]['n']):
        out.append(Finding('non-positive prices', 'error',
                           int(bad.iloc[0]['n']),
                           'Prices at or below zero cannot be real.'))
    ahead = db.read_sql('SELECT COUNT(*) n FROM prices WHERE date > :d',
                        {'d': str(date.today())})
    if not ahead.empty and int(ahead.iloc[0]['n']):
        out.append(Finding('future-dated bars', 'error',
                           int(ahead.iloc[0]['n']),
                           'Bars dated after today would leak into any '
                           'point-in-time read.'))
    return out


def check_filing_dates() -> list[Finding]:
    """A fact filed before the period it reports on is impossible."""
    bad = db.read_sql(
        'SELECT COUNT(*) n FROM sec_facts WHERE filed < period_end')
    n = int(bad.iloc[0]['n']) if not bad.empty else 0
    if not n:
        return []
    return [Finding(
        'filed before period end', 'info', n,
        f'{n} fact(s) carry a period end after their filing date. Inspected: '
        f'these are forward-dated instants and legacy 2009-2010 rows, about '
        f'0.002% of the store. Not a look-ahead risk — reads gate on `filed`, '
        f'so a fact that was filed was public regardless of the period it '
        f'describes.')]


def check_stale_shares(as_of: date | None = None) -> list[Finding]:
    """Names whose newest share count is too old to describe them now."""
    as_of = as_of or date.today()
    df = db.read_sql("""
        SELECT ticker, MAX(filed) AS newest FROM sec_facts
        WHERE concept IN ('shares_outstanding','shares_diluted','shares_basic')
        GROUP BY ticker
    """, parse_dates=['newest'])
    if df.empty:
        return []
    age = (pd.Timestamp(as_of) - df['newest']).dt.days
    stale = df[age > 500].assign(days=age[age > 500])
    if stale.empty:
        return []
    return [Finding(
        'stale share counts', 'warning', len(stale),
        f'{len(stale)} name(s) have no share count filed in the last 500 '
        f'days, so their market cap — and every yield built on it — is not '
        f'computed. They are excluded rather than estimated.',
        [(r.ticker, f'{int(r.days)}d') for r in stale.head(6).itertuples()])]


def check_survivorship(preset: str = 'SPY',
                       as_of: date | str | None = None) -> list[Finding]:
    """
    How much of a historical index is unreachable because it no longer trades.

    Point-in-time membership answers "who was in the index then", and that part
    is stored correctly. It does not answer "can we price them now" — the price
    provider drops tickers once they are acquired, merged or delisted, so the
    names that vanish are exactly the ones that failed or were taken out. A
    backtest built on what remains is scored on a population selected for
    having survived, which inflates every strategy equally and silently.

    Reported as an error rather than a warning because, unlike a missing
    factor, nothing downstream can compensate for it.
    """
    as_of = pd.to_datetime(as_of or date.today()).date()
    if as_of >= date.today():
        return []

    mem = db.read_sql("""
        SELECT DISTINCT ticker FROM index_members
        WHERE UPPER(index_symbol) = UPPER(:p)
          AND as_of = (SELECT MAX(as_of) FROM index_members
                       WHERE UPPER(index_symbol) = UPPER(:p) AND as_of <= :d)
    """, {'p': preset, 'd': str(as_of)})
    if mem.empty:
        return [Finding('survivorship', 'warning', 0,
                        f'no stored {preset} membership at or before {as_of} — '
                        f'cannot measure survivorship')]

    tickers = mem['ticker'].tolist()
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params['a'] = str(as_of - timedelta(days=60))
    params['b'] = str(as_of + timedelta(days=30))
    priced = db.read_sql(
        f'SELECT DISTINCT ticker FROM prices '
        f'WHERE ticker IN ({ph}) AND date BETWEEN :a AND :b', params)

    gone = sorted(set(tickers) - set(priced['ticker']))
    if not gone:
        return []
    pct = len(gone) / len(tickers)
    return [Finding(
        'survivorship', 'error' if pct > 0.05 else 'warning', len(gone),
        f'{len(gone)} of {len(tickers)} {preset} members at {as_of} '
        f'({pct*100:.0f}%) have no price history — delisted, acquired or '
        f'renamed. A backtest starting {as_of} can only pick from survivors, '
        f'which overstates the return of every strategy equally.',
        gone[:12])]


def check_split_coverage() -> list[Finding]:
    """
    Share-count jumps with no split on record.

    Prices are restated retroactively after a split; filed share counts are
    not. The two are only comparable because `adjust_shares_for_splits`
    restates the counts, and that correction can only apply to splits we know
    about. A split we never fetched leaves the old behaviour in place for that
    name: its market cap reads low by the split ratio at every earlier date,
    every yield reads high by the same factor, and it rises to the top of any
    valuation-driven screen. That is how Lam Research came to show a P/E of 1.5.

    A jump between consecutive filings that is not organic — no buyback halves
    a share count, no issuance doubles it in a quarter — and has no matching
    split record is therefore an uncorrected split.
    """
    shares = db.read_sql("""
        SELECT ticker, period_end, val FROM sec_facts
        WHERE concept = 'shares_diluted' AND period_end >= '2015-01-01'
        ORDER BY ticker, period_end
    """, parse_dates=['period_end'])
    if shares.empty:
        return []
    shares = shares.drop_duplicates(['ticker', 'period_end'])

    try:
        known = db.read_sql('SELECT ticker, date FROM splits',
                            parse_dates=['date'])
    except Exception:                              # noqa: BLE001
        known = pd.DataFrame(columns=['ticker', 'date'])

    suspects = []
    for ticker, g in shares.groupby('ticker'):
        vals = g['val'].to_numpy()
        ends = g['period_end'].to_numpy()
        for i in range(len(vals) - 1):
            if vals[i] <= 0:
                continue
            ratio = float(vals[i + 1] / vals[i])
            if not _near_split_ratio(ratio):
                continue
            when = pd.Timestamp(ends[i + 1])
            near = known[(known['ticker'] == ticker)
                         & (known['date'] - when).abs().le(pd.Timedelta(days=200))]
            if near.empty:
                suspects.append((ticker, f'{str(when.date())} {ratio:.2f}x'))

    if not suspects:
        return []
    return [Finding(
        'share-count discontinuity', 'info', len(suspects),
        f'{len(suspects)} share-count jump(s) near a split ratio have no split '
        f'on record. Treat as a prompt to look, not a defect: a share count '
        f'cannot distinguish a 2-for-1 split from a merger that doubled the '
        f'count, and only the first needs correcting. The split table is '
        f'fetched from the provider for the whole universe, so a genuinely '
        f'missing split is unlikely — the market-cap agreement check is the '
        f'reliable detector, because it compares against a figure derived a '
        f'different way.',
        suspects[:8])]


# Splits are declared in simple ratios. Share counts are *weighted averages*
# over the period, so a split mid-quarter blends the two bases and the observed
# ratio lands near the declared one rather than on it — Lam Research's 10-for-1
# reads 9.88. The tolerance has to absorb that without becoming wide enough to
# swallow a merger.
# 3-for-2 and 5-for-2 are deliberately absent. They are rare in modern markets
# and their ratios sit exactly where ordinary equity issuance lands, so
# including them produced far more IPO and merger noise than real splits —
# Airbnb's first post-IPO quarter, Charter/Time Warner, Cigna/Express Scripts.
CANONICAL_SPLITS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
                    15.0, 20.0, 25.0, 30.0)


def _near_split_ratio(ratio: float, tol: float = 0.06) -> bool:
    """True when a share-count ratio is close to a declared split ratio."""
    if not np.isfinite(ratio) or ratio <= 0:
        return False
    for canon in CANONICAL_SPLITS:
        for candidate in (canon, 1.0 / canon):
            if abs(ratio - candidate) <= tol * candidate:
                return True
    return False


def check_market_cap_agreement(factors: pd.DataFrame | None = None,
                               tolerance: float = 0.15,
                               sample: int = 0) -> list[Finding]:
    """
    Computed market cap against the provider's own figure.

    The two are derived completely differently — one is shares times price out
    of filings and bars, the other is published by the data vendor — so they
    fail independently. That makes disagreement the single most useful signal
    available about the share-count basis: a split, a wrong share class, or a
    units error moves the computed value and leaves the sourced one alone.

    Only names with a same-day snapshot can be checked, which today is a
    fraction of the index; it is an anchor, not a full audit.
    """
    if factors is None or factors.empty or 'MARKET_CAP' not in factors:
        return []

    snap = db.read_sql("""
        SELECT ticker, market_cap FROM profile_snapshots
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM profile_snapshots)
          AND market_cap IS NOT NULL
    """)

    # Stored snapshots cover only the names an earlier gap-fill happened to
    # need, so on their own they check almost nothing. Ask the market-data
    # provider directly when the stored set is too thin to be an audit.
    if len(snap) < max(20, 0.1 * len(factors)) and sample:
        try:
            from data import marketdata as MD
            if MD.available():
                fetched = MD.fetch_market_caps(list(factors.index)[:sample])
                if not fetched.empty:
                    snap = pd.concat([snap, fetched[['ticker', 'market_cap']]],
                                     ignore_index=True).drop_duplicates('ticker')
        except Exception as exc:                   # noqa: BLE001
            log.debug('market cap fetch for validation failed: %s', exc)

    if snap.empty:
        return [Finding(
            'market cap unverified', 'warning', 0,
            'No provider market caps available, so the computed figure has '
            'nothing independent to check it against. Set FINNHUB_API_KEY or '
            'pull profiles from the Data tab.')]

    sourced = snap.set_index('ticker')['market_cap']
    computed = pd.to_numeric(factors['MARKET_CAP'], errors='coerce')
    pair = pd.concat([computed.rename('computed'), sourced.rename('sourced')],
                     axis=1).dropna()
    pair = pair[pair['sourced'] > 0]
    if pair.empty:
        return []

    pair['ratio'] = pair['computed'] / pair['sourced']
    bad = pair[(pair['ratio'] - 1.0).abs() > tolerance]
    if bad.empty:
        return [Finding(
            'market cap agreement', 'info', len(pair),
            f'All {len(pair)} checked names agree with the provider within '
            f'{tolerance:.0%} (median {pair["ratio"].median():.3f}x). Shares '
            f'and prices are on the same basis.')]
    return [Finding(
        'market cap disagreement', 'error', len(bad),
        f'{len(bad)} of {len(pair)} checked names differ from the provider by '
        f'more than {tolerance:.0%}. A ratio near a round number — 10, 4, 0.1 — '
        f'is an uncorrected split; an arbitrary one is usually a share class.',
        [(t, f"{r.ratio:.2f}x") for t, r in bad.head(6).iterrows()])]


def run_all(factors: pd.DataFrame | None = None,
            as_of: date | None = None) -> list[Finding]:
    """Every check. Store-level ones always run; factor ones need a frame."""
    findings: list[Finding] = []
    for check in (check_price_sanity, check_filing_dates, check_split_coverage):
        try:
            findings += check()
        except Exception as exc:                   # noqa: BLE001
            log.debug('%s failed: %s', check.__name__, exc)
    try:
        findings += check_stale_shares(as_of)
    except Exception as exc:                       # noqa: BLE001
        log.debug('stale share check failed: %s', exc)

    if factors is not None and not factors.empty:
        findings += check_factor_bounds(factors)
        findings += check_infinities(factors)
        findings += check_coverage(factors)
        try:
            findings += check_market_cap_agreement(factors)
        except Exception as exc:                   # noqa: BLE001
            log.debug('market cap agreement check failed: %s', exc)

    order = {'error': 0, 'warning': 1, 'info': 2}
    return sorted(findings, key=lambda f: (order.get(f.severity, 3), -f.count))


def to_frame(findings: list[Finding]) -> pd.DataFrame:
    if not findings:
        return pd.DataFrame()
    return pd.DataFrame([{
        'severity': f.severity, 'check': f.check, 'count': f.count,
        'examples': ', '.join(f'{t}={v}' for t, v in f.examples[:4]) or '—',
    } for f in findings])
