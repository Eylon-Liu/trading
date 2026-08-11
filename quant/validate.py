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
from datetime import date

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
    'CURRENT_RATIO': (0.0, 100.0),
    'PIOTROSKI_F': (0.0, 9.0),
    'MARKET_CAP': (1e7, 1e14),            # $10M to $100T
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


def run_all(factors: pd.DataFrame | None = None,
            as_of: date | None = None) -> list[Finding]:
    """Every check. Store-level ones always run; factor ones need a frame."""
    findings: list[Finding] = []
    for check in (check_price_sanity, check_filing_dates):
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

    order = {'error': 0, 'warning': 1, 'info': 2}
    return sorted(findings, key=lambda f: (order.get(f.severity, 3), -f.count))


def to_frame(findings: list[Finding]) -> pd.DataFrame:
    if not findings:
        return pd.DataFrame()
    return pd.DataFrame([{
        'severity': f.severity, 'check': f.check, 'count': f.count,
        'examples': ', '.join(f'{t}={v}' for t, v in f.examples[:4]) or '—',
    } for f in findings])
