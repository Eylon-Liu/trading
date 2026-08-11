"""
SEC EDGAR provider — point-in-time fundamentals, corporate events, insiders.

This is the backbone of the whole system. Every fact EDGAR returns carries a
`filed` date, which is what makes honest backtesting possible: we can ask what
was *knowable* on a given day rather than what was ultimately true.

Three streams:
  companyfacts  XBRL financial facts -> sec_facts (one row per concept /
                period / filing, so restatements accumulate rather than
                overwrite)
  submissions   filing index -> sec_filings + corporate_events. 8-K item codes
                are already structured, so corporate events need no NLP.
  Form 4        insider transactions -> insider_txns

No API key. SEC asks for a descriptive User-Agent and throttles at 10 req/s;
core.http handles both.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
from lxml import etree

import config
from core import db, http

log = logging.getLogger(__name__)

FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
SUBS_URL = 'https://data.sec.gov/submissions/CIK{cik}.json'
ARCHIVE = 'https://www.sec.gov/Archives/edgar/data/{cik}/{acc}'

# Reverse lookup: us-gaap tag -> our concept name.
_TAG_TO_CONCEPT = {
    tag: concept
    for concept, tags in config.SEC_TAG_MAP.items()
    for tag in tags
}


# ─────────────────────────────────────────────
# TICKER <-> CIK
# ─────────────────────────────────────────────

_cik_cache: dict[str, str] | None = None


def ticker_cik_map() -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK, from the SEC's own file."""
    global _cik_cache
    if _cik_cache is not None:
        return _cik_cache

    payload = http.fetch_json('https://www.sec.gov/files/company_tickers.json',
                              category='members', ttl=60 * 60 * 24 * 7)
    out: dict[str, str] = {}
    if payload:
        for row in payload.values():
            tk = str(row.get('ticker', '')).upper().replace('.', '-')
            if tk:
                out[tk] = str(row['cik_str']).zfill(10)
    _cik_cache = out
    log.info('CIK map: %d tickers', len(out))
    return out


def cik_for(ticker: str) -> str | None:
    """The CIK currently registered against this ticker."""
    return ticker_cik_map().get(str(ticker).upper().replace('.', '-'))


_lineage_cache: dict[str, list[str]] = {}
_CIK_RE = re.compile(r'CIK=(\d{10})')


def cik_lineage(ticker: str) -> list[str]:
    """
    Every CIK that has filed under this ticker, newest registrant first.

    A single CIK is not enough. When a company reorganizes under a holding
    company it becomes a *successor issuer*: a brand-new CIK takes over the
    ticker while the entire filing history stays behind on the old one.
    ExxonMobil did exactly this in July 2026 — `company_tickers.json` maps XOM
    to CIK 2115436 ("ExxonMobil Holdings Corp", 28 filings, 5 weeks old),
    while CIK 34088 ("EXXON MOBIL CORP") holds a thousand filings of history.
    Trusting the ticker file alone silently reduced XOM to 56 facts and made
    every fundamental NaN.

    The two sources disagree usefully: `company_tickers.json` knows who files
    *today*, and EDGAR's company browse knows who holds the *history*. We take
    the union and let the point-in-time gate pick the right row per date.
    """
    key = str(ticker).upper().replace('.', '-')
    if key in _lineage_cache:
        return _lineage_cache[key]

    ciks: list[str] = []
    primary = cik_for(key)
    if primary:
        ciks.append(primary)

    atom = http.fetch_text(
        'https://www.sec.gov/cgi-bin/browse-edgar', category='submissions',
        ttl=60 * 60 * 24 * 7,
        params={'action': 'getcompany', 'CIK': key, 'type': '10-K',
                'dateb': '', 'owner': 'include', 'count': '1', 'output': 'atom'},
    )
    if atom:
        for found in _CIK_RE.findall(atom):
            if found not in ciks:
                ciks.append(found)

    if len(ciks) > 1:
        log.info('%s spans %d CIKs (successor-issuer reorg): %s',
                 key, len(ciks), ciks)

    _lineage_cache[key] = ciks
    return ciks


# ─────────────────────────────────────────────
# COMPANYFACTS -> POINT-IN-TIME FACTS
# ─────────────────────────────────────────────

def _fact_rows(cik: str, ticker: str, payload: dict) -> list[dict]:
    """
    Flatten companyfacts into rows, keeping only the concepts we model.

    Both `start` and `end` are preserved. That matters: AAPL's FY2026 Q3
    NetIncomeLoss returns two facts sharing an end date of 2026-06-27 —
    29,789M for the quarter and 101,464M for the nine months to date. Without
    the span you cannot tell them apart, and every TTM figure built on them
    would be wrong.
    """
    gaap = (payload.get('facts') or {}).get('us-gaap') or {}
    rows: list[dict] = []

    for tag, concept in _TAG_TO_CONCEPT.items():
        node = gaap.get(tag)
        if not node:
            continue
        for unit, facts in (node.get('units') or {}).items():
            for f in facts:
                end = f.get('end')
                filed = f.get('filed')
                val = f.get('val')
                if not end or not filed or val is None:
                    continue
                rows.append({
                    'cik': cik, 'tag': tag, 'unit': unit,
                    'period_start': _d(f.get('start')) or _d(end),
                    'period_end': _d(end),
                    'filed': _d(filed),
                    'form': f.get('form') or '',
                    'ticker': ticker, 'concept': concept,
                    'fy': f.get('fy'), 'fp': f.get('fp'),
                    'val': float(val),
                })
    return rows


def _d(s):
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except (ValueError, TypeError):
        return None


def update_facts(tickers: list[str]) -> int:
    """
    Download and store XBRL facts for each ticker.

    Fetches every CIK in the ticker's lineage, so a company that reorganized
    under a new holding company keeps its pre-reorg history.
    """
    jobs: list[tuple[str, str]] = []
    missing: list[str] = []
    for t in sorted(set(tickers)):
        ciks = cik_lineage(t)
        if not ciks:
            missing.append(t)
            continue
        jobs.extend((t, c) for c in ciks)

    if missing:
        log.info('no CIK for %d tickers (likely non-US): %s',
                 len(missing), missing[:8])

    written = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {
            pool.submit(http.fetch_json, FACTS_URL.format(cik=cik),
                        category='sec_facts'): (t, cik)
            for t, cik in jobs
        }
        for fut in as_completed(futures):
            ticker, cik = futures[fut]
            try:
                payload = fut.result()
            except Exception as exc:              # noqa: BLE001
                log.warning('companyfacts failed for %s: %s', ticker, exc)
                db.record_ingest('sec_facts', ticker, error=str(exc)[:200])
                continue
            if not payload:
                continue
            rows = _fact_rows(cik, ticker, payload)
            if rows:
                written += db.upsert(db.sec_facts, rows)
                db.record_ingest('sec_facts', f'{ticker}:{cik}', rows=len(rows))
                log.info('facts %s (CIK %s): %d rows', ticker, cik, len(rows))

    return written


# ─────────────────────────────────────────────
# SUBMISSIONS -> FILINGS + 8-K CORPORATE EVENTS
# ─────────────────────────────────────────────

def _submissions(cik: str) -> dict | None:
    return http.fetch_json(SUBS_URL.format(cik=cik), category='submissions')


def update_securities(tickers: list[str]) -> int:
    """
    Populate the securities table (name, sector, industry, exchange, country)
    from SEC submissions.

    Deliberately not sourced from yfinance `.info`: that endpoint is the most
    heavily rate-limited provider in the stack and returned nothing during this
    project's first full ingest, which left every name classified "Unknown" and
    silently collapsed the sector-relative z-scores into a single bucket. SEC
    submissions cost a request we already make and never throttle.
    """
    from data.classify import industry_from_description, sector_from_sic

    rows = []
    jobs = [(t, cik_lineage(t)) for t in sorted(set(tickers))]

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {}
        for ticker, ciks in jobs:
            if not ciks:
                continue
            # The first CIK is the current registrant and carries today's
            # classification; later ones are predecessors.
            futures[pool.submit(_submissions, ciks[0])] = (ticker, ciks[0])

        for fut in as_completed(futures):
            ticker, cik = futures[fut]
            payload = fut.result()
            if not payload:
                continue

            addresses = payload.get('addresses') or {}
            business = addresses.get('business') or {}
            exchanges = payload.get('exchanges') or []

            rows.append({
                'ticker': ticker, 'cik': cik,
                'name': payload.get('name') or ticker,
                'sector': sector_from_sic(payload.get('sic')),
                'industry': industry_from_description(payload.get('sicDescription')),
                'exchange': exchanges[0] if exchanges else None,
                'country': business.get('country') or 'US',
                'is_adr': 0,
                'first_seen': date.today(),
                'last_updated': pd.Timestamp.utcnow().to_pydatetime().replace(tzinfo=None),
            })

    n = db.upsert(db.securities, rows)
    log.info('securities: classified %d names from SEC SIC codes', n)
    db.record_ingest('securities', str(date.today()), rows=n)
    return n


def update_filings(tickers: list[str], forms: tuple[str, ...] =
                   ('8-K', '10-K', '10-Q', '4')) -> int:
    """
    Store the filing index and derive corporate events from 8-K item codes.

    The SEC has already classified these, so an executive departure or a
    restructuring charge is a lookup rather than an NLP problem.
    """
    jobs = [(t, c) for t in sorted(set(tickers)) for c in cik_lineage(t)]
    written = 0

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(_submissions, cik): (t, cik) for t, cik in jobs}
        for fut in as_completed(futures):
            ticker, cik = futures[fut]
            payload = fut.result()
            if not payload:
                continue

            recent = (payload.get('filings') or {}).get('recent') or {}
            n = len(recent.get('accessionNumber', []))
            if not n:
                continue

            filing_rows, event_rows = [], []
            for i in range(n):
                form = recent['form'][i]
                if forms and not any(form == f or form.startswith(f + '/')
                                     for f in forms):
                    continue

                acc = recent['accessionNumber'][i]
                filed = _d(recent['filingDate'][i])
                items = (recent.get('items') or [''] * n)[i] or ''

                filing_rows.append({
                    'accession': acc, 'cik': cik, 'ticker': ticker,
                    'form': form, 'items': items, 'filing_date': filed,
                    'report_date': _d((recent.get('reportDate') or [''] * n)[i]),
                    'primary_doc': (recent.get('primaryDocument') or [''] * n)[i],
                })

                if form.startswith('8-K') and items:
                    for code in {c.strip() for c in items.split(',') if c.strip()}:
                        mapped = config.EIGHTK_ITEMS.get(code)
                        if not mapped:
                            continue
                        category, subtype = mapped
                        event_rows.append({
                            'ticker': ticker, 'filed': filed, 'accession': acc,
                            'item_code': code, 'category': category,
                            'subtype': subtype, 'detail': None,
                        })

            if filing_rows:
                written += db.upsert(db.sec_filings, filing_rows)
            if event_rows:
                db.upsert(db.corporate_events, event_rows)
            db.record_ingest('filings', ticker, rows=len(filing_rows))
            log.info('filings %s: %d filings, %d events',
                     ticker, len(filing_rows), len(event_rows))

    return written


# ─────────────────────────────────────────────
# FORM 4 -> INSIDER TRANSACTIONS
# ─────────────────────────────────────────────

F4_NS_STRIP = re.compile(r'\sxmlns="[^"]+"')
_XSL_PREFIX = re.compile(r'^xsl[^/]*/', re.I)


def _raw_form_doc(primary_doc: str | None) -> str:
    """
    Strip EDGAR's XSL rendering prefix to reach the machine-readable filing.

    A Form 4's `primaryDocument` is reported as `xslF345X06/form4.xml`, which
    is the *styled HTML view* — fetching it returns a `<!DOCTYPE html>` page
    that no XML parser will accept. The raw XML sits at the same accession
    path with the prefix removed.
    """
    doc = str(primary_doc or 'form4.xml')
    return _XSL_PREFIX.sub('', doc)


def _parse_form4(raw: bytes, ticker: str, cik: str, acc: str,
                 filed: date) -> list[dict]:
    """Extract non-derivative transactions from a Form 4 ownership document."""
    try:
        text = raw.decode('utf-8', errors='replace')
        text = F4_NS_STRIP.sub('', text, count=1)   # Form 4 namespacing is inconsistent
        root = etree.fromstring(text.encode('utf-8'))
    except (etree.XMLSyntaxError, ValueError):
        return []

    owner = root.findtext('.//reportingOwnerId/rptOwnerName') or 'unknown'
    roles = []
    for tag, label in (('isDirector', 'Director'), ('isOfficer', 'Officer'),
                       ('isTenPercentOwner', '10% Owner')):
        v = root.findtext(f'.//reportingOwnerRelationship/{tag}')
        if v and v.strip() in ('1', 'true'):
            roles.append(label)
    title = root.findtext('.//reportingOwnerRelationship/officerTitle')
    if title:
        roles.append(title.strip())

    rows = []
    for txn in root.findall('.//nonDerivativeTransaction'):
        code = txn.findtext('.//transactionCoding/transactionCode')
        if not code:
            continue
        txn_date = _d(txn.findtext('.//transactionDate/value'))
        shares = _num(txn.findtext('.//transactionShares/value'))
        price = _num(txn.findtext('.//transactionPricePerShare/value'))
        acq_disp = txn.findtext('.//transactionAcquiredDisposedCode/value')

        if shares is not None and acq_disp == 'D':
            shares = -abs(shares)

        rows.append({
            'accession': acc, 'ticker': ticker, 'insider': owner[:120],
            'txn_date': txn_date or filed, 'txn_code': code.strip(),
            'cik': cik, 'filed': filed, 'role': ', '.join(roles)[:120] or None,
            'shares': shares, 'price': price,
            'value': (abs(shares) * price) if (shares is not None and price) else None,
        })
    return rows


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def update_insiders(tickers: list[str], lookback_days: int = 730,
                    max_per_ticker: int = 250) -> int:
    """
    Parse recent Form 4 filings into individual transactions.

    Bounded deliberately: AAPL alone has 587 Form 4s on file, so an unbounded
    backfill across a large universe would be tens of thousands of requests.
    The window covers the horizon the insider factor actually uses.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    written = 0

    for ticker in sorted(set(tickers)):
        if not cik_lineage(ticker):
            continue

        # Join back to the filing's own CIK: Form 4s for a company that
        # reorganized are split across the lineage.
        pending = db.read_sql("""
            SELECT accession, filing_date, primary_doc, cik FROM sec_filings
            WHERE ticker = :t AND form = '4' AND filing_date >= :c
            ORDER BY filing_date DESC LIMIT :n
        """, {'t': ticker, 'c': str(cutoff), 'n': max_per_ticker})
        if pending.empty:
            continue

        done = set(db.read_sql(
            'SELECT DISTINCT accession FROM insider_txns WHERE ticker = :t',
            {'t': ticker})['accession'])

        jobs = [r for _i, r in pending.iterrows() if r['accession'] not in done]
        if not jobs:
            continue

        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futures = {}
            for r in jobs:
                acc_clean = r['accession'].replace('-', '')
                url = (ARCHIVE.format(cik=int(r['cik']), acc=acc_clean)
                       + '/' + _raw_form_doc(r['primary_doc']))
                futures[pool.submit(http.fetch, url, category='submissions',
                                    ttl=-1)] = r
            for fut in as_completed(futures):
                r = futures[fut]
                raw = fut.result()
                if not raw:
                    continue
                rows.extend(_parse_form4(
                    raw, ticker, r['cik'], r['accession'],
                    pd.to_datetime(r['filing_date']).date()))

        if rows:
            written += db.upsert(db.insider_txns, rows)
            log.info('insiders %s: %d transactions from %d filings',
                     ticker, len(rows), len(jobs))
        db.record_ingest('insiders', ticker, rows=len(rows))

    return written


# ─────────────────────────────────────────────
# READS  (point-in-time gated)
# ─────────────────────────────────────────────

def facts_asof(tickers: list[str], as_of: date | str,
               concepts: list[str] | None = None) -> pd.DataFrame:
    """
    Every fact knowable at `as_of`.

    Filters on `filed`, never `period_end` — the gap between them is real
    (AAPL's Q3 ending 2026-06-27 was not public until 2026-07-31) and ignoring
    it is the most common way a backtest silently cheats.
    """
    if not tickers:
        return pd.DataFrame()

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params['ts'] = str(pd.to_datetime(as_of).date())

    clause = ''
    if concepts:
        cph = ','.join(f':c{i}' for i in range(len(concepts)))
        params.update({f'c{i}': c for i, c in enumerate(concepts)})
        clause = f'AND concept IN ({cph})'

    return db.read_sql(f"""
        SELECT ticker, concept, tag, unit, period_start, period_end,
               filed, form, fy, fp, val
        FROM sec_facts
        WHERE ticker IN ({ph}) AND filed <= :ts {clause}
        ORDER BY ticker, concept, period_end, filed
    """, params, parse_dates=['period_start', 'period_end', 'filed'])


def events_asof(tickers: list[str], as_of: date | str,
                lookback_days: int = 90) -> pd.DataFrame:
    """8-K corporate events filed in the window ending at `as_of`."""
    if not tickers:
        return pd.DataFrame()
    end = pd.to_datetime(as_of).date()
    start = end - timedelta(days=lookback_days)

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params.update({'lo': str(start), 'hi': str(end)})

    return db.read_sql(f"""
        SELECT ticker, filed, item_code, category, subtype, accession
        FROM corporate_events
        WHERE ticker IN ({ph}) AND filed BETWEEN :lo AND :hi
        ORDER BY filed DESC
    """, params, parse_dates=['filed'])


def insiders_asof(tickers: list[str], as_of: date | str,
                  lookback_days: int = 180) -> pd.DataFrame:
    """Insider transactions filed in the window ending at `as_of`."""
    if not tickers:
        return pd.DataFrame()
    end = pd.to_datetime(as_of).date()
    start = end - timedelta(days=lookback_days)

    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params.update({'lo': str(start), 'hi': str(end)})

    return db.read_sql(f"""
        SELECT ticker, filed, txn_date, insider, role, txn_code,
               shares, price, value
        FROM insider_txns
        WHERE ticker IN ({ph}) AND filed BETWEEN :lo AND :hi
        ORDER BY filed DESC
    """, params, parse_dates=['filed', 'txn_date'])
