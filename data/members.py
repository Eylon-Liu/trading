"""
Point-in-time index membership.

Using today's constituent list to score a past date is survivorship bias: the
list only contains names that survived, so any backtest built on it is quietly
loaded with winners. The original app had this bug in full — hardcoded Dow and
Russell lists, plus a Wikipedia scrape of the *current* table.

Three free sources, each with a different tradeoff:

  wiki_revision  MediaWiki revision API — fetch the constituents page as it
                 stood on any past date. Long history (~2007+), day-level
                 granularity, gives tickers directly. This is the spine.
  nport          The ETF's own filed holdings (SEC N-PORT). Authoritative and
                 carries real weights, but quarterly, 2019+, and lags the
                 period end by ~2 months. Holdings identify securities by
                 name/CUSIP/ISIN with no ticker, so it is used to validate and
                 weight rather than to enumerate.
  wiki_changes   The "Selected changes" table — cheap incremental deltas.

Rows land in index_members(index_symbol, ticker, as_of, source, confidence),
and `as_of` is always the date the membership was *knowable*, which for N-PORT
is the filing date and not the period end.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

import pandas as pd
from bs4 import BeautifulSoup
from lxml import etree

import config
from core import db, http

log = logging.getLogger(__name__)

WIKI_API = 'https://en.wikipedia.org/w/api.php'
WIKI_REV = 'https://en.wikipedia.org/w/index.php'
NPORT_NS = {'n': 'http://www.sec.gov/edgar/nport'}

# Header labels that identify the ticker column across the various pages and
# across a decade of edits to them.
_TICKER_HEADERS = {'symbol', 'ticker', 'ticker symbol', 'symbol/ticker'}
_TICKER_RE = re.compile(r'^[A-Z][A-Z0-9]{0,5}([.\-][A-Z])?$')


# ─────────────────────────────────────────────
# TICKER NORMALIZATION
# ─────────────────────────────────────────────

def normalize_ticker(raw: str | None) -> str | None:
    """
    Canonicalize to the Yahoo convention (BRK.B -> BRK-B).

    Returns None for anything that does not look like a ticker, which keeps
    footnote markers and stray table text out of the universe.
    """
    if not raw:
        return None
    t = str(raw).strip().upper()
    t = re.sub(r'\[.*?\]', '', t)            # strip Wikipedia footnote markers
    t = t.replace('.', '-').replace('​', '').strip()
    if not t or not _TICKER_RE.match(t):
        return None
    return t


# Corporate-form words that carry no identifying information. Deliberately
# excludes bare single letters: "A O Smith" and "BF-B" need theirs.
_CORP_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'the',
    'plc', 'ltd', 'limited', 'holdings', 'holding', 'group', 'class', 'cl',
    'sa', 'nv', 'ag', 'lp', 'llc', 'lc', 'trust', 'adr', 'and', 'of', 'new',
    'com', 'cos', 'intl', 'international',
}


def _normalize_company(name: str) -> str:
    """
    Squash a company name to a matchable key for N-PORT reconciliation.

    N-PORT and the SEC ticker file disagree in four systematic ways, all
    handled here: SEC appends a state-of-incorporation suffix ("APPLIED
    MATERIALS INC /DE"), one side writes "&" where the other writes "and",
    SEC uses surname-first legal ordering ("SMITH A O CORP" vs "A O Smith
    Corp"), and corporate-form words vary freely. Sorting the surviving
    tokens makes word order irrelevant.
    """
    n = (name or '').lower()
    n = re.sub(r'[/\\][a-z]{2}[/\\]?\s*$', ' ', n)   # trailing /de, /ma/, \de\
    n = n.replace('&', ' and ')
    n = re.sub(r'\bclass\s+[abc]\b', ' ', n)         # "class a", not bare "a"
    n = re.sub(r'[^a-z0-9 ]', ' ', n)

    tokens = [t for t in n.split() if t and t not in _CORP_STOPWORDS]
    return ' '.join(sorted(set(tokens)))


# ─────────────────────────────────────────────
# SOURCE 1 — MEDIAWIKI REVISIONS (the spine)
# ─────────────────────────────────────────────

def _revision_at(page: str, as_of: date) -> tuple[int, datetime] | None:
    """Latest revision id of `page` at or before `as_of`."""
    payload = http.fetch_json(
        WIKI_API, category='members',
        params={
            'action': 'query', 'prop': 'revisions', 'titles': page,
            'rvlimit': 1, 'rvdir': 'older', 'rvprop': 'ids|timestamp',
            'rvstart': f'{as_of.isoformat()}T23:59:59Z', 'format': 'json',
        },
    )
    if not payload:
        return None
    try:
        pages = payload['query']['pages']
        revs = next(iter(pages.values())).get('revisions') or []
        if not revs:
            return None
        rev = revs[0]
        ts = datetime.strptime(rev['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
        return int(rev['revid']), ts
    except (KeyError, StopIteration, ValueError) as exc:
        log.warning('revision lookup failed for %s @ %s: %s', page, as_of, exc)
        return None


def _parse_constituent_table(html: str) -> list[str]:
    """
    Pull tickers out of a constituents page.

    Deliberately structural rather than keyed on `id="constituents"`: that id
    did not exist on older revisions, and the whole point here is reading
    decade-old markup.
    """
    soup = BeautifulSoup(html, 'lxml')
    best: list[str] = []

    for table in soup.find_all('table', {'class': 'wikitable'}):
        header_row = table.find('tr')
        if not header_row:
            continue
        headers = [th.get_text(' ', strip=True).lower()
                   for th in header_row.find_all(['th', 'td'])]

        col = next((i for i, h in enumerate(headers) if h in _TICKER_HEADERS), None)
        if col is None:
            col = next((i for i, h in enumerate(headers)
                        if 'symbol' in h or 'ticker' in h), None)
        if col is None:
            continue

        found: list[str] = []
        for tr in table.find_all('tr')[1:]:
            cells = tr.find_all(['td', 'th'])
            if len(cells) <= col:
                continue
            t = normalize_ticker(cells[col].get_text(' ', strip=True))
            if t:
                found.append(t)

        # A constituents table beats a "recent changes" table, which also has
        # a Symbol column but far fewer rows.
        if len(found) > len(best):
            best = found

    # Preserve order, drop duplicates.
    seen, out = set(), []
    for t in best:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def members_from_wiki(index_symbol: str, as_of: date) -> list[str]:
    """Constituents of `index_symbol` as the page recorded them on `as_of`."""
    page = config.INDEX_WIKI_PAGE.get(index_symbol)
    if not page:
        return []

    rev = _revision_at(page, as_of)
    if rev is None:
        return []
    revid, _ts = rev

    # A given revision's content is immutable, so cache it forever.
    html = http.fetch_text(WIKI_REV, params={'oldid': revid},
                           category='members', ttl=-1)
    if not html:
        return []
    return _parse_constituent_table(html)


# ─────────────────────────────────────────────
# SOURCE 2 — SEC N-PORT (authoritative, lagged)
# ─────────────────────────────────────────────

def _cusip_to_ticker_map() -> dict[str, str]:
    """Name-key -> ticker, built from the SEC's own ticker file."""
    payload = http.fetch_json('https://www.sec.gov/files/company_tickers.json',
                              category='members', ttl=60 * 60 * 24 * 7)
    if not payload:
        return {}
    out = {}
    for row in payload.values():
        key = _normalize_company(row.get('title', ''))
        tk = normalize_ticker(row.get('ticker'))
        if key and tk:
            out.setdefault(key, tk)
    return out


def nport_holdings(index_symbol: str, max_filings: int = 40) -> pd.DataFrame:
    """
    Holdings from the ETF's N-PORT filings, keyed on the *filing* date.

    N-PORT identifies securities by name, CUSIP, and ISIN but never by ticker,
    so names are reconciled against the SEC ticker file. Unmatched names are
    reported rather than silently dropped.
    """
    cik = config.INDEX_ETF_CIK.get(index_symbol)
    if not cik:
        return pd.DataFrame()

    subs = http.fetch_json(f'https://data.sec.gov/submissions/CIK{cik}.json',
                           category='submissions')
    if not subs:
        return pd.DataFrame()

    recent = subs.get('filings', {}).get('recent', {})
    forms = recent.get('form', [])
    idxs = [i for i, f in enumerate(forms) if str(f).startswith('NPORT-P')][:max_filings]
    if not idxs:
        return pd.DataFrame()

    name_map = _cusip_to_ticker_map()
    rows, unmatched = [], 0

    for i in idxs:
        acc_raw = recent['accessionNumber'][i]
        acc = acc_raw.replace('-', '')
        filed = pd.to_datetime(recent['filingDate'][i]).date()
        period = pd.to_datetime(recent['reportDate'][i]).date()

        url = (f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}'
               f'/primary_doc.xml')
        raw = http.fetch(url, category='members', ttl=-1)  # immutable filing
        if not raw:
            continue

        try:
            root = etree.fromstring(raw)
        except etree.XMLSyntaxError as exc:
            log.warning('bad N-PORT XML %s: %s', acc_raw, exc)
            continue

        for sec in root.findall('.//n:invstOrSec', NPORT_NS):
            name = (sec.findtext('n:name', namespaces=NPORT_NS) or '').strip()
            asset_cat = sec.findtext('n:assetCat', namespaces=NPORT_NS)
            if asset_cat and asset_cat != 'EC':      # equity common only
                continue

            ticker = name_map.get(_normalize_company(name))
            if not ticker:
                unmatched += 1
                continue

            try:
                pct = float(sec.findtext('n:pctVal', namespaces=NPORT_NS) or 'nan')
            except ValueError:
                pct = float('nan')

            rows.append({
                'index_symbol': index_symbol, 'ticker': ticker,
                'as_of': filed,               # knowable date, not period end
                'period_end': period, 'source': 'nport',
                'confidence': 1.0, 'weight': pct,
            })

    if unmatched:
        log.info('%s N-PORT: %d holdings unmatched to a ticker', index_symbol, unmatched)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# SOURCE 3 — WIKIPEDIA CHANGES TABLE
# ─────────────────────────────────────────────

def membership_changes(index_symbol: str) -> pd.DataFrame:
    """Add/remove events from the page's 'Selected changes' table."""
    page = config.INDEX_WIKI_PAGE.get(index_symbol)
    if not page:
        return pd.DataFrame()

    html = http.fetch_text(WIKI_REV, params={'title': page}, category='members')
    if not html:
        return pd.DataFrame()

    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table', {'id': 'changes'})
    if table is None:
        for t in soup.find_all('table', {'class': 'wikitable'}):
            head = t.get_text(' ', strip=True)[:250].lower()
            if 'added' in head and 'removed' in head:
                table = t
                break
    if table is None:
        return pd.DataFrame()

    rows = []
    for tr in table.find_all('tr')[2:]:          # two header rows
        cells = [c.get_text(' ', strip=True) for c in tr.find_all('td')]
        if len(cells) < 3:
            continue
        try:
            eff = pd.to_datetime(cells[0], errors='coerce')
        except Exception:
            continue
        if pd.isna(eff):
            continue
        added = normalize_ticker(cells[1]) if len(cells) > 1 else None
        removed = normalize_ticker(cells[3]) if len(cells) > 3 else None
        if added:
            rows.append({'index_symbol': index_symbol, 'ticker': added,
                         'effective': eff.date(), 'action': 'added'})
        if removed:
            rows.append({'index_symbol': index_symbol, 'ticker': removed,
                         'effective': eff.date(), 'action': 'removed'})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# BACKFILL & READ
# ─────────────────────────────────────────────

def _membership_changed(index_symbol: str, source: str, as_of: date,
                        tickers: list[str]) -> bool:
    """
    True when `tickers` differs from the newest stored snapshot before `as_of`.

    Also true when there is no prior snapshot — the first observation is always
    worth keeping.
    """
    prior = db.read_sql("""
        SELECT MAX(as_of) AS d FROM index_members
        WHERE index_symbol = :i AND source = :s AND as_of < :t
    """, {'i': index_symbol, 's': source, 't': as_of.isoformat()})

    if prior.empty or pd.isna(prior.iloc[0]['d']):
        return True

    previous = db.read_sql("""
        SELECT ticker FROM index_members
        WHERE index_symbol = :i AND source = :s AND as_of = :d
    """, {'i': index_symbol, 's': source, 'd': prior.iloc[0]['d']})

    return set(previous['ticker']) != set(tickers)


def backfill(index_symbol: str, start: date, end: date | None = None,
             freq: str = 'ME') -> int:
    """
    Snapshot membership at each period end between `start` and `end`.

    Writes wiki_revision rows for every date and, where the ETF files them,
    N-PORT rows too. Both live in the table; disagreements surface through
    `compare_sources` rather than being silently reconciled.
    """
    end = end or date.today()
    stamps = pd.date_range(start=start, end=end, freq=freq)
    if len(stamps) == 0 or stamps[-1].date() < end:
        stamps = stamps.append(pd.DatetimeIndex([pd.Timestamp(end)]))

    written = 0
    skipped = 0
    for ts in stamps:
        as_of = ts.date()
        tickers = members_from_wiki(index_symbol, as_of)
        if not tickers:
            continue

        # Only snapshot when the constituent set actually changed. An index
        # reconstitutes a few times a year, so a daily pull would otherwise
        # write ~500 identical rows a day — roughly 180k rows a year saying
        # nothing. Resolution is unaffected: members_asof takes the most
        # recent snapshot at or before a date, and the most recent *change* is
        # exactly the state on that date.
        if not _membership_changed(index_symbol, 'wiki_revision', as_of, tickers):
            skipped += 1
            continue

        written += db.upsert(db.index_members, [{
            'index_symbol': index_symbol, 'ticker': t, 'as_of': as_of,
            'source': 'wiki_revision', 'confidence': 0.9, 'weight': None,
        } for t in tickers])
        log.info('%s @ %s -> %d members (changed)', index_symbol, as_of,
                 len(tickers))

    if skipped:
        log.info('%s: %d snapshot(s) unchanged, not stored', index_symbol,
                 skipped)

    nport = nport_holdings(index_symbol)
    if not nport.empty:
        written += db.upsert(db.index_members, [
            {k: v for k, v in r.items() if k != 'period_end'}
            for r in nport.to_dict('records')
        ])
        log.info('%s N-PORT -> %d holding rows', index_symbol, len(nport))

    db.record_ingest('members', index_symbol, rows=written)
    return written


def members_asof(index_symbol: str, as_of: date | str,
                 source: str = 'auto') -> list[str]:
    """
    Constituents of `index_symbol` as knowable on `as_of` — the point-in-time
    read every caller should use.

    Takes the most recent snapshot at or before `as_of`. Because N-PORT rows
    are stamped with the filing date, respecting the ~2-month reporting lag
    falls out of the same comparison rather than needing a special case.
    """
    as_of = pd.to_datetime(as_of).date()

    if source == 'auto':
        src_clause = ''
        params = {'idx': index_symbol, 'ts': as_of.isoformat()}
    else:
        src_clause = 'AND source = :src'
        params = {'idx': index_symbol, 'ts': as_of.isoformat(), 'src': source}

    snap = db.read_sql(f"""
        SELECT as_of, source FROM index_members
        WHERE index_symbol = :idx AND as_of <= :ts {src_clause}
        ORDER BY as_of DESC LIMIT 1
    """, params)
    if snap.empty:
        return []

    chosen_date, chosen_src = snap.iloc[0]['as_of'], snap.iloc[0]['source']
    rows = db.read_sql("""
        SELECT DISTINCT ticker FROM index_members
        WHERE index_symbol = :idx AND as_of = :d AND source = :s
        ORDER BY ticker
    """, {'idx': index_symbol, 'd': str(chosen_date), 's': chosen_src})
    return rows['ticker'].tolist()


def compare_sources(index_symbol: str, as_of: date | str) -> dict:
    """
    Cross-validate the crowd-sourced spine against the ETF's own filing.

    Surfaced in the Data tab: disagreement is information, so it is reported
    rather than resolved behind the scenes.
    """
    wiki = set(members_asof(index_symbol, as_of, source='wiki_revision'))
    nport = set(members_asof(index_symbol, as_of, source='nport'))
    if not wiki or not nport:
        return {'comparable': False, 'wiki_n': len(wiki), 'nport_n': len(nport)}

    both = wiki & nport
    return {
        'comparable': True,
        'wiki_n': len(wiki), 'nport_n': len(nport),
        'agree_n': len(both),
        'agreement_pct': round(100 * len(both) / max(len(wiki | nport), 1), 1),
        'wiki_only': sorted(wiki - nport)[:25],
        'nport_only': sorted(nport - wiki)[:25],
    }


def latest_members(index_symbol: str) -> list[str]:
    """Today's constituents, fetching and storing them if absent."""
    today = date.today()
    existing = members_asof(index_symbol, today)
    if existing:
        return existing

    tickers = members_from_wiki(index_symbol, today)
    if tickers:
        db.upsert(db.index_members, [{
            'index_symbol': index_symbol, 'ticker': t, 'as_of': today,
            'source': 'wiki_revision', 'confidence': 0.9, 'weight': None,
        } for t in tickers])
    return tickers


def coverage(index_symbol: str) -> pd.DataFrame:
    """Snapshot dates held per source — for the Data tab."""
    return db.read_sql("""
        SELECT source, COUNT(DISTINCT as_of) AS snapshots,
               MIN(as_of) AS earliest, MAX(as_of) AS latest,
               COUNT(*) AS rows
        FROM index_members WHERE index_symbol = :idx
        GROUP BY source ORDER BY source
    """, {'idx': index_symbol})
