"""
LLM earnings review — read each new results release and grade it.

For every 8-K Item 2.02 ("results of operations") the company files, the
press release (Exhibit 99) is pulled from EDGAR and handed to the configured
model together with figures the quant engine has already computed: growth,
margins, returns on capital, valuation yields, the market's reaction to the
release, the consensus EPS estimate where stored, and recent headlines. The
model answers one question for a 6–18 month holder — *is this worth buying?*
— as structured JSON, which is stored and becomes the LLM_EARNINGS_VIEW
factor.

Why this is forward-only, and must stay so
------------------------------------------
A review is stamped with the day it was *written* (`assessed_on`), and only
that date is used point-in-time. Asking a model today about a 2023 release
is not a 2023 opinion: the model's training data already contains what the
stock did next. Back-dating such reviews would leak the future into every
backtest that touched them. So the factor has history only from the day the
routine started writing reviews, and its value is established by the live
track record, not by a backtest.

Cost control: one review per filing (cached by accession), only for names
that just reported, prioritised portfolio → watchlist → current picks, and
capped per run (QUANT_LLM_MAX_REVIEWS, default 25).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from core import db, http
from nlp import providers

log = logging.getLogger(__name__)

VERSION = 1
MAX_RELEASE_CHARS = 14_000
LOOKBACK_DAYS = 10
VIEW_WINDOW_DAYS = 120

VERDICTS = ('buy', 'accumulate', 'hold', 'avoid')
GUIDANCE = ('raised', 'maintained', 'lowered', 'withdrawn', 'none')
VALUATION = ('cheap', 'fair', 'expensive', 'unclear')
CONVICTION_WEIGHT = {'low': 0.5, 'medium': 0.8, 'high': 1.0}

INDEX_URL = ('https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/'
             '{acc}-index.htm')
DOC_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}'

SYSTEM = """\
You are a buy-side equity analyst reviewing a company's latest quarterly \
results for a long-only investor with a 6-18 month holding period. You are \
given the earnings press release and figures already computed from SEC \
filings and market data. Judge only from what you are given; do not use \
outside knowledge of later events. Answer with a single JSON object and \
nothing else — no prose, no markdown fences."""

PROMPT = """Company: {name} ({ticker}) — sector: {sector}
Results filed with the SEC on {filed}. Today is {today}.

Computed figures (from filings through the previous quarter, and prices):
{figures}

Market reaction: {reaction}
Consensus EPS estimate before the release: {estimate}

Recent headlines:
{headlines}

Earnings press release (may be truncated):
<<<
{release}
>>>

Return JSON with exactly these keys:
{{
  "quarter_quality": number from -2 to 2 (versus the estimate, the prior-year quarter and the company's own trend),
  "guidance": one of "raised", "maintained", "lowered", "withdrawn", "none",
  "outlook": number from -2 to 2 (direction of the fundamentals over the next 2-4 quarters),
  "valuation_view": one of "cheap", "fair", "expensive", "unclear" (from the yields given, relative to the quality),
  "verdict": one of "buy", "accumulate", "hold", "avoid" (for a 6-18 month holder, at today's price),
  "conviction": one of "low", "medium", "high",
  "score": number from -2 to 2 (overall attractiveness; 0 = no edge),
  "positives": up to 3 short strings,
  "negatives": up to 3 short strings,
  "summary": one or two sentences, under 60 words
}}"""


# ─────────────────────────────────────────────
# SEC: THE PRESS RELEASE
# ─────────────────────────────────────────────

def _acc_parts(cik: str, accession: str) -> dict:
    return {'cik': int(str(cik).lstrip('0') or 0), 'acc': accession,
            'acc_nodash': accession.replace('-', '')}


def filing_documents(cik: str, accession: str) -> list[dict]:
    """Documents in a filing, from its EDGAR index page: type, name, url."""
    from bs4 import BeautifulSoup
    parts = _acc_parts(cik, accession)
    page = http.fetch_text(INDEX_URL.format(**parts), category='submissions')
    if not page:
        return []
    soup = BeautifulSoup(page, 'lxml')
    out = []
    for tr in soup.select('table.tableFile tr'):
        tds = tr.find_all('td')
        if len(tds) < 4:
            continue
        link = tds[2].find('a')
        if not link or not link.get('href'):
            continue
        name = link.get('href').rsplit('/', 1)[-1]
        if name.startswith('ix?doc='):
            name = name.split('=', 1)[1].rsplit('/', 1)[-1]
        out.append({'type': tds[3].get_text(strip=True).upper(), 'name': name,
                    'description': tds[1].get_text(strip=True),
                    'url': DOC_URL.format(doc=name, **parts)})
    return out


def html_to_text(raw: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, 'lxml')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(' ')
    return re.sub(r'\s+', ' ', text).strip()


def press_release_text(cik: str, accession: str,
                       primary_doc: str | None = None,
                       max_chars: int = MAX_RELEASE_CHARS) -> str | None:
    """Exhibit 99 text of a results 8-K; falls back to the 8-K body."""
    docs = filing_documents(cik, accession)
    ex99 = sorted((d for d in docs if d['type'].startswith('EX-99')),
                  key=lambda d: (d['type'] != 'EX-99.1', d['type']))
    urls = [d['url'] for d in ex99[:2]]
    if not urls and primary_doc:
        urls = [DOC_URL.format(doc=primary_doc, **_acc_parts(cik, accession))]
    texts = []
    for u in urls:
        raw = http.fetch_text(u, category='submissions')
        if raw:
            texts.append(html_to_text(raw))
    text = '\n\n'.join(t for t in texts if t)
    return text[:max_chars] if text else None


# ─────────────────────────────────────────────
# CONTEXT
# ─────────────────────────────────────────────

_FIGURES = [
    ('REVENUE_GROWTH_1Y', 'Revenue growth, trailing year', 'pct'),
    ('EARNINGS_GROWTH_1Y', 'Net income growth, trailing year', 'pct'),
    ('OPERATING_MARGIN', 'Operating margin', 'pct'),
    ('ROIC', 'Return on invested capital', 'pct'),
    ('FCF_YIELD', 'Free-cash-flow yield', 'pct'),
    ('EARNINGS_YIELD', 'Earnings yield (E/P)', 'pct'),
    ('NET_DEBT_TO_EQUITY', 'Net debt / equity', 'x'),
    ('PIOTROSKI_F', 'Piotroski F-score (0-9)', 'n'),
    ('MOM_12_1', '12-1 month price momentum', 'pct'),
    ('PCT_VS_MA200', 'Price vs 200-day average', 'pct'),
]


def _fmt(v, kind) -> str:
    if v is None or not np.isfinite(v):
        return 'n/a'
    return f'{v:+.1%}' if kind == 'pct' else f'{v:.2f}' if kind == 'x' else f'{v:.0f}'


def build_context(ticker: str, filed: date, as_of: date) -> dict:
    """Everything the prompt needs except the release text."""
    from quant import factors as FA

    meta = db.read_sql('SELECT name, sector FROM securities WHERE ticker = :t',
                       {'t': ticker})
    name = meta['name'].iloc[0] if not meta.empty and meta['name'].iloc[0] else ticker
    sector = meta['sector'].iloc[0] if not meta.empty and meta['sector'].iloc[0] else 'Unknown'

    figures, reaction = [], 'not available'
    try:
        raw = FA.build_all([ticker], as_of)
        if not raw.empty and ticker in raw.index:
            row = raw.loc[ticker]
            figures = [f'- {label}: {_fmt(row.get(col), kind)}'
                       for col, label, kind in _FIGURES]
            r = row.get('EARNINGS_REACTION')
            if r is not None and np.isfinite(r):
                reaction = (f'{r:+.1%} versus the S&P 500 over the three '
                            f'sessions around the release')
    except Exception as exc:                       # noqa: BLE001
        log.debug('factor context failed for %s: %s', ticker, exc)

    est = db.read_sql("""
        SELECT eps_estimate FROM earnings_calendar
        WHERE ticker = :t AND report_date BETWEEN :lo AND :hi
          AND eps_estimate IS NOT NULL
        ORDER BY report_date DESC LIMIT 1
    """, {'t': ticker, 'lo': str(filed - timedelta(days=4)),
          'hi': str(filed + timedelta(days=2))})
    estimate = (f'{est["eps_estimate"].iloc[0]:.2f} per share'
                if not est.empty else 'not available')

    news = db.read_sql("""
        SELECT published, title FROM news
        WHERE ticker = :t AND published >= :lo AND published < :hi
        ORDER BY published DESC LIMIT 8
    """, {'t': ticker, 'lo': str(filed - timedelta(days=2)),
          'hi': str(as_of + timedelta(days=1))}, parse_dates=['published'])
    headlines = '\n'.join(f'- {p:%Y-%m-%d} {t[:140]}'
                          for p, t in zip(news['published'], news['title'])) \
        or '- none stored'

    return {'name': name, 'sector': sector,
            'figures': '\n'.join(figures) or '- not available',
            'reaction': reaction, 'estimate': estimate, 'headlines': headlines}


# ─────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────

def _num(v, lo=-2.0, hi=2.0) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return float(np.clip(f, lo, hi)) if np.isfinite(f) else None


def _choice(v, allowed, default) -> str:
    s = str(v or '').strip().lower()
    return s if s in allowed else default


def parse_review(text: str | None) -> dict | None:
    """
    Model output -> validated review, or None when it is not usable.

    Tolerates markdown fences and prose around the object; clamps numbers to
    the scale and maps unknown labels to neutral ones. A response with no
    usable `score` is rejected outright rather than stored as zero — a
    missing opinion must not read as a neutral one.
    """
    if not text:
        return None
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    score = _num(obj.get('score'))
    if score is None:
        return None

    def strings(v):
        if not isinstance(v, list):
            return []
        return [str(x).strip()[:160] for x in v if str(x).strip()][:3]

    return {
        'score': score,
        'quarter_quality': _num(obj.get('quarter_quality')),
        'outlook': _num(obj.get('outlook')),
        'guidance': _choice(obj.get('guidance'), GUIDANCE, 'none'),
        'valuation_view': _choice(obj.get('valuation_view'), VALUATION, 'unclear'),
        'verdict': _choice(obj.get('verdict'), VERDICTS, 'hold'),
        'conviction': _choice(obj.get('conviction'), CONVICTION_WEIGHT, 'low'),
        'positives': strings(obj.get('positives')),
        'negatives': strings(obj.get('negatives')),
        'summary': str(obj.get('summary') or '').strip()[:400],
    }


# ─────────────────────────────────────────────
# RUNNING
# ─────────────────────────────────────────────

def enabled() -> bool:
    flag = (config.QUANT_LLM_REVIEWS or 'auto').lower()
    if flag in ('0', 'off', 'false', 'no'):
        return False
    return providers.available()


def pending(as_of: date | None = None, tickers: list[str] | None = None,
            lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Results filings in the window that have no review yet."""
    as_of = as_of or date.today()
    clause, params = '', {'lo': str(as_of - timedelta(days=lookback_days)),
                          'hi': str(as_of)}
    if tickers:
        ph = ','.join(f':t{i}' for i in range(len(tickers)))
        clause = f'AND e.ticker IN ({ph})'
        params.update({f't{i}': t for i, t in enumerate(tickers)})
    return db.read_sql(f"""
        SELECT e.ticker, e.accession, e.filed, f.cik, f.primary_doc
        FROM corporate_events e
        JOIN sec_filings f ON f.accession = e.accession
        WHERE e.item_code = '2.02' AND f.form = '8-K'
          AND e.filed BETWEEN :lo AND :hi {clause}
          AND NOT EXISTS (SELECT 1 FROM llm_reviews r WHERE r.accession = e.accession)
        ORDER BY e.filed DESC
    """, params, parse_dates=['filed'])


def _priority(todo: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    rank = {t: i for i, t in enumerate(first)}
    todo = todo.copy()
    todo['_p'] = todo['ticker'].map(rank).fillna(len(rank))
    return todo.sort_values(['_p', 'filed'], ascending=[True, False]).drop(columns='_p')


def review_filing(ticker: str, accession: str, filed: date, cik: str,
                  primary_doc: str | None = None,
                  as_of: date | None = None) -> dict | None:
    """Review one results filing and store it. Returns the stored row or None."""
    as_of = as_of or date.today()
    release = press_release_text(cik, accession, primary_doc)
    if not release:
        log.info('review %s %s: no press release text found', ticker, accession)
        return None
    ctx = build_context(ticker, filed, as_of)
    prompt = PROMPT.format(ticker=ticker, filed=filed, today=as_of,
                           release=release, **ctx)
    text = providers.complete(prompt, SYSTEM, 1500)
    review = parse_review(text)
    if review is None:
        log.warning('review %s %s: model returned no usable JSON', ticker, accession)
        return None
    row = {
        'accession': accession, 'ticker': ticker, 'filed': filed,
        'assessed_on': as_of, 'model': providers.model_name(), 'version': VERSION,
        **{k: review[k] for k in ('score', 'conviction', 'verdict', 'quarter_quality',
                                  'outlook', 'guidance', 'valuation_view', 'summary')},
        'detail_json': json.dumps({'positives': review['positives'],
                                   'negatives': review['negatives']}),
        'created_at': config.utc_now(),
    }
    db.upsert(db.llm_reviews, [row])
    return row


def review_recent(as_of: date | None = None, tickers: list[str] | None = None,
                  priority: list[str] | None = None,
                  max_reviews: int | None = None,
                  lookback_days: int = LOOKBACK_DAYS,
                  progress=None) -> list[dict]:
    """
    Review results filings from the last `lookback_days` that have none yet.

    `priority` names go first (portfolio, watchlist, current picks); the
    rest follow newest-first until `max_reviews` (QUANT_LLM_MAX_REVIEWS).
    """
    if not enabled():
        return []
    as_of = as_of or date.today()
    cap = config.QUANT_LLM_MAX_REVIEWS if max_reviews is None else max_reviews
    todo = _priority(pending(as_of, tickers, lookback_days), priority or [])
    done = []
    for _i, r in todo.head(cap).iterrows():
        if progress:
            progress(f'reviewing {r["ticker"]} results filed {r["filed"].date()}')
        try:
            row = review_filing(r['ticker'], r['accession'], r['filed'].date(),
                                r['cik'], r['primary_doc'], as_of)
        except Exception as exc:                   # noqa: BLE001
            log.warning('review %s failed: %s', r['ticker'], exc)
            continue
        if row:
            done.append(row)
    return done


# ─────────────────────────────────────────────
# READS (point-in-time on assessed_on)
# ─────────────────────────────────────────────

def latest_reviews(tickers: list[str] | None = None, as_of: date | None = None,
                   days: int = VIEW_WINDOW_DAYS) -> pd.DataFrame:
    """The newest review per ticker written on or before `as_of`."""
    as_of = pd.to_datetime(as_of or date.today()).date()
    clause, params = '', {'lo': str(as_of - timedelta(days=days)), 'hi': str(as_of)}
    if tickers:
        ph = ','.join(f':t{i}' for i in range(len(tickers)))
        clause = f'AND ticker IN ({ph})'
        params.update({f't{i}': t for i, t in enumerate(tickers)})
    df = db.read_sql(f"""
        SELECT * FROM llm_reviews
        WHERE assessed_on BETWEEN :lo AND :hi {clause}
        ORDER BY assessed_on DESC, filed DESC
    """, params, parse_dates=['filed', 'assessed_on'])
    return df.drop_duplicates('ticker', keep='first').reset_index(drop=True)


def view_factor(tickers: list[str], as_of: date | str,
                days: int = VIEW_WINDOW_DAYS) -> pd.Series:
    """LLM_EARNINGS_VIEW: score scaled by stated conviction, -2 .. +2."""
    df = latest_reviews(tickers, pd.to_datetime(as_of).date(), days)
    if df.empty:
        return pd.Series(dtype=float)
    w = df['conviction'].map(CONVICTION_WEIGHT).fillna(0.5)
    return pd.Series((df['score'] * w).values, index=df['ticker'].values, dtype=float)
