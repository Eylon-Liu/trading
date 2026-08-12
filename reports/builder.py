"""
Daily and monthly reports, rendered from stored runs.

Reports read the database rather than refetching, so they are fast, cheap and
reproducible — running yesterday's report tomorrow gives the same answer.

The daily report is built around *change*: what entered, what left, whose
signal moved, which mid-term plans hit a stop or a target. A ranked list on
its own does not tell you what to do differently today.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta

import pandas as pd

import config
from core import db
from data import news as NEWS
from data import policy as POL
from data.universe import PRESETS, UniverseSpec
from quant import engine as EN
from quant import research as RS
from quant import strategies as ST
from quant import tradeplan as TP

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DATA GATHERING
# ─────────────────────────────────────────────

def _latest_run(strategy: str, on_or_before: date) -> pd.Series | None:
    df = db.read_sql("""
        SELECT run_id, as_of, strategy, universe_n FROM runs
        WHERE strategy = :s AND as_of <= :d
        ORDER BY as_of DESC, created_at DESC LIMIT 1
    """, {'s': strategy, 'd': str(on_or_before)}, parse_dates=['as_of'])
    return df.iloc[0] if not df.empty else None


def _run_scores(run_id: str) -> pd.DataFrame:
    return db.read_sql("""
        SELECT ticker, composite, rank, signal, signal_score, reasons, sector
        FROM scores WHERE run_id = :r ORDER BY rank
    """, {'r': run_id})


def gather(strategy: str, as_of: date, compare_days: int = 1) -> dict:
    """Assemble everything a report needs, running the screen if needed."""
    current = _latest_run(strategy, as_of)
    if current is None:
        log.info('no stored run for %s — running screen now', strategy)
        spec = PRESETS.get('spy', UniverseSpec(preset='SPY'))
        result = EN.run(spec, strategy, as_of=as_of, persist=True)
        if result.scores.empty:
            return {'error': f'screen for {strategy} produced no results'}
        current = _latest_run(strategy, as_of)
        if current is None:
            return {'error': f'screen ran but no stored run found for {strategy}'}

    prior = _latest_run(strategy, as_of - timedelta(days=compare_days))
    now = _run_scores(current['run_id'])

    payload: dict = {
        'strategy': strategy,
        'strategy_name': ST.get(strategy).name if strategy in ST.ALL_STRATEGIES else strategy,
        'horizon': ST.get(strategy).horizon if strategy in ST.ALL_STRATEGIES else 'long',
        'as_of': as_of,
        'run_date': pd.to_datetime(current['as_of']).date(),
        'universe_n': int(current['universe_n']),
        'top': now.head(20),
        'signal_counts': now['signal'].value_counts().to_dict() if not now.empty else {},
        'sector_counts': now.head(20)['sector'].value_counts().to_dict()
        if not now.empty else {},
    }

    if prior is not None and prior['run_id'] != current['run_id']:
        comp = RS.compare_runs(prior['run_id'], current['run_id'])
        if not comp.empty:
            payload['compare_date'] = pd.to_datetime(prior['as_of']).date()
            payload['entered'] = comp[comp['status'] == 'entered'].head(10)
            payload['dropped'] = comp[comp['status'] == 'dropped'].head(10)
            held = comp[comp['status'] == 'held']
            payload['risers'] = held.nlargest(8, 'rank_change')
            payload['fallers'] = held.nsmallest(8, 'rank_change')
            payload['signal_changes'] = held[held['signal_changed']].head(10)

    return payload


def gather_news(tickers: list[str], as_of: date, limit: int = 12) -> pd.DataFrame:
    """Most material recent coverage across the shortlist."""
    df = NEWS.news_asof(tickers, as_of, lookback_days=7)
    if df.empty:
        return df
    df = df.copy()
    df['abs_sent'] = df['sentiment'].abs()
    df['has_event'] = df['events'].fillna('').str.len() > 0
    return (df.sort_values(['has_event', 'abs_sent'], ascending=False)
              .head(limit))


# ─────────────────────────────────────────────
# HTML RENDERING
# ─────────────────────────────────────────────

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
     background:#0f1117;color:#e0e0e0;margin:0;padding:24px}
.wrap{max-width:1000px;margin:0 auto}
h1{color:#38ef7d;font-size:1.7rem;margin:0 0 4px;letter-spacing:-.5px}
h2{color:#11998e;font-size:1.1rem;margin:28px 0 10px;
   border-bottom:1px solid rgba(17,153,142,.35);padding-bottom:6px}
.sub{color:#8b93a7;font-size:.85rem;margin-bottom:18px}
table{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:8px}
th{background:linear-gradient(135deg,#11998e,#38ef7d);color:#000;
   padding:8px 10px;text-align:left;font-weight:600}
td{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.07)}
tr:nth-child(even) td{background:rgba(255,255,255,.02)}
.tk{font-weight:700;color:#38ef7d}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:#38ef7d}.neg{color:#ff6b6b}.warn{color:#ffd93d}
.badge{display:inline-block;padding:4px 10px;border-radius:12px;
       background:rgba(17,153,142,.18);margin:0 6px 6px 0;font-size:.8rem}
.card{background:#1a1d2e;border:1px solid rgba(255,255,255,.08);
      border-radius:10px;padding:14px 16px;margin-bottom:14px}
.muted{color:#8b93a7;font-size:.78rem}
.disc{margin-top:32px;padding-top:14px;border-top:1px solid #2a2e42;
      color:#6b7280;font-size:.72rem;line-height:1.5}
"""


def _table(df: pd.DataFrame, cols: dict[str, str], numeric: set[str] | None = None) -> str:
    """Render a DataFrame as an HTML table using `cols` as header mapping."""
    if df is None or df.empty:
        return '<p class="muted">Nothing to report.</p>'
    numeric = numeric or set()

    head = ''.join(f'<th>{label}</th>' for label in cols.values())
    body = []
    for _i, r in df.iterrows():
        cells = []
        for key in cols:
            v = r.get(key, '')
            if isinstance(v, float):
                v = '—' if pd.isna(v) else f'{v:,.2f}'
            cls = ' class="num"' if key in numeric else ''
            if key == 'ticker':
                cls = ' class="tk"'
            cells.append(f'<td{cls}>{v}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def render_daily(payload: dict, news_df: pd.DataFrame | None = None,
                 policy_df: pd.DataFrame | None = None,
                 exits: pd.DataFrame | None = None) -> str:
    """Render the daily report to standalone HTML."""
    if 'error' in payload:
        return f'<html><body><p>{payload["error"]}</p></body></html>'

    horizon_label = ST.HORIZONS.get(payload['horizon'], {}).get('label', '')
    parts = [
        f'<h1>📊 Daily Research Brief</h1>',
        f'<div class="sub">{payload["strategy_name"]} &nbsp;·&nbsp; {horizon_label}'
        f' &nbsp;·&nbsp; {payload["as_of"]} &nbsp;·&nbsp; '
        f'{payload["universe_n"]} names screened</div>',
    ]

    badges = ''.join(f'<span class="badge">{k}: {v}</span>'
                     for k, v in payload.get('signal_counts', {}).items())
    if badges:
        parts.append(f'<div class="card">{badges}</div>')

    top = payload['top'].head(12).copy()
    if not top.empty:
        top['composite'] = top['composite'].round(3)
        parts.append('<h2>Top ranked</h2>')
        parts.append(_table(top, {
            'rank': '#', 'ticker': 'Ticker', 'sector': 'Sector',
            'composite': 'Score', 'signal': 'Signal', 'reasons': 'Why',
        }, numeric={'rank', 'composite'}))

    if 'compare_date' in payload:
        parts.append(f'<h2>What changed since {payload["compare_date"]}</h2>')
        for key, title in (('entered', 'New entrants'), ('dropped', 'Dropped out'),
                           ('risers', 'Biggest rank gains'),
                           ('fallers', 'Biggest rank falls')):
            sub = payload.get(key)
            if sub is not None and not sub.empty:
                s = sub.reset_index()
                parts.append(f'<div class="card"><b>{title}</b>')
                parts.append(_table(s.head(6), {
                    'ticker': 'Ticker', 'rank_now': 'Rank',
                    'rank_change': 'Δ Rank', 'signal_now': 'Signal',
                }, numeric={'rank_now', 'rank_change'}))
                parts.append('</div>')

    if exits is not None and not exits.empty:
        closed = exits[exits['status'] != 'open']
        if not closed.empty:
            parts.append('<h2>Trade plan events</h2>')
            parts.append(_table(closed.reset_index(), {
                'ticker': 'Ticker', 'reason': 'Event', 'entry': 'Entry',
                'last': 'Last', 'pnl_pct': 'P&L %', 'r_realized': 'R',
            }, numeric={'entry', 'last', 'pnl_pct', 'r_realized'}))

    if news_df is not None and not news_df.empty:
        parts.append('<h2>Notable coverage</h2><div class="card">')
        for _i, r in news_df.head(8).iterrows():
            ev = f' <span class="muted">[{r["events"]}]</span>' if r.get('events') else ''
            parts.append(
                f'<div style="margin-bottom:9px"><span class="tk">{r["ticker"]}</span> '
                f'{r.get("sentiment_label","")} {r["title"]}{ev}</div>')
        parts.append('</div>')

    if policy_df is not None and not policy_df.empty:
        parts.append('<h2>Policy &amp; regulation</h2><div class="card">')
        for _i, r in policy_df.head(6).iterrows():
            sect = r.get('affected_sectors') or '—'
            parts.append(
                f'<div style="margin-bottom:9px"><b>{r["doc_type"]}</b> '
                f'{r["title"][:130]}<br><span class="muted">{r["agencies"][:80]} '
                f'→ {sect}</span></div>')
        parts.append('</div>')

    parts.append(
        '<div class="disc"><b>⚠️ Disclaimer.</b> Generated by an automated '
        'quantitative research tool for informational and educational purposes '
        'only. Not investment advice, and not a recommendation to buy or sell '
        'any security. Signals, scores and any entry, stop or target levels are '
        'mechanical outputs of published rules, not judgements about your '
        'circumstances. Backtested results are hypothetical and carry no '
        'guarantee of future performance. Do your own research and consider '
        'consulting a licensed adviser.</div>')

    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>Daily Brief — {payload["as_of"]}</title>'
            f'<style>{CSS}</style></head><body><div class="wrap">'
            f'{"".join(parts)}</div></body></html>')


def render_monthly(payload: dict, backtest_metrics: dict | None = None) -> str:
    """Render the monthly review to standalone HTML."""
    if 'error' in payload:
        return f'<html><body><p>{payload["error"]}</p></body></html>'

    parts = [
        '<h1>📈 Monthly Review</h1>',
        f'<div class="sub">{payload["strategy_name"]} &nbsp;·&nbsp; '
        f'period ending {payload["as_of"]}</div>',
    ]

    if backtest_metrics:
        rows = []
        pretty = {
            'cagr': 'CAGR', 'volatility': 'Volatility', 'sharpe': 'Sharpe',
            'sortino': 'Sortino', 'max_drawdown': 'Max drawdown',
            'turnover': 'Turnover (ann.)', 'benchmark_cagr': 'Benchmark CAGR',
            'excess_cagr': 'Excess vs benchmark', 'mean_ic': 'Mean rank IC',
            'ic_ir': 'IC information ratio', 'ic_hit_rate': 'IC hit rate',
            'win_rate': 'Win rate', 'avg_r': 'Average R', 'n_trades': 'Trades',
        }
        for k, label in pretty.items():
            if k in backtest_metrics:
                v = backtest_metrics[k]
                shown = f'{v:,.3f}' if isinstance(v, float) else v
                rows.append(f'<tr><td>{label}</td><td class="num">{shown}</td></tr>')
        if rows:
            parts.append('<h2>Performance</h2><table><thead><tr><th>Metric</th>'
                         '<th>Value</th></tr></thead><tbody>'
                         + ''.join(rows) + '</tbody></table>')
            if 'mean_ic' in backtest_metrics and abs(backtest_metrics['mean_ic']) < 0.02:
                parts.append(
                    '<div class="card"><b class="warn">⚠️ Weak rank IC.</b> '
                    '<span class="muted">The mean information coefficient is near '
                    'zero, meaning the ranking showed little predictive power over '
                    'this window. Any outperformance is more likely to be a few '
                    'positions than a working signal — treat with caution and '
                    'consider a broader universe.</span></div>')

    top = payload['top'].head(20).copy()
    if not top.empty:
        top['composite'] = top['composite'].round(3)
        parts.append('<h2>Current ranking</h2>')
        parts.append(_table(top, {
            'rank': '#', 'ticker': 'Ticker', 'sector': 'Sector',
            'composite': 'Score', 'signal': 'Signal',
        }, numeric={'rank', 'composite'}))

    if payload.get('sector_counts'):
        chips = ''.join(f'<span class="badge">{k}: {v}</span>'
                        for k, v in payload['sector_counts'].items())
        parts.append(f'<h2>Sector distribution</h2><div class="card">{chips}</div>')

    parts.append(
        '<div class="disc"><b>⚠️ Disclaimer.</b> Automated quantitative research '
        'output for informational and educational purposes only. Not investment '
        'advice. Backtested and hypothetical results do not reflect actual '
        'trading and are no guarantee of future performance.</div>')

    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>Monthly Review — {payload["as_of"]}</title>'
            f'<style>{CSS}</style></head><body><div class="wrap">'
            f'{"".join(parts)}</div></body></html>')


# ─────────────────────────────────────────────
# BUILD & PERSIST
# ─────────────────────────────────────────────

def build_report(strategy: str, kind: str = 'daily',
                 as_of: date | str | None = None,
                 include_news: bool = True,
                 include_policy: bool = True,
                 backtest_metrics: dict | None = None) -> tuple[str, str]:
    """
    Build a report and write it to disk.

    Returns (path, html).
    """
    as_of = pd.to_datetime(as_of or date.today()).date()
    compare_days = 1 if kind == 'daily' else 30
    payload = gather(strategy, as_of, compare_days)

    news_df = policy_df = None
    if 'error' not in payload:
        tickers = payload['top']['ticker'].tolist()
        if include_news and tickers:
            news_df = gather_news(tickers, as_of)
        if include_policy:
            sectors = list(payload.get('sector_counts', {}))
            policy_df = POL.policy_asof(as_of, lookback_days=7, sectors=sectors or None)

    html = (render_daily(payload, news_df, policy_df) if kind == 'daily'
            else render_monthly(payload, backtest_metrics))

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / f'{kind}_{strategy}_{as_of}.html'
    path.write_text(html, encoding='utf-8')

    db.upsert(db.reports, [{
        'report_id': str(uuid.uuid4()), 'kind': kind,
        'period_start': as_of - timedelta(days=compare_days),
        'period_end': as_of, 'path': str(path),
        'emailed_at': None, 'created_at': datetime.utcnow(),
    }])

    log.info('report written: %s', path)
    return str(path), html


def list_reports(limit: int = 30) -> pd.DataFrame:
    return db.read_sql("""
        SELECT report_id, kind, period_end, path, emailed_at, created_at
        FROM reports ORDER BY created_at DESC LIMIT :n
    """, {'n': limit}, parse_dates=['period_end', 'created_at'])
