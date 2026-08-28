"""
Cross-strategy market reports.

Instead of one strategy at a time, each report aggregates across every
strategy that has a stored run, so the trader gets a single view of where
the models agree, where they diverge, and what changed.

Daily  → what to pay attention to *today*: consensus picks, signal shifts,
         1-day returns, news, policy.
Monthly → what happened *this month*: which strategies called it right,
          consensus turnover, MTD returns, sector rotation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from core import db
from data import news as NEWS
from data import policy as POL
from data import yahoo
from data.universe import UniverseSpec
from quant import custom as CU
from quant import engine as ENG
from quant import research as RS
from quant import strategies as ST

log = logging.getLogger(__name__)

TOP_N = 20


def _signal_bucket(sig: str) -> str:
    s = str(sig).lower()
    if 'buy' in s or 'enter' in s or 'strong buy' in s:
        return 'Buy'
    if 'sell' in s or 'trim' in s or 'avoid' in s:
        return 'Sell'
    if 'hold' in s or 'add' in s or 'watch' in s:
        return 'Hold'
    return 'Neutral'


# ─────────────────────────────────────────────
# DATA GATHERING
# ─────────────────────────────────────────────

def _all_strategy_keys() -> list[str]:
    keys = list(ST.ALL_STRATEGIES.keys())
    try:
        custom = CU.summary()
        if not custom.empty:
            keys.extend(custom['key'].tolist())
    except Exception:  # noqa: BLE001
        pass
    return keys


def _latest_run(strategy: str, on_or_before: date) -> pd.Series | None:
    df = db.read_sql("""
        SELECT run_id, as_of, strategy, universe_n, universe_spec_json
        FROM runs
        WHERE strategy = :s AND as_of <= :d
        ORDER BY as_of DESC, created_at DESC LIMIT 1
    """, {'s': strategy, 'd': str(on_or_before)}, parse_dates=['as_of'])
    return df.iloc[0] if not df.empty else None


def _run_scores(run_id: str) -> pd.DataFrame:
    return db.read_sql("""
        SELECT ticker, composite, rank, signal, signal_score, reasons, sector
        FROM scores WHERE run_id = :r ORDER BY rank
    """, {'r': run_id})


def _price_returns(tickers: list[str], as_of: date,
                   days: int) -> pd.Series:
    start = as_of - timedelta(days=days * 2 + 5)
    px = yahoo.price_history(tickers, start=start, end=as_of, field='adj_close')
    if px.empty or len(px) < 2:
        return pd.Series(dtype=float)
    end_px = px.iloc[-1]
    start_idx = max(0, len(px) - 1 - days)
    start_px = px.iloc[start_idx]
    return (end_px / start_px - 1).dropna()


BENCH_TICKERS = ['SPY', 'QQQ', 'DIA', 'IWM']
BENCH_LABELS = {
    'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100',
    'DIA': 'Dow 30', 'IWM': 'Russell 2000',
}


def _index_performance(as_of: date) -> dict | None:
    """Fetch benchmark ETF performance for the report header."""
    try:
        yahoo.update_prices(BENCH_TICKERS, period='1mo')
    except Exception:  # noqa: BLE001
        pass
    start = as_of - timedelta(days=40)
    px = yahoo.price_history(BENCH_TICKERS, start=start, end=as_of,
                             field='adj_close')
    if px.empty or len(px) < 2:
        return None
    result = {}
    for sym in BENCH_TICKERS:
        if sym not in px.columns:
            continue
        col = px[sym].dropna()
        if len(col) < 2:
            continue
        r1d = col.iloc[-1] / col.iloc[-2] - 1 if len(col) >= 2 else None
        r5d = (col.iloc[-1] / col.iloc[-6] - 1
               if len(col) >= 6 else None)
        idx_1m = max(0, len(col) - 22)
        r1m = col.iloc[-1] / col.iloc[idx_1m] - 1
        result[sym] = {
            'label': BENCH_LABELS.get(sym, sym),
            'price': col.iloc[-1],
            '1d': r1d, '5d': r5d, '1m': r1m,
        }
    return result if result else None


def _strategy_name(key: str) -> str:
    try:
        return ST.get(key).name
    except (KeyError, Exception):  # noqa: BLE001
        return key.replace('_', ' ').title()


def _auto_run_prior(key: str, prior_date: date,
                    spec_json: str) -> tuple[pd.Series | None, str | None]:
    """Try to auto-run a strategy for a prior date using stored data.

    Returns (run_row, warning).  Exactly one will be non-None.
    """
    try:
        spec = UniverseSpec.from_json(spec_json)
        result = ENG.run(spec, key, as_of=prior_date,
                         persist=True, sync=False)
        if result.scores.empty:
            return None, (f'{_strategy_name(key)}: auto-run for {prior_date}'
                          f' produced no scores (insufficient stored data)')
        new_run = _latest_run(key, prior_date)
        if new_run is not None:
            return new_run, None
        return None, (f'{_strategy_name(key)}: auto-run for {prior_date}'
                      f' did not persist')
    except Exception as exc:  # noqa: BLE001
        log.warning('auto-run %s as_of %s failed: %s', key, prior_date, exc)
        return None, (f'{_strategy_name(key)}: auto-run for {prior_date}'
                      f' failed — {type(exc).__name__}: {str(exc)[:120]}')


def _gather_cross_strategy(as_of: date, compare_days: int = 1,
                           max_staleness: int = 3) -> dict:
    """Collect the latest run for every strategy and aggregate."""
    cutoff = as_of - timedelta(days=max_staleness)
    strategies_used: list[dict] = []
    all_scores: list[pd.DataFrame] = []
    prior_scores: list[pd.DataFrame] = []
    data_warnings: list[str] = []

    for key in _all_strategy_keys():
        run = _latest_run(key, as_of)
        if run is None:
            continue
        run_date = pd.to_datetime(run['as_of']).date()
        if run_date < cutoff:
            continue
        scores = _run_scores(run['run_id'])
        if scores.empty:
            continue
        scores = scores.copy()
        scores['strategy'] = key
        scores['strategy_name'] = _strategy_name(key)
        all_scores.append(scores)
        strategies_used.append({
            'key': key, 'name': _strategy_name(key),
            'run_date': run_date, 'run_id': run['run_id'],
            'universe_n': int(run['universe_n']),
        })

        prior_date = as_of - timedelta(days=compare_days)
        prior_cutoff = prior_date - timedelta(days=max_staleness)
        prior = _latest_run(key, prior_date)
        prior_ok = (prior is not None
                    and prior['run_id'] != run['run_id']
                    and pd.to_datetime(prior['as_of']).date() >= prior_cutoff)
        if prior_ok:
            ps = _run_scores(prior['run_id'])
            if not ps.empty:
                ps = ps.copy()
                ps['strategy'] = key
                prior_scores.append(ps)
        else:
            stale_note = ''
            if prior is not None and prior['run_id'] != run['run_id']:
                stale_date = pd.to_datetime(prior['as_of']).date()
                stale_note = f' (nearest is {stale_date}, too old)'
            log.info('no recent prior run for %s near %s%s — attempting '
                     'auto-run', key, prior_date, stale_note)
            auto_run, warning = _auto_run_prior(
                key, prior_date, run['universe_spec_json'])
            if auto_run is not None:
                ps = _run_scores(auto_run['run_id'])
                if not ps.empty:
                    ps = ps.copy()
                    ps['strategy'] = key
                    prior_scores.append(ps)
                    log.info('auto-run succeeded for %s on %s', key, prior_date)
            elif warning:
                data_warnings.append(warning)

    if not all_scores:
        return {'error': 'No stored runs found. Run a screen first.'}

    combined = pd.concat(all_scores, ignore_index=True)
    prior_combined = (pd.concat(prior_scores, ignore_index=True)
                      if prior_scores else pd.DataFrame())

    # Consensus: how many strategies put each ticker in top-N
    top_tickers = combined[combined['rank'] <= TOP_N].copy()
    consensus = (top_tickers.groupby('ticker')
                 .agg(n_strategies=('strategy', 'nunique'),
                      avg_rank=('rank', 'mean'),
                      best_rank=('rank', 'min'),
                      strategies=('strategy_name',
                                  lambda x: ', '.join(sorted(set(x)))),
                      sector=('sector', 'first'),
                      signals=('signal',
                               lambda x: ', '.join(sorted(set(
                                   _signal_bucket(s) for s in x)))))
                 .sort_values(['n_strategies', 'avg_rank'],
                              ascending=[False, True])
                 .reset_index())

    # Signal distribution across all strategies — group by broad category
    signal_dist = combined['signal'].map(_signal_bucket).value_counts().to_dict()

    # Strategy-level stats
    strat_stats = []
    for info in strategies_used:
        s = combined[combined['strategy'] == info['key']]
        top = s[s['rank'] <= TOP_N]
        buys = s['signal'].str.contains('Buy|Enter|Strong Buy',
                                          case=False, na=False).sum()
        strat_stats.append({
            'strategy': info['name'], 'key': info['key'],
            'run_date': info['run_date'],
            'universe_n': info['universe_n'],
            'n_buy': int(buys),
            'top_tickers': ', '.join(top.head(5)['ticker'].tolist()),
        })
    strat_stats_df = pd.DataFrame(strat_stats)

    payload: dict = {
        'as_of': as_of,
        'compare_days': compare_days,
        'n_strategies': len(strategies_used),
        'strategies': strategies_used,
        'strat_stats': strat_stats_df,
        'consensus': consensus,
        'signal_dist': signal_dist,
        'data_warnings': data_warnings,
    }

    # DOD / MOM consensus changes — only compare strategies present on
    # both dates so adding a new strategy doesn't show everything as
    # "gained" and removing one doesn't show everything as "lost".
    if not prior_combined.empty:
        paired_keys = set(prior_combined['strategy'].unique())
        paired_current = combined[combined['strategy'].isin(paired_keys)]
        paired_top = paired_current[paired_current['rank'] <= TOP_N]
        paired_consensus = (paired_top.groupby('ticker')
                            .agg(n_strategies=('strategy', 'nunique'))
                            .reset_index())

        prior_top = prior_combined[prior_combined['rank'] <= TOP_N]
        prior_consensus = (prior_top.groupby('ticker')
                           .agg(n_strategies_prev=('strategy', 'nunique'),
                                avg_rank_prev=('rank', 'mean'))
                           .reset_index())
        merged = paired_consensus.merge(
            prior_consensus, on='ticker', how='outer',
            suffixes=('', '_prev'))
        merged['n_strategies'] = merged['n_strategies'].fillna(0).astype(int)
        merged['n_strategies_prev'] = (merged['n_strategies_prev']
                                       .fillna(0).astype(int))
        merged['consensus_change'] = (merged['n_strategies']
                                      - merged['n_strategies_prev'])
        n_paired = len(paired_keys)
        payload['n_paired'] = n_paired

        new_consensus = (merged[(merged['n_strategies'] >= 2)
                                & (merged['n_strategies_prev'] == 0)]
                         .sort_values('n_strategies', ascending=False).head(10))
        lost_consensus = (merged[(merged['n_strategies'] == 0)
                                 & (merged['n_strategies_prev'] >= 2)]
                          .sort_values('n_strategies_prev', ascending=False)
                          .head(10))
        momentum_up = (merged[(merged['consensus_change'] > 0)
                              & (merged['n_strategies'] >= 2)]
                       .sort_values('consensus_change', ascending=False)
                       .head(10))
        momentum_down = (merged[(merged['consensus_change'] < 0)
                                & (merged['n_strategies_prev'] >= 2)]
                         .sort_values('consensus_change').head(10))

        compare_label = ('yesterday' if compare_days <= 1
                         else f'{compare_days} days ago')
        payload['compare_label'] = (
            f'{compare_label} ({n_paired} paired '
            f'{"strategy" if n_paired == 1 else "strategies"})')
        payload['new_consensus'] = new_consensus
        payload['lost_consensus'] = lost_consensus
        payload['momentum_up'] = momentum_up
        payload['momentum_down'] = momentum_down

        # Per-strategy signal shifts
        signal_shifts = []
        for info in strategies_used:
            prev_run = _latest_run(info['key'],
                                   as_of - timedelta(days=compare_days))
            if prev_run is None or prev_run['run_id'] == info['run_id']:
                continue
            comp = RS.compare_runs(prev_run['run_id'], info['run_id'])
            if comp.empty:
                continue
            changed = comp[comp['signal_changed']
                           & (comp['rank_now'] <= TOP_N)]
            for ticker, row in changed.iterrows():
                signal_shifts.append({
                    'ticker': ticker, 'strategy': info['name'],
                    'was': row.get('signal_then', '?'),
                    'now': row.get('signal_now', '?'),
                    'rank': (int(row['rank_now'])
                             if pd.notna(row['rank_now']) else None),
                })
        if signal_shifts:
            payload['signal_shifts'] = pd.DataFrame(signal_shifts)

    # Price returns for consensus picks
    consensus_tickers = consensus.head(30)['ticker'].tolist()
    if consensus_tickers:
        try:
            ret_1d = _price_returns(consensus_tickers, as_of, 1)
            if not ret_1d.empty:
                payload['returns_1d'] = ret_1d
        except Exception:  # noqa: BLE001
            pass
        if compare_days >= 20:
            try:
                ret_1m = _price_returns(consensus_tickers, as_of, 21)
                if not ret_1m.empty:
                    payload['returns_1m'] = ret_1m
            except Exception:  # noqa: BLE001
                pass

    return payload


def gather_news(tickers: list[str], as_of: date,
                limit: int = 12) -> pd.DataFrame:
    df = NEWS.news_asof(tickers, as_of, lookback_days=7)
    if df.empty:
        return df
    df = df.copy()
    df['abs_sent'] = df['sentiment'].abs()
    df['has_event'] = df['events'].fillna('').str.len() > 0
    return (df.sort_values(['has_event', 'abs_sent'], ascending=False)
              .head(limit))


def _monthly_stability(as_of: date,
                       max_staleness: int = 3) -> list[dict]:
    """Top-10 stability per strategy over the past 30 days.

    If only one run exists in the window, auto-run the strategy for
    the start of the window so we have something to compare against.
    """
    cutoff = as_of - timedelta(days=max_staleness)
    window_start = as_of - timedelta(days=35)
    results = []
    for key in _all_strategy_keys():
        runs = db.read_sql("""
            SELECT run_id, as_of FROM runs
            WHERE strategy = :s AND as_of BETWEEN :start AND :end
            ORDER BY as_of
        """, {'s': key, 'start': str(window_start),
              'end': str(as_of)}, parse_dates=['as_of'])
        if runs.empty:
            continue
        last_date = pd.to_datetime(runs.iloc[-1]['as_of']).date()
        if last_date < cutoff:
            continue

        if len(runs) < 2:
            current = _latest_run(key, as_of)
            if current is None:
                continue
            auto_run, _warn = _auto_run_prior(
                key, window_start, current['universe_spec_json'])
            if auto_run is not None:
                runs = db.read_sql("""
                    SELECT run_id, as_of FROM runs
                    WHERE strategy = :s AND as_of BETWEEN :start AND :end
                    ORDER BY as_of
                """, {'s': key, 'start': str(window_start),
                      'end': str(as_of)}, parse_dates=['as_of'])
            if len(runs) < 2:
                continue

        first_scores = _run_scores(runs.iloc[0]['run_id'])
        last_scores = _run_scores(runs.iloc[-1]['run_id'])
        if first_scores.empty or last_scores.empty:
            continue
        top_start = set(first_scores.head(10)['ticker'])
        top_end = set(last_scores.head(10)['ticker'])
        overlap = len(top_start & top_end)
        results.append({
            'strategy': _strategy_name(key), 'key': key,
            'overlap': overlap, 'n_runs': len(runs),
            'entered': sorted(top_end - top_start),
            'exited': sorted(top_start - top_end),
        })
    return sorted(results, key=lambda r: -r['overlap'])


# ─────────────────────────────────────────────
# HTML RENDERING
# ─────────────────────────────────────────────

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
     background:#0f1117;color:#e0e0e0;margin:0;padding:24px}
.wrap{max-width:1060px;margin:0 auto}
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
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
             gap:10px;margin:10px 0 16px}
.metric{background:#252836;border:1px solid rgba(255,255,255,.06);
        border-radius:8px;padding:12px 14px;text-align:center}
.metric-val{font-size:1.3rem;font-weight:800;font-variant-numeric:tabular-nums;
            letter-spacing:-.5px}
.metric-lbl{font-size:.65rem;color:#8b93a7;text-transform:uppercase;
            letter-spacing:.5px;margin-top:2px}
.consensus-bar{display:flex;gap:2px;margin:4px 0}
.consensus-bar span{height:8px;border-radius:4px;min-width:8px}
"""

_DISCLAIMER = (
    '<div class="disc"><b>Disclaimer.</b> Generated by an automated '
    'quantitative research tool for informational and educational purposes '
    'only. Not investment advice and not a recommendation to buy or sell '
    'any security. Signals, scores and rankings are mechanical outputs of '
    'published rules, not judgements about your circumstances. Do your own '
    'research and consider consulting a licensed adviser.</div>')


def _color_pct(v: float) -> str:
    if not np.isfinite(v):
        return '—'
    cls = 'pos' if v >= 0 else 'neg'
    return f'<span class="{cls}">{v:+.1%}</span>'


def _metric_card(value: str, label: str, color: str = '#e0e0e0') -> str:
    return (f'<div class="metric"><div class="metric-val" style="color:{color}">'
            f'{value}</div><div class="metric-lbl">{label}</div></div>')


def _table(df: pd.DataFrame, cols: dict[str, str],
           numeric: set[str] | None = None) -> str:
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


def _consensus_bar(n: int, total: int) -> str:
    filled = min(n, total)
    bars = ''.join('<span style="background:#38ef7d;flex:1"></span>'
                   for _ in range(filled))
    bars += ''.join('<span style="background:#2a2e42;flex:1"></span>'
                    for _ in range(total - filled))
    return f'<div class="consensus-bar">{bars}</div>'


def _render_bench(bench: dict | None) -> str:
    """Render index performance bar for the report header."""
    if not bench:
        return ''
    parts = ['<div class="metric-grid">']
    for sym in BENCH_TICKERS:
        info = bench.get(sym)
        if not info:
            continue
        r1d = info.get('1d')
        r5d = info.get('5d')
        if r1d is None:
            continue
        color = '#38ef7d' if r1d >= 0 else '#ff6b6b'
        week_str = f'<span class="muted" style="font-size:.65rem">'
        if r5d is not None:
            w_color = '#38ef7d' if r5d >= 0 else '#ff6b6b'
            week_str += (f'5D <span style="color:{w_color}">'
                         f'{r5d:+.1%}</span>')
        r1m = info.get('1m')
        if r1m is not None:
            m_color = '#38ef7d' if r1m >= 0 else '#ff6b6b'
            week_str += (f' &nbsp;1M <span style="color:{m_color}">'
                         f'{r1m:+.1%}</span>')
        week_str += '</span>'
        parts.append(
            f'<div class="metric">'
            f'<div class="metric-val" style="color:{color}">'
            f'{r1d:+.2%}</div>'
            f'<div class="metric-lbl">{info["label"]}</div>'
            f'{week_str}</div>')
    parts.append('</div>')
    return ''.join(parts)


def _render_policy(policy_df: pd.DataFrame | None) -> str:
    """Render policy section with full context, filtering out noise."""
    if policy_df is None or policy_df.empty:
        return ''
    # Filter out routine items (safety zones, local regulations, notices)
    noise = ['safety zone', 'special local regulation', 'drawbridge',
             'anchorage', 'security zone', 'regulated navigation area']
    filtered = policy_df.copy()
    title_low = filtered['title'].str.lower()
    for phrase in noise:
        filtered = filtered[~title_low.str.contains(phrase, na=False)]
        title_low = filtered['title'].str.lower()
    if filtered.empty:
        return ''
    parts = ['<h2>Policy &amp; regulation</h2>']
    for _i, r in filtered.head(6).iterrows():
        sect = r.get('affected_sectors') or ''
        themes = r.get('themes') or ''
        abstract = str(r.get('abstract', '') or '')[:300]
        badges = ''
        if sect:
            for s in sect.split(','):
                s = s.strip()
                if s:
                    badges += f'<span class="badge">{s}</span>'
        if themes:
            for t in themes.split(','):
                t = t.strip()
                if t:
                    badges += (f'<span class="badge" style="background:'
                               f'rgba(255,217,61,.15);color:#ffd93d">'
                               f'{t}</span>')
        parts.append(
            f'<div class="card" style="margin-bottom:10px">'
            f'<div style="margin-bottom:4px">'
            f'<span style="background:rgba(17,153,142,.3);padding:2px 8px;'
            f'border-radius:4px;font-size:.72rem;font-weight:600;'
            f'text-transform:uppercase">{r["doc_type"]}</span></div>'
            f'<div style="font-weight:600;font-size:.88rem;'
            f'margin-bottom:4px">{r["title"]}</div>')
        if abstract:
            parts.append(
                f'<div style="font-size:.8rem;color:#b0b8cc;'
                f'line-height:1.5;margin-bottom:6px">{abstract}'
                f'{"…" if len(str(r.get("abstract", ""))) > 300 else ""}'
                f'</div>')
        if badges:
            parts.append(f'<div>{badges}</div>')
        agency = r.get('agencies', '') or ''
        if agency:
            parts.append(
                f'<div class="muted" style="margin-top:4px">{agency}</div>')
        parts.append('</div>')
    return ''.join(parts)


def render_daily(payload: dict, news_df: pd.DataFrame | None = None,
                 policy_df: pd.DataFrame | None = None,
                 bench: dict | None = None) -> str:
    if 'error' in payload:
        return f'<html><body><p>{payload["error"]}</p></body></html>'

    parts = [
        '<h1>Market Intelligence Brief</h1>',
        f'<div class="sub">{payload["as_of"]} &nbsp;·&nbsp; '
        f'{payload["n_strategies"]} strategies compared</div>',
    ]

    # Index performance
    if bench:
        parts.append('<h2>Market snapshot</h2>')
        parts.append(_render_bench(bench))

    consensus = payload['consensus']
    signal_dist = payload.get('signal_dist', {})

    # Headline metrics
    n_buy = signal_dist.get('Buy', 0)
    n_sell = signal_dist.get('Sell', 0)
    high_conviction = (len(consensus[consensus['n_strategies'] >= 3])
                       if not consensus.empty else 0)
    metrics = [
        _metric_card(str(payload['n_strategies']), 'Strategies', '#00d4ff'),
        _metric_card(str(high_conviction), 'High conviction', '#38ef7d'),
        _metric_card(str(n_buy), 'Buy signals', '#38ef7d'),
        _metric_card(str(n_sell), 'Sell signals', '#ff6b6b'),
    ]
    if 'compare_label' in payload:
        new_c = payload.get('new_consensus')
        lost_c = payload.get('lost_consensus')
        n_new = len(new_c) if new_c is not None and not new_c.empty else 0
        n_lost = len(lost_c) if lost_c is not None and not lost_c.empty else 0
        metrics.append(
            _metric_card(f'+{n_new}/−{n_lost}', 'Consensus chg', '#ffd93d'))
    parts.append(f'<div class="metric-grid">{"".join(metrics)}</div>')

    # Cross-strategy consensus picks
    if not consensus.empty:
        top = consensus[consensus['n_strategies'] >= 2].head(25).copy()
        if not top.empty:
            ret_1d = payload.get('returns_1d')
            parts.append('<h2>Consensus picks (2+ strategies agree)</h2>')
            head = ('<tr><th>Ticker</th><th>Sector</th>'
                    '<th># Agree</th><th>Avg rank</th><th>Best rank</th>'
                    '<th>1D return</th><th>Signals</th><th>Agreement</th></tr>')
            rows = []
            for _, r in top.iterrows():
                ret = '—'
                if ret_1d is not None and r['ticker'] in ret_1d.index:
                    ret = _color_pct(ret_1d[r['ticker']])
                bar = _consensus_bar(int(r['n_strategies']),
                                     payload['n_strategies'])
                rows.append(
                    f'<tr><td class="tk">{r["ticker"]}</td>'
                    f'<td>{r.get("sector", "")}</td>'
                    f'<td class="num">{r["n_strategies"]}</td>'
                    f'<td class="num">{r["avg_rank"]:.0f}</td>'
                    f'<td class="num">{r["best_rank"]}</td>'
                    f'<td class="num">{ret}</td>'
                    f'<td>{r["signals"]}</td>'
                    f'<td style="min-width:80px">{bar}</td></tr>')
            parts.append(f'<table><thead>{head}</thead>'
                         f'<tbody>{"".join(rows)}</tbody></table>')

    # DOD consensus changes
    if 'compare_label' in payload:
        parts.append(
            f'<h2>Consensus changes vs {payload["compare_label"]}</h2>')

        new_c = payload.get('new_consensus')
        if new_c is not None and not new_c.empty:
            parts.append('<div class="card"><b>🟢 Gained consensus</b> '
                         '<span class="muted">— appeared in 2+ top lists'
                         '</span>')
            parts.append(_table(new_c, {
                'ticker': 'Ticker', 'n_strategies': '# strategies',
                'avg_rank': 'Avg rank',
            }, numeric={'n_strategies', 'avg_rank'}))
            parts.append('</div>')

        lost_c = payload.get('lost_consensus')
        if lost_c is not None and not lost_c.empty:
            parts.append('<div class="card"><b>🔴 Lost consensus</b> '
                         '<span class="muted">— dropped from 2+ top lists'
                         '</span>')
            parts.append(_table(lost_c, {
                'ticker': 'Ticker',
                'n_strategies_prev': 'Was in # strategies',
            }, numeric={'n_strategies_prev'}))
            parts.append('</div>')

        mu = payload.get('momentum_up')
        if mu is not None and not mu.empty:
            parts.append('<div class="card"><b>📈 Gaining momentum</b> '
                         '<span class="muted">— picked by more strategies'
                         '</span>')
            parts.append(_table(mu, {
                'ticker': 'Ticker', 'n_strategies': 'Now',
                'n_strategies_prev': 'Was', 'consensus_change': 'Δ',
            }, numeric={'n_strategies', 'n_strategies_prev',
                        'consensus_change'}))
            parts.append('</div>')

        shifts = payload.get('signal_shifts')
        if shifts is not None and not shifts.empty:
            parts.append(
                '<div class="card"><b>🔄 Signal shifts in top-20</b>')
            parts.append(_table(shifts.head(12), {
                'ticker': 'Ticker', 'strategy': 'Strategy',
                'was': 'Was', 'now': 'Now', 'rank': 'Rank',
            }, numeric={'rank'}))
            parts.append('</div>')

    # Strategy heat map
    strat_stats = payload.get('strat_stats')
    if strat_stats is not None and not strat_stats.empty:
        parts.append('<h2>Strategy signals today</h2>')
        head = ('<tr><th>Strategy</th><th>Run date</th>'
                '<th>Buys</th><th>Top 5</th></tr>')
        rows = []
        for _, r in strat_stats.sort_values('n_buy',
                                             ascending=False).iterrows():
            rows.append(
                f'<tr><td>{r["strategy"]}</td>'
                f'<td class="num">{r["run_date"]}</td>'
                f'<td class="num"><span class="pos">{r["n_buy"]}</span></td>'
                f'<td class="muted" style="font-size:.76rem">'
                f'{r["top_tickers"]}</td></tr>')
        parts.append(f'<table><thead>{head}</thead>'
                     f'<tbody>{"".join(rows)}</tbody></table>')

    # Sector concentration
    if not consensus.empty:
        top_c = consensus[consensus['n_strategies'] >= 2]
        if not top_c.empty:
            sc = top_c['sector'].value_counts().to_dict()
            chips = ''.join(f'<span class="badge">{k}: {v}</span>'
                            for k, v in sc.items()
                            if k and k != 'Unknown')
            if chips:
                parts.append('<h2>Sector concentration (consensus picks)</h2>'
                             f'<div class="card">{chips}</div>')

    # News
    if news_df is not None and not news_df.empty:
        parts.append('<h2>Notable coverage</h2><div class="card">')
        for _i, r in news_df.head(8).iterrows():
            sent = r.get('sentiment_label', '')
            ev = (f' <span class="muted">[{r["events"]}]</span>'
                  if r.get('events') else '')
            parts.append(
                f'<div style="margin-bottom:9px">'
                f'<span class="tk">{r["ticker"]}</span> '
                f'{sent} {r["title"]}{ev}</div>')
        parts.append('</div>')

    # Policy (filtered and expanded)
    parts.append(_render_policy(policy_df))

    # Data availability warnings
    warnings = payload.get('data_warnings', [])
    if warnings:
        parts.append('<h2>⚠️ Data availability</h2><div class="card">'
                     '<p class="muted">Some strategies could not be compared '
                     'against a prior date. The report auto-ran screens for '
                     'missing dates where possible; below are strategies that '
                     'still lack comparison data.</p>')
        for w in warnings:
            parts.append(f'<div class="warn" style="margin-bottom:6px;'
                         f'font-size:.82rem">• {w}</div>')
        parts.append('</div>')

    parts.append(_DISCLAIMER)
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>Market Brief — {payload["as_of"]}</title>'
            f'<style>{CSS}</style></head><body><div class="wrap">'
            f'{"".join(parts)}</div></body></html>')


def render_monthly(payload: dict, stability: list[dict] | None = None,
                   backtest_metrics: dict | None = None,
                   bench: dict | None = None) -> str:
    if 'error' in payload:
        return f'<html><body><p>{payload["error"]}</p></body></html>'

    parts = [
        '<h1>Monthly Review</h1>',
        f'<div class="sub">Period ending {payload["as_of"]} &nbsp;·&nbsp; '
        f'{payload["n_strategies"]} strategies compared</div>',
    ]

    # Index performance
    if bench:
        parts.append('<h2>Market snapshot</h2>')
        parts.append(_render_bench(bench))

    consensus = payload['consensus']
    ret_1m = payload.get('returns_1m')

    # Headline metrics
    metrics = []
    if ret_1m is not None and not ret_1m.empty:
        top_c = (consensus[consensus['n_strategies'] >= 2]
                 if not consensus.empty else pd.DataFrame())
        if not top_c.empty:
            matched = ret_1m.reindex(top_c['ticker']).dropna()
            if not matched.empty:
                avg_ret = matched.mean()
                best = matched.idxmax()
                worst = matched.idxmin()
                color = '#38ef7d' if avg_ret >= 0 else '#ff6b6b'
                metrics.extend([
                    _metric_card(f'{avg_ret:+.1%}',
                                 'Avg MTD (consensus)', color),
                    _metric_card(f'{matched.max():+.1%}',
                                 f'Best: {best}', '#38ef7d'),
                    _metric_card(f'{matched.min():+.1%}',
                                 f'Worst: {worst}', '#ff6b6b'),
                    _metric_card(f'{(matched > 0).mean():.0%}',
                                 'Win rate', '#00d4ff'),
                ])
    if stability:
        avg_ov = np.mean([s['overlap'] for s in stability])
        metrics.append(_metric_card(f'{avg_ov:.0f}/10',
                                    'Avg stability', '#ffd93d'))
    if metrics:
        parts.append(f'<div class="metric-grid">{"".join(metrics)}</div>')

    # Strategy performance comparison
    if ret_1m is not None and not ret_1m.empty:
        strat_perf = []
        strat_stats = payload.get('strat_stats')
        if strat_stats is not None and not strat_stats.empty:
            for _, info in strat_stats.iterrows():
                key = info['key']
                run = _latest_run(key, payload['as_of'])
                if run is None:
                    continue
                scores = _run_scores(run['run_id'])
                if scores.empty:
                    continue
                top5 = scores.head(5)['ticker'].tolist()
                rets = ret_1m.reindex(top5).dropna()
                if rets.empty:
                    continue
                strat_perf.append({
                    'strategy': info['strategy'],
                    'avg_mtd': rets.mean(),
                    'best_pick': rets.idxmax(),
                    'best_ret': rets.max(),
                    'n_positive': int((rets > 0).sum()),
                    'n_picks': len(rets),
                })
        if strat_perf:
            sp = pd.DataFrame(strat_perf).sort_values(
                'avg_mtd', ascending=False)
            parts.append(
                '<h2>Strategy performance (top-5 picks, MTD)</h2>')
            head = ('<tr><th>Strategy</th><th>Avg MTD</th>'
                    '<th>Best pick</th><th>Best ret</th>'
                    '<th>Win/Total</th></tr>')
            rows = []
            for _, r in sp.iterrows():
                rows.append(
                    f'<tr><td>{r["strategy"]}</td>'
                    f'<td class="num">{_color_pct(r["avg_mtd"])}</td>'
                    f'<td class="tk">{r["best_pick"]}</td>'
                    f'<td class="num">{_color_pct(r["best_ret"])}</td>'
                    f'<td class="num">{r["n_positive"]}/{r["n_picks"]}'
                    f'</td></tr>')
            parts.append(f'<table><thead>{head}</thead>'
                         f'<tbody>{"".join(rows)}</tbody></table>')

    # MOM consensus changes
    if 'compare_label' in payload:
        parts.append('<h2>Month-over-month changes</h2>')
        new_c = payload.get('new_consensus')
        if new_c is not None and not new_c.empty:
            parts.append(
                '<div class="card"><b>🟢 Gained consensus this month</b>')
            show_cols: dict[str, str] = {
                'ticker': 'Ticker', 'n_strategies': '# strategies',
            }
            if ret_1m is not None:
                new_c = new_c.copy()
                new_c['mtd'] = new_c['ticker'].map(ret_1m).apply(
                    lambda v: _color_pct(v) if pd.notna(v) else '—')
                show_cols['mtd'] = 'MTD'
            parts.append(_table(new_c, show_cols,
                                numeric={'n_strategies'}))
            parts.append('</div>')

        lost_c = payload.get('lost_consensus')
        if lost_c is not None and not lost_c.empty:
            parts.append(
                '<div class="card"><b>🔴 Lost consensus this month</b>')
            parts.append(_table(lost_c, {
                'ticker': 'Ticker',
                'n_strategies_prev': 'Was in # strategies',
            }, numeric={'n_strategies_prev'}))
            parts.append('</div>')

    # Top-10 stability per strategy
    if stability:
        parts.append('<h2>Top-10 stability by strategy</h2>')
        head = ('<tr><th>Strategy</th><th>Overlap</th><th>Runs</th>'
                '<th>Entered</th><th>Exited</th></tr>')
        rows = []
        for s in stability:
            bar = _consensus_bar(s['overlap'], 10)
            entered = ', '.join(s['entered'][:5]) or '—'
            exited = ', '.join(s['exited'][:5]) or '—'
            rows.append(
                f'<tr><td>{s["strategy"]}</td>'
                f'<td style="min-width:100px">{bar} '
                f'<span class="num">{s["overlap"]}/10</span></td>'
                f'<td class="num">{s["n_runs"]}</td>'
                f'<td class="muted" style="font-size:.76rem">{entered}</td>'
                f'<td class="muted" style="font-size:.76rem">{exited}'
                f'</td></tr>')
        parts.append(f'<table><thead>{head}</thead>'
                     f'<tbody>{"".join(rows)}</tbody></table>')

    # Current consensus with MTD returns
    if not consensus.empty:
        top = consensus[consensus['n_strategies'] >= 2].head(25).copy()
        if not top.empty:
            parts.append('<h2>Current consensus ranking</h2>')
            if ret_1m is not None:
                top['mtd'] = top['ticker'].map(ret_1m).apply(
                    lambda v: _color_pct(v) if pd.notna(v) else '—')
            else:
                top['mtd'] = '—'
            head = ('<tr><th>Ticker</th><th>Sector</th>'
                    '<th># Strategies</th><th>Avg rank</th>'
                    '<th>MTD return</th><th>Signals</th></tr>')
            rows = []
            for _, r in top.iterrows():
                rows.append(
                    f'<tr><td class="tk">{r["ticker"]}</td>'
                    f'<td>{r.get("sector", "")}</td>'
                    f'<td class="num">{r["n_strategies"]}</td>'
                    f'<td class="num">{r["avg_rank"]:.0f}</td>'
                    f'<td class="num">{r["mtd"]}</td>'
                    f'<td>{r["signals"]}</td></tr>')
            parts.append(f'<table><thead>{head}</thead>'
                         f'<tbody>{"".join(rows)}</tbody></table>')

    # Sector distribution
    if not consensus.empty:
        top_c = consensus[consensus['n_strategies'] >= 2]
        if not top_c.empty:
            sc = top_c['sector'].value_counts().to_dict()
            chips = ''.join(f'<span class="badge">{k}: {v}</span>'
                            for k, v in sc.items()
                            if k and k != 'Unknown')
            if chips:
                parts.append('<h2>Sector distribution</h2>'
                             f'<div class="card">{chips}</div>')

    # Backtest metrics (if provided)
    if backtest_metrics:
        pretty = {
            'cagr': 'CAGR', 'volatility': 'Volatility',
            'sharpe': 'Sharpe', 'sortino': 'Sortino',
            'max_drawdown': 'Max drawdown',
        }
        rows = []
        for k, label in pretty.items():
            if k in backtest_metrics:
                v = backtest_metrics[k]
                shown = f'{v:,.3f}' if isinstance(v, float) else v
                rows.append(
                    f'<tr><td>{label}</td><td class="num">{shown}</td></tr>')
        if rows:
            parts.append(
                '<h2>Backtest reference</h2><table><thead><tr><th>Metric</th>'
                '<th>Value</th></tr></thead><tbody>'
                + ''.join(rows) + '</tbody></table>')

    # Data availability warnings
    warnings = payload.get('data_warnings', [])
    if warnings:
        parts.append('<h2>⚠️ Data availability</h2><div class="card">'
                     '<p class="muted">Some strategies could not be compared '
                     'against a prior period. The report auto-ran screens for '
                     'missing dates where possible; below are strategies that '
                     'still lack comparison data.</p>')
        for w in warnings:
            parts.append(f'<div class="warn" style="margin-bottom:6px;'
                         f'font-size:.82rem">• {w}</div>')
        parts.append('</div>')

    parts.append(_DISCLAIMER)
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>Monthly Review — {payload["as_of"]}</title>'
            f'<style>{CSS}</style></head><body><div class="wrap">'
            f'{"".join(parts)}</div></body></html>')


# ─────────────────────────────────────────────
# BUILD & PERSIST
# ─────────────────────────────────────────────

def build_report(strategy: str | None = None, kind: str = 'daily',
                 as_of: date | str | None = None,
                 include_news: bool = True,
                 include_policy: bool = True,
                 backtest_metrics: dict | None = None) -> tuple[str, str]:
    """Build a cross-strategy report. `strategy` is accepted for backward
    compatibility but ignored — reports aggregate all strategies."""
    as_of = pd.to_datetime(as_of or date.today()).date()
    compare_days = 1 if kind == 'daily' else 30
    payload = _gather_cross_strategy(as_of, compare_days)

    news_df = policy_df = None
    bench = _index_performance(as_of)
    if 'error' not in payload:
        consensus = payload['consensus']
        tickers = (consensus[consensus['n_strategies'] >= 2]['ticker']
                   .head(20).tolist()) if not consensus.empty else []
        if include_news and tickers:
            news_df = gather_news(tickers, as_of,
                                  limit=8 if kind == 'daily' else 12)
        if include_policy:
            sectors = list({s for s in consensus['sector'].tolist()
                            if s and s != 'Unknown'}
                           ) if not consensus.empty else []
            policy_df = POL.policy_asof(as_of, lookback_days=7,
                                        sectors=sectors or None)

    if kind == 'daily':
        html_out = render_daily(payload, news_df, policy_df, bench)
    else:
        stability = _monthly_stability(as_of)
        html_out = render_monthly(payload, stability, backtest_metrics,
                                  bench)

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / f'{kind}_cross_strategy_{as_of}.html'
    path.write_text(html_out, encoding='utf-8')

    db.upsert(db.reports, [{
        'report_id': str(uuid.uuid4()), 'kind': kind,
        'period_start': as_of - timedelta(days=compare_days),
        'period_end': as_of, 'path': str(path),
        'emailed_at': None, 'created_at': config.utc_now(),
    }])

    log.info('report written: %s', path)
    return str(path), html_out


def delete_report(report_id: str) -> bool:
    row = db.read_sql('SELECT path FROM reports WHERE report_id = :r',
                      {'r': report_id})
    if row.empty:
        return False
    path = row.iloc[0]['path']
    deleted = db.execute('DELETE FROM reports WHERE report_id = :r',
                         {'r': report_id})
    if not deleted:
        return False
    still_used = db.read_sql(
        'SELECT COUNT(*) AS n FROM reports WHERE path = :p', {'p': path}
    ).iloc[0]['n']
    if not still_used and path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as exc:
            log.warning('deleted row but could not remove %s: %s', path, exc)
    log.info('deleted report %s', report_id[:8])
    return True


def list_reports(limit: int = 30) -> pd.DataFrame:
    return db.read_sql("""
        SELECT report_id, kind, period_end, path, emailed_at, created_at
        FROM reports ORDER BY created_at DESC LIMIT :n
    """, {'n': limit}, parse_dates=['period_end', 'created_at'])
