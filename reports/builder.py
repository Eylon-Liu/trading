"""
Cross-strategy market reports.

Instead of one strategy at a time, each report aggregates across every
strategy that has a stored run, so the trader gets a single view of where
the models agree, where they diverge, and what changed.

Daily  → what to pay attention to *today*: consensus picks, signal shifts,
         1-day returns, news, policy.
News   → the brief for a day the market is shut (weekends, holidays): only
         what was published since the last session — no prices, no screens,
         because nothing traded and re-sending Friday's numbers on Saturday
         would read as if they were new.
Monthly → what happened *this month*: which strategies called it right,
          consensus turnover, MTD returns, sector rotation.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

import config
from core import db
from core import market_calendar as MC
from data import earnings as ER
from data import macro as MACRO
from data import news as NEWS
from data import policy as POL
from data import yahoo
from data.universe import UniverseSpec
from quant import custom as CU
from quant import engine as ENG
from quant import portfolio as PF
from quant import research as RS
from quant import strategies as ST
from quant import track as TRACK

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


def _is_meta(key: str) -> bool:
    """True for a strategy built from other strategies (the consensus)."""
    try:
        return bool(ST.get(key).is_meta)
    except Exception:                              # noqa: BLE001
        return False


def _is_avoid(key: str) -> bool:
    """True for an inverted strategy — a list of names to avoid, worst first."""
    try:
        return bool(ST.get(key).invert)
    except Exception:                              # noqa: BLE001
        return False


def _latest_run(strategy: str, on_or_before: date,
                spec_json: str | None = None) -> pd.Series | None:
    """
    Newest stored run of `strategy` dated on or before `on_or_before`.

    `spec_json` pins the universe. The scheduled brief passes it so that an
    ad-hoc screen run in the UI on a narrower universe (one sector, a
    custom filter) cannot quietly become the run the email reports on.
    """
    clause = 'AND universe_spec_json = :u' if spec_json else ''
    params = {'s': strategy, 'd': str(on_or_before)}
    if spec_json:
        params['u'] = spec_json
    df = db.read_sql(f"""
        SELECT run_id, as_of, strategy, universe_n, universe_spec_json
        FROM runs
        WHERE strategy = :s AND as_of <= :d {clause}
        ORDER BY as_of DESC, created_at DESC LIMIT 1
    """, params, parse_dates=['as_of'])
    return df.iloc[0] if not df.empty else None


def _session_on_or_before(d: date) -> date:
    return d if MC.is_trading_day(d) else MC.previous_trading_day(d)


def _cutoff(d: date, max_staleness: int, sessions: int = 2) -> date:
    """
    Oldest run date still treated as current for a report dated `d`.

    The more lenient of `max_staleness` calendar days and `sessions` trading
    sessions. Calendar days alone dropped every strategy on the Tuesday after
    a holiday Monday, when the newest run is legitimately four days old.
    """
    s = _session_on_or_before(d)
    for _ in range(sessions):
        s = MC.previous_trading_day(s)
    return min(d - timedelta(days=max_staleness), s)


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
                           max_staleness: int = 3,
                           spec_json: str | None = None) -> dict:
    """
    Collect the latest run for every strategy and aggregate.

    A daily report compares each run with the run for the *previous trading
    session* — not "as_of minus one day", which on a Monday pointed at Sunday
    and after a morning-and-evening schedule compared two runs over the same
    closing prices.
    """
    cutoff = _cutoff(as_of, max_staleness)
    strategies_used: list[dict] = []
    all_scores: list[pd.DataFrame] = []
    prior_scores: list[pd.DataFrame] = []
    prior_runs: dict[str, str] = {}
    data_warnings: list[str] = []
    prior_dates: set[date] = set()

    avoid_rows: list[pd.DataFrame] = []
    consensus_run = pd.DataFrame()
    for key in _all_strategy_keys():
        run = _latest_run(key, as_of, spec_json)
        if run is None:
            continue
        run_date = pd.to_datetime(run['as_of']).date()
        if run_date < cutoff:
            continue
        scores = _run_scores(run['run_id'])
        if scores.empty:
            continue
        if _is_meta(key):
            # The consensus is itself a count of agreement; counting it as
            # one more voter would double-count every strategy inside it.
            top = scores.head(25).copy()
            top['strategy_name'] = _strategy_name(key)
            consensus_run = top
            continue
        if _is_avoid(key):
            # An avoid list ranks the worst names first. Counting its top 20
            # as "agreement" put flagged companies into the consensus picks.
            top = scores.head(15).copy()
            top['strategy'] = key
            top['strategy_name'] = _strategy_name(key)
            avoid_rows.append(top)
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

        if compare_days <= 1:
            prior_date = MC.previous_trading_day(run_date)
            prior_cutoff = _cutoff(prior_date, max_staleness, sessions=1)
        else:
            prior_date = as_of - timedelta(days=compare_days)
            prior_cutoff = prior_date - timedelta(days=max_staleness)
        prior_dates.add(prior_date)
        prior = _latest_run(key, prior_date, spec_json)
        prior_ok = (prior is not None
                    and prior['run_id'] != run['run_id']
                    and pd.to_datetime(prior['as_of']).date() >= prior_cutoff)
        if prior_ok:
            ps = _run_scores(prior['run_id'])
            if not ps.empty:
                ps = ps.copy()
                ps['strategy'] = key
                prior_scores.append(ps)
                prior_runs[key] = prior['run_id']
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
                    prior_runs[key] = auto_run['run_id']
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
        'combined': combined,
        'signal_dist': signal_dist,
        'data_warnings': data_warnings,
        'data_through': max(s['run_date'] for s in strategies_used),
        'avoid': (pd.concat(avoid_rows, ignore_index=True)
                  if avoid_rows else pd.DataFrame()),
        'consensus_run': consensus_run,
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

        if compare_days <= 1 and len(prior_dates) == 1:
            compare_label = f'the previous session ({_fmt_day(next(iter(prior_dates)))})'
        elif compare_days <= 1:
            compare_label = 'the previous session'
        else:
            compare_label = f'{compare_days} days ago'
        payload['compare_label'] = (
            f'{compare_label} ({n_paired} paired '
            f'{"strategy" if n_paired == 1 else "strategies"})')
        payload['new_consensus'] = new_consensus
        payload['lost_consensus'] = lost_consensus
        payload['momentum_up'] = momentum_up
        payload['momentum_down'] = momentum_down

        # Per-strategy signal shifts, against the same prior run the
        # consensus comparison used. Names that only just entered the list
        # have no earlier signal to shift from, so they are not shifts.
        signal_shifts = []
        for info in strategies_used:
            prev_id = prior_runs.get(info['key'])
            if prev_id is None or prev_id == info['run_id']:
                continue
            comp = RS.compare_runs(prev_id, info['run_id'])
            if comp.empty:
                continue
            changed = comp[comp['signal_changed']
                           & comp['signal_then'].notna()
                           & comp['signal_now'].notna()
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
        if _is_avoid(key) or _is_meta(key):
            continue
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
# CONTEXT: DATES, FRESHNESS, WATCHLIST
# ─────────────────────────────────────────────

def _safe(fn, default=None):
    """
    Run an optional section's gatherer; on failure log and return `default`.

    One broken add-on (a missing table, a provider hiccup) must not cost the
    whole email — the brief is still worth sending without that section.
    """
    try:
        return fn()
    except Exception as exc:                       # noqa: BLE001
        log.warning('report section skipped: %s: %s', type(exc).__name__, exc)
        return default


def _recent_reviews(since: date, as_of: date, held: list[str],
                    watch: list[str]) -> pd.DataFrame:
    """LLM reviews written since `since`; your names first, then by score."""
    rv = db.read_sql("""
        SELECT * FROM llm_reviews WHERE assessed_on BETWEEN :lo AND :hi
        ORDER BY assessed_on DESC
    """, {'lo': str(since), 'hi': str(as_of)}, parse_dates=['filed', 'assessed_on'])
    if rv.empty:
        return rv
    mine = set(held) | set(watch)
    rv['_mine'] = ~rv['ticker'].isin(mine)
    rv['_abs'] = -rv['score'].abs()
    return rv.sort_values(['_mine', '_abs']).drop(columns=['_mine', '_abs'])


def _first_session_of_week(d: date) -> bool:
    """True on the first trading day of an ISO week (usually Monday)."""
    if not MC.is_trading_day(d):
        return False
    prev = MC.previous_trading_day(d)
    return prev.isocalendar()[:2] != d.isocalendar()[:2]


def _fmt_day(d) -> str:
    """'Fri Nov 6' — portable (no %-d)."""
    d = pd.to_datetime(d).date()
    return f'{d:%a %b} {d.day}'


def _fmt_et(ts) -> str:
    """A naive-UTC timestamp as New York time, e.g. 'Sat 9:15 AM ET'."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    t = t.tz_convert(MC.ET)
    hour = t.hour % 12 or 12
    return f'{t:%a} {hour}:{t:%M} {"AM" if t.hour < 12 else "PM"} ET'


def _esc(v) -> str:
    """HTML-escape provider text. Headlines carry '&' and '<' routinely."""
    return html_lib.escape('' if v is None or (isinstance(v, float) and np.isnan(v))
                           else str(v))


def expected_session(as_of: date) -> date:
    """
    The session whose close a report dated `as_of` should reflect.

    For today, whatever has completed by now. For a past date the time of
    day is unknown, so assume a morning brief — the session before — which
    is the lenient choice and never flags a correct report as stale.
    """
    if as_of >= date.today():
        return MC.last_completed_session()
    return MC.previous_trading_day(as_of)


def _newest_bar(ticker: str) -> date | None:
    row = db.read_sql('SELECT MAX(date) AS d FROM prices WHERE ticker = :t',
                      {'t': ticker})
    if row.empty or pd.isna(row['d'].iloc[0]):
        return None
    return pd.to_datetime(row['d'].iloc[0]).date()


def freshness_warnings(as_of: date, data_through: date | None) -> list[str]:
    """
    Plain statements of anything stale, for the top of the email.

    A brief built on yesterday's numbers because last night's ingest failed
    looks exactly like a current one; the only defence is to say so.
    """
    expected = expected_session(as_of)
    out = []
    newest = _newest_bar(config.BENCHMARK_TICKER)
    if newest is None:
        out.append('No price data is stored yet — run '
                   '`python cli.py ingest --index SPY --full`.')
    elif newest < expected:
        out.append(f'Prices are current only to the {_fmt_day(newest)} close; '
                   f'the last completed session was {_fmt_day(expected)}. '
                   f'The price ingest may have failed — check logs/.')
    if data_through is not None and data_through < expected:
        out.append(f'Screens were last run for {_fmt_day(data_through)}, not '
                   f'{_fmt_day(expected)}. Rankings and signals below are '
                   f'from that run.')
    return out + feed_warnings()


def feed_warnings(now: datetime | None = None) -> list[str]:
    """
    News and policy feeds that could not be reached at their last refresh.

    Without this a dead feed is invisible: the weekend digest simply shows
    fewer (or no) articles, which reads as a quiet weekend. Three signals,
    all from stored bookkeeping — nothing is fetched here:
      * the last news refresh failed for every ticker, or for a large share;
      * articles are stored, but none newer than three days;
      * the Federal Register refresh failed.
    """
    now = now or config.utc_now()
    out: list[str] = []
    try:
        log_rows = db.read_sql(
            "SELECT source, key, last_success, last_error FROM ingest_log "
            "WHERE source IN ('news', 'policy')", parse_dates=['last_success'])
    except Exception as exc:                       # noqa: BLE001
        log.debug('feed warnings unavailable: %s', exc)
        return out
    news = log_rows[log_rows['source'] == 'news']
    if not news.empty:
        failed = int(news['last_error'].notna().sum())
        if failed == len(news):
            out.append('News could not be fetched at the last refresh — no '
                       'source answered (Yahoo, Google News). Articles below, '
                       'if any, are from earlier refreshes.')
        elif failed >= max(5, 0.2 * len(news)):
            out.append(f'News could not be fetched for {failed} of {len(news)} '
                       f'tickers at the last refresh; coverage below is partial.')
        else:
            newest = db.read_sql('SELECT MAX(published) AS p FROM news',
                                 parse_dates=['p'])['p'].iloc[0]
            if pd.notna(newest) and (now - newest) > timedelta(days=3):
                out.append(f'No article newer than {newest:%a %b %d} is stored — '
                           f'the news refresh may not be running.')
    policy = log_rows[log_rows['source'] == 'policy']
    if not policy.empty and policy['last_error'].notna().all():
        out.append('The Federal Register could not be reached at the last '
                   'refresh; the policy section may be out of date.')
    return out


def _returns_table(tickers: list[str], as_of: date) -> pd.DataFrame:
    """Last close and 1D / 5D / 1M returns, from stored bars only."""
    if not tickers:
        return pd.DataFrame()
    px = yahoo.price_history(tickers, start=as_of - timedelta(days=45),
                             end=as_of, field='adj_close')
    rows = []
    for t in tickers:
        col = px[t].dropna() if t in getattr(px, 'columns', []) else pd.Series(dtype=float)
        n = len(col)
        rows.append({
            'ticker': t,
            'price': col.iloc[-1] if n else np.nan,
            'r1d': col.iloc[-1] / col.iloc[-2] - 1 if n >= 2 else np.nan,
            'r5d': col.iloc[-1] / col.iloc[-6] - 1 if n >= 6 else np.nan,
            'r1m': col.iloc[-1] / col.iloc[max(0, n - 22)] - 1 if n >= 2 else np.nan,
        })
    return pd.DataFrame(rows).set_index('ticker')


def watchlist_summary(watchlist: list[str], combined: pd.DataFrame | None,
                      as_of: date) -> pd.DataFrame:
    """Per watched name: returns, and how every strategy currently ranks it."""
    if not watchlist:
        return pd.DataFrame()
    out = _returns_table(watchlist, as_of)
    best_rank, best_strat, n_top, votes = {}, {}, {}, {}
    if combined is not None and not combined.empty:
        for t, grp in combined[combined['ticker'].isin(watchlist)].groupby('ticker'):
            g = grp.sort_values('rank')
            best_rank[t] = int(g['rank'].iloc[0])
            best_strat[t] = g['strategy_name'].iloc[0]
            n_top[t] = int((g['rank'] <= TOP_N).sum())
            b = g['signal'].map(_signal_bucket).value_counts()
            votes[t] = ' · '.join(f'{k} {b[k]}' for k in ('Buy', 'Hold', 'Neutral', 'Sell')
                                  if k in b)
    out['best_rank'] = pd.Series(best_rank)
    out['best_strategy'] = pd.Series(best_strat)
    out['n_top'] = pd.Series(n_top)
    out['votes'] = pd.Series(votes)
    return out


# ─────────────────────────────────────────────
# NEWS DIGEST — for days the market is closed
# ─────────────────────────────────────────────

_ROUTINE_8K = {'governance', 'disclosure'}     # vote results, Reg FD, "other"


def _last_emailed() -> datetime | None:
    row = db.read_sql("SELECT MAX(emailed_at) AS t FROM reports "
                      "WHERE kind IN ('daily', 'news')", parse_dates=['t'])
    if row.empty or pd.isna(row['t'].iloc[0]):
        return None
    return row['t'].iloc[0].to_pydatetime()


def digest_window(as_of: date, now: datetime | None = None
                  ) -> tuple[datetime, datetime, date]:
    """
    (start, end, last_session) for a news digest, as naive UTC.

    Coverage starts at the last session's close — nothing earlier is news
    to someone who saw that day's brief — and, on a second closed day in a
    row, at most 24 hours back so Sunday does not repeat Saturday. If an
    email was missed (the machine was asleep), the window reaches back to
    the last one that went out, but never past the close.
    """
    # An explicit `now` always wins. Only when none is given and `as_of` is
    # in the past does the window close at the end of that day — otherwise a
    # digest rebuilt the next morning reached into the following day.
    if now is not None:
        end = now
    elif as_of >= date.today():
        end = config.utc_now()
    else:
        end = datetime.combine(as_of + timedelta(days=1), datetime.min.time())
    last = MC.last_completed_session(end)
    close = MC.session_close(last).astimezone(timezone.utc).replace(tzinfo=None)
    lower = end - timedelta(hours=24)
    emailed = _last_emailed()
    if emailed is not None and emailed < lower:
        lower = emailed
    return max(close, lower), end, last


def _current_picks(as_of: date, spec_json: str | None,
                   min_agree: int = 2, limit: int = 25) -> list[str]:
    """
    Current picks from the latest stored runs — no re-screening.

    The multi-strategy consensus, when it has a run: names with at least
    `min_agree` family votes. Otherwise names in `min_agree` top-20 lists.
    """
    meta = next((k for k in _all_strategy_keys() if _is_meta(k)), None)
    if meta:
        run = _latest_run(meta, as_of, spec_json)
        if run is not None:
            sc = _run_scores(run['run_id'])
            picked = sc[sc['signal_score'] >= min_agree]
            if not picked.empty:
                return picked.head(limit)['ticker'].tolist()
    rows = []
    for key in _all_strategy_keys():
        if _is_avoid(key) or _is_meta(key):
            continue
        run = _latest_run(key, as_of, spec_json)
        if run is None:
            continue
        sc = _run_scores(run['run_id'])
        top = sc[sc['rank'] <= TOP_N][['ticker', 'rank']].copy()
        top['strategy'] = key
        rows.append(top)
    if not rows:
        return []
    allc = pd.concat(rows, ignore_index=True)
    agg = (allc.groupby('ticker').agg(n=('strategy', 'nunique'), r=('rank', 'mean'))
           .sort_values(['n', 'r'], ascending=[False, True]))
    return agg[agg['n'] >= min_agree].head(limit).index.tolist()


def gather_digest(as_of: date, now: datetime | None = None,
                  watchlist: list[str] | None = None,
                  spec_json: str | None = None) -> dict:
    """Everything published since the last session, grouped by why it matters."""
    start, end, last = digest_window(as_of, now)
    watchlist = watchlist or []

    news = db.read_sql("""
        SELECT ticker, published, source, title, url, body_summary,
               sentiment, sentiment_label, events
        FROM news WHERE published >= :lo AND published < :hi
        ORDER BY published DESC
    """, {'lo': str(start), 'hi': str(end)}, parse_dates=['published'])

    if not news.empty:
        # One article is often filed under several tickers ("Apple and
        # Microsoft ..."); show it once, tagged with all of them.
        news['key'] = news['title'].map(NEWS.normalize_title)
        tickers = news.groupby('key')['ticker'].agg(lambda x: sorted(set(x)))
        news = news.drop_duplicates('key').copy()
        news['tickers'] = news['key'].map(tickers)
        news['events'] = news['events'].fillna('')
        news['n_events'] = news['events'].map(lambda e: len([x for x in e.split(',') if x]))
        news['abs_sent'] = news['sentiment'].abs().fillna(0)

    picks = _current_picks(as_of, spec_json)

    def touching(names: list[str]) -> pd.DataFrame:
        if news.empty or not names:
            return pd.DataFrame()
        wanted = set(names)
        return news[news['tickers'].map(lambda ts: bool(wanted & set(ts)))]

    watch_news = touching(watchlist)
    shown = set(watch_news['key']) if not watch_news.empty else set()
    pick_news = touching([p for p in picks if p not in watchlist])
    if not pick_news.empty:
        pick_news = pick_news[~pick_news['key'].isin(shown)]
        shown |= set(pick_news['key'])
    material = pd.DataFrame()
    if not news.empty:
        material = (news[(news['n_events'] > 0) & ~news['key'].isin(shown)]
                    .sort_values(['n_events', 'abs_sent'], ascending=False))

    filings = db.read_sql("""
        SELECT e.ticker, e.filed, e.item_code, e.category, e.subtype,
               s.name
        FROM corporate_events e LEFT JOIN securities s ON s.ticker = e.ticker
        WHERE e.filed >= :lo AND e.filed <= :hi
        ORDER BY e.filed DESC, e.ticker
    """, {'lo': str(start.date()), 'hi': str(end.date())}, parse_dates=['filed'])
    if not filings.empty:
        filings = filings[~filings['category'].isin(_ROUTINE_8K)]

    # Filings and Federal Register documents carry a date, not a time, so
    # they are cut at the window's first day: Friday's late filings reach
    # Saturday's digest, and Sunday's does not repeat them.
    policy = POL.policy_asof(end.date(),
                             lookback_days=max(1, (end.date() - start.date()).days))
    if policy is not None and not policy.empty:
        policy = policy[policy['published'].dt.date >= start.date()]

    return {
        'as_of': as_of, 'start': start, 'end': end, 'last_session': last,
        'closed_reason': MC.closed_reason(as_of),
        'next_session': (as_of if MC.is_trading_day(as_of)
                         else MC.next_trading_day(as_of)),
        'news': news, 'watchlist': watchlist, 'picks': picks,
        'watch_news': watch_news, 'pick_news': pick_news,
        'material': material, 'filings': filings, 'policy': policy,
    }


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
a{color:#7fdbca;text-decoration:none}
.alert{background:rgba(255,217,61,.10);border:1px solid rgba(255,217,61,.45);
       color:#ffd93d;border-radius:8px;padding:10px 14px;margin:0 0 14px;
       font-size:.84rem}
.note{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.3);
      border-radius:8px;padding:10px 14px;margin:0 0 14px;font-size:.84rem;
      color:#b0e9ff}
.art{margin-bottom:12px;line-height:1.45}
.art .meta{color:#8b93a7;font-size:.74rem;margin-top:2px}
.art .sum{color:#b0b8cc;font-size:.8rem;margin-top:3px}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:.7rem;
     background:rgba(255,255,255,.08);margin-right:4px}
.tag.pos{background:rgba(56,239,125,.15)}.tag.neg{background:rgba(255,107,107,.15)}
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
            f'text-transform:uppercase">{_esc(r["doc_type"])}</span></div>'
            f'<div style="font-weight:600;font-size:.88rem;'
            f'margin-bottom:4px">{_link(r.get("title"), r.get("url"))}</div>')
        if abstract:
            parts.append(
                f'<div style="font-size:.8rem;color:#b0b8cc;'
                f'line-height:1.5;margin-bottom:6px">{_esc(abstract)}'
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


def _render_alerts(alerts: list[dict] | None) -> str:
    if not alerts:
        return ''
    icon = {'risk': '🔴', 'watch': '🟠', 'info': '🔵'}
    rows = ''.join(
        f'<div style="margin-bottom:6px">{icon.get(a["level"], "•")} '
        f'<span class="tk">{_esc(a["ticker"])}</span> {_esc(a["text"])}</div>'
        for a in alerts[:25])
    return f'<h2>🚨 Alerts</h2><div class="card">{rows}</div>'


def _fmt_money(v) -> str:
    if v is None or not np.isfinite(v):
        return '—'
    return f'{"-" if v < 0 else ""}${abs(v):,.0f}'


def _render_portfolio(h: pd.DataFrame | None, s: dict | None,
                      n_strategies: int) -> str:
    if h is None or h.empty:
        return ''
    s = s or {}
    color = lambda v: '#38ef7d' if v >= 0 else '#ff6b6b'     # noqa: E731
    cards = [
        _metric_card(_fmt_money(s.get('market_value')), 'Market value', '#00d4ff'),
        _metric_card(_fmt_money(s.get('day_pnl')), 'Last session',
                     color(s.get('day_pnl') or 0)),
        _metric_card(_fmt_money(s.get('pnl')), 'Unrealised P&amp;L',
                     color(s.get('pnl') or 0)),
        _metric_card(f'{s.get("pnl_pct", 0):+.1%}' if np.isfinite(s.get('pnl_pct', np.nan))
                     else '—', 'On cost', color(s.get('pnl_pct') or 0)),
    ]
    head = ('<tr><th>Ticker</th><th>Shares</th><th>Avg cost</th><th>Last</th>'
            '<th>1D</th><th>P&amp;L</th><th>Weight</th><th>Stop</th>'
            '<th>Target</th><th>Top-20 in</th><th>Signals</th></tr>')
    rows = []
    for t, r in h.iterrows():
        def lvl(v, dist):
            if not np.isfinite(v):
                return '—'
            d = f' <span class="muted">({dist:+.1%})</span>' if np.isfinite(dist) else ''
            return f'{v:,.2f}{d}'
        n_top = int(r['n_top']) if pd.notna(r.get('n_top')) else 0
        votes = r.get('votes')
        rows.append(
            f'<tr><td class="tk">{_esc(t)}</td>'
            f'<td class="num">{r["shares"]:,.4g}</td>'
            f'<td class="num">{r["avg_cost"]:,.2f}</td>'
            f'<td class="num">{r["last"]:,.2f}</td>'
            f'<td class="num">{_color_pct(r["day_chg"])}</td>'
            f'<td class="num">{_color_pct(r["pnl_pct"])} '
            f'<span class="muted">{_fmt_money(r["pnl"])}</span></td>'
            f'<td class="num">{r["weight"]:.0%}</td>'
            f'<td class="num">{lvl(r["stop"], -r["to_stop"] if np.isfinite(r["to_stop"]) else np.nan)}</td>'
            f'<td class="num">{lvl(r["target"], r["to_target"])}</td>'
            f'<td class="muted">{n_top}/{n_strategies}</td>'
            f'<td class="muted">{_esc(votes) if isinstance(votes, str) else "—"}</td></tr>')
    return ('<h2>💼 Your portfolio</h2>'
            f'<div class="metric-grid">{"".join(cards)}</div>'
            f'<table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
            '<p class="muted">Price return on the raw close against your cost; '
            'splits applied, dividends not included.</p>')


def _render_macro(snap: pd.DataFrame | None, flags: list | None) -> str:
    if (snap is None or snap.empty) and not flags:
        return ''
    parts = ['<h2>🌐 Macro &amp; regime</h2>']
    if flags:
        icon = {'ok': '🟢', 'watch': '🟠', 'risk': '🔴'}
        parts.append('<div class="card">' + ''.join(
            f'<div style="margin-bottom:4px">{icon.get(l, "•")} {_esc(t)}</div>'
            for l, t in flags) + '</div>')
    if snap is not None and not snap.empty:
        def fmt(v, unit, sign=False):
            if not np.isfinite(v):
                return '—'
            body = f'{v:+.2f}' if sign else f'{v:.2f}'
            return body + ('%' if unit == '%' and not sign else
                           'pp' if unit in ('%', 'pp') else '')
        rows = ''.join(
            f'<tr><td>{_esc(r["label"])}</td>'
            f'<td class="num">{fmt(r["value"], r["unit"])}</td>'
            f'<td class="num">{fmt(r["chg_1w"], r["unit"], True)}</td>'
            f'<td class="num">{fmt(r["chg_1m"], r["unit"], True)}</td>'
            f'<td class="muted">{_fmt_day(r["date"])}</td></tr>'
            for _i, r in snap.iterrows())
        parts.append('<table><thead><tr><th>Series</th><th>Level</th><th>1W</th>'
                     f'<th>1M</th><th>As of</th></tr></thead><tbody>{rows}</tbody></table>'
                     '<p class="muted">Source: FRED. Context only — none of this '
                     'changes a score.</p>')
    return ''.join(parts)


def _render_earnings(er: pd.DataFrame | None, held: set | None = None,
                     watch: set | None = None, title: str = '📅 Earnings ahead'
                     ) -> str:
    if er is None or er.empty:
        return ''
    held, watch = held or set(), watch or set()
    rows = []
    for _i, e in er.head(40).iterrows():
        tag = ('<span class="tag neg">held</span>' if e['ticker'] in held else
               '<span class="tag">watch</span>' if e['ticker'] in watch else '')
        when = {'bmo': 'before open', 'amc': 'after close'}.get(e['time'], '—')
        est = e.get('eps_estimate')
        rows.append(
            f'<tr><td class="tk">{_esc(e["ticker"])}</td><td>{tag}</td>'
            f'<td>{_fmt_day(e["report_date"])}</td><td class="muted">{when}</td>'
            f'<td class="num">{f"{est:.2f}" if est is not None and np.isfinite(est) else "—"}</td>'
            f'<td class="muted">{"confirmed" if e["confirmed"] else "estimated"}</td></tr>')
    return (f'<h2>{title}</h2><table><thead><tr><th>Ticker</th><th></th><th>Date</th>'
            '<th>When</th><th>EPS est.</th><th>Source</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            '<p class="muted">Estimated dates come from the company’s own filing '
            'cadence (8-K Item 2.02) and can be off by days.</p>')


def _render_consensus(cons: pd.DataFrame | None, held: set | None = None,
                      watch: set | None = None) -> str:
    """The multi-strategy consensus: names several families agree on."""
    if cons is None or cons.empty:
        return ''
    held, watch = held or set(), watch or set()
    top = cons[cons['signal_score'] >= 2]
    if top.empty:
        return ('<h2>🏆 Multi-strategy consensus</h2><p class="muted">No name has '
                'two or more strategy families agreeing today.</p>')
    rows = []
    for _i, r in top.head(15).iterrows():
        tag = ('<span class="tag neg">held</span>' if r['ticker'] in held else
               '<span class="tag">watch</span>' if r['ticker'] in watch else '')
        rows.append(
            f'<tr><td class="tk">{_esc(r["ticker"])}</td><td>{tag}</td>'
            f'<td>{_esc(r.get("sector") or "")}</td>'
            f'<td>{_esc(r["signal"])}</td>'
            f'<td class="muted" style="font-size:.74rem">{_esc(r.get("reasons") or "")}</td></tr>')
    return ('<h2>🏆 Multi-strategy consensus</h2>'
            '<p class="muted" style="margin-top:0">Names that independent strategy '
            'families rank in their top 20% at once. Correlated strategies vote '
            'once per family; the avoid list costs a vote. ✓ = this family votes '
            'for it, with its percentile.</p>'
            '<table><thead><tr><th>Ticker</th><th></th><th>Sector</th><th>Agreement</th>'
            f'<th>Families</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _render_reviews(rv: pd.DataFrame | None, held: set | None = None,
                    watch: set | None = None,
                    title: str = '🧠 AI earnings reviews') -> str:
    """New LLM reviews of results releases."""
    if rv is None or rv.empty:
        return ''
    held, watch = held or set(), watch or set()
    colour = {'buy': 'pos', 'accumulate': 'pos', 'avoid': 'neg'}
    parts = [f'<h2>{title}</h2><div class="card">'
             '<p class="muted" style="margin-top:0">A language model read each '
             'release with the computed figures and judged it for a 6–18 month '
             'holder. Opinion, not data — it feeds the AI review vote and the AI '
             'Earnings strategy, and its record is graded live.</p>']
    for _i, r in rv.head(12).iterrows():
        try:
            extra = json.loads(r.get('detail_json') or '{}')
        except (TypeError, ValueError):
            extra = {}
        tag = (' <span class="tag neg">held</span>' if r['ticker'] in held else
               ' <span class="tag">watch</span>' if r['ticker'] in watch else '')
        verdict = str(r.get('verdict') or '').lower()
        pros = '; '.join(extra.get('positives') or [])
        cons = '; '.join(extra.get('negatives') or [])
        parts.append(
            f'<div class="art"><span class="tk">{_esc(r["ticker"])}</span>{tag} '
            f'<span class="tag {colour.get(verdict, "")}">{_esc(verdict.upper())}</span>'
            f'<span class="muted">score {r["score"]:+.1f} · {_esc(r.get("conviction"))} '
            f'conviction · guidance {_esc(r.get("guidance"))} · valuation '
            f'{_esc(r.get("valuation_view"))}</span>'
            f'<div class="sum">{_esc(r.get("summary") or "")}</div>'
            + (f'<div class="meta">＋ {_esc(pros)}</div>' if pros else '')
            + (f'<div class="meta">－ {_esc(cons)}</div>' if cons else '')
            + f'<div class="meta">results filed {_fmt_day(r["filed"])} · '
              f'{_esc(r.get("model") or "")}</div></div>')
    parts.append('</div>')
    return ''.join(parts)


def _render_avoid(av: pd.DataFrame | None, held: set | None = None,
                  picks: set | None = None, watch: set | None = None) -> str:
    """Names the avoid-list strategies flag, with any overlap called out."""
    if av is None or av.empty:
        return ''
    held, picks, watch = held or set(), picks or set(), watch or set()
    av = av[av['signal'].fillna('').str.contains('Avoid|Caution')]
    if av.empty:
        return ''
    rows = []
    for _i, r in av.iterrows():
        flags = []
        if r['ticker'] in held:
            flags.append('<span class="tag neg">you hold it</span>')
        if r['ticker'] in watch:
            flags.append('<span class="tag">watchlist</span>')
        if r['ticker'] in picks:
            flags.append('<span class="tag">also a consensus pick</span>')
        rows.append(
            f'<tr><td class="tk">{_esc(r["ticker"])}</td>'
            f'<td class="num">{int(r["rank"])}</td><td>{_esc(r["signal"])}</td>'
            f'<td class="muted" style="font-size:.76rem">{_esc(r.get("reasons") or "")}</td>'
            f'<td>{" ".join(flags)}</td></tr>')
    names = ', '.join(sorted(set(av['strategy_name'])))
    return (f'<h2>⚠️ Avoid list — {_esc(names)}</h2>'
            '<p class="muted" style="margin-top:0">Worst-first: weakening '
            'fundamentals and trend together. Not counted in the consensus; '
            'listed so a name you hold or watch is not a surprise.</p>'
            '<table><thead><tr><th>Ticker</th><th>Rank</th><th>Signal</th>'
            f'<th>Why</th><th></th></tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _render_track(tr: pd.DataFrame | None) -> str:
    if tr is None or tr.empty:
        return ''
    rows = []
    for _i, r in tr.iterrows():
        rows.append(
            f'<tr><td>{_esc(r["name"])}</td><td>{r["horizon"]}</td>'
            f'<td class="num">{r["cohorts"]} '
            f'<span class="muted">(~{r["independent"]} indep.)</span></td>'
            f'<td class="num">{_color_pct(r["mean_pick_return"])}</td>'
            f'<td class="num">{_color_pct(r["mean_excess"])}</td>'
            f'<td class="num">{r["hit_rate"]:.0%}</td>'
            f'<td class="num">{r["mean_ic"]:+.3f}</td></tr>'
            if pd.notna(r['mean_ic']) else
            f'<tr><td>{_esc(r["name"])}</td><td>{r["horizon"]}</td>'
            f'<td class="num">{r["cohorts"]}</td>'
            f'<td class="num">{_color_pct(r["mean_pick_return"])}</td>'
            f'<td class="num">{_color_pct(r["mean_excess"])}</td>'
            f'<td class="num">{r["hit_rate"]:.0%}</td><td class="num">—</td></tr>')
    return ('<h2>📈 Live track record</h2>'
            '<p class="muted" style="margin-top:0">What each strategy’s top-10 '
            'picks did after the screens this routine actually stored, bought at '
            'the next close. Overlapping daily cohorts are not independent — '
            'judge by the “indep.” count, not the cohort count. “Worked” is the '
            'share of cohorts that beat SPY — or, for an avoid list, that lagged '
            'it; rank IC is signed so that positive always means the list '
            'pointed the right way.</p>'
            '<table><thead><tr><th>Strategy</th><th>Horizon</th><th>Cohorts</th>'
            '<th>Picks</th><th>vs SPY</th><th>Worked</th><th>Rank IC</th></tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table>')


def render_daily(payload: dict, news_df: pd.DataFrame | None = None,
                 policy_df: pd.DataFrame | None = None,
                 bench: dict | None = None) -> str:
    if 'error' in payload:
        return f'<html><body><p>{payload["error"]}</p></body></html>'

    through = payload.get('data_through')
    parts = [
        '<h1>Market Intelligence Brief</h1>',
        f'<div class="sub">{_fmt_day(payload["as_of"])} &nbsp;·&nbsp; '
        + (f'data through the {_fmt_day(through)} close &nbsp;·&nbsp; '
           if through else '')
        + f'{payload["n_strategies"]} strategies compared</div>',
    ]
    for w in payload.get('freshness_warnings') or []:
        parts.append(f'<div class="alert">⚠️ {_esc(w)}</div>')
    if payload.get('calendar_note'):
        parts.append(f'<div class="note">🗓️ {_esc(payload["calendar_note"])}</div>')

    parts.append(_render_alerts(payload.get('alerts')))

    # Index performance
    if bench:
        parts.append('<h2>Market snapshot</h2>')
        parts.append(_render_bench(bench))
    parts.append(_render_macro(payload.get('macro'), payload.get('regime')))

    parts.append(_render_portfolio(payload.get('portfolio'),
                                   payload.get('portfolio_summary'),
                                   payload['n_strategies']))
    parts.append(_render_watchlist(payload.get('watchlist_table'),
                                   payload['n_strategies']))
    watch_news = payload.get('watch_news')
    if watch_news is not None and not watch_news.empty:
        parts.append('<h2>News on your names (last 3 days)</h2><div class="card">')
        parts.extend(_article_html(r) for _i, r in watch_news.iterrows())
        parts.append('</div>')
    parts.append(_render_consensus(payload.get('consensus_run'),
                                   payload.get('earnings_held'),
                                   payload.get('earnings_watch')))
    parts.append(_render_reviews(payload.get('reviews'),
                                 payload.get('earnings_held'),
                                 payload.get('earnings_watch')))
    parts.append(_render_earnings(payload.get('earnings'),
                                  payload.get('earnings_held'),
                                  payload.get('earnings_watch')))

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
            parts.append('<h2>Strategy overlap (2+ strategies rank it top 20)</h2>')
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

    if not consensus.empty:
        pick_set = set(consensus[consensus['n_strategies'] >= 2]['ticker'])
    else:
        pick_set = set()
    parts.append(_render_avoid(payload.get('avoid'), payload.get('earnings_held'),
                               pick_set, payload.get('earnings_watch')))

    # News
    if news_df is not None and not news_df.empty:
        parts.append('<h2>Notable coverage</h2><div class="card">')
        parts.extend(_article_html(r) for _i, r in news_df.head(8).iterrows())
        parts.append('</div>')

    # Policy (filtered and expanded)
    parts.append(_render_policy(policy_df))
    parts.append(_render_track(payload.get('track')))

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

    parts.append(_render_track(payload.get('track')))

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


def _link(text, url) -> str:
    t = _esc(text)
    u = str(url or '')
    if u.startswith(('http://', 'https://')):
        return f'<a href="{html_lib.escape(u, quote=True)}">{t}</a>'
    return t


def _article_html(r) -> str:
    """One headline: tickers, tone, linked title, source/time, events, summary."""
    tickers = r.get('tickers')
    if not isinstance(tickers, (list, tuple)):
        tickers = [r.get('ticker')] if r.get('ticker') else []
    chips = ' '.join(f'<span class="tk">{_esc(t)}</span>' for t in tickers)
    label = str(r.get('sentiment_label') or '').strip()
    tone = ''
    if label:
        cls = ('pos' if 'pos' in label.lower()
               else 'neg' if 'neg' in label.lower() else '')
        tone = f'<span class="tag {cls}">{_esc(label)}</span>'
    meta = [_esc(r.get('source') or '')]
    if pd.notna(r.get('published')):
        meta.append(_fmt_et(r['published']))
    events = str(r.get('events') or '')
    if events:
        meta.append(' · '.join(_esc(e.replace('_', ' '))
                               for e in events.split(',') if e))
    summary = str(r.get('body_summary') or '').strip()
    return (f'<div class="art">{chips} {tone}{_link(r.get("title"), r.get("url"))}'
            f'<div class="meta">{" · ".join(m for m in meta if m)}</div>'
            + (f'<div class="sum">{_esc(summary[:280])}</div>' if summary else '')
            + '</div>')


def _render_watchlist(table: pd.DataFrame | None, n_strategies: int) -> str:
    if table is None or table.empty:
        return ''
    head = ('<tr><th>Ticker</th><th>Price</th><th>1D</th><th>5D</th><th>1M</th>'
            '<th>Top-20 in</th><th>Best rank</th><th>Signals</th></tr>')
    rows = []
    for t, r in table.iterrows():
        price = f'{r["price"]:,.2f}' if pd.notna(r['price']) else '—'
        n_top = int(r['n_top']) if pd.notna(r.get('n_top')) else 0
        best = (f'#{int(r["best_rank"])} · {_esc(r["best_strategy"])}'
                if pd.notna(r.get('best_rank')) else 'not ranked')
        votes = r.get('votes')
        rows.append(
            f'<tr><td class="tk">{_esc(t)}</td><td class="num">{price}</td>'
            f'<td class="num">{_color_pct(r["r1d"])}</td>'
            f'<td class="num">{_color_pct(r["r5d"])}</td>'
            f'<td class="num">{_color_pct(r["r1m"])}</td>'
            f'<td style="min-width:80px">{_consensus_bar(n_top, n_strategies)}'
            f'<span class="muted">{n_top}/{n_strategies}</span></td>'
            f'<td class="muted">{best}</td>'
            f'<td class="muted">{_esc(votes) if isinstance(votes, str) else "—"}</td></tr>')
    return ('<h2>Your watchlist</h2>'
            f'<table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>')


def _calendar_note(today: date) -> str | None:
    """Closures and half-days in the next few days, so none is a surprise."""
    notes = []
    if MC.is_early_close(today):
        notes.append('today the market closes early at 1 p.m. ET')
    for i in range(1, 5):
        d = today + timedelta(days=i)
        reason = MC.holiday_name(d)
        if reason and d.weekday() < 5:
            notes.append(f'market closed {_fmt_day(d)} ({reason})')
        elif MC.is_early_close(d):
            notes.append(f'{_fmt_day(d)} closes early at 1 p.m. ET')
    return ('Heads-up: ' + '; '.join(notes) + '.') if notes else None


def render_news_digest(d: dict) -> str:
    """The closed-market email: news, filings and policy — nothing priced."""
    reason = d.get('closed_reason')
    if reason == 'Weekend':
        title = 'Weekend News Digest'
    elif reason:
        title = f'Market Closed — {_esc(reason)}'
    else:
        title = 'News Digest'
    nxt = d['next_session']
    early = ' (closes 1 p.m. ET)' if MC.is_early_close(nxt) else ''
    parts = [
        f'<h1>{title}</h1>',
        f'<div class="sub">{_fmt_day(d["as_of"])} &nbsp;·&nbsp; coverage since '
        f'{_fmt_et(d["start"])} &nbsp;·&nbsp; next session '
        f'{_fmt_day(nxt)}{early}</div>',
        '<div class="note">The market is closed, so this email carries news '
        'only — no prices, rankings or signals. The full brief resumes on the '
        'next trading day.</div>',
    ]
    for w in d.get('feed_warnings') or []:
        parts.append(f'<div class="alert">⚠️ {_esc(w)}</div>')

    news, filings, policy = d['news'], d['filings'], d['policy']
    n_tickers = (len({t for ts in news['tickers'] for t in ts})
                 if not news.empty else 0)
    n_events = int((news['n_events'] > 0).sum()) if not news.empty else 0
    n_filings = 0 if filings is None else len(filings)
    parts.append('<div class="metric-grid">' + ''.join([
        _metric_card(str(len(news)), 'Articles', '#00d4ff'),
        _metric_card(str(n_tickers), 'Tickers covered', '#e0e0e0'),
        _metric_card(str(n_events), 'With events', '#ffd93d'),
        _metric_card(str(n_filings), '8-K items', '#38ef7d'),
    ]) + '</div>')

    def section(heading: str, df: pd.DataFrame | None, limit: int,
                blurb: str = '', empty: str | None = None) -> None:
        if (df is None or df.empty) and empty is None:
            return
        parts.append(f'<h2>{heading}</h2><div class="card">')
        if blurb:
            parts.append(f'<p class="muted" style="margin-top:0">{blurb}</p>')
        if df is None or df.empty:
            parts.append(f'<p class="muted">{empty}</p>')
        else:
            parts.extend(_article_html(r) for _i, r in df.head(limit).iterrows())
        parts.append('</div>')

    if d['watchlist']:
        section('⭐ Your portfolio &amp; watchlist', d['watch_news'], 20,
                empty='No new coverage of your names in this window.')
    section('🎯 Current picks in the news', d['pick_news'], 15,
            blurb='Names that two or more strategies rank in their top 20 '
                  '(from the last session’s screens).')
    section('⚡ Material events', d['material'], 15,
            blurb='Guidance, capital allocation, M&amp;A, management and legal '
                  'news across the universe, strongest first.')

    if n_filings:
        shown = filings.head(25).assign(
            filed=filings['filed'].map(_fmt_day),
            name=filings['name'].fillna('').map(_esc),
            subtype=filings['subtype'].fillna('').map(_esc))
        parts.append(f'<h2>📄 SEC 8-K filings since {_fmt_day(d["start"])}</h2>')
        parts.append(_table(shown, {
            'ticker': 'Ticker', 'name': 'Company', 'filed': 'Filed',
            'item_code': 'Item', 'category': 'Category', 'subtype': 'What',
        }))

    parts.append(_render_policy(policy))
    parts.append(_render_reviews(d.get('reviews'), set(d.get('held') or []),
                                 set(d['watchlist'])))
    parts.append(_render_earnings(d.get('earnings'), title='📅 Reporting soon'))

    if not news.empty:
        counts = pd.Series([t for ts in news['tickers'] for t in ts]).value_counts()
        chips = ''.join(f'<span class="badge">{_esc(t)} · {n}</span>'
                        for t, n in counts.head(15).items())
        parts.append(f'<h2>Most-covered names</h2><div class="card">{chips}</div>')

    if news.empty and not n_filings and (policy is None or policy.empty):
        parts.append('<p class="muted">No new coverage, filings or policy '
                     f'documents since {_fmt_et(d["start"])}.</p>')

    parts.append(_DISCLAIMER)
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<title>{title} — {d["as_of"]}</title>'
            f'<style>{CSS}</style></head><body><div class="wrap">'
            f'{"".join(parts)}</div></body></html>')


# ─────────────────────────────────────────────
# BUILD & PERSIST
# ─────────────────────────────────────────────

REPORT_KINDS = ('daily', 'news', 'monthly')


def auto_kind(day: date | None = None) -> str:
    """'daily' on a trading day, 'news' whenever the market is closed."""
    return 'daily' if MC.is_trading_day(day or date.today()) else 'news'


def report_id_for(kind: str, as_of: date | str) -> str:
    """One history row per kind and date: rebuilding replaces, never piles up."""
    return f'{kind}-{pd.to_datetime(as_of).date()}'


def subject_for(kind: str, as_of: date | str) -> str:
    d = pd.to_datetime(as_of).date()
    if kind == 'news':
        reason = MC.closed_reason(d)
        if reason == 'Weekend':
            return f'📰 Weekend News — {_fmt_day(d)}'
        if reason:
            return f'📰 Market Closed ({reason}) — News, {_fmt_day(d)}'
        return f'📰 News Digest — {_fmt_day(d)}'
    if kind == 'monthly':
        return f'📊 Monthly Review — {d:%B %Y}'
    return f'📈 Market Brief — {_fmt_day(d)}'


def emailed_at(report_id: str) -> datetime | None:
    row = db.read_sql('SELECT emailed_at FROM reports WHERE report_id = :r',
                      {'r': report_id}, parse_dates=['emailed_at'])
    if row.empty or pd.isna(row['emailed_at'].iloc[0]):
        return None
    return row['emailed_at'].iloc[0].to_pydatetime()


def build_report(strategy: str | None = None, kind: str = 'daily',
                 as_of: date | str | None = None,
                 include_news: bool = True,
                 include_policy: bool = True,
                 backtest_metrics: dict | None = None,
                 universe: UniverseSpec | str | None = None,
                 watchlist: list[str] | None = None,
                 now: datetime | None = None) -> tuple[str, str]:
    """
    Build a cross-strategy report and record it. Returns (path, html).

    `kind` is 'daily', 'news' (closed-market digest) or 'monthly'; 'auto'
    picks daily or news from the exchange calendar. `universe` pins which
    stored runs are read (see `_latest_run`); `watchlist` defaults to
    QUANT_WATCHLIST. `strategy` is accepted for backward compatibility and
    ignored — reports aggregate all strategies.
    """
    as_of = pd.to_datetime(as_of or date.today()).date()
    if kind == 'auto':
        kind = auto_kind(as_of)
    if kind not in REPORT_KINDS:
        raise ValueError(f'unknown report kind {kind!r}; expected {REPORT_KINDS}')
    watchlist = list(PF.watchlist() if watchlist is None else watchlist)
    spec_json = universe.to_json() if isinstance(universe, UniverseSpec) else universe
    try:
        held = PF.holdings(as_of) if kind != 'monthly' else pd.DataFrame()
    except Exception as exc:                       # noqa: BLE001
        log.warning('portfolio valuation failed: %s', exc)
        held = pd.DataFrame()
    held_names = list(held.index) if not held.empty else []

    if kind == 'news':
        digest = gather_digest(as_of, now=now,
                               watchlist=sorted(set(watchlist) | set(held_names)),
                               spec_json=spec_json)
        digest['earnings'] = _safe(lambda: ER.upcoming(
            sorted(set(watchlist) | set(held_names) | set(digest['picks'])),
            as_of, days=10))
        digest['held'] = held_names
        digest['feed_warnings'] = _safe(feed_warnings) or []
        digest['reviews'] = _safe(lambda: _recent_reviews(
            digest['start'].date(), as_of, held_names, watchlist), pd.DataFrame())
        html_out = render_news_digest(digest)
        period_start = digest['last_session']
        filename = f'news_digest_{as_of}.html'
    else:
        compare_days = 1 if kind == 'daily' else 30
        payload = _gather_cross_strategy(as_of, compare_days, spec_json=spec_json)
        news_df = policy_df = None
        bench = _index_performance(as_of)
        if 'error' not in payload:
            payload['freshness_warnings'] = freshness_warnings(
                as_of, payload.get('data_through'))
            payload['calendar_note'] = _calendar_note(as_of) if kind == 'daily' else None
            only_watch = [t for t in watchlist if t not in held_names]
            if only_watch:
                payload['watchlist_table'] = watchlist_summary(
                    only_watch, payload.get('combined'), as_of)
            followed = sorted(set(watchlist) | set(held_names))
            if followed:
                wn = NEWS.news_asof(followed, as_of, lookback_days=3)
                if not wn.empty:
                    payload['watch_news'] = (wn.sort_values('published', ascending=False)
                                             .groupby('ticker').head(3))
            if kind == 'daily':
                if not held.empty:
                    ranks = watchlist_summary(held_names, payload.get('combined'), as_of)
                    payload['portfolio'] = held.join(
                        ranks[['best_rank', 'best_strategy', 'n_top', 'votes']])
                    payload['portfolio_summary'] = PF.summary(held)
                payload['alerts'] = _safe(lambda: PF.alerts(
                    as_of, held, watchlist, spec_json), [])
                consensus_top = (payload['consensus'][payload['consensus']['n_strategies'] >= 2]
                                 ['ticker'].head(20).tolist()
                                 if not payload['consensus'].empty else [])
                payload['earnings'] = _safe(lambda: ER.upcoming(
                    sorted(set(followed) | set(consensus_top)), as_of, days=14))
                payload['earnings_held'] = set(held_names)
                payload['earnings_watch'] = set(watchlist)
                payload['reviews'] = _safe(lambda: _recent_reviews(
                    MC.previous_trading_day(as_of), as_of, held_names, watchlist),
                    pd.DataFrame())
                snap = _safe(lambda: MACRO.snapshot(as_of), pd.DataFrame())
                payload['macro'] = snap
                payload['regime'] = _safe(lambda: MACRO.regime(as_of, snap), [])
                if _first_session_of_week(as_of):
                    payload['track'] = _safe(lambda: TRACK.summary(
                        as_of=as_of, spec_json=spec_json), pd.DataFrame())
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
            period_start = MC.previous_trading_day(as_of)
        else:
            stability = _monthly_stability(as_of)
            if 'error' not in payload:
                payload['track'] = _safe(lambda: TRACK.summary(
                    as_of=as_of, spec_json=spec_json), pd.DataFrame())
            html_out = render_monthly(payload, stability, backtest_metrics,
                                      bench)
            period_start = as_of - timedelta(days=compare_days)
        filename = f'{kind}_cross_strategy_{as_of}.html'

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / filename
    path.write_text(html_out, encoding='utf-8')

    report_id = report_id_for(kind, as_of)
    with db.connect() as conn:
        # Rows from before ids were deterministic point at the same file.
        conn.execute(text('DELETE FROM reports WHERE path = :p AND report_id != :r'),
                     {'p': str(path), 'r': report_id})
        # emailed_at is deliberately not written: rebuilding a report that
        # already went out must not make it look unsent.
        db.upsert(db.reports, [{
            'report_id': report_id, 'kind': kind,
            'period_start': period_start, 'period_end': as_of,
            'path': str(path), 'created_at': config.utc_now(),
        }], conn=conn)

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
