"""
Point-in-time backtest engine.

Two execution models, matching the two horizons:

  long  Periodic rebalance into the top-N ranked names, equal or inverse-vol
        weighted, held until the next rebalance. Exit is a re-rank, not a price.
  mid   Every entry carries a stop, a target and a time stop, and the path is
        walked bar by bar so a position can be closed between rebalances.
        Backtesting a stop-based strategy with month-end snapshots would score
        trades that a real stop would have closed weeks earlier.

The look-ahead discipline is structural rather than advisory: at each
rebalance date the engine calls the same scoring path the live screen uses,
gated to that date. Nothing filed later is visible, so a backtest cannot use
information the strategy could not have had. Orders are filled from the
session *after* the scoring date, never at the close the score was computed
from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from data import sec, yahoo
from data.universe import UniverseSpec
from quant import availability as AV
from quant import engine as EN
from quant import factors as FA
from quant import factorcache
from quant import strategies as ST
from quant import tradeplan as TP
from quant.fundamentals import adjust_shares_for_splits

log = logging.getLogger(__name__)

REBALANCE_FREQ = {'M': 'ME', 'Q': 'QE', 'SA': '2QE', 'A': 'YE'}


class DegradedBacktest(RuntimeError):
    """
    Raised when the data cannot support the strategy over the window asked for.

    A backtest that quietly drops the factors it has no data for still produces
    a number, and that number reads as a result. Refusing is the only honest
    option: an insider strategy scored over years with no insider filings is
    not a weak result for that strategy, it is a result for a different one.
    """


def _require_honest_window(spec: UniverseSpec, strategy_key,
                           start: date, strict: bool) -> None:
    """
    Refuse a start date the stored data cannot support.

    This catches what the per-run factor check cannot: survivorship. A factor
    that is missing announces itself as a gap, but a universe missing the
    companies that failed looks complete — every name in it has full data. The
    only signal is that the ones which went bankrupt or were acquired are not
    there to be picked, and the effect is to inflate every strategy at once.
    """
    if not strict:
        return
    try:
        window = AV.backtest_window(strategy_key, spec.preset or 'SPY')
    except Exception as exc:                       # noqa: BLE001
        log.debug('availability check failed (%s) — not gating', exc)
        return

    if not window.testable:
        raise DegradedBacktest(
            f'{window.strategy} cannot be backtested at all: '
            + '; '.join(window.blockers))
    if start < window.earliest:
        raise DegradedBacktest(
            f'{window.strategy} is not backtestable before {window.earliest} '
            f'(asked for {start}): ' + '; '.join(window.notes)
            + '. Earlier windows are survivorship-inflated or missing factors, '
              'so any number would overstate the result.')


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    holdings: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    ic_series: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        m = self.metrics
        return (f"CAGR {m.get('cagr', 0)*100:.1f}%  "
                f"Sharpe {m.get('sharpe', 0):.2f}  "
                f"MaxDD {m.get('max_drawdown', 0)*100:.1f}%  "
                f"Turnover {m.get('turnover', 0)*100:.0f}%/yr")


# ─────────────────────────────────────────────
# SHARED PRE-LOAD AND POINT-IN-TIME SCORING
# ─────────────────────────────────────────────

@dataclass
class _Preload:
    """Everything a walk-forward needs, read once for the whole window."""
    facts: pd.DataFrame
    splits: pd.DataFrame
    data_version: str
    ohlc: tuple                      # (adj_close, adj_high, adj_low, raw_close)
    insiders: pd.DataFrame
    events: pd.DataFrame
    prev_fund: pd.DataFrame | None = None
    prev_date: date | None = None


def _preload(spec: UniverseSpec, dates: pd.DatetimeIndex, start: date,
             end: date) -> _Preload:
    """
    One read per source covering every rebalance in the window.

    Rows dated after a given rebalance are present in these frames, so every
    consumer slices them to the as-of date — `facts` by `filed` in
    `_score_at`, prices/insiders/events inside quant.factors. Nothing past
    `end` is loaded at all.
    """
    tickers: set[str] = set()
    for ts in dates:
        tickers.update(spec.resolve(ts.date()))
    tickers_sorted = sorted(tickers)
    last = dates[-1].date()

    splits = (yahoo.split_factors(tickers_sorted) if tickers_sorted
              else pd.DataFrame(columns=['ticker', 'date', 'ratio']))
    facts = sec.facts_asof(tickers_sorted, last) if tickers_sorted else pd.DataFrame()
    if not facts.empty:
        facts = adjust_shares_for_splits(facts, splits)

    # The benchmark rides along so BETA / IDIO_VOL can be computed: it is
    # rarely a member of the universe being tested.
    px_tickers = sorted(tickers | {config.BENCHMARK_TICKER})
    ohlc = yahoo.bulk_ohlc(px_tickers,
                           start=start - timedelta(days=FA.PRICE_LOOKBACK_DAYS),
                           end=end)
    insiders = sec.insiders_asof(
        tickers_sorted, last,
        lookback_days=(last - start).days + 180) if tickers_sorted else pd.DataFrame()
    events = sec.events_asof(
        tickers_sorted, last,
        lookback_days=(last - start).days + 90) if tickers_sorted else pd.DataFrame()
    log.info('pre-loaded %d tickers, %d price bars, %d facts',
             len(tickers_sorted), len(ohlc[0]), len(facts))
    return _Preload(facts, splits, factorcache.data_version(), ohlc,
                    insiders, events)


def _score_at(pre: _Preload, spec: UniverseSpec, strategy_key, d0: date,
              top_n: int, plan_params: TP.PlanParams | None = None
              ) -> EN.RunResult:
    """Run the live scoring path gated to `d0`, reusing unchanged fundamentals."""
    facts = pre.facts
    if facts.empty:
        facts_d0, changed = None, None
    else:
        filed = facts['filed']
        facts_d0 = facts[filed <= pd.Timestamp(d0)]
        changed = (None if pre.prev_date is None else set(
            facts.loc[(filed > pd.Timestamp(pre.prev_date))
                      & (filed <= pd.Timestamp(d0)), 'ticker'].unique()))

    result = EN.run(spec, strategy_key, as_of=d0, top_n=top_n, persist=False,
                    plan_params=plan_params,
                    _bt_facts=facts_d0,
                    _bt_data_version=pre.data_version,
                    _bt_prev_fund=pre.prev_fund,
                    _bt_changed_tickers=changed,
                    _bt_splits=pre.splits,
                    _bt_splits_applied=True,
                    _bt_ohlc=pre.ohlc,
                    _bt_insiders=pre.insiders,
                    _bt_events=pre.events)
    pre.prev_fund, pre.prev_date = result._fund_raw, d0
    return result


def _check_degraded(result: EN.RunResult, strategy: ST.Strategy, d0: date,
                    strict: bool) -> None:
    if strict and result.degraded:
        raise DegradedBacktest(
            f'{strategy.key} cannot be honestly backtested at {d0}: '
            + '; '.join(result.warnings)
            + '  (pass strict=False to measure the degraded variant anyway)')


def _next_session(sessions: pd.DatetimeIndex, d: date) -> pd.Timestamp | None:
    """
    First trading session strictly after `d`.

    Scores use everything dated `d`, including filings that land after that
    day's close, so the earliest honest fill is the following session.
    Trading at the very close the signal was computed from would quietly
    assume a filing published at 6pm was known at 4pm.
    """
    pos = sessions.searchsorted(pd.Timestamp(d), side='right')
    return sessions[pos] if pos < len(sessions) else None


# ─────────────────────────────────────────────
# WEIGHTING
# ─────────────────────────────────────────────

def _weights(scores: pd.DataFrame, n: int, scheme: str,
             vols: pd.Series | None = None) -> pd.Series:
    picks = scores.head(n)
    if picks.empty:
        return pd.Series(dtype=float)

    if scheme == 'inverse_vol' and vols is not None:
        v = vols.reindex(picks.index).replace(0, np.nan)
        if v.notna().sum() >= 2:
            w = (1.0 / v).fillna(0)
            if w.sum() > 0:
                return w / w.sum()
    elif scheme == 'score' and 'composite' in picks:
        s = picks['composite'].clip(lower=0)
        if s.sum() > 0:
            return s / s.sum()

    return pd.Series(1.0 / len(picks), index=picks.index)


# ─────────────────────────────────────────────
# LONG-HORIZON BACKTEST
# ─────────────────────────────────────────────

def run_backtest(spec: UniverseSpec, strategy_key: str | ST.Strategy,
                 start: date | str, end: date | str | None = None,
                 n_holdings: int | None = None,
                 rebalance: str | None = None,
                 weighting: str = 'equal',
                 cost_bps: float | None = None,
                 strict: bool = True) -> BacktestResult:
    """
    Walk forward, re-scoring at each rebalance with only the data of that date.

    Scores are computed as of each rebalance date and the portfolio is traded
    at the close of the *next* session, held to the close of the session after
    the following rebalance date.

    For an inverted (avoid-list) strategy the portfolio holds the names it
    flags, so underperforming the benchmark is the strategy working; the IC is
    reported on the un-inverted score so that positive still means "works".

    `strict` (the default) refuses to return a number when the data cannot
    support the strategy at a rebalance date — see `DegradedBacktest`. Pass
    strict=False only to inspect *how* a strategy degrades; the equity curve
    that comes back is then not the strategy you configured.
    """
    strategy = ST.get(strategy_key)
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    rebalance = rebalance or ST.HORIZONS[strategy.horizon]['rebalance']
    n_holdings = n_holdings or ST.HORIZONS[strategy.horizon]['default_holdings']
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    round_trip = (cost_bps + config.SLIPPAGE_BPS) / 10_000.0

    _require_honest_window(spec, strategy_key, start, strict)

    dates = pd.date_range(start, end, freq=REBALANCE_FREQ.get(rebalance, 'ME'))
    if len(dates) < 2:
        raise ValueError('backtest window too short for the rebalance frequency')

    log.info('backtest %s | %s -> %s | %d rebalances | top %d',
             strategy.name, start, end, len(dates), n_holdings)

    pre = _preload(spec, dates, start, end)
    adj_close = pre.ohlc[0]
    if adj_close.empty:
        raise ValueError('no stored prices for this universe and window')
    sessions = adj_close.index
    last_session = sessions[-1]

    def fill_at(d: date) -> pd.Timestamp:
        return _next_session(sessions, d) or last_session

    ic_sign = -1.0 if strategy.invert else 1.0
    prev_w = pd.Series(dtype=float)
    equity, turnover_log, holdings_log, trades, ics = [1.0], [], [], [], []
    equity_dates = [fill_at(dates[0].date())]

    for i in range(len(dates) - 1):
        d0, d1 = dates[i].date(), dates[i + 1].date()
        t0, t1 = fill_at(d0), fill_at(d1)

        if t1 <= t0:
            break                       # no session left to trade in

        result = _score_at(pre, spec, strategy_key, d0, n_holdings)
        _check_degraded(result, strategy, d0, strict)

        w = (_weights(result.scores, n_holdings, weighting,
                      result.scores.get('VOL_1Y'))
             if not result.scores.empty else pd.Series(dtype=float))
        px = adj_close.loc[t0:t1]
        if w.empty:
            # Nothing held for the period: carry the curve flat, and liquidate
            # what was held before, which costs a trade.
            exit_cost = float(prev_w.abs().sum()) * round_trip
            equity.append(equity[-1] * (1.0 - exit_cost))
            equity_dates.append(t1)
            if exit_cost:
                turnover_log.append(float(prev_w.abs().sum()))
            prev_w = pd.Series(dtype=float)
            continue

        # Forward-filled only inside the holding window, so a name that stops
        # trading keeps its last price rather than borrowing a future one.
        held = px.reindex(columns=w.index).ffill()
        period_ret = (held.iloc[-1] / held.iloc[0] - 1.0).fillna(0.0)
        gross = float((w * period_ret).sum())

        traded = float((w.subtract(prev_w, fill_value=0.0)).abs().sum())
        equity.append(equity[-1] * (1.0 + gross - traded * round_trip))
        equity_dates.append(t1)
        turnover_log.append(traded)

        # Rank IC over the same window the portfolio was exposed to.
        if len(result.scores) >= 8:
            cols = px.columns.intersection(result.scores.index)
            all_px = px[cols].ffill()
            fwd = all_px.iloc[-1] / all_px.iloc[0] - 1.0
            pair = pd.concat([result.scores['composite'] * ic_sign, fwd],
                             axis=1).dropna()
            if len(pair) >= 8:
                ics.append({'date': d1, 'ic': pair.iloc[:, 0].corr(
                    pair.iloc[:, 1], method='spearman')})

        for t, weight in w.items():
            holdings_log.append({'date': d0, 'ticker': t, 'weight': weight,
                                 'period_return': period_ret.get(t, 0.0)})
        for t in w.index.difference(prev_w.index):
            trades.append({'date': d0, 'ticker': t, 'action': 'buy',
                           'weight': w[t]})
        for t in prev_w.index.difference(w.index):
            trades.append({'date': d0, 'ticker': t, 'action': 'sell',
                           'weight': prev_w[t]})
        # Weights drift with returns over the period; the next rebalance trades
        # from the drifted book, not from the targets set at this one.
        drifted = w * (1.0 + period_ret)
        prev_w = drifted / drifted.sum() if drifted.sum() > 0 else w

    eq = pd.Series(equity, index=pd.DatetimeIndex(equity_dates), name='strategy')
    eq = eq[~eq.index.duplicated(keep='last')]
    bench = _benchmark_curve(eq.index[0].date(), eq.index[-1].date(), eq.index)

    metrics = compute_metrics(eq, bench)
    metrics['turnover'] = float(np.mean(turnover_log)) * _periods_per_year(rebalance) \
        if turnover_log else 0.0
    metrics['n_rebalances'] = len(dates) - 1
    metrics['avg_holdings'] = n_holdings
    if strategy.invert:
        metrics['inverted'] = True

    ic_series = pd.Series(dtype=float)
    if ics:
        ic_df = pd.DataFrame(ics).dropna()
        if not ic_df.empty:
            ic_series = ic_df.set_index('date')['ic']
            metrics['mean_ic'] = float(ic_series.mean())
            metrics['ic_ir'] = float(ic_series.mean() / ic_series.std()) \
                if ic_series.std() else 0.0
            metrics['ic_hit_rate'] = float((ic_series > 0).mean())

    return BacktestResult(
        equity=eq, benchmark=bench,
        holdings=pd.DataFrame(holdings_log), trades=pd.DataFrame(trades),
        metrics=metrics, ic_series=ic_series)


# ─────────────────────────────────────────────
# MID-HORIZON BACKTEST  (stop / target aware)
# ─────────────────────────────────────────────

# Sessions a pullback limit order stays working before it is cancelled.
ENTRY_WINDOW_SESSIONS = 5


def _simulate_trade(plan: pd.Series, hi: pd.Series, lo: pd.Series,
                    cl: pd.Series, time_stop_days: int) -> dict | None:
    """
    Walk one plan through the sessions after its scan date.

    `hi`/`lo`/`cl` start at the first session after the scan and are on the
    same adjusted basis as the plan's levels. Returns None when the order is
    never filled or the setup is already broken at the fill.

      market    filled at the first session's close.
      pullback  a limit at `entry`, working for ENTRY_WINDOW_SESSIONS; it
                fills only on a session whose low reaches it, at the lower of
                the limit and that session's high (a gap down fills below).
                Assuming every limit fills — and at the limit — books the
                discount on exactly the trades where price never came back.

    After the fill, a session that touches the stop exits at the stop, or at
    the session's high when it gapped through; one that touches the target
    exits at the target. When both are touched on the same day the stop is
    assumed, since daily bars cannot reveal the order. Otherwise the position
    is closed at the close `time_stop_days` sessions after the fill.
    """
    entry, stop, target = (float(plan['entry']), float(plan['stop']),
                           float(plan['target']))
    if cl.empty:
        return None

    limit = plan.get('entry_type') == 'pullback'
    if limit:
        window = lo.iloc[:ENTRY_WINDOW_SESSIONS]
        touched = window.index[window <= entry]
        if len(touched) == 0:
            return None
        fill_day = touched[0]
        day_hi = float(hi.get(fill_day, np.nan))
        fill = min(entry, day_hi) if np.isfinite(day_hi) else entry
    else:
        fill_day = cl.index[0]
        fill = float(cl.iloc[0])

    if not (stop < fill < target):
        return None                                    # setup broken before entry

    fill_pos = cl.index.get_loc(fill_day)
    planned_exit = fill_pos + time_stop_days
    last_pos = min(planned_exit, len(cl) - 1)
    exit_price, exit_day = float(cl.iloc[last_pos]), cl.index[last_pos]
    reason = 'time' if planned_exit <= len(cl) - 1 else 'end'   # window ran out

    # A market fill happens at the close, so the walk starts next session. A
    # limit fills intraday: its own session can still stop out, but a high on
    # that day may have printed before the dip, so it cannot count as target.
    first_walk = fill_pos if limit else fill_pos + 1
    for day in cl.index[first_walk:last_pos + 1]:
        day_lo, day_hi = lo.get(day, np.nan), hi.get(day, np.nan)
        if np.isfinite(day_lo) and day_lo <= stop:
            gapped = np.isfinite(day_hi) and day_hi < stop
            exit_price, reason, exit_day = (day_hi if gapped else stop), 'stop', day
            break
        if day != fill_day and np.isfinite(day_hi) and day_hi >= target:
            exit_price, reason, exit_day = target, 'target', day
            break

    return {'fill_date': fill_day.date(), 'fill': fill, 'exit_date': exit_day.date(),
            'exit': exit_price, 'reason': reason}


def run_trade_backtest(spec: UniverseSpec, strategy_key: str | ST.Strategy,
                       start: date | str, end: date | str | None = None,
                       n_positions: int = 10,
                       params: TP.PlanParams | None = None,
                       cost_bps: float | None = None,
                       strict: bool = True) -> BacktestResult:
    """
    Backtest a mid-term strategy honouring its own exit rules.

    At each month-end scan the engine builds plans from data of that date,
    then walks the following sessions to see whether the entry filled and
    which of stop, target or time stop came first (see `_simulate_trade`).

    The book is one portfolio: a name already held is not bought again, and a
    new position only takes the weight still uninvested, so overlapping scans
    can never add up to more than 100% exposure.
    """
    strategy = ST.get(strategy_key)
    if strategy.horizon != 'mid':
        raise ValueError(f'{strategy.key} is a long-horizon strategy; '
                         f'use run_backtest instead')
    if strategy.invert:
        raise ValueError(f'{strategy.key} is an avoid list, not a trade; '
                         f'use run_backtest to test whether its names lag')

    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    params = params or TP.PlanParams()
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    round_trip = (cost_bps + config.SLIPPAGE_BPS) / 10_000.0

    _require_honest_window(spec, strategy_key, start, strict)

    scan_dates = pd.date_range(start, end, freq='ME')
    if len(scan_dates) == 0:
        raise ValueError('backtest window too short for a monthly scan')

    pre = _preload(spec, scan_dates, start, end)
    adj_c, adj_h, adj_l, _raw = pre.ohlc
    if adj_c.empty:
        raise ValueError('no stored prices for this universe and window')

    trades: list[dict] = []
    open_book: list[dict] = []          # {'ticker', 'exit_date', 'weight'}
    unfilled = 0

    for ts in scan_dates:
        d0 = ts.date()
        result = _score_at(pre, spec, strategy_key, d0, n_positions,
                           plan_params=params)
        _check_degraded(result, strategy, d0, strict)
        if result.plans.empty:
            continue

        # Positions whose exit is on or before the scan are closed by now.
        open_book = [p for p in open_book if p['exit_date'] > d0]
        held = {p['ticker'] for p in open_book}
        capacity = 1.0 - sum(p['weight'] for p in open_book)

        after = adj_c.index[adj_c.index > pd.Timestamp(d0)]
        if len(after) == 0:
            continue
        # Rank order, not the order the plans happened to be built in.
        for t in [x for x in result.scores.index if x in result.plans.index]:
            if capacity <= 1e-9:
                break
            if t in held or t not in adj_c.columns:
                continue
            plan = result.plans.loc[t]
            cl = adj_c.loc[after, t].dropna()
            hi = adj_h[t].reindex(cl.index) if t in adj_h else pd.Series(np.nan, index=cl.index)
            lo = adj_l[t].reindex(cl.index) if t in adj_l else pd.Series(np.nan, index=cl.index)
            sim = _simulate_trade(plan, hi, lo, cl, params.time_stop_days)
            if sim is None:
                unfilled += 1
                continue

            weight = min(float(plan['weight_pct']) / 100.0, capacity)
            capacity -= weight
            open_book.append({'ticker': t, 'exit_date': sim['exit_date'],
                              'weight': weight})

            gross = sim['exit'] / sim['fill'] - 1.0
            risk = (sim['fill'] - float(plan['stop'])) / sim['fill']
            trades.append({
                'entry_date': sim['fill_date'], 'exit_date': sim['exit_date'],
                'ticker': t, 'entry': round(sim['fill'], 2),
                'exit': round(sim['exit'], 2), 'reason': sim['reason'],
                'weight': weight, 'gross_return': gross,
                'net_return': gross - 2 * round_trip,
                'r_multiple': (gross / risk) if risk > 0 else np.nan,
                'days_held': (sim['exit_date'] - sim['fill_date']).days,
            })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return BacktestResult(pd.Series(dtype=float), pd.Series(dtype=float),
                              pd.DataFrame(), pd.DataFrame(),
                              {'note': 'no trades generated',
                               'unfilled_orders': unfilled})

    # Equity: weighted P&L booked on each exit date.
    trades_df = trades_df.sort_values('exit_date')
    daily = (trades_df.assign(contrib=trades_df['net_return'] * trades_df['weight'])
                      .groupby('exit_date')['contrib'].sum())
    eq = (1.0 + daily).cumprod()
    eq.index = pd.to_datetime(eq.index)
    eq = pd.concat([pd.Series([1.0], index=[pd.Timestamp(start)]), eq])

    bench = _benchmark_curve(start, end, eq.index)
    metrics = compute_metrics(eq, bench)
    wins = trades_df['net_return'] > 0
    metrics.update({
        'n_trades': int(len(trades_df)),
        'unfilled_orders': unfilled,
        'win_rate': float(wins.mean()),
        'avg_win': float(trades_df.loc[wins, 'net_return'].mean()) if wins.any() else 0.0,
        'avg_loss': float(trades_df.loc[~wins, 'net_return'].mean()) if (~wins).any() else 0.0,
        'avg_r': float(trades_df['r_multiple'].mean()),
        'avg_days_held': float(trades_df['days_held'].mean()),
        'exit_mix': trades_df['reason'].value_counts().to_dict(),
    })
    if metrics['avg_loss'] != 0:
        metrics['profit_factor'] = abs(
            (metrics['avg_win'] * wins.sum()) /
            (metrics['avg_loss'] * (~wins).sum()))

    return BacktestResult(equity=eq, benchmark=bench, holdings=pd.DataFrame(),
                          trades=trades_df, metrics=metrics)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Standard performance statistics for an equity curve."""
    if equity is None or len(equity) < 2:
        return {}

    eq = equity.dropna()
    rets = eq.pct_change().dropna()
    if rets.empty:
        return {}

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)

    ppy = len(rets) / years
    vol = float(rets.std() * np.sqrt(ppy)) if len(rets) > 1 else 0.0
    rf = config.RISK_FREE_RATE
    sharpe = float((cagr - rf) / vol) if vol > 0 else 0.0

    downside = rets[rets < 0]
    dvol = float(downside.std() * np.sqrt(ppy)) if len(downside) > 1 else 0.0
    sortino = float((cagr - rf) / dvol) if dvol > 0 else 0.0

    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())

    out = {
        'total_return': total, 'cagr': cagr, 'volatility': vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd,
        'calmar': float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        'best_period': float(rets.max()), 'worst_period': float(rets.min()),
        'hit_rate': float((rets > 0).mean()), 'years': years,
    }

    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            b_cagr = float((b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1.0)
            out['benchmark_cagr'] = b_cagr
            out['excess_cagr'] = cagr - b_cagr
            joined = pd.concat([rets, b.pct_change()], axis=1).dropna()
            if len(joined) > 2:
                active = joined.iloc[:, 0] - joined.iloc[:, 1]
                te = float(active.std() * np.sqrt(ppy))
                out['tracking_error'] = te
                out['information_ratio'] = float(active.mean() * ppy / te) if te > 0 else 0.0
    return out


def _benchmark_curve(start: date, end: date, index) -> pd.Series:
    px = yahoo.price_history([config.BENCHMARK_TICKER], start=start, end=end,
                             field='adj_close')
    if px.empty:
        return pd.Series(dtype=float)
    s = px.iloc[:, 0].ffill()
    s = s / s.iloc[0]
    return s.reindex(index).ffill().bfill()


def _periods_per_year(rebalance: str) -> int:
    return {'M': 12, 'Q': 4, 'SA': 2, 'A': 1}.get(rebalance, 12)
