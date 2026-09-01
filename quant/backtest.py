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
information the strategy could not have had.
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
    rebalance_dates: list
    ic_series: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        m = self.metrics
        return (f"CAGR {m.get('cagr', 0)*100:.1f}%  "
                f"Sharpe {m.get('sharpe', 0):.2f}  "
                f"MaxDD {m.get('max_drawdown', 0)*100:.1f}%  "
                f"Turnover {m.get('turnover', 0)*100:.0f}%/yr")


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

    # ── pre-fetch: load everything once for the entire backtest window ──
    all_tickers: set[str] = set()
    for ts in dates:
        all_tickers.update(spec.resolve(ts.date()))
    all_tickers_sorted = sorted(all_tickers)
    facts_full = sec.facts_asof(all_tickers_sorted, dates[-1].date())
    dv = factorcache.data_version()
    splits = yahoo.split_factors(all_tickers_sorted) if all_tickers_sorted else pd.DataFrame()
    facts_full = adjust_shares_for_splits(facts_full, splits)

    # Prices: one SQL read for the entire window (lookback + forward).
    price_lookback = int(config.TRADING_DAYS['3Y'] * 1.6)
    price_start = start - timedelta(days=price_lookback)
    bt_ohlc = yahoo.bulk_ohlc(all_tickers_sorted, start=price_start, end=end)
    adj_close_all = bt_ohlc[0].ffill() if not bt_ohlc[0].empty else pd.DataFrame()
    log.info('pre-loaded prices: %d tickers, %d bars',
             len(adj_close_all.columns) if not adj_close_all.empty else 0,
             len(adj_close_all))

    # Insiders and events: one read each covering the full lookback.
    ins_start = start - timedelta(days=180)
    bt_insiders = sec.insiders_asof(all_tickers_sorted, dates[-1].date(),
                                     lookback_days=(dates[-1].date() - ins_start).days)
    evt_start = start - timedelta(days=90)
    bt_events = sec.events_asof(all_tickers_sorted, dates[-1].date(),
                                 lookback_days=(dates[-1].date() - evt_start).days)

    prev_fund: pd.DataFrame | None = None
    prev_d0: date | None = None

    prev_w = pd.Series(dtype=float)
    equity, turnover_log, holdings_log, trades, ics = [1.0], [], [], [], []
    equity_dates = [dates[0].date()]
    degraded_dates: list = []

    for i in range(len(dates) - 1):
        d0, d1 = dates[i].date(), dates[i + 1].date()

        facts_d0 = facts_full[facts_full['filed'] <= pd.Timestamp(d0)]
        if prev_d0 is not None:
            new_filings = facts_full[
                (facts_full['filed'] > pd.Timestamp(prev_d0))
                & (facts_full['filed'] <= pd.Timestamp(d0))]
            changed = set(new_filings['ticker'].unique())
        else:
            changed = None

        result = EN.run(spec, strategy_key, as_of=d0,
                        top_n=n_holdings, persist=False,
                        _bt_facts=facts_d0,
                        _bt_data_version=dv,
                        _bt_prev_fund=prev_fund,
                        _bt_changed_tickers=changed,
                        _bt_splits=splits,
                        _bt_splits_applied=True,
                        _bt_ohlc=bt_ohlc,
                        _bt_insiders=bt_insiders,
                        _bt_events=bt_events)
        prev_fund = result._fund_raw
        prev_d0 = d0
        if strict and result.degraded:
            raise DegradedBacktest(
                f'{strategy.key} cannot be honestly backtested at {d0}: '
                + '; '.join(result.warnings)
                + '  (pass strict=False to measure the degraded variant anyway)')
        degraded_dates.extend([d0] * bool(result.degraded))
        if result.scores.empty:
            equity.append(equity[-1]); equity_dates.append(d1)
            continue

        vols = result.scores.get('VOL_1Y')
        w = _weights(result.scores, n_holdings, weighting, vols)
        if w.empty:
            equity.append(equity[-1]); equity_dates.append(d1)
            continue

        # Forward return from pre-loaded prices (no SQL per rebalance).
        if not adj_close_all.empty:
            avail = adj_close_all.columns.intersection(w.index)
            px = adj_close_all.loc[pd.Timestamp(d0):pd.Timestamp(d1), avail]
        else:
            px = pd.DataFrame()
        if px.empty or len(px) < 2:
            equity.append(equity[-1]); equity_dates.append(d1)
            continue
        period_ret = (px.iloc[-1] / px.iloc[0] - 1.0).reindex(w.index).fillna(0.0)

        gross = float((w * period_ret).sum())

        traded = float((w.subtract(prev_w, fill_value=0.0)).abs().sum())
        cost = traded * round_trip
        equity.append(equity[-1] * (1.0 + gross - cost))
        equity_dates.append(d1)
        turnover_log.append(traded)

        # Rank IC from pre-loaded prices (no SQL per rebalance).
        if len(result.scores) >= 8 and not adj_close_all.empty:
            ic_tickers = adj_close_all.columns.intersection(result.scores.index)
            all_px = adj_close_all.loc[pd.Timestamp(d0):pd.Timestamp(d1), ic_tickers]
            if not all_px.empty and len(all_px) >= 2:
                fwd = all_px.iloc[-1] / all_px.iloc[0] - 1.0
                pair = pd.concat([result.scores['composite'], fwd], axis=1).dropna()
                pair.columns = ['score', 'fwd']
                if len(pair) >= 8:
                    ics.append({'date': d1,
                                'ic': pair['score'].corr(pair['fwd'], method='spearman')})

        for t, weight in w.items():
            holdings_log.append({'date': d0, 'ticker': t, 'weight': weight,
                                 'period_return': period_ret.get(t, 0.0)})
        for t in set(w.index) - set(prev_w.index):
            trades.append({'date': d0, 'ticker': t, 'action': 'buy',
                           'weight': w[t]})
        for t in set(prev_w.index) - set(w.index):
            trades.append({'date': d0, 'ticker': t, 'action': 'sell',
                           'weight': prev_w[t]})
        prev_w = w

    eq = pd.Series(equity, index=pd.to_datetime(equity_dates), name='strategy')
    bench = _benchmark_curve(dates[0].date(), dates[-1].date(), eq.index)

    metrics = compute_metrics(eq, bench)
    metrics['turnover'] = float(np.mean(turnover_log)) * _periods_per_year(rebalance) \
        if turnover_log else 0.0
    metrics['n_rebalances'] = len(dates) - 1
    metrics['avg_holdings'] = n_holdings

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
        metrics=metrics, rebalance_dates=[d.date() for d in dates],
        ic_series=ic_series)


# ─────────────────────────────────────────────
# BUY AND HOLD  (concentrated, no rebalance)
# ─────────────────────────────────────────────

def run_buy_and_hold(spec: UniverseSpec, strategy_key, start: date | str,
                     end: date | str | None = None, n_holdings: int = 3,
                     cost_bps: float | None = None,
                     initial_cash: float = 10_000.0,
                     strict: bool = True) -> BacktestResult:
    """
    Score once, buy the top `n_holdings` with equal cash, never touch them again.

    This asks a different question from `run_backtest`. A quarterly rebalance
    measures the *ranking rule* — it keeps re-picking, so a bad name is only
    ever held for one period. Buying three names and holding for a decade
    measures the *picks*: whether the businesses this screen put at the top
    were actually worth owning. That is the question a long-horizon screen
    claims to answer, and a rebalance quietly hides the answer by continually
    correcting itself.

    Weights are equal in *dollars at entry* and then left to drift, which is
    what a real buy-and-hold portfolio does — a winner becoming an outsized
    share of the portfolio is a result, not an error to be corrected.

    Costs are charged once, on entry.
    """
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    entry_cost = (cost_bps + config.SLIPPAGE_BPS) / 10_000.0

    _require_honest_window(spec, strategy_key, start, strict)

    result = EN.run(spec, strategy_key, as_of=start, top_n=n_holdings,
                    persist=False)
    if result.scores.empty:
        raise ValueError(f'{strategy_key} produced no picks at {start}')
    if result.degraded:
        raise DegradedBacktest(
            f'{ST.get(strategy_key).key} cannot be honestly backtested at '
            f'{start}: ' + '; '.join(result.warnings))

    picks = result.scores.head(n_holdings)
    px = yahoo.price_history(list(picks.index), start=start, end=end,
                             field='adj_close')
    if px.empty:
        raise ValueError(f'no price history for {list(picks.index)}')
    px = px.ffill().dropna(axis=1, how='all')
    if px.empty:
        raise ValueError('no usable price history for the picks')

    # Equal cash per name, so the share count differs and the weights drift.
    cash_each = initial_cash / len(px.columns)
    shares = (cash_each * (1.0 - entry_cost)) / px.iloc[0]
    values = px.mul(shares, axis=1)
    eq = values.sum(axis=1)
    eq.name = 'strategy'

    bench = _benchmark_curve(start, end, eq.index)
    if not bench.empty:
        bench = bench / bench.iloc[0] * initial_cash

    metrics = compute_metrics(eq, bench)
    metrics.update({
        'initial_cash': initial_cash,
        'final_cash': float(eq.iloc[-1]),
        'final_cash_benchmark': float(bench.iloc[-1]) if not bench.empty else np.nan,
        'n_holdings': int(len(px.columns)),
        'turnover': 0.0,
        'n_rebalances': 0,
    })

    entry, exit_ = px.iloc[0], px.iloc[-1]
    holdings = pd.DataFrame({
        'ticker': px.columns,
        'entry_price': entry.values,
        'exit_price': exit_.values,
        'shares': shares.values,
        'cash_in': cash_each * (1.0 - entry_cost),
        'cash_out': values.iloc[-1].values,
        'total_return': (exit_ / entry - 1.0).values,
    }).sort_values('total_return', ascending=False).reset_index(drop=True)

    return BacktestResult(
        equity=eq, benchmark=bench, holdings=holdings,
        trades=pd.DataFrame(), metrics=metrics,
        rebalance_dates=[start])


def compare_buy_and_hold(spec: UniverseSpec, strategy_keys: list,
                         start: date | str, end: date | str | None = None,
                         n_holdings: int = 3,
                         initial_cash: float = 10_000.0) -> pd.DataFrame:
    """
    Terminal cash per strategy from one concentrated buy-and-hold, side by side.

    The headline column is `final_cash`: what the same starting cash became.
    `picks` is carried through because with three names the answer is driven by
    which three, and an average hides that.
    """
    rows = []
    for key in strategy_keys:
        name = key if isinstance(key, str) else getattr(key, 'key', str(key))
        try:
            r = run_buy_and_hold(spec, key, start, end, n_holdings=n_holdings,
                                 initial_cash=initial_cash)
        except DegradedBacktest as exc:
            log.warning('%s not backtestable at %s: %s', name, start, exc)
            rows.append({'strategy': name, 'error': 'INSUFFICIENT DATA',
                         'detail': str(exc)[:160]})
            continue
        except Exception as exc:                   # noqa: BLE001
            log.warning('buy-and-hold failed for %s: %s', name, exc)
            rows.append({'strategy': name, 'error': str(exc)[:80]})
            continue
        m = r.metrics
        rows.append({
            'strategy': name,
            'final_cash': m.get('final_cash'),
            'benchmark_cash': m.get('final_cash_benchmark'),
            'total_return': m.get('total_return'),
            'cagr': m.get('cagr'),
            'sharpe': m.get('sharpe'),
            'max_drawdown': m.get('max_drawdown'),
            'picks': ', '.join(r.holdings['ticker']),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MID-HORIZON BACKTEST  (stop / target aware)
# ─────────────────────────────────────────────

def run_trade_backtest(spec: UniverseSpec, strategy_key: str,
                       start: date | str, end: date | str | None = None,
                       n_positions: int = 10,
                       params: TP.PlanParams | None = None,
                       cost_bps: float | None = None) -> BacktestResult:
    """
    Backtest a mid-term strategy honouring its own exit rules.

    At each monthly scan the engine builds plans, then walks daily bars to see
    whether the stop, the target, or the time stop was reached first. When a
    single day trades through both stop and target, daily bars cannot reveal
    the order, so the stop is assumed — the pessimistic read, chosen so the
    results are not flattered by an ambiguity.
    """
    strategy = ST.get(strategy_key)
    if strategy.horizon != 'mid':
        raise ValueError(f'{strategy_key} is a long-horizon strategy; '
                         f'use run_backtest instead')

    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    params = params or TP.PlanParams()
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    round_trip = (cost_bps + config.SLIPPAGE_BPS) / 10_000.0

    scan_dates = pd.date_range(start, end, freq='ME')
    trades: list[dict] = []

    # ── pre-fetch: load everything once for the entire window ──
    all_tickers_mid: set[str] = set()
    for ts in scan_dates:
        all_tickers_mid.update(spec.resolve(ts.date()))
    all_tickers_mid_sorted = sorted(all_tickers_mid)
    facts_full_mid = sec.facts_asof(all_tickers_mid_sorted, scan_dates[-1].date()) \
        if all_tickers_mid_sorted else pd.DataFrame()
    dv_mid = factorcache.data_version()
    splits_mid = yahoo.split_factors(all_tickers_mid_sorted) \
        if all_tickers_mid_sorted else pd.DataFrame()
    if not facts_full_mid.empty:
        facts_full_mid = adjust_shares_for_splits(facts_full_mid, splits_mid)

    price_lookback_mid = int(config.TRADING_DAYS['3Y'] * 1.6)
    bt_ohlc_mid = yahoo.bulk_ohlc(all_tickers_mid_sorted,
                                   start=start - timedelta(days=price_lookback_mid),
                                   end=end + timedelta(days=int(params.time_stop_days * 1.5)))
    ins_start_mid = start - timedelta(days=180)
    bt_insiders_mid = sec.insiders_asof(
        all_tickers_mid_sorted, scan_dates[-1].date(),
        lookback_days=(scan_dates[-1].date() - ins_start_mid).days) \
        if all_tickers_mid_sorted else pd.DataFrame()
    bt_events_mid = sec.events_asof(
        all_tickers_mid_sorted, scan_dates[-1].date(),
        lookback_days=(scan_dates[-1].date() - (start - timedelta(days=90))).days) \
        if all_tickers_mid_sorted else pd.DataFrame()

    prev_fund_mid: pd.DataFrame | None = None
    prev_d0_mid: date | None = None

    for ts in scan_dates:
        d0 = ts.date()
        facts_d0_mid = facts_full_mid[facts_full_mid['filed'] <= pd.Timestamp(d0)] \
            if not facts_full_mid.empty else pd.DataFrame()
        if prev_d0_mid is not None and not facts_full_mid.empty:
            new_mid = facts_full_mid[
                (facts_full_mid['filed'] > pd.Timestamp(prev_d0_mid))
                & (facts_full_mid['filed'] <= pd.Timestamp(d0))]
            changed_mid = set(new_mid['ticker'].unique())
        else:
            changed_mid = None

        result = EN.run(spec, strategy_key, as_of=d0, top_n=n_positions,
                        persist=False, plan_params=params,
                        _bt_facts=facts_d0_mid if not facts_d0_mid.empty else None,
                        _bt_data_version=dv_mid,
                        _bt_prev_fund=prev_fund_mid,
                        _bt_changed_tickers=changed_mid,
                        _bt_splits=splits_mid,
                        _bt_splits_applied=True,
                        _bt_ohlc=bt_ohlc_mid,
                        _bt_insiders=bt_insiders_mid,
                        _bt_events=bt_events_mid)
        prev_fund_mid = result._fund_raw
        prev_d0_mid = d0
        if result.plans.empty:
            continue

        horizon_end = min(end, d0 + timedelta(days=int(params.time_stop_days * 1.5)))
        tickers = list(result.plans.index)
        # Adjusted OHLC: raw highs/lows sit on a different basis to the
        # adjusted price the plan levels were derived from.
        highs, lows, closes = yahoo.adjusted_ohlc(tickers, start=d0, end=horizon_end)
        if closes.empty:
            continue

        for t in tickers:
            if t not in closes.columns:
                continue
            plan = result.plans.loc[t]
            entry, stop, target = (float(plan['entry']), float(plan['stop']),
                                   float(plan['target']))

            hi = highs[t].dropna() if t in highs.columns else pd.Series(dtype=float)
            lo = lows[t].dropna() if t in lows.columns else pd.Series(dtype=float)
            cl = closes[t].dropna()
            if cl.empty:
                continue

            exit_price, exit_reason, exit_date = None, 'time', cl.index[-1].date()
            for day in cl.index:
                day_low = lo.get(day, np.nan)
                day_high = hi.get(day, np.nan)
                if np.isfinite(day_low) and day_low <= stop:
                    exit_price, exit_reason, exit_date = stop, 'stop', day.date()
                    break
                if np.isfinite(day_high) and day_high >= target:
                    exit_price, exit_reason, exit_date = target, 'target', day.date()
                    break
            if exit_price is None:
                exit_price = float(cl.iloc[-1])

            gross = exit_price / entry - 1.0
            net = gross - 2 * round_trip
            risk = (entry - stop) / entry
            trades.append({
                'entry_date': d0, 'exit_date': exit_date, 'ticker': t,
                'entry': entry, 'exit': round(exit_price, 2),
                'reason': exit_reason, 'weight': float(plan['weight_pct']) / 100.0,
                'gross_return': gross, 'net_return': net,
                'r_multiple': (gross / risk) if risk > 0 else np.nan,
                'days_held': (exit_date - d0).days,
            })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return BacktestResult(pd.Series(dtype=float), pd.Series(dtype=float),
                              pd.DataFrame(), pd.DataFrame(),
                              {'note': 'no trades generated'}, [])

    # Equity curve: sum weighted P&L by exit date.
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
            (metrics['avg_loss'] * (~wins).sum())) if (~wins).any() else np.inf

    return BacktestResult(equity=eq, benchmark=bench, holdings=pd.DataFrame(),
                          trades=trades_df, metrics=metrics,
                          rebalance_dates=[d.date() for d in scan_dates])


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
    sharpe = float(cagr / vol) if vol > 0 else 0.0

    downside = rets[rets < 0]
    dvol = float(downside.std() * np.sqrt(ppy)) if len(downside) > 1 else 0.0
    sortino = float(cagr / dvol) if dvol > 0 else 0.0

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
