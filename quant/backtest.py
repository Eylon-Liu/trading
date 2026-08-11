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
from data import yahoo
from data.universe import UniverseSpec
from quant import engine as EN
from quant import strategies as ST
from quant import tradeplan as TP

log = logging.getLogger(__name__)

REBALANCE_FREQ = {'M': 'ME', 'Q': 'QE', 'SA': '2QE', 'A': 'YE'}


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

def run_backtest(spec: UniverseSpec, strategy_key: str,
                 start: date | str, end: date | str | None = None,
                 n_holdings: int | None = None,
                 rebalance: str | None = None,
                 weighting: str = 'equal',
                 cost_bps: float | None = None) -> BacktestResult:
    """
    Walk forward, re-scoring at each rebalance with only the data of that date.
    """
    strategy = ST.get(strategy_key)
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    rebalance = rebalance or ST.HORIZONS[strategy.horizon]['rebalance']
    n_holdings = n_holdings or ST.HORIZONS[strategy.horizon]['default_holdings']
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    round_trip = (cost_bps + config.SLIPPAGE_BPS) / 10_000.0

    dates = pd.date_range(start, end, freq=REBALANCE_FREQ.get(rebalance, 'ME'))
    if len(dates) < 2:
        raise ValueError('backtest window too short for the rebalance frequency')

    log.info('backtest %s | %s -> %s | %d rebalances | top %d',
             strategy.name, start, end, len(dates), n_holdings)

    prev_w = pd.Series(dtype=float)
    equity, turnover_log, holdings_log, trades, ics = [1.0], [], [], [], []
    equity_dates = [dates[0].date()]

    for i in range(len(dates) - 1):
        d0, d1 = dates[i].date(), dates[i + 1].date()

        result = EN.run(spec, strategy_key, as_of=d0,
                        top_n=n_holdings, persist=False)
        if result.scores.empty:
            equity.append(equity[-1]); equity_dates.append(d1)
            continue

        vols = result.scores.get('VOL_1Y')
        w = _weights(result.scores, n_holdings, weighting, vols)
        if w.empty:
            equity.append(equity[-1]); equity_dates.append(d1)
            continue

        # Forward return over the holding period — the only forward-looking
        # step, and it is the realised outcome, not an input to the decision.
        px = yahoo.price_history(list(w.index), start=d0 - timedelta(days=7),
                                 end=d1, field='adj_close')
        if px.empty:
            equity.append(equity[-1]); equity_dates.append(d1)
            continue
        px = px.ffill()
        period_ret = (px.iloc[-1] / px.iloc[0] - 1.0).reindex(w.index).fillna(0.0)

        gross = float((w * period_ret).sum())

        traded = float((w.subtract(prev_w, fill_value=0.0)).abs().sum())
        cost = traded * round_trip
        equity.append(equity[-1] * (1.0 + gross - cost))
        equity_dates.append(d1)
        turnover_log.append(traded)

        # Rank IC: correlation between the score and the return that followed.
        if len(result.scores) >= 8:
            all_px = yahoo.price_history(list(result.scores.index),
                                         start=d0 - timedelta(days=7), end=d1,
                                         field='adj_close')
            if not all_px.empty:
                all_px = all_px.ffill()
                fwd = (all_px.iloc[-1] / all_px.iloc[0] - 1.0)
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

    for ts in scan_dates:
        d0 = ts.date()
        result = EN.run(spec, strategy_key, as_of=d0, top_n=n_positions,
                        persist=False, plan_params=params)
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
