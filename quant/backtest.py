"""
Point-in-time backtest engine.

Two execution models, matching the two horizons:

  long  Periodic rebalance into the top-N ranked names, equal or inverse-vol
        weighted, held until the next rebalance. Exit is a re-rank, not a price.
  mid   Every entry carries a stop, a target and a time stop. The portfolio is
        simulated session by session: orders fill only if the market actually
        reaches them, exits are checked on every bar, capital is finite, and
        the equity curve is marked to market daily.

The look-ahead discipline is structural rather than advisory: at each
rebalance date the engine calls the same scoring path the live screen uses,
gated to that date. Nothing filed later is visible, and trades are filled no
earlier than the session *after* the signal was computed.

Execution conventions (both models)
  * Signals are computed after the close of a real trading session — never a
    calendar date that happens to be a weekend.
  * Long horizon: fills at the close `execution_lag` sessions later (1).
  * Mid horizon: market orders fill at the next session's open; pullback
    limits fill only when a later bar trades down to them, within
    `fill_window` sessions, or the order expires unfilled.
  * A stop that the market gaps through fills at the open, not at the stop.
  * Costs (commission + slippage) are charged on every trade, both sides.
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
# SHARED: SESSIONS, PRE-LOADING, SCORING
# ─────────────────────────────────────────────

def rebalance_positions(sessions: pd.DatetimeIndex, rebalance: str) -> list[int]:
    """
    Index positions of the rebalance sessions: the first session, then the
    last session of each period (month, quarter, half-year, year) after it.

    Calendar period-ends were used before. When one fell on a weekend the
    period ended at Friday's close and the next began at Monday's, so the
    return between them belonged to neither period and silently vanished.
    """
    if len(sessions) == 0:
        return []
    s = pd.Series(range(len(sessions)), index=sessions)
    if rebalance == 'Q':
        key = sessions.year * 10 + sessions.quarter
    elif rebalance == 'SA':
        key = sessions.year * 10 + (sessions.month > 6).astype(int)
    elif rebalance == 'A':
        key = sessions.year
    else:
        key = sessions.year * 100 + sessions.month
    last = s.groupby(np.asarray(key)).max().tolist()
    return sorted({0, *[p for p in last if p > 0]})


@dataclass
class _Preload:
    tickers: list[str]
    facts: pd.DataFrame
    version: str
    splits: pd.DataFrame
    ohlc: tuple
    insiders: pd.DataFrame
    events: pd.DataFrame


def _preload(spec: UniverseSpec, start: date, end: date,
             rebalance: str = 'M') -> _Preload:
    """Everything the scoring path needs, read once for the whole window."""
    tickers: set[str] = set(spec.resolve(start))
    for ts in pd.date_range(start, end, freq=REBALANCE_FREQ.get(rebalance, 'ME')):
        tickers.update(spec.resolve(ts.date()))
    names = sorted(tickers)
    facts = sec.facts_asof(names, end) if names else pd.DataFrame()
    splits = yahoo.split_factors(names) if names else pd.DataFrame()
    if not facts.empty:
        facts = adjust_shares_for_splits(facts, splits)
    lookback = int(config.TRADING_DAYS['3Y'] * 1.6)
    # The benchmark rides along: BETA, IDIO_VOL and EARNINGS_REACTION are all
    # measured against it. Loading only the universe left those factors empty
    # in every backtest — silently, for the mid-term path, which never ran
    # the degraded-run check.
    ohlc = yahoo.bulk_ohlc(sorted(set(names) | {config.BENCHMARK_TICKER}),
                           start=start - timedelta(days=lookback), end=end)
    insiders = sec.insiders_asof(names, end,
                                 lookback_days=(end - start).days + 180) \
        if names else pd.DataFrame()
    events = sec.events_asof(names, end, lookback_days=(end - start).days + 90) \
        if names else pd.DataFrame()
    return _Preload(names, facts, factorcache.data_version(), splits, ohlc,
                    insiders, events)


class _Scorer:
    """Runs the live scoring path at successive dates, reusing prior work."""

    def __init__(self, spec, strategy_key, pre: _Preload):
        self.spec, self.key, self.pre = spec, strategy_key, pre
        self.prev_fund = None
        self.prev_d0: date | None = None

    def run(self, d0: date, top_n: int, plan_params=None):
        facts = self.pre.facts
        facts_d0 = facts[facts['filed'] <= pd.Timestamp(d0)] if not facts.empty \
            else None
        changed = None
        if self.prev_d0 is not None and not facts.empty:
            new = facts[(facts['filed'] > pd.Timestamp(self.prev_d0))
                        & (facts['filed'] <= pd.Timestamp(d0))]
            changed = set(new['ticker'].unique())
        result = EN.run(self.spec, self.key, as_of=d0, top_n=top_n,
                        persist=False, plan_params=plan_params,
                        _bt_facts=facts_d0,
                        _bt_data_version=self.pre.version,
                        _bt_prev_fund=self.prev_fund,
                        _bt_changed_tickers=changed,
                        _bt_splits=self.pre.splits,
                        _bt_splits_applied=True,
                        _bt_ohlc=self.pre.ohlc,
                        _bt_insiders=self.pre.insiders,
                        _bt_events=self.pre.events)
        self.prev_fund = result._fund_raw
        self.prev_d0 = d0
        return result


def _risk_free(start: date, end: date) -> float:
    """Average 3-month T-bill yield over the window; the config rate if unknown."""
    try:
        from data import macro
        return macro.risk_free_rate(start, end)
    except Exception as exc:                       # noqa: BLE001
        log.debug('risk-free lookup failed (%s) — using config rate', exc)
        return config.RISK_FREE_RATE


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
                 strict: bool = True,
                 execution_lag: int = 1) -> BacktestResult:
    """
    Walk forward, re-scoring at each rebalance with only the data of that date.

    The signal is computed after the close of a rebalance session; trades fill
    at the close `execution_lag` sessions later (default 1). Filling at the
    very close the signal was computed from assumes you could see that close
    and trade on it in the same instant — and lets a filing published after
    the bell on the signal day count as tradeable that day.

    The equity curve is marked to market daily, so drawdown, volatility and
    Sharpe reflect what happened between rebalances, not only at them.

    `strict` (the default) refuses to return a number when the data cannot
    support the strategy at a rebalance date — see `DegradedBacktest`. Pass
    strict=False only to inspect *how* a strategy degrades; the equity curve
    that comes back is then not the strategy you configured.
    """
    strategy = ST.get(strategy_key)
    if strategy.invert:
        raise ValueError(f'{strategy.name} is an avoid list; a long-only '
                         f'backtest of it is meaningless.')
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    meta_notes: list[str] = []
    if strategy.is_meta:
        # Only the families whose data reaches back to `start` take part,
        # for the whole run; forward-only votes (the AI review) never do.
        strategy, meta_notes = AV.meta_members_at(strategy, start,
                                                  spec.preset or 'SPY')
        if len(strategy.components) < 2:
            raise DegradedBacktest(
                f'{strategy.name}: fewer than two strategy families are '
                f'backtestable from {start}: ' + '; '.join(meta_notes))
        strategy_key = strategy
    rebalance = rebalance or ST.HORIZONS[strategy.horizon]['rebalance']
    n_holdings = n_holdings or ST.HORIZONS[strategy.horizon]['default_holdings']
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    # Charged on every unit of weight traded, buys and sells alike.
    one_way = (cost_bps + config.SLIPPAGE_BPS) / 10_000.0
    lag = max(0, int(execution_lag))

    _require_honest_window(spec, strategy_key, start, strict)

    pre = _preload(spec, start, end, rebalance)
    adj_close_all = pre.ohlc[0].ffill() if not pre.ohlc[0].empty else pd.DataFrame()
    if adj_close_all.empty:
        raise ValueError('no stored prices for this universe and window')

    sessions = adj_close_all.index[(adj_close_all.index >= pd.Timestamp(start))
                                   & (adj_close_all.index <= pd.Timestamp(end))]
    signal_pos = rebalance_positions(sessions, rebalance)
    if len(signal_pos) < 2 or signal_pos[0] + lag >= len(sessions) - 1:
        raise ValueError('backtest window too short for the rebalance frequency')
    dates = [sessions[p] for p in signal_pos]
    log.info('backtest %s | %s -> %s | %d rebalances | top %d | lag %d',
             strategy.name, start, end, len(dates), n_holdings, lag)

    scorer = _Scorer(spec, strategy_key, pre)
    prev_w = pd.Series(dtype=float)        # drifted weights held going in
    curve: list[pd.Series] = []
    equity_now = 1.0
    turnover_log, holdings_log, trades, ics = [], [], [], []

    for k, p0 in enumerate(signal_pos):
        d0 = sessions[p0].date()
        # Hold from this rebalance's fill to the next one's fill.
        p_in = p0 + lag
        p_out = (min(signal_pos[k + 1] + lag, len(sessions) - 1)
                 if k + 1 < len(signal_pos) else len(sessions) - 1)
        if p_in >= p_out:
            break

        result = scorer.run(d0, n_holdings)
        if strict and result.degraded:
            raise DegradedBacktest(
                f'{strategy.key} cannot be honestly backtested at {d0}: '
                + '; '.join(result.warnings)
                + '  (pass strict=False to measure the degraded variant anyway)')

        seg_index = sessions[p_in:p_out + 1]
        w = pd.Series(dtype=float)
        if not result.scores.empty:
            w = _weights(result.scores, n_holdings, weighting,
                         result.scores.get('VOL_1Y'))
        seg = pd.DataFrame(index=seg_index)
        if not w.empty:
            # A name with no price at the fill is not bought; its share is cash.
            seg = adj_close_all.reindex(columns=w.index).loc[seg_index]
            priced = (seg.iloc[0].notna() & (seg.iloc[0] > 0)).to_numpy()
            w, seg = w[priced], seg.loc[:, priced]

        if curve:
            # The fill bar closed the previous segment; it opens this one
            # with the post-cost value instead of appearing twice.
            curve[-1] = curve[-1].iloc[:-1]

        if w.empty:
            traded = float(prev_w.abs().sum())
            equity_now *= 1.0 - traded * one_way
            curve.append(pd.Series(equity_now, index=seg_index))
            prev_w = pd.Series(dtype=float)
            continue

        rel = seg.ffill() / seg.iloc[0]
        cash_w = max(0.0, 1.0 - float(w.sum()))
        port_rel = rel.mul(w, axis=1).sum(axis=1) + cash_w

        traded = float(w.subtract(prev_w, fill_value=0.0).abs().sum())
        start_value = equity_now * (1.0 - traded * one_way)
        seg_curve = start_value * port_rel
        curve.append(seg_curve)
        equity_now = float(seg_curve.iloc[-1])
        if k > 0:
            turnover_log.append(traded)

        prior = prev_w
        period_ret = rel.iloc[-1] - 1.0
        # Weights drift with prices until the next rebalance; turnover is
        # measured against what is actually held then, not the old targets.
        prev_w = (w * rel.iloc[-1]) / float(port_rel.iloc[-1])

        # Rank IC over the same holding window, across every scored name.
        if len(result.scores) >= 8:
            cols = adj_close_all.columns.intersection(result.scores.index)
            px = adj_close_all.loc[seg_index, cols]
            fwd = px.iloc[-1] / px.iloc[0] - 1.0
            pair = pd.concat([result.scores['composite'], fwd], axis=1).dropna()
            pair.columns = ['score', 'fwd']
            if len(pair) >= 8:
                ics.append({'date': seg_index[-1].date(),
                            'ic': pair['score'].corr(pair['fwd'], method='spearman')})

        for t, weight in w.items():
            holdings_log.append({'date': d0, 'ticker': t, 'weight': weight,
                                 'period_return': float(period_ret.get(t, 0.0))})
        fill_day = seg_index[0].date()
        for t in set(w.index) - set(prior.index):
            trades.append({'date': fill_day, 'ticker': t, 'action': 'buy',
                           'weight': float(w[t])})
        for t in set(prior.index) - set(w.index):
            trades.append({'date': fill_day, 'ticker': t, 'action': 'sell',
                           'weight': float(prior[t])})

    eq = pd.concat(curve) if curve else pd.Series(dtype=float)
    eq = eq[~eq.index.duplicated(keep='last')]
    eq.name = 'strategy'
    bench = (_benchmark_curve(eq.index[0].date(), eq.index[-1].date(), eq.index)
             if len(eq) else pd.Series(dtype=float))

    rf = _risk_free(start, end)
    metrics = compute_metrics(eq, bench, rf=rf)
    metrics['turnover'] = float(np.mean(turnover_log)) * _periods_per_year(rebalance) \
        if turnover_log else 0.0
    metrics['n_rebalances'] = len(dates)
    metrics['avg_holdings'] = n_holdings
    metrics['execution_lag'] = lag
    if strategy.is_meta:
        metrics['families_used'] = ', '.join(strategy.components)
        if meta_notes:
            metrics['families_note'] = '; '.join(meta_notes)

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

    Scored at `start`; bought at the close of the next session. Costs are
    charged once, on entry. A pick with no price on the entry day is dropped
    and its cash shared among the rest — it used to be bought with NaN shares
    and its slice of the starting cash silently disappeared.
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
    px = yahoo.price_history(list(picks.index), start=start + timedelta(days=1),
                             end=end, field='adj_close')
    if px.empty:
        raise ValueError(f'no price history for {list(picks.index)}')
    px = px.ffill()
    entry_row = px.iloc[0]
    px = px.loc[:, entry_row.notna() & (entry_row > 0)]
    if px.empty:
        raise ValueError('no usable price history for the picks')

    # Equal cash per name, so the share count differs and the weights drift.
    cash_each = initial_cash / len(px.columns)
    shares = (cash_each * (1.0 - entry_cost)) / px.iloc[0]
    values = px.mul(shares, axis=1)
    eq = values.sum(axis=1)
    eq.name = 'strategy'

    bench = _benchmark_curve(eq.index[0].date(), end, eq.index)
    if not bench.empty:
        bench = bench / bench.iloc[0] * initial_cash

    metrics = compute_metrics(eq, bench, rf=_risk_free(start, end))
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

def simulate_trades(bars: dict[str, pd.DataFrame], signals: dict,
                    params: TP.PlanParams, one_way: float,
                    fill_window: int = 5, max_gross_pct: float = 100.0
                    ) -> tuple[pd.Series, pd.DataFrame, dict]:
    """
    Session-by-session portfolio simulation of trade plans.

    `bars` holds adjusted open/high/low/close wide frames over the sessions to
    simulate; `signals` maps a session Timestamp to the plans produced after
    that session's close (a frame indexed by ticker, best first, with entry,
    entry_type, stop, target and weight_pct).

    Returns (daily equity starting at 1.0, closed trades, stats).

    What the previous version got wrong, and this one does not:
      * every plan was assumed filled at its entry price — including pullback
        limits set below the market that price never came back to;
      * the bar walk began *on the signal day*, so that day's range, which
        happened before the signal existed, could trigger a stop or target;
      * positions overlapped across monthly scans with no capital limit, so
        exposure could reach several hundred percent;
      * P&L was booked only at exits, so open losses never showed in the
        drawdown;
      * a gap through the stop was filled at the stop instead of the open.
    """
    o, h, l, c = bars['open'], bars['high'], bars['low'], bars['close']
    sessions = c.index
    closes_ff = c.ffill()
    cash = 1.0
    positions: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    closed: list[dict] = []
    equity = []
    stats = {'orders': 0, 'filled': 0, 'expired': 0, 'skipped_capacity': 0,
             'skipped_gap_below_stop': 0}
    exposure = []

    def value_of(i: int) -> float:
        row = closes_ff.iloc[i]
        return cash + sum(p['shares'] * float(row.get(t, p['entry']))
                          for t, p in positions.items())

    def close_position(t: str, price: float, reason: str, i: int) -> None:
        nonlocal cash
        p = positions.pop(t)
        cash += p['shares'] * price * (1.0 - one_way)
        gross = price / p['entry'] - 1.0
        risk = (p['entry'] - p['stop']) / p['entry']
        closed.append({
            'signal_date': p['signal_date'], 'entry_date': p['entry_date'],
            'exit_date': sessions[i].date(), 'ticker': t,
            'entry': round(p['entry'], 4), 'exit': round(price, 4),
            'stop': p['stop'], 'target': p['target'], 'reason': reason,
            'weight': p['weight'], 'gross_return': gross,
            'net_return': (1 + gross) * (1 - one_way) / (1 + one_way) - 1,
            'r_multiple': gross / risk if risk > 0 else np.nan,
            'days_held': i - p['bar'],
        })

    for i, day in enumerate(sessions):
        eq_prev = equity[-1] if equity else 1.0

        # 1. fills — orders placed after an earlier close
        for t, order in list(pending.items()):
            if i > order['expires']:
                stats['expired'] += 1
                del pending[t]
                continue
            op, lo_ = o.at[day, t] if t in o else np.nan, l.at[day, t] if t in l else np.nan
            if not (np.isfinite(op) and np.isfinite(lo_)):
                continue
            if order['entry_type'] == 'market':
                fill = op
            elif op <= order['entry']:
                fill = op                       # opened below the limit
            elif lo_ <= order['entry']:
                fill = order['entry']
            else:
                continue
            del pending[t]
            if fill <= order['stop']:
                # Already through the stop before we were in: the setup is
                # invalid, and nobody enters a trade below its own stop.
                stats['skipped_gap_below_stop'] += 1
                continue
            invested = sum(p['shares'] * float(closes_ff.iloc[max(i - 1, 0)].get(t2, p['entry']))
                           for t2, p in positions.items())
            room = max(0.0, max_gross_pct / 100.0 * eq_prev - invested)
            want = order['weight'] * eq_prev
            alloc = min(want, room, cash / (1.0 + one_way))
            if alloc < params.min_weight_pct / 100.0 * eq_prev:
                stats['skipped_capacity'] += 1
                continue
            shares = alloc / fill
            cash -= alloc * (1.0 + one_way)
            positions[t] = {'shares': shares, 'entry': fill, 'stop': order['stop'],
                            'target': order['target'], 'bar': i,
                            'limit': order['entry_type'] != 'market',
                            'entry_date': day.date(), 'signal_date': order['signal_date'],
                            'weight': alloc / eq_prev}
            stats['filled'] += 1

        # 2. exits — stop first: within one bar the order is unknowable, so
        #    assume the worse outcome.
        for t, p in list(positions.items()):
            op = o.at[day, t] if t in o else np.nan
            hi = h.at[day, t] if t in h else np.nan
            lo_ = l.at[day, t] if t in l else np.nan
            cl = c.at[day, t] if t in c else np.nan
            if not np.isfinite(cl):
                continue
            fresh = p['bar'] == i
            if not fresh and np.isfinite(op) and op <= p['stop']:
                close_position(t, op, 'stop', i)
            elif np.isfinite(lo_) and lo_ <= p['stop']:
                close_position(t, p['stop'], 'stop', i)
            elif not fresh and np.isfinite(op) and op >= p['target']:
                close_position(t, op, 'target', i)
            elif (np.isfinite(hi) and hi >= p['target']
                  and not (fresh and p['limit'])):
                close_position(t, p['target'], 'target', i)
            elif i - p['bar'] >= params.time_stop_days:
                close_position(t, cl, 'time', i)

        # 3. mark to market
        v = value_of(i)
        equity.append(v)
        exposure.append((v - cash) / v if v > 0 else 0.0)

        # 4. new signals after this close → orders from the next session
        plans = signals.get(day)
        if plans is not None and not plans.empty:
            for t, plan in plans.iterrows():
                if t in positions or t in pending or t not in c.columns:
                    continue
                pending[t] = {'entry': float(plan['entry']),
                              'entry_type': str(plan.get('entry_type', 'market')),
                              'stop': float(plan['stop']),
                              'target': float(plan['target']),
                              'weight': float(plan['weight_pct']) / 100.0,
                              'expires': i + fill_window,
                              'signal_date': day.date()}
                stats['orders'] += 1

    # Anything still open is closed at the last close of the window.
    last = len(sessions) - 1
    for t in list(positions):
        px = float(closes_ff.iloc[last].get(t, positions[t]['entry']))
        close_position(t, px, 'end', last)
    if equity:
        equity[-1] = cash

    stats['avg_exposure'] = float(np.mean(exposure)) if exposure else 0.0
    stats['fill_rate'] = stats['filled'] / stats['orders'] if stats['orders'] else 0.0
    return (pd.Series(equity, index=sessions, name='strategy'),
            pd.DataFrame(closed), stats)


def run_trade_backtest(spec: UniverseSpec, strategy_key: str,
                       start: date | str, end: date | str | None = None,
                       n_positions: int = 10,
                       params: TP.PlanParams | None = None,
                       cost_bps: float | None = None,
                       fill_window: int = 5,
                       max_gross_pct: float = 100.0,
                       strict: bool = True) -> BacktestResult:
    """
    Backtest a mid-term strategy honouring its own entry and exit rules.

    Scans after the last session of each month; the resulting plans become
    orders for the following sessions and the portfolio is simulated bar by
    bar (see `simulate_trades`). Gross exposure never exceeds `max_gross_pct`
    of equity, and a name already held or on order is not bought again.

    `strict` refuses a scan date where the strategy's factors are missing,
    exactly as the long-horizon backtest does; without it a strategy that
    had lost half its weight was reported under its own name.
    """
    strategy = ST.get(strategy_key)
    if strategy.invert:
        raise ValueError(f'{strategy.name} is an avoid list (rank 1 is the worst '
                         f'name); a long-only backtest of it is meaningless. Its '
                         f'live track record grades it the right way round.')
    if strategy.horizon != 'mid':
        raise ValueError(f'{strategy_key} is a long-horizon strategy; '
                         f'use run_backtest instead')

    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end or date.today()).date()
    params = params or TP.PlanParams()
    cost_bps = config.TRANSACTION_COST_BPS if cost_bps is None else cost_bps
    one_way = (cost_bps + config.SLIPPAGE_BPS) / 10_000.0

    pre = _preload(spec, start, end, 'M')
    bars = yahoo.adjusted_bars(pre.tickers, start=start, end=end)
    if bars['close'].empty:
        return BacktestResult(pd.Series(dtype=float), pd.Series(dtype=float),
                              pd.DataFrame(), pd.DataFrame(),
                              {'note': 'no stored prices in the window'}, [])
    sessions = bars['close'].index
    scan_pos = rebalance_positions(sessions, 'M')

    _require_honest_window(spec, strategy_key, start, strict)
    scorer = _Scorer(spec, strategy_key, pre)
    signals: dict = {}
    for p in scan_pos:
        d0 = sessions[p]
        result = scorer.run(d0.date(), n_positions, plan_params=params)
        if strict and result.degraded:
            raise DegradedBacktest(
                f'{strategy.key} cannot be honestly backtested at {d0.date()}: '
                + '; '.join(result.warnings)
                + '  (pass strict=False to measure the degraded variant anyway)')
        if result.plans.empty:
            continue
        order = [t for t in result.scores.index if t in result.plans.index]
        signals[d0] = result.plans.loc[order].head(n_positions)

    eq, trades_df, stats = simulate_trades(bars, signals, params, one_way,
                                           fill_window, max_gross_pct)
    scan_dates = [sessions[p].date() for p in scan_pos]
    if trades_df.empty:
        return BacktestResult(eq, pd.Series(dtype=float), pd.DataFrame(),
                              pd.DataFrame(),
                              {'note': 'no trades filled', **stats}, scan_dates)

    bench = _benchmark_curve(eq.index[0].date(), eq.index[-1].date(), eq.index)
    metrics = compute_metrics(eq, bench, rf=_risk_free(start, end))
    wins = trades_df['net_return'] > 0
    metrics.update({
        'n_trades': int(len(trades_df)),
        'win_rate': float(wins.mean()),
        'avg_win': float(trades_df.loc[wins, 'net_return'].mean()) if wins.any() else 0.0,
        'avg_loss': float(trades_df.loc[~wins, 'net_return'].mean()) if (~wins).any() else 0.0,
        'avg_r': float(trades_df['r_multiple'].mean()),
        'avg_days_held': float(trades_df['days_held'].mean()),
        'exit_mix': trades_df['reason'].value_counts().to_dict(),
        **stats,
    })
    gains = trades_df.loc[wins, 'net_return'].sum()
    losses = trades_df.loc[~wins, 'net_return'].sum()
    metrics['profit_factor'] = float(gains / abs(losses)) if losses < 0 else np.inf

    return BacktestResult(equity=eq, benchmark=bench, holdings=pd.DataFrame(),
                          trades=trades_df, metrics=metrics,
                          rebalance_dates=scan_dates)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(equity: pd.Series, benchmark: pd.Series | None = None,
                    rf: float | None = None) -> dict:
    """
    Standard performance statistics for an equity curve.

    Sharpe is the mean periodic excess return over its standard deviation,
    annualised — the textbook definition. (CAGR over volatility, used before,
    understates by roughly half the variance and penalises volatile
    strategies twice.) Sortino divides by the downside *deviation* below the
    risk-free rate, counting every period, rather than by the standard
    deviation of the losing periods alone. `rf` is annual; it defaults to
    config.RISK_FREE_RATE.
    """
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
    rf = config.RISK_FREE_RATE if rf is None else float(rf)
    rf_p = (1.0 + rf) ** (1.0 / ppy) - 1.0
    excess = rets - rf_p
    sd = float(rets.std()) if len(rets) > 1 else 0.0
    vol = sd * np.sqrt(ppy)
    sharpe = float(excess.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0

    downside = np.sqrt(float((np.minimum(excess, 0.0) ** 2).mean()))
    sortino = float(excess.mean() / downside * np.sqrt(ppy)) if downside > 0 else 0.0

    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())

    out = {
        'total_return': total, 'cagr': cagr, 'volatility': vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd,
        'calmar': float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        'best_period': float(rets.max()), 'worst_period': float(rets.min()),
        'hit_rate': float((rets > 0).mean()), 'years': years,
        'risk_free': rf,
    }

    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.reindex(eq.index).ffill().dropna()
        if len(b) > 1:
            b_years = max((b.index[-1] - b.index[0]).days / 365.25, 1e-9)
            b_cagr = float((b.iloc[-1] / b.iloc[0]) ** (1 / b_years) - 1.0)
            out['benchmark_cagr'] = b_cagr
            out['excess_cagr'] = cagr - b_cagr
            joined = pd.concat([rets, b.pct_change()], axis=1).dropna()
            if len(joined) > 2:
                active = joined.iloc[:, 0] - joined.iloc[:, 1]
                te = float(active.std() * np.sqrt(ppy))
                out['tracking_error'] = te
                out['information_ratio'] = float(active.mean() * ppy / te) if te > 0 else 0.0
                bvar = float(joined.iloc[:, 1].var())
                if bvar > 0:
                    out['beta'] = float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / bvar)
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
