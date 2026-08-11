"""
Scoring engine — universe + strategy + date -> ranked, persisted result.

Signals are horizon-aware, because the two horizons are answering different
questions:

  long  "Is this worth owning?" — graded on business quality and valuation.
        Exit is a broken thesis, so no price levels are emitted.
  mid   "Is this worth trading now?" — graded on trend, setup validity and
        timing, and every candidate carries a full plan with entry, stop,
        target, time stop and size.

Every run is written to the database so a later date can be compared against
it, which is what makes the -1M / -1Y comparison possible at all.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

import config
from core import db
from data import sync as SY
from data.universe import UniverseSpec
from quant import factors as FA
from quant import strategies as ST
from quant import tradeplan as TP
from quant import transforms as T

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    as_of: date
    strategy: ST.Strategy
    scores: pd.DataFrame          # ranked, one row per name
    factor_detail: pd.DataFrame   # per factor/ticker scoring intermediates
    plans: pd.DataFrame           # mid-term only; empty for long-term
    universe: list[str]
    sync: SY.SyncReport | None = None     # what the pull phase did

    @property
    def horizon(self) -> str:
        return self.strategy.horizon

    def top(self, n: int = 20) -> pd.DataFrame:
        return self.scores.head(n)


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

def score_universe(raw: pd.DataFrame, strategy: ST.Strategy
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn raw factor values into sector-relative scores and a composite.

    Returns (per-name scores, tidy per-factor detail).
    """
    sectors = raw['sector'] if 'sector' in raw else pd.Series('Unknown', index=raw.index)
    size = raw['MARKET_CAP'] if 'MARKET_CAP' in raw else None

    scored, detail = {}, []
    for name, weight in strategy.weights.items():
        if name not in raw.columns:
            log.debug('%s: factor %s unavailable', strategy.key, name)
            continue
        prepared = T.prepare_factor(
            raw[name], sectors=sectors, size=size,
            higher_is_better=FA.FACTOR_DIRECTION.get(name, True),
            method=strategy.neutralize,
        )
        scored[name] = prepared['sector_z']
        tidy = prepared.reset_index().rename(columns={'index': 'ticker'})
        tidy['factor'] = name
        detail.append(tidy)

    if not scored:
        return pd.DataFrame(), pd.DataFrame()

    factor_scores = pd.DataFrame(scored)
    weights = {k: v for k, v in strategy.weights.items() if k in factor_scores}
    composite, coverage = T.composite_score(factor_scores, weights)

    out = pd.DataFrame({
        'composite': composite,
        'coverage': coverage,
        'sector': sectors.reindex(composite.index),
    })
    out['rank'] = out['composite'].rank(ascending=False, method='min')
    out = out.sort_values('composite', ascending=False, na_position='last')

    detail_df = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return out, detail_df


# ─────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────

def apply_filters(raw: pd.DataFrame, strategy: ST.Strategy) -> pd.Index:
    """Names that satisfy the strategy's hard gates."""
    keep = pd.Series(True, index=raw.index)
    f = strategy.filters

    if f.get('min_above_ma200') and 'ABOVE_MA200' in raw:
        keep &= raw['ABOVE_MA200'].fillna(0) >= 1
    if f.get('rsi_max') and 'RSI_14' in raw:
        keep &= raw['RSI_14'].fillna(50) <= f['rsi_max']
    if f.get('rsi_min') and 'RSI_14' in raw:
        keep &= raw['RSI_14'].fillna(50) >= f['rsi_min']
    if f.get('min_market_cap') and 'MARKET_CAP' in raw:
        keep &= raw['MARKET_CAP'].fillna(0) >= f['min_market_cap']

    return raw.index[keep]


# ─────────────────────────────────────────────
# SIGNALS — LONG TERM
# ─────────────────────────────────────────────

def _long_term_signal(row: pd.Series) -> tuple[str, float, str]:
    """
    Grade a long-term holding on business quality and valuation.

    Deliberately ignores short-term price action: a good business does not
    stop being one because it is below its 50-day average.
    """
    score, reasons = 0.0, []
    comp = row.get('composite', np.nan)

    if np.isfinite(comp):
        if comp > 1.0:
            score += 3; reasons.append('Top-decile factor score')
        elif comp > 0.5:
            score += 2; reasons.append('Strong factor score')
        elif comp > 0:
            score += 1; reasons.append('Above-average factor score')
        elif comp > -0.5:
            score -= 1; reasons.append('Below-average factor score')
        else:
            score -= 2; reasons.append('Weak factor score')

    roic = row.get('ROIC')
    if roic is not None and np.isfinite(roic):
        if roic > 0.15:
            score += 1.5; reasons.append(f'High ROIC ({roic*100:.0f}%)')
        elif roic < 0:
            score -= 1.5; reasons.append('Negative return on capital')

    pf = row.get('PIOTROSKI_F')
    if pf is not None and np.isfinite(pf):
        if pf >= 7:
            score += 1; reasons.append(f'Piotroski {pf:.0f}/9')
        elif pf <= 3:
            score -= 1.5; reasons.append(f'Weak Piotroski {pf:.0f}/9')

    de = row.get('DEBT_TO_EQUITY')
    if de is not None and np.isfinite(de):
        if de > 2.5:
            score -= 1.5; reasons.append(f'High leverage (D/E {de:.1f})')
        elif de < 0.5:
            score += 0.5; reasons.append('Conservative balance sheet')

    acc = row.get('ACCRUALS')
    if acc is not None and np.isfinite(acc) and acc > 0.1:
        score -= 1; reasons.append('Earnings outpacing cash flow')

    ey = row.get('EARNINGS_YIELD')
    if ey is not None and np.isfinite(ey):
        if ey > 0.08:
            score += 1; reasons.append(f'Cheap ({ey*100:.1f}% earnings yield)')
        elif 0 < ey < 0.02:
            score -= 0.5; reasons.append('Richly valued')

    labels = [(5, '🟢 Strong Buy — accumulate'), (3, '🟢 Buy'),
              (1, '🟡 Hold / add on weakness'), (-1, '⚪ Neutral'),
              (-3, '🟠 Trim'), (float('-inf'), '🔴 Avoid')]
    label = next(l for t, l in labels if score >= t)
    return label, score, '; '.join(reasons)


# ─────────────────────────────────────────────
# SIGNALS — MID TERM
# ─────────────────────────────────────────────

def _mid_term_signal(row: pd.Series) -> tuple[str, float, str]:
    """
    Grade a mid-term trade on setup quality and timing.

    Trend and momentum dominate; fundamentals only act as a sanity filter.
    """
    score, reasons = 0.0, []
    comp = row.get('composite', np.nan)

    if np.isfinite(comp):
        if comp > 1.0:
            score += 3; reasons.append('Top-ranked setup')
        elif comp > 0.5:
            score += 2; reasons.append('Strong setup')
        elif comp > 0:
            score += 1; reasons.append('Positive setup')
        else:
            score -= 2; reasons.append('Weak setup')

    ma200 = row.get('PCT_VS_MA200')
    if ma200 is not None and np.isfinite(ma200):
        if ma200 > 0:
            score += 1.5; reasons.append(f'Uptrend (+{ma200*100:.0f}% vs 200d)')
        else:
            score -= 2; reasons.append('Below 200-day MA — trend against')

    ma50 = row.get('PCT_VS_MA50')
    if ma50 is not None and np.isfinite(ma50) and ma50 > 0:
        score += 0.5; reasons.append('Above 50-day MA')

    rsi = row.get('RSI_14')
    if rsi is not None and np.isfinite(rsi):
        if 45 <= rsi <= 65:
            score += 1; reasons.append(f'Constructive RSI ({rsi:.0f})')
        elif rsi > 78:
            score -= 1.5; reasons.append(f'Extended (RSI {rsi:.0f})')
        elif rsi < 30:
            score += 0.5; reasons.append(f'Oversold (RSI {rsi:.0f})')

    atr_pct = row.get('ATR_PCT')
    if atr_pct is not None and np.isfinite(atr_pct):
        if atr_pct > 0.06:
            score -= 1; reasons.append(f'Very volatile (ATR {atr_pct*100:.1f}%/day)')
        elif atr_pct < 0.015:
            score += 0.5; reasons.append('Orderly volatility')

    ins = row.get('INSIDER_NET_BUY')
    if ins is not None and np.isfinite(ins) and ins > 0.3:
        score += 1; reasons.append('Insiders net buyers')

    labels = [(5, '🟢 Enter — full size'), (3.5, '🟢 Enter — half size'),
              (2, '🟡 Watch for trigger'), (0, '⚪ No setup'),
              (float('-inf'), '🔴 Avoid')]
    label = next(l for t, l in labels if score >= t)
    return label, score, '; '.join(reasons)


def generate_signals(scores: pd.DataFrame, raw: pd.DataFrame,
                     horizon: str) -> pd.DataFrame:
    """Attach horizon-appropriate signal, score and reasoning."""
    if scores.empty:
        return scores

    merged = scores.join(raw.drop(columns=['sector'], errors='ignore'), how='left')
    fn = _mid_term_signal if horizon == 'mid' else _long_term_signal

    results = [fn(row) for _i, row in merged.iterrows()]
    scores = scores.copy()
    scores['signal'] = [r[0] for r in results]
    scores['signal_score'] = [r[1] for r in results]
    scores['reasons'] = [r[2] for r in results]
    return scores


# ─────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────

def run(spec: UniverseSpec, strategy_key: str, as_of: date | str | None = None,
        top_n: int | None = None, persist: bool = True,
        plan_params: TP.PlanParams | None = None,
        sync: bool = False, sync_progress=None) -> RunResult:
    """
    Score a universe from stored data.

    Reading and pulling are separate operations, and this is the read.
    Fetching lives in one place — the Data tab's refresh button, or
    `cli.py ingest` — so that screening, backtesting and comparing are pure
    database work: fast, reproducible, and unable to stall on a rate-limited
    provider mid-click.

    `sync=True` opts a caller into refreshing stale sources first; the CLI
    uses it for unattended runs. Historical as-of dates ignore it either way,
    since fetching *today's* data cannot inform a past date and would only
    invalidate caches.
    """
    as_of = pd.to_datetime(as_of or date.today()).date()
    strategy = ST.get(strategy_key)
    top_n = top_n or ST.HORIZONS[strategy.horizon]['default_holdings']

    # ── phase 1: pull ────────────────────────────────────────────
    # Membership first: the universe cannot be resolved until it is current.
    sync_report = None
    is_historical = as_of < date.today()
    if sync and not is_historical:
        sync_report = SY.sync(
            [], index=(spec.preset or None), sources=['members'],
            progress=sync_progress)

    universe = spec.resolve(as_of)
    if not universe:
        log.warning('universe resolved to zero names for %s', spec.describe())
        return RunResult(str(uuid.uuid4()), as_of, strategy,
                         pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [],
                         sync_report)

    if sync and not is_historical:
        rest = SY.sync(universe, index=(spec.preset or None),
                       sources=[s for s in SY.ESSENTIAL + SY.OPTIONAL
                                if s != 'members'],
                       progress=sync_progress)
        sync_report.results.extend(rest.results)
        sync_report.seconds += rest.seconds
    elif is_historical:
        log.info('as_of %s is historical — reading stored data only', as_of)

    # ── phase 2: read ────────────────────────────────────────────
    log.info('run: %s | %s | %d names | as_of %s',
             strategy.name, spec.describe(), len(universe), as_of)

    raw = FA.build_all(universe, as_of)
    if raw.empty:
        return RunResult(str(uuid.uuid4()), as_of, strategy,
                         pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), universe,
                         sync_report)

    eligible = apply_filters(raw, strategy)
    if len(eligible) == 0:
        log.warning('all %d names filtered out by %s gates', len(raw), strategy.key)
        return RunResult(str(uuid.uuid4()), as_of, strategy,
                         pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), universe,
                         sync_report)
    log.info('filters: %d/%d names eligible', len(eligible), len(raw))

    raw_eligible = raw.loc[eligible]
    scores, detail = score_universe(raw_eligible, strategy)
    if scores.empty:
        return RunResult(str(uuid.uuid4()), as_of, strategy,
                         pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), universe,
                         sync_report)

    scores = generate_signals(scores, raw_eligible, strategy.horizon)
    scores = scores.dropna(subset=['composite'])
    scores['rank'] = range(1, len(scores) + 1)

    # Carry through the display columns the UI needs.
    for col in ('name', 'PRICE', 'MARKET_CAP', 'RSI_14', 'PCT_VS_MA200',
                'ATR_PCT', 'MA50', 'MA200', 'ATR_14', 'EARNINGS_YIELD',
                'ROIC', 'PIOTROSKI_F', 'DIVIDEND_YIELD', 'MOM_12_1', 'VOL_1Y'):
        if col in raw_eligible.columns:
            scores[col] = raw_eligible[col].reindex(scores.index)

    plans = pd.DataFrame()
    if strategy.needs_trade_plan:
        shortlist = scores.head(top_n)
        plans = TP.build_plans(
            raw_eligible.loc[raw_eligible.index.intersection(shortlist.index)],
            as_of, setup=strategy.setup, params=plan_params)
        if not plans.empty:
            for col in ('entry', 'stop', 'target', 'weight_pct',
                        'stop_pct', 'target_pct', 'entry_type', 'invalidation'):
                scores[col] = plans[col].reindex(scores.index)

    run_id = str(uuid.uuid4())
    if persist:
        _persist(run_id, as_of, spec, strategy, universe, scores, detail)

    return RunResult(run_id, as_of, strategy, scores, detail, plans, universe,
                     sync_report)


def _persist(run_id: str, as_of: date, spec: UniverseSpec,
             strategy: ST.Strategy, universe: list[str],
             scores: pd.DataFrame, detail: pd.DataFrame) -> None:
    """Store the run so future dates can be compared against it."""
    db.upsert(db.runs, [{
        'run_id': run_id, 'as_of': as_of, 'strategy': strategy.key,
        'universe_spec_json': spec.to_json(),
        'universe_tickers': ','.join(universe),
        'params_json': pd.io.json.ujson_dumps(strategy.weights)
        if hasattr(pd.io.json, 'ujson_dumps') else str(strategy.weights),
        'universe_n': len(universe), 'created_at': datetime.utcnow(),
    }])

    db.upsert(db.scores, [{
        'run_id': run_id, 'ticker': t,
        'composite': _num(r.get('composite')), 'rank': int(r.get('rank', 0)),
        'signal': r.get('signal'), 'signal_score': _num(r.get('signal_score')),
        'reasons': r.get('reasons'), 'coverage': _num(r.get('coverage')),
        'sector': r.get('sector'),
    } for t, r in scores.iterrows()])

    if not detail.empty:
        db.upsert(db.factor_scores, [{
            'run_id': run_id, 'ticker': r['ticker'], 'factor': r['factor'],
            'raw': _num(r.get('raw')), 'winsorized': _num(r.get('winsorized')),
            'z': _num(r.get('z')), 'sector_z': _num(r.get('sector_z')),
            'pct_rank': _num(r.get('pct_rank')),
        } for _i, r in detail.iterrows()])

    log.info('persisted run %s (%d names)', run_id[:8], len(scores))


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
