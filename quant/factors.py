"""
Factor library — price, fundamental, and alternative-data signals.

Several constructions here differ deliberately from the original app:

  * Value uses *yields*, not dollar EPS. Trailing EPS of $10 says nothing
    about cheapness without a price attached; earnings yield (E/P) does. The
    original ranked companies on raw EPS, which is not comparable across names.
  * Momentum is 12-1, skipping the most recent month. A plain 1-month return
    is short-term *reversal* in the literature — close to the opposite signal —
    so it is kept separately and signed correctly as REVERSAL_1M.
  * Lookbacks are in trading days from a proper calendar. The original indexed
    35 rows back and called it "1 month"; 35 sessions is about seven weeks.
  * RSI uses Wilder's smoothing. The original's docstring claimed Wilder while
    the code used a simple rolling mean.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from core import db
from data import sec, yahoo
from quant import factorcache
from quant import fundamentals as F

log = logging.getLogger(__name__)

TD = config.TRADING_DAYS


# ─────────────────────────────────────────────
# PRICE-BASED INDICATORS
# ─────────────────────────────────────────────

def wilder_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Wilder's RSI — exponential smoothing with alpha = 1/period.

    A simple rolling mean (what the original used) reacts differently and does
    not match any published RSI value.
    """
    s = prices.dropna()
    if len(s) < period + 1:
        return np.nan
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = gain.iloc[-1] / last_loss
    val = 100.0 - (100.0 / (1.0 + rs))
    return float(val) if np.isfinite(val) else np.nan


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> float:
    """Average True Range (Wilder) — the volatility unit for stops and sizing."""
    df = pd.DataFrame({'h': high, 'l': low, 'c': close}).dropna()
    if len(df) < period + 1:
        return np.nan
    prev_close = df['c'].shift(1)
    tr = pd.concat([
        df['h'] - df['l'],
        (df['h'] - prev_close).abs(),
        (df['l'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    val = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(val) if np.isfinite(val) else np.nan


def _ret(px: pd.DataFrame, lookback: int, skip: int = 0) -> pd.Series:
    """Total return over `lookback` sessions, ending `skip` sessions ago."""
    if px.empty or len(px) < lookback + skip + 1:
        return pd.Series(np.nan, index=px.columns, dtype=float)
    end = px.iloc[-1 - skip] if skip else px.iloc[-1]
    start = px.iloc[-1 - skip - lookback]
    with np.errstate(divide='ignore', invalid='ignore'):
        return (end / start - 1.0).replace([np.inf, -np.inf], np.nan)


def price_factors(tickers: list[str], as_of: date | str,
                  benchmark: str | None = None) -> pd.DataFrame:
    """
    Every price-derived factor, computed only from bars at or before `as_of`.
    """
    as_of = pd.to_datetime(as_of).date()
    benchmark = benchmark or config.BENCHMARK_TICKER
    start = as_of - timedelta(days=int(TD['3Y'] * 1.6))

    universe = sorted(set(tickers) | {benchmark})
    px = yahoo.price_history(universe, start=start, end=as_of, field='adj_close')
    if px.empty:
        return pd.DataFrame()
    px = px.ffill()

    # ATR must sit on the same basis as PRICE (adjusted), or a stop placed N
    # ATRs below an adjusted price is measured in the wrong units.
    high, low, close = yahoo.adjusted_ohlc(universe, start=start, end=as_of)

    # NB: a DataFrame with an index but no columns reports .empty as True, so
    # the guard has to look at the index directly.
    out = pd.DataFrame(index=[t for t in tickers if t in px.columns])
    if len(out.index) == 0:
        return out

    # ── momentum ──────────────────────────────────────────────────
    # 12-1 skips the most recent month: the last month is dominated by
    # short-term reversal, which contaminates a momentum signal.
    out['MOM_12_1'] = _ret(px, TD['1Y'] - TD['1M'], skip=TD['1M']).reindex(out.index)
    out['MOM_6_1'] = _ret(px, TD['6M'] - TD['1M'], skip=TD['1M']).reindex(out.index)
    out['RETURN_3M'] = _ret(px, TD['3M']).reindex(out.index)
    out['RETURN_6M'] = _ret(px, TD['6M']).reindex(out.index)
    out['RETURN_1Y'] = _ret(px, TD['1Y']).reindex(out.index)

    # Kept, but named and signed for what it actually is.
    out['REVERSAL_1M'] = -_ret(px, TD['1M']).reindex(out.index)

    # ── risk ──────────────────────────────────────────────────────
    rets = px.pct_change(fill_method=None)
    ann = np.sqrt(252)
    out['VOL_1Y'] = (rets.tail(TD['1Y']).std() * ann).reindex(out.index)
    out['VOL_3M'] = (rets.tail(TD['3M']).std() * ann).reindex(out.index)

    if benchmark in rets.columns:
        bench = rets[benchmark].tail(TD['1Y'])
        betas, idio = {}, {}
        for t in out.index:
            if t not in rets.columns:
                continue
            pair = pd.concat([rets[t].tail(TD['1Y']), bench], axis=1).dropna()
            if len(pair) < 60:
                continue
            cov = pair.cov().iloc[0, 1]
            var = pair.iloc[:, 1].var()
            if var and np.isfinite(var) and var != 0:
                b = cov / var
                betas[t] = b
                idio[t] = float((pair.iloc[:, 0] - b * pair.iloc[:, 1]).std() * ann)
        out['BETA'] = pd.Series(betas)
        out['IDIO_VOL'] = pd.Series(idio)

    # Vol-scaled momentum: return per unit of risk taken to earn it.
    out['MOM_VOL_ADJ'] = out['MOM_12_1'] / out['VOL_1Y'].replace(0, np.nan)

    # ── drawdown & trend ──────────────────────────────────────────
    win = px.tail(TD['1Y'])
    running_max = win.cummax()
    out['MAX_DD_1Y'] = ((win / running_max - 1.0).min()).reindex(out.index)

    last = px.iloc[-1]
    out['PCT_FROM_52W_HIGH'] = (last / win.max() - 1.0).reindex(out.index)
    out['PCT_FROM_52W_LOW'] = (last / win.min() - 1.0).reindex(out.index)

    for w, name in ((50, 'MA50'), (200, 'MA200')):
        ma = px.rolling(w).mean().iloc[-1]
        out[f'PCT_VS_{name}'] = (last / ma - 1.0).reindex(out.index)
        out[name] = ma.reindex(out.index)

    out['ABOVE_MA200'] = (out['PCT_VS_MA200'] > 0).astype(float)

    # ── oscillators & volatility units ────────────────────────────
    out['RSI_14'] = pd.Series(
        {t: wilder_rsi(px[t]) for t in out.index if t in px.columns})

    if not high.empty and not low.empty and not close.empty:
        out['ATR_14'] = pd.Series({
            t: atr(high[t], low[t], close[t])
            for t in out.index
            if t in high.columns and t in low.columns and t in close.columns
        })
        out['ATR_PCT'] = out['ATR_14'] / last.reindex(out.index)

    out['PRICE'] = last.reindex(out.index)
    return out


# ─────────────────────────────────────────────
# FUNDAMENTAL FACTORS
# ─────────────────────────────────────────────

def fundamental_factors(tickers: list[str], as_of: date | str) -> pd.DataFrame:
    """
    Valuation, quality, and growth factors from point-in-time SEC data.

    Market cap comes from the profile snapshot, which cannot be backfilled, so
    for historical dates the yield factors fall back to a shares-outstanding
    times price estimate built entirely from filed data and stored bars.
    """
    as_of = pd.to_datetime(as_of).date()
    # Fetched once and shared with the F-score below: at S&P 500 scale this
    # read alone was costing several seconds per duplicate call.
    facts_all = sec.facts_asof(tickers, as_of)
    fund = F.build_fundamentals(tickers, as_of, facts_all=facts_all)
    if fund.empty:
        return pd.DataFrame()

    prof = yahoo.profile_asof(tickers, as_of)
    px = yahoo.latest_prices(tickers, as_of)

    mcap = pd.Series(np.nan, index=fund.index, dtype=float)
    if not prof.empty and 'market_cap' in prof:
        mcap = prof['market_cap'].reindex(fund.index)

    # Reconstruct market cap where the snapshot is missing (any historical
    # date before the app started running).
    est = fund['shares_diluted'].reindex(fund.index) * px.reindex(fund.index)
    mcap = mcap.fillna(est)

    ev = mcap + fund['debt'].fillna(0) - fund['cash'].fillna(0)

    out = pd.DataFrame(index=fund.index)
    out['MARKET_CAP'] = mcap

    # ── value: yields, not raw dollar figures ─────────────────────
    out['EARNINGS_YIELD'] = _div(fund['net_income_ttm'], mcap)
    out['FCF_YIELD'] = _div(fund['fcf_ttm'], mcap)
    out['SALES_YIELD'] = _div(fund['revenue_ttm'], mcap)
    out['BOOK_TO_MARKET'] = _div(fund['equity'], mcap)
    out['EBIT_TO_EV'] = _div(fund['operating_income_ttm'], ev)

    # ── quality ───────────────────────────────────────────────────
    out['ROE'] = fund['roe']
    out['ROIC'] = fund['roic']
    out['GROSS_PROFITABILITY'] = fund['gross_profitability']
    out['NET_MARGIN'] = fund['net_margin']
    out['OPERATING_MARGIN'] = fund['operating_margin']
    out['ACCRUALS'] = fund['accruals']          # lower is better
    out['DEBT_TO_EQUITY'] = fund['debt_to_equity']
    out['CURRENT_RATIO'] = fund['current_ratio']
    out['ASSET_TURNOVER'] = fund['asset_turnover']

    # ── growth ────────────────────────────────────────────────────
    out['REVENUE_GROWTH_1Y'] = fund['revenue_growth_1y']
    out['REVENUE_CAGR_3Y'] = fund['revenue_cagr_3y']
    out['EARNINGS_GROWTH_1Y'] = fund['earnings_growth_1y']
    out['EQUITY_CAGR_3Y'] = fund['equity_cagr_3y']

    # ── income ────────────────────────────────────────────────────
    # This used to rebuild the whole fundamentals frame a second time to read
    # a column that was never produced, so the factor was always NaN and the
    # rebuild cost as much as the rest of the screen combined.
    out['BUYBACK_YIELD'] = _div(fund['buybacks_ttm'], mcap)
    if not prof.empty:
        out['DIVIDEND_YIELD'] = prof.get('dividend_yield').reindex(fund.index) \
            if 'dividend_yield' in prof else np.nan
        out['PAYOUT_RATIO'] = prof.get('payout_ratio').reindex(fund.index) \
            if 'payout_ratio' in prof else np.nan
        # Forward-looking, analyst-derived: live screen only, never historical.
        out['FORWARD_PE'] = prof.get('forward_pe').reindex(fund.index) \
            if 'forward_pe' in prof else np.nan
        out['TRAILING_PE'] = prof.get('trailing_pe').reindex(fund.index) \
            if 'trailing_pe' in prof else np.nan

    out['PIOTROSKI_F'] = F.piotroski_f(
        tickers, as_of, facts_all=facts_all).reindex(fund.index)
    out['DATA_AS_OF'] = fund['last_filed']
    return out


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


# ─────────────────────────────────────────────
# ALTERNATIVE-DATA FACTORS
# ─────────────────────────────────────────────

def alt_factors(tickers: list[str], as_of: date | str) -> pd.DataFrame:
    """
    Signals from filings and attention data.

    All of these carry a filing or observation date, so unlike news sentiment
    they are backtestable from day one.
    """
    as_of = pd.to_datetime(as_of).date()
    out = pd.DataFrame(index=pd.Index(tickers, name='ticker'))

    # ── insider activity (Form 4, open-market only) ───────────────
    ins = sec.insiders_asof(tickers, as_of, lookback_days=180)
    if not ins.empty:
        signal = ins[ins['txn_code'].isin(config.INSIDER_SIGNAL_CODES)].copy()
        if not signal.empty:
            signal['buy_val'] = np.where(signal['shares'] > 0,
                                         signal['value'].fillna(0), 0.0)
            signal['sell_val'] = np.where(signal['shares'] < 0,
                                          signal['value'].fillna(0), 0.0)
            agg = signal.groupby('ticker').agg(
                buy=('buy_val', 'sum'), sell=('sell_val', 'sum'),
                n_buyers=('insider', lambda s: s[signal.loc[s.index, 'shares'] > 0].nunique()),
            )
            total = agg['buy'] + agg['sell']
            out['INSIDER_NET_BUY'] = ((agg['buy'] - agg['sell'])
                                      / total.replace(0, np.nan)).reindex(out.index)
            # Several distinct insiders buying at once is the stronger form.
            out['INSIDER_CLUSTER'] = agg['n_buyers'].reindex(out.index).fillna(0)

    # ── 8-K event intensity ───────────────────────────────────────
    ev = sec.events_asof(tickers, as_of, lookback_days=90)
    if not ev.empty:
        counts = ev.groupby('ticker').size()
        out['EVENT_INTENSITY_90D'] = counts.reindex(out.index).fillna(0)
        for cat in ('management', 'restructuring', 'strategy', 'results'):
            sub = ev[ev['category'] == cat].groupby('ticker').size()
            out[f'EVENT_{cat.upper()}'] = sub.reindex(out.index).fillna(0)

    # ── retail attention (Wikipedia pageviews) ────────────────────
    att = _attention_zscore(tickers, as_of)
    if att is not None:
        out['ATTENTION_Z'] = att.reindex(out.index)

    return out


def attention_zscore(views: pd.DataFrame, recent_days: int = 14,
                     min_history: int = 60) -> pd.Series | None:
    """
    Pure: recent pageviews versus their own trailing baseline, as a z-score.

    Takes a tidy (ticker, date, wiki_views) frame and returns one score per
    ticker. No database access, so it can be tested against a fixture and
    reasoned about without knowing where the rows came from.

    Vectorised rather than looped: the previous version iterated groups and
    re-sorted inside each one, which is the slow way to do a groupby.
    """
    if views is None or views.empty:
        return None

    # Reset the index first: callers may hand over a concatenated or filtered
    # frame whose index repeats, and positional alignment below would then
    # raise or, worse, silently mismatch rows.
    df = views.sort_values(['ticker', 'date']).reset_index(drop=True)
    grouped = df.groupby('ticker')['wiki_views']

    # Rank within ticker so the tail can be selected without a Python loop.
    df['_order'] = grouped.cumcount(ascending=False)   # 0 = most recent
    df['_count'] = grouped.transform('size')

    eligible = df[df['_count'] >= min_history]
    if eligible.empty:
        return None

    is_recent = eligible['_order'] < recent_days
    recent_mean = eligible[is_recent].groupby('ticker')['wiki_views'].mean()
    base = eligible[~is_recent].groupby('ticker')['wiki_views']
    base_mean, base_sd = base.mean(), base.std()

    z = (recent_mean - base_mean) / base_sd.where(base_sd > 0)
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    return z if len(z) else None


def _read_attention(tickers: list[str], as_of: date,
                    lookback_days: int = 400) -> pd.DataFrame:
    """I/O only: pageview rows in the window ending at `as_of`."""
    if not tickers:
        return pd.DataFrame()
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params: dict = {f't{i}': t for i, t in enumerate(tickers)}
    params.update({'lo': str(as_of - timedelta(days=lookback_days)),
                   'hi': str(as_of)})
    return db.read_sql(
        f'SELECT ticker, date, wiki_views FROM attention '
        f'WHERE ticker IN ({ph}) AND date BETWEEN :lo AND :hi',
        params, parse_dates=['date'])


def _attention_zscore(tickers: list[str], as_of: date) -> pd.Series | None:
    """Read, then compute — the two steps kept separate and each testable."""
    return attention_zscore(_read_attention(tickers, as_of))


# ─────────────────────────────────────────────
# ASSEMBLY
# ─────────────────────────────────────────────

def build_all(tickers: list[str], as_of: date | str,
              use_cache: bool = True) -> pd.DataFrame:
    """
    All factor families joined into one frame, plus sector labels.

    Cached on disk by (universe, as_of, data version). The version moves
    whenever anything is ingested, so a hit is always as fresh as a rebuild —
    this is a speed optimisation only, never a staleness trade.
    """
    as_of = pd.to_datetime(as_of).date()

    cache_key = None
    if use_cache and tickers:
        cache_key = factorcache.key(tickers, as_of)
        cached = factorcache.load(cache_key)
        if cached is not None:
            log.info('factors: cache hit (%d names, as_of %s)',
                     len(cached), as_of)
            return cached

    started = time.perf_counter()
    frames = [
        price_factors(tickers, as_of),
        fundamental_factors(tickers, as_of),
        alt_factors(tickers, as_of),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how='outer')

    if tickers:
        ph = ','.join(f':t{i}' for i in range(len(tickers)))
        meta = db.read_sql(
            f'SELECT ticker, sector, name, industry FROM securities '
            f'WHERE ticker IN ({ph})',
            {f't{i}': t for i, t in enumerate(tickers)})
        if not meta.empty:
            out = out.join(meta.set_index('ticker'), how='left')

    if 'sector' not in out:
        out['sector'] = 'Unknown'
    out['sector'] = out['sector'].fillna('Unknown')

    log.info('factors: built %d names in %.1fs', len(out),
             time.perf_counter() - started)
    if cache_key is not None:
        factorcache.save(cache_key, out)
    return out


# Direction of preference for every factor: True when larger is better.
FACTOR_DIRECTION: dict[str, bool] = {
    'EARNINGS_YIELD': True, 'FCF_YIELD': True, 'SALES_YIELD': True,
    'BOOK_TO_MARKET': True, 'EBIT_TO_EV': True,
    'ROE': True, 'ROA': True, 'ROIC': True, 'GROSS_PROFITABILITY': True,
    'NET_MARGIN': True, 'OPERATING_MARGIN': True, 'ASSET_TURNOVER': True,
    'CURRENT_RATIO': True, 'PIOTROSKI_F': True,
    'EVENT_RESULTS': True,          # a recent results filing opens the drift window
    'ACCRUALS': False, 'DEBT_TO_EQUITY': False,
    'REVENUE_GROWTH_1Y': True, 'REVENUE_CAGR_3Y': True,
    'EARNINGS_GROWTH_1Y': True, 'EQUITY_CAGR_3Y': True,
    'MOM_12_1': True, 'MOM_6_1': True, 'MOM_VOL_ADJ': True,
    'RETURN_3M': True, 'RETURN_6M': True, 'RETURN_1Y': True,
    'REVERSAL_1M': True,          # already sign-flipped at construction
    'PCT_FROM_52W_HIGH': True, 'PCT_VS_MA200': True, 'PCT_VS_MA50': True,
    'VOL_1Y': False, 'VOL_3M': False, 'BETA': False, 'IDIO_VOL': False,
    'MAX_DD_1Y': True,            # less negative is better
    'DIVIDEND_YIELD': True, 'BUYBACK_YIELD': True, 'PAYOUT_RATIO': False,
    'INSIDER_NET_BUY': True, 'INSIDER_CLUSTER': True,
    'ATTENTION_Z': True, 'EVENT_INTENSITY_90D': False,
    'EVENT_MANAGEMENT': False, 'EVENT_RESTRUCTURING': False,
}
