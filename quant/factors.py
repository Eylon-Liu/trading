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
from concurrent.futures import ThreadPoolExecutor
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


def _vectorized_rsi(px: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's RSI for every column at once — replaces per-ticker loop."""
    delta = px.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    result = rsi.iloc[-1]
    result[px.count() < period + 1] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def _vectorized_atr(high: pd.DataFrame, low: pd.DataFrame,
                    close: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR for every column at once — replaces per-ticker loop."""
    prev_close = close.shift(1)
    tr = np.maximum(
        np.maximum(high - low, (high - prev_close).abs()),
        (low - prev_close).abs())
    atr_vals = tr.ewm(alpha=1 / period, adjust=False).mean()
    result = atr_vals.iloc[-1]
    result[close.count() < period + 1] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def price_factors(tickers: list[str], as_of: date | str,
                  benchmark: str | None = None,
                  _bt_ohlc: tuple | None = None) -> pd.DataFrame:
    """
    Every price-derived factor, computed only from bars at or before `as_of`.
    """
    as_of = pd.to_datetime(as_of).date()
    benchmark = benchmark or config.BENCHMARK_TICKER

    if _bt_ohlc is not None:
        adj_c, adj_h, adj_l, _raw_c = _bt_ohlc
        cutoff = pd.Timestamp(as_of)
        px = adj_c.loc[:cutoff].copy()
        high = adj_h.loc[:cutoff].copy()
        low = adj_l.loc[:cutoff].copy()
        adj_close = px
        needed = sorted(set(tickers) | {benchmark})
        avail = px.columns.intersection(needed)
        px, high, low, adj_close = px[avail], high[avail], low[avail], adj_close[avail]
    else:
        start = as_of - timedelta(days=int(TD['3Y'] * 1.6))
        universe = sorted(set(tickers) | {benchmark})
        px, high, low, adj_close = yahoo.price_and_ohlc(universe, start=start,
                                                         end=as_of)
    if px.empty:
        return pd.DataFrame()
    px = px.ffill()

    out = pd.DataFrame(index=[t for t in tickers if t in px.columns])
    if len(out.index) == 0:
        return out

    # ── momentum ──────────────────────────────────────────────────
    out['MOM_12_1'] = _ret(px, TD['1Y'] - TD['1M'], skip=TD['1M']).reindex(out.index)
    out['MOM_6_1'] = _ret(px, TD['6M'] - TD['1M'], skip=TD['1M']).reindex(out.index)
    out['RETURN_1M'] = _ret(px, TD['1M']).reindex(out.index)
    out['RETURN_3M'] = _ret(px, TD['3M']).reindex(out.index)
    out['RETURN_6M'] = _ret(px, TD['6M']).reindex(out.index)
    out['RETURN_1Y'] = _ret(px, TD['1Y']).reindex(out.index)
    out['REVERSAL_1M'] = -out['RETURN_1M']

    # ── risk ──────────────────────────────────────────────────────
    rets = px.pct_change(fill_method=None)
    ann = np.sqrt(252)
    out['VOL_1Y'] = (rets.tail(TD['1Y']).std() * ann).reindex(out.index)
    out['VOL_3M'] = (rets.tail(TD['3M']).std() * ann).reindex(out.index)

    if benchmark in rets.columns:
        b_rets = rets[benchmark].tail(TD['1Y']).dropna()
        s_cols = [t for t in out.index if t in rets.columns]
        s_rets = rets[s_cols].tail(TD['1Y']).reindex(b_rets.index)

        b_dm = b_rets - b_rets.mean()
        s_dm = s_rets.sub(s_rets.mean())
        products = s_dm.mul(b_dm, axis=0)
        n_valid = products.notna().sum()
        cov_sb = products.sum() / (n_valid - 1).clip(lower=1)
        var_b = float(b_dm.var())

        if var_b and np.isfinite(var_b) and var_b != 0:
            betas = cov_sb / var_b
            betas[n_valid < 60] = np.nan
            out['BETA'] = betas.reindex(out.index)

            resid = s_rets.sub(betas.values * b_rets.values[:, np.newaxis],
                               axis=0)
            idio = resid.std() * ann
            idio[n_valid < 60] = np.nan
            out['IDIO_VOL'] = idio.reindex(out.index)

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
    px_for_rsi = px[[t for t in out.index if t in px.columns]]
    out['RSI_14'] = _vectorized_rsi(px_for_rsi).reindex(out.index)

    common = out.index.intersection(high.columns).intersection(
        low.columns).intersection(adj_close.columns)
    if len(common):
        out['ATR_14'] = _vectorized_atr(
            high[common], low[common], adj_close[common]).reindex(out.index)
        out['ATR_PCT'] = out['ATR_14'] / last.reindex(out.index)

    out['PRICE'] = last.reindex(out.index)
    return out


# ─────────────────────────────────────────────
# FUNDAMENTAL FACTORS
# ─────────────────────────────────────────────

def fundamental_factors(tickers: list[str], as_of: date | str,
                        _facts_all: pd.DataFrame | None = None,
                        _prev_fund: pd.DataFrame | None = None,
                        _changed_tickers: set[str] | None = None,
                        _splits: pd.DataFrame | None = None,
                        _splits_applied: bool = False,
                        _fund_stash: dict | None = None,
                        _bt_close: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Valuation, quality, and growth factors from point-in-time SEC data.

    Market cap comes from the profile snapshot, which cannot be backfilled, so
    for historical dates the yield factors fall back to a shares-outstanding
    times price estimate built entirely from filed data and stored bars.
    """
    as_of = pd.to_datetime(as_of).date()
    facts_all = _facts_all if _facts_all is not None else sec.facts_asof(tickers, as_of)
    fund = F.build_fundamentals(tickers, as_of, facts_all=facts_all,
                                splits=_splits,
                                _prev_result=_prev_fund,
                                _changed_tickers=_changed_tickers,
                                _splits_applied=_splits_applied)
    if _fund_stash is not None:
        _fund_stash['result'] = fund
    if fund.empty:
        return pd.DataFrame()

    prof = yahoo.profile_asof(tickers, as_of)

    mcap = pd.Series(np.nan, index=fund.index, dtype=float)
    mcap_source = pd.Series('none', index=fund.index)
    if not prof.empty and 'market_cap' in prof:
        provider = prof['market_cap'].reindex(fund.index)
        has_provider = provider.notna()
        mcap = provider
        mcap_source = mcap_source.where(~has_provider, 'provider')

    # Reconstruct market cap where the snapshot is missing — which is every
    # historical date, since the provider publishes no market-cap history.
    #
    # `close` rather than `adj_close` removes the *dividend* adjustment only.
    # Both columns are restated retroactively for splits, so choosing between
    # them does nothing about a split, and an earlier comment here claiming
    # otherwise was wrong. The share count is what has to move: it is restated
    # onto the current basis in fundamentals.adjust_shares_for_splits, so that
    # by this point shares and price already share one basis. Without that,
    # Lam Research showed a $3.1B market cap in 2018 against ~$30B actual.
    if _bt_close is not None:
        cutoff = pd.Timestamp(as_of)
        close_slice = _bt_close.loc[:cutoff]
        px_close = yahoo.last_close(close_slice).reindex(tickers)
    else:
        px_close = yahoo.latest_prices(tickers, as_of, field='close')
    est = fund['shares_diluted'].reindex(fund.index) * px_close.reindex(fund.index)
    filled_from_est = mcap.isna() & est.notna()
    mcap = mcap.fillna(est)
    mcap_source = mcap_source.where(~filled_from_est, 'computed')

    ev = mcap + fund['debt'].fillna(0) - fund['cash'].fillna(0)

    out = pd.DataFrame(index=fund.index)
    out['MARKET_CAP'] = mcap
    out['MCAP_SOURCE'] = mcap_source

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
    out['NET_DEBT_TO_EQUITY'] = fund['net_debt_to_equity']
    out['FCF_CONVERSION'] = fund['fcf_conversion']
    out['CURRENT_RATIO'] = fund['current_ratio']
    out['ROA'] = fund['roa']
    out['ASSET_TURNOVER'] = fund['asset_turnover']

    # ── growth ────────────────────────────────────────────────────
    out['REVENUE_GROWTH_1Y'] = fund['revenue_growth_1y']
    out['REVENUE_CAGR_3Y'] = fund['revenue_cagr_3y']
    out['EARNINGS_GROWTH_1Y'] = fund['earnings_growth_1y']
    out['EQUITY_CAGR_3Y'] = fund['equity_cagr_3y']
    out['EPS_CAGR_3Y'] = fund['eps_cagr_3y']

    # ── income ────────────────────────────────────────────────────
    # This used to rebuild the whole fundamentals frame a second time to read
    # a column that was never produced, so the factor was always NaN and the
    # rebuild cost as much as the rest of the screen combined.
    out['BUYBACK_YIELD'] = _div(fund['buybacks_ttm'], mcap)

    # Dividends come from the cash flow statement, not the provider snapshot.
    #
    # Reading them from profile_snapshots covered 5% of the S&P 500 and existed
    # only for today, so Total Shareholder Yield was scoring a third of its
    # weight on almost nothing and could not be backtested at all. The cash
    # actually paid is in the filings — the same place BUYBACK_YIELD already
    # comes from — which makes both halves of "shareholder yield" consistent,
    # point-in-time, and available for the whole history.
    #
    # It is also a *decimal* like every other yield here. The provider reported
    # percentages, so a 1.66% yield arrived as 1.66 and sat three orders of
    # magnitude above earnings yield in the same composite.
    out['DIVIDEND_YIELD'] = _div(fund['dividends_paid_ttm'], mcap)
    out['PAYOUT_RATIO'] = _div(fund['dividends_paid_ttm'],
                               fund['net_income_ttm'])

    if not prof.empty:
        # Forward-looking, analyst-derived: live screen only, never historical.
        out['FORWARD_PE'] = prof.get('forward_pe').reindex(fund.index) \
            if 'forward_pe' in prof else np.nan
        out['TRAILING_PE'] = prof.get('trailing_pe').reindex(fund.index) \
            if 'trailing_pe' in prof else np.nan

    out['PIOTROSKI_F'] = fund['piotroski_f']
    out['DATA_AS_OF'] = fund['last_filed']
    return out


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


# ─────────────────────────────────────────────
# ALTERNATIVE-DATA FACTORS
# ─────────────────────────────────────────────

def alt_factors(tickers: list[str], as_of: date | str,
                _bt_insiders: pd.DataFrame | None = None,
                _bt_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Signals from filings and attention data.

    All of these carry a filing or observation date, so unlike news sentiment
    they are backtestable from day one.
    """
    as_of = pd.to_datetime(as_of).date()
    out = pd.DataFrame(index=pd.Index(tickers, name='ticker'))

    # ── insider activity (Form 4, open-market only) ───────────────
    if _bt_insiders is not None:
        cutoff = pd.Timestamp(as_of)
        lo = cutoff - pd.Timedelta(days=180)
        mask = ((_bt_insiders['filed'] >= lo)
                & (_bt_insiders['filed'] <= cutoff)
                & (_bt_insiders['ticker'].isin(tickers)))
        ins = _bt_insiders[mask]
    else:
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
    if _bt_events is not None:
        cutoff = pd.Timestamp(as_of)
        lo = cutoff - pd.Timedelta(days=90)
        mask = ((_bt_events['filed'] >= lo)
                & (_bt_events['filed'] <= cutoff)
                & (_bt_events['ticker'].isin(tickers)))
        ev = _bt_events[mask]
    else:
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
              use_cache: bool = True,
              _facts_all: pd.DataFrame | None = None,
              _data_version: str | None = None,
              _prev_fund: pd.DataFrame | None = None,
              _changed_tickers: set[str] | None = None,
              _splits: pd.DataFrame | None = None,
              _splits_applied: bool = False,
              _fund_stash: dict | None = None,
              _bt_ohlc: tuple | None = None,
              _bt_insiders: pd.DataFrame | None = None,
              _bt_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    All factor families joined into one frame, plus sector labels.

    Cached on disk by (universe, as_of, data version). The version moves
    whenever anything is ingested, so a hit is always as fresh as a rebuild —
    this is a speed optimisation only, never a staleness trade.
    """
    as_of = pd.to_datetime(as_of).date()

    cache_key = None
    if use_cache and tickers:
        cache_key = factorcache.key(tickers, as_of, version=_data_version)
        cached = factorcache.load(cache_key)
        if cached is not None:
            log.info('factors: cache hit (%d names, as_of %s)',
                     len(cached), as_of)
            return cached

    started = time.perf_counter()
    inner_stash: dict = {}

    bt_close = _bt_ohlc[3] if _bt_ohlc is not None else None

    def _build_price():
        return price_factors(tickers, as_of, _bt_ohlc=_bt_ohlc)

    def _build_fund():
        return fundamental_factors(tickers, as_of,
                                   _facts_all=_facts_all,
                                   _prev_fund=_prev_fund,
                                   _changed_tickers=_changed_tickers,
                                   _splits=_splits,
                                   _splits_applied=_splits_applied,
                                   _fund_stash=inner_stash,
                                   _bt_close=bt_close)

    def _build_alt():
        return alt_factors(tickers, as_of,
                           _bt_insiders=_bt_insiders,
                           _bt_events=_bt_events)

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_pf = pool.submit(_build_price)
        fut_ff = pool.submit(_build_fund)
        fut_af = pool.submit(_build_alt)
        pf = fut_pf.result()
        ff = fut_ff.result()
        af = fut_af.result()

    log.info('factors breakdown: total %.1fs (parallel)',
             time.perf_counter() - started)

    frames = [pf, ff, af]
    if _fund_stash is not None:
        _fund_stash.update(inner_stash)
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


# Factors sourced from live provider snapshots rather than filings or prices.
#
# profile_snapshots only ever holds *today* — the provider does not publish a
# history and nothing backfills one — so these are unknowable at any past date.
# A historical screen that scores them is not reading a stale value, it is
# reading nothing at all, and the weight silently redistributes onto whatever
# else the strategy holds. Naming them lets the engine say so out loud.
# Dividend yield and payout ratio used to belong here. They now derive from
# the cash flow statement, so they have a real history and came off the list.
LIVE_ONLY_FACTORS: frozenset[str] = frozenset({
    'FORWARD_PE', 'TRAILING_PE',
})

# Below this share of the universe a factor is treated as absent rather than
# sparse: it is a data gap to report, not a ranking to trust.
MIN_FACTOR_PRESENCE = 0.10


def factor_availability(raw: pd.DataFrame, weights: dict[str, float],
                        as_of: date | str | None = None) -> pd.DataFrame:
    """
    Per-factor coverage across the universe, for a strategy's weight vector.

    The composite already applies a *per-name* coverage floor, which catches a
    company missing half its factors. It cannot catch a factor missing for
    every company: each name then looks uniformly a little thin, passes the
    floor, and the strategy quietly ranks on the factors that remain. That is
    how a backtest of an insider strategy ran over a period with no insider
    data and reported a result anyway.
    """
    is_hist = (as_of is not None
               and pd.to_datetime(as_of).date() < date.today())
    rows = []
    for name, weight in weights.items():
        cov = float(raw[name].notna().mean()) if name in raw.columns else 0.0
        if name not in raw.columns:
            status = 'missing'
        elif cov < MIN_FACTOR_PRESENCE:
            status = 'empty'
        elif cov < 0.5:
            status = 'sparse'
        else:
            status = 'ok'
        if status != 'ok' and is_hist and name in LIVE_ONLY_FACTORS:
            status = 'live_only'
        rows.append({'factor': name, 'weight': float(weight),
                     'coverage': cov, 'status': status})
    return pd.DataFrame(rows).sort_values('coverage')


# Direction of preference for every factor: True when larger is better.
FACTOR_DIRECTION: dict[str, bool] = {
    'EARNINGS_YIELD': True, 'FCF_YIELD': True, 'SALES_YIELD': True,
    'BOOK_TO_MARKET': True, 'EBIT_TO_EV': True,
    'ROE': True, 'ROA': True, 'ROIC': True, 'GROSS_PROFITABILITY': True,
    'NET_MARGIN': True, 'OPERATING_MARGIN': True, 'ASSET_TURNOVER': True,
    'CURRENT_RATIO': True, 'PIOTROSKI_F': True, 'FCF_CONVERSION': True,
    'EVENT_RESULTS': True,          # a recent results filing opens the drift window
    'ACCRUALS': False, 'DEBT_TO_EQUITY': False, 'NET_DEBT_TO_EQUITY': False,
    'REVENUE_GROWTH_1Y': True, 'REVENUE_CAGR_3Y': True,
    'EARNINGS_GROWTH_1Y': True, 'EQUITY_CAGR_3Y': True, 'EPS_CAGR_3Y': True,
    'MOM_12_1': True, 'MOM_6_1': True, 'MOM_VOL_ADJ': True,
    'RETURN_1M': True, 'RETURN_3M': True, 'RETURN_6M': True, 'RETURN_1Y': True,
    'REVERSAL_1M': True,          # already sign-flipped at construction
    'PCT_FROM_52W_HIGH': True, 'PCT_VS_MA200': True, 'PCT_VS_MA50': True,
    'VOL_1Y': False, 'VOL_3M': False, 'BETA': False, 'IDIO_VOL': False,
    'MAX_DD_1Y': True,            # less negative is better
    'DIVIDEND_YIELD': True, 'BUYBACK_YIELD': True, 'PAYOUT_RATIO': False,
    # Multiples, so cheaper is better. Undeclared they would default to
    # higher-is-better and rank the most expensive names first.
    'FORWARD_PE': False, 'TRAILING_PE': False,
    'INSIDER_NET_BUY': True, 'INSIDER_CLUSTER': True,
    'ATTENTION_Z': True, 'EVENT_INTENSITY_90D': False,
    'EVENT_MANAGEMENT': False, 'EVENT_RESTRUCTURING': False,
}
