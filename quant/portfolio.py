"""
Your portfolio and watchlist — the names you actually own or follow.

Everything else in this project ranks a universe. This module answers the
question you wake up with instead: *what is happening to my positions?*

  watchlist  names you follow; no levels, no money
  positions  lots you hold: shares, cost per share, open date, and optional
             stop / target alert levels

P&L is measured on the raw (unadjusted) close against your cost per share —
that is what you paid — with share counts and cost restated through any
split since the lot was opened, so a 10-for-1 split does not read as a 90%
loss. Dividends are not included (price return only).

Alerts are deterministic checks run against stored data:

  stop hit / target hit       last session's low / high crossed your level
  near stop                   last close within 3% of the stop
  big move                    |1-day move| ≥ max(5%, 2.5 × the stock's ATR%)
  earnings soon               a report within the next 7 days
  strategies turned negative  more strategies call it Sell/Avoid than Buy
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

import config
from core import db

log = logging.getLogger(__name__)

NEAR_STOP = 0.03
BIG_MOVE_FLOOR = 0.05
EARNINGS_DAYS = 7


def _norm(t: str) -> str:
    return str(t).strip().upper().replace('.', '-')


# ─────────────────────────────────────────────
# WATCHLIST
# ─────────────────────────────────────────────

def watch_add(tickers: list[str], note: str | None = None) -> list[str]:
    names = sorted({_norm(t) for t in tickers if str(t).strip()})
    if names:
        db.upsert(db.watchlist, [{'ticker': t, 'note': note,
                                  'added_at': config.utc_now()} for t in names])
    return names


def watch_remove(tickers: list[str]) -> int:
    n = 0
    for t in {_norm(x) for x in tickers}:
        n += db.execute('DELETE FROM watchlist WHERE ticker = :t', {'t': t})
    return n


def watchlist_table() -> pd.DataFrame:
    return db.read_sql('SELECT ticker, note, added_at FROM watchlist ORDER BY ticker',
                       parse_dates=['added_at'])


def watchlist() -> list[str]:
    """Stored watchlist plus QUANT_WATCHLIST from .env, de-duplicated."""
    stored = watchlist_table()['ticker'].tolist()
    return sorted(set(stored) | set(config.WATCHLIST))


# ─────────────────────────────────────────────
# POSITIONS
# ─────────────────────────────────────────────

def add_lot(ticker: str, shares: float, cost_basis: float,
            opened: date | str | None = None, stop: float | None = None,
            target: float | None = None, strategy: str | None = None,
            note: str | None = None) -> str:
    """Record a purchase. Returns the new lot id."""
    if shares is None or float(shares) <= 0:
        raise ValueError('shares must be positive')
    if cost_basis is None or float(cost_basis) <= 0:
        raise ValueError('cost per share must be positive')
    stop = float(stop) if stop not in (None, '') else None
    target = float(target) if target not in (None, '') else None
    if stop is not None and stop >= float(cost_basis) * 3:
        raise ValueError('stop looks wrong — it is far above the cost basis')
    lot_id = uuid.uuid4().hex[:12]
    db.upsert(db.positions, [{
        'lot_id': lot_id, 'ticker': _norm(ticker), 'shares': float(shares),
        'cost_basis': float(cost_basis),
        'opened': pd.to_datetime(opened or date.today()).date(),
        'stop': stop, 'target': target, 'strategy': strategy, 'note': note,
        'status': 'open', 'updated_at': config.utc_now(),
    }])
    return lot_id


def close(ticker_or_lot: str, exit_price: float | None = None,
          closed: date | str | None = None) -> int:
    """
    Close a lot (by id) or every open lot of a ticker.

    Without `exit_price` the last stored close is used.
    """
    key = str(ticker_or_lot).strip()
    lots_ = db.read_sql(
        "SELECT lot_id, ticker FROM positions WHERE status = 'open' "
        "AND (lot_id = :k OR ticker = :t)", {'k': key, 't': _norm(key)})
    if lots_.empty:
        return 0
    when = pd.to_datetime(closed or date.today()).date()
    if exit_price is None:
        last = _last_close(lots_['ticker'].unique().tolist(), when)
    rows = []
    for _i, r in lots_.iterrows():
        px = exit_price if exit_price is not None else last.get(r['ticker'])
        rows.append({'lot_id': r['lot_id'], 'status': 'closed', 'closed': when,
                     'exit_price': float(px) if px is not None and np.isfinite(px) else None,
                     'updated_at': config.utc_now()})
    db.upsert(db.positions, rows)
    return len(rows)


def set_levels(ticker: str, stop: float | None = None,
               target: float | None = None) -> int:
    """Set alert levels on every open lot of a ticker."""
    sets, params = [], {'t': _norm(ticker)}
    if stop is not None:
        sets.append('stop = :s'); params['s'] = float(stop) if stop >= 0 else None
    if target is not None:
        sets.append('target = :g'); params['g'] = float(target) if target >= 0 else None
    if not sets:
        return 0
    return db.execute(f"UPDATE positions SET {', '.join(sets)} "
                      f"WHERE ticker = :t AND status = 'open'", params)


def delete_lot(lot_id: str) -> int:
    return db.execute('DELETE FROM positions WHERE lot_id = :l', {'l': lot_id})


def lots(status: str | None = 'open') -> pd.DataFrame:
    where = 'WHERE status = :s' if status else ''
    return db.read_sql(f'SELECT * FROM positions {where} ORDER BY ticker, opened',
                       {'s': status} if status else {},
                       parse_dates=['opened', 'closed'])


def import_csv(path: str | Path) -> int:
    """
    Load lots from a CSV with columns ticker, shares, cost_basis and,
    optionally, opened, stop, target, strategy, note. Returns lots added.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {'symbol': 'ticker', 'qty': 'shares', 'quantity': 'shares',
               'cost': 'cost_basis', 'price': 'cost_basis', 'date': 'opened'}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    missing = {'ticker', 'shares', 'cost_basis'} - set(df.columns)
    if missing:
        raise ValueError(f'CSV is missing column(s): {", ".join(sorted(missing))}')
    n = 0
    for _i, r in df.iterrows():
        add_lot(r['ticker'], r['shares'], r['cost_basis'],
                opened=r.get('opened') if pd.notna(r.get('opened')) else None,
                stop=r.get('stop') if pd.notna(r.get('stop')) else None,
                target=r.get('target') if pd.notna(r.get('target')) else None,
                strategy=r.get('strategy') if pd.notna(r.get('strategy')) else None,
                note=r.get('note') if pd.notna(r.get('note')) else None)
        n += 1
    return n


# ─────────────────────────────────────────────
# VALUATION
# ─────────────────────────────────────────────

def _last_close(tickers: list[str], as_of: date) -> dict[str, float]:
    from data import yahoo
    s = yahoo.latest_prices(tickers, as_of, field='close')
    return {k: float(v) for k, v in s.items() if pd.notna(v)}


def _split_factor(tickers: list[str], opened: dict[str, date]) -> dict[str, float]:
    """Product of split ratios after each lot opened (share multiplier)."""
    from data import yahoo
    sp = yahoo.split_factors(tickers)
    out = {}
    for t, d in opened.items():
        rows = sp[(sp['ticker'] == t) & (sp['date'].dt.date > d)] if not sp.empty \
            else pd.DataFrame()
        out[t] = float(np.prod(rows['ratio'])) if not rows.empty else 1.0
    return out


def holdings(as_of: date | None = None) -> pd.DataFrame:
    """
    Open positions aggregated per ticker, valued at the last stored close.

    Columns: shares, avg_cost, cost_value, last, last_date, market_value,
    pnl, pnl_pct, day_chg, weight, stop, target, to_stop, to_target, lots.
    """
    as_of = as_of or date.today()
    open_lots = lots('open')
    if open_lots.empty:
        return pd.DataFrame()

    open_lots['opened_d'] = open_lots['opened'].dt.date
    mult = {}
    for _i, r in open_lots.iterrows():
        f = _split_factor([r['ticker']], {r['ticker']: r['opened_d']})[r['ticker']]
        mult[r['lot_id']] = f
    open_lots['adj_shares'] = open_lots['shares'] * open_lots['lot_id'].map(mult)
    open_lots['cost_value'] = open_lots['shares'] * open_lots['cost_basis']

    tickers = sorted(open_lots['ticker'].unique())
    from data import yahoo
    px = yahoo.price_history(tickers, start=as_of - timedelta(days=30), end=as_of,
                             field='close')
    rows = []
    for t, grp in open_lots.groupby('ticker'):
        shares = float(grp['adj_shares'].sum())
        cost_value = float(grp['cost_value'].sum())
        col = px[t].dropna() if t in getattr(px, 'columns', []) else pd.Series(dtype=float)
        last = float(col.iloc[-1]) if len(col) else np.nan
        prev = float(col.iloc[-2]) if len(col) >= 2 else np.nan
        mv = shares * last if np.isfinite(last) else np.nan
        stops = grp['stop'].dropna()
        targets = grp['target'].dropna()
        stop = float(stops.max()) if len(stops) else np.nan        # tightest
        target = float(targets.min()) if len(targets) else np.nan  # nearest
        rows.append({
            'ticker': t, 'shares': shares, 'avg_cost': cost_value / shares,
            'cost_value': cost_value, 'last': last,
            'last_date': col.index[-1].date() if len(col) else None,
            'market_value': mv, 'pnl': mv - cost_value,
            'pnl_pct': mv / cost_value - 1 if cost_value else np.nan,
            'day_chg': last / prev - 1 if np.isfinite(prev) and prev else np.nan,
            'stop': stop, 'target': target,
            'to_stop': last / stop - 1 if np.isfinite(stop) and stop else np.nan,
            'to_target': target / last - 1 if np.isfinite(target) and last else np.nan,
            'lots': len(grp),
            'opened': grp['opened_d'].min(),
            'strategy': ', '.join(sorted({s for s in grp['strategy'].dropna()})),
        })
    out = pd.DataFrame(rows).set_index('ticker')
    total = out['market_value'].sum(min_count=1)
    out['weight'] = out['market_value'] / total if total else np.nan
    return out.sort_values('market_value', ascending=False)


def summary(h: pd.DataFrame) -> dict:
    """
    Totals over positions that have a price. A name with no stored close is
    left out of value *and* cost — counting its cost against a zero value
    would report a total loss.
    """
    if h is None or h.empty:
        return {}
    priced = h[h['market_value'].notna()]
    mv = float(priced['market_value'].sum())
    cost = float(priced['cost_value'].sum())
    day = (priced['market_value'] * priced['day_chg']
           / (1 + priced['day_chg'])).sum(min_count=1)
    return {'market_value': mv if len(priced) else np.nan,
            'cost': cost, 'pnl': mv - cost if len(priced) else np.nan,
            'pnl_pct': mv / cost - 1 if cost else np.nan,
            'day_pnl': float(day) if pd.notna(day) else np.nan,
            'positions': int(len(h)), 'unpriced': int(len(h) - len(priced))}


def realized(since: date | None = None) -> pd.DataFrame:
    """Closed lots with their realised P&L."""
    c = lots('closed')
    if c.empty:
        return c
    if since:
        c = c[c['closed'].dt.date >= since]
    c['pnl'] = (c['exit_price'] - c['cost_basis']) * c['shares']
    c['pnl_pct'] = c['exit_price'] / c['cost_basis'] - 1
    return c


# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────

def strategy_view(tickers: list[str], as_of: date,
                  spec_json: str | None = None) -> pd.DataFrame:
    """How the latest stored run of every strategy ranks each ticker."""
    if not tickers:
        return pd.DataFrame()
    from reports import builder as RB
    rows = []
    for key in RB._all_strategy_keys():
        run = RB._latest_run(key, as_of, spec_json)
        if run is None:
            continue
        sc = RB._run_scores(run['run_id'])
        sc = sc[sc['ticker'].isin(tickers)]
        avoid = RB._is_avoid(key)
        for _i, r in sc.iterrows():
            rows.append({'ticker': r['ticker'], 'strategy': key,
                         'strategy_name': RB._strategy_name(key),
                         'rank': r['rank'], 'signal': r['signal'],
                         'reasons': r.get('reasons'), 'avoid': avoid,
                         'bucket': RB._signal_bucket(r['signal'])})
    return pd.DataFrame(rows)


def alerts(as_of: date | None = None, h: pd.DataFrame | None = None,
           watch: list[str] | None = None,
           spec_json: str | None = None) -> list[dict]:
    """
    Things about your positions (and watchlist) that need a look today.

    Each alert: {level: 'risk'|'watch'|'info', ticker, text}. Ordered risk
    first. Uses stored data only; nothing is fetched.
    """
    from data import earnings as ER
    from data import yahoo

    as_of = as_of or date.today()
    h = holdings(as_of) if h is None else h
    watch = watchlist() if watch is None else watch
    out: list[dict] = []

    if h is not None and not h.empty:
        tickers = list(h.index)
        hi_, lo_, cl_ = yahoo.adjusted_ohlc(tickers, start=as_of - timedelta(days=40),
                                             end=as_of)
        raw = yahoo.price_history(tickers, start=as_of - timedelta(days=10),
                                  end=as_of, field='close')
        for t, r in h.iterrows():
            last = r['last']
            if not np.isfinite(last):
                continue
            # Levels are in raw dollars; compare with the raw session range.
            day_lo = day_hi = np.nan
            if t in lo_.columns and t in cl_.columns and t in raw.columns:
                ratio = (raw[t] / cl_[t]).dropna()
                if len(ratio):
                    k = float(ratio.iloc[-1])
                    day_lo = float(lo_[t].dropna().iloc[-1]) * k
                    day_hi = float(hi_[t].dropna().iloc[-1]) * k
            stop, target = r['stop'], r['target']
            if np.isfinite(stop) and (last <= stop or (np.isfinite(day_lo) and day_lo <= stop)):
                out.append({'level': 'risk', 'ticker': t,
                            'text': f'Stop {stop:,.2f} hit — last {last:,.2f} '
                                    f'({r["pnl_pct"]:+.1%} on cost)'})
            elif np.isfinite(stop) and last / stop - 1 <= NEAR_STOP:
                out.append({'level': 'watch', 'ticker': t,
                            'text': f'{last / stop - 1:.1%} above your stop '
                                    f'({stop:,.2f})'})
            if np.isfinite(target) and (last >= target or (np.isfinite(day_hi) and day_hi >= target)):
                out.append({'level': 'info', 'ticker': t,
                            'text': f'Target {target:,.2f} reached — last {last:,.2f} '
                                    f'({r["pnl_pct"]:+.1%}); decide: take profit or '
                                    f'raise the stop'})
            # Big move relative to the stock's own volatility.
            if np.isfinite(r['day_chg']) and t in cl_.columns:
                c = cl_[t].dropna()
                atr_pct = (hi_[t] - lo_[t]).dropna().tail(14).mean() / c.iloc[-1] \
                    if len(c) and t in hi_.columns else np.nan
                limit = max(BIG_MOVE_FLOOR, 2.5 * atr_pct) if np.isfinite(atr_pct) \
                    else BIG_MOVE_FLOOR
                if abs(r['day_chg']) >= limit:
                    out.append({'level': 'watch', 'ticker': t,
                                'text': f'Moved {r["day_chg"]:+.1%} in the last '
                                        f'session — unusually large for it'})

    names = sorted(set(h.index if h is not None and not h.empty else []) | set(watch))
    if names:
        er = ER.upcoming(names, as_of, days=EARNINGS_DAYS)
        held = set(h.index) if h is not None and not h.empty else set()
        for _i, e in er.iterrows():
            when = {'bmo': 'before the open', 'amc': 'after the close'}.get(e['time'], '')
            tag = 'confirmed' if e['confirmed'] else 'estimated'
            out.append({'level': 'watch' if e['ticker'] in held else 'info',
                        'ticker': e['ticker'],
                        'text': f'Earnings {e["report_date"]:%a %b} '
                                f'{e["report_date"].day} {when} ({tag})'
                                + (' — you hold it' if e['ticker'] in held else '')})

        view = strategy_view(names, as_of, spec_json)
        held_set = set(h.index) if h is not None and not h.empty else set()
        if not view.empty:
            for t, g in view.groupby('ticker'):
                ranked = g[~g['avoid']]
                sells = int((ranked['bucket'] == 'Sell').sum())
                buys = int((ranked['bucket'] == 'Buy').sum())
                if sells > buys and sells >= 2 and t in held_set:
                    out.append({'level': 'watch', 'ticker': t,
                                'text': f'{sells} strategies now rate it Sell/Avoid '
                                        f'(vs {buys} Buy)'})
                flagged = g[g['avoid'] & g['signal'].fillna('').str.contains('Avoid')]
                for _i, f in flagged.iterrows():
                    why = f' — {f["reasons"]}' if f.get('reasons') else ''
                    out.append({'level': 'watch' if t in held_set else 'info',
                                'ticker': t,
                                'text': f'On the {f["strategy_name"]} avoid list '
                                        f'(#{int(f["rank"])}){why}'})

    order = {'risk': 0, 'watch': 1, 'info': 2}
    return sorted(out, key=lambda a: (order[a['level']], a['ticker']))
