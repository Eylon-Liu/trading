"""
Portfolio tab — your positions, watchlist, alerts, earnings dates, macro
regime and the live track record of the strategies.

Everything here reads the database; nothing on this page fetches. Prices,
the earnings calendar and macro series arrive through the normal ingest
(`cli.py daily` or the Data tab).
"""

from __future__ import annotations

import logging
from datetime import date

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)


def _input(id_: str, placeholder: str, type_: str = 'text', width: int = 2):
    return dbc.Col(dbc.Input(id=id_, placeholder=placeholder, type=type_,
                             size='sm'), lg=width, md=4, className='mb-2')


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id='pf-version', data=0),
        C.card('💼 Positions', [
            dbc.Row([
                _input('pf-ticker', 'Ticker'),
                _input('pf-shares', 'Shares', 'number'),
                _input('pf-cost', 'Cost / share', 'number'),
                _input('pf-date', 'Opened YYYY-MM-DD'),
                _input('pf-stop', 'Stop (optional)', 'number', 1),
                _input('pf-target', 'Target (optional)', 'number', 1),
                dbc.Col(dbc.Button('Add lot', id='pf-add', color='success',
                                   size='sm', className='w-100'),
                        lg=2, md=4, className='mb-2'),
            ], className='g-2'),
            dbc.Row([
                _input('pf-act-ticker', 'Ticker or lot id'),
                _input('pf-act-price', 'Exit price (blank = last close)', 'number'),
                dbc.Col(dbc.Button('Close position', id='pf-close', color='danger',
                                   outline=True, size='sm', className='w-100'),
                        lg=2, md=4, className='mb-2'),
                _input('pf-act-stop', 'New stop', 'number', 1),
                _input('pf-act-target', 'New target', 'number', 1),
                dbc.Col(dbc.Button('Update levels', id='pf-levels', color='info',
                                   outline=True, size='sm', className='w-100'),
                        lg=2, md=4, className='mb-2'),
            ], className='g-2'),
            html.Div(id='pf-status', style={'color': TH.MUTED, 'fontSize': '0.8rem',
                                            'marginBottom': '8px'}),
            html.Div(id='pf-holdings'),
        ], subtitle='Stored locally. P&L is price return on the raw close vs your '
                    'cost, split-adjusted; dividends excluded.', className='mb-3'),

        C.card('🚨 Alerts', html.Div(id='pf-alerts'), className='mb-3'),

        C.card('⭐ Watchlist', [
            dbc.Row([
                _input('wl-tickers', 'Tickers, comma-separated', width=4),
                _input('wl-note', 'Note (optional)', width=3),
                dbc.Col(dbc.Button('Add', id='wl-add', color='success', size='sm',
                                   className='w-100'), lg=1, md=4, className='mb-2'),
                dbc.Col(dbc.Button('Remove', id='wl-remove', color='danger',
                                   outline=True, size='sm', className='w-100'),
                        lg=1, md=4, className='mb-2'),
            ], className='g-2'),
            html.Div(id='wl-table'),
        ], subtitle='Watched names lead the daily email and are ingested even '
                    'outside your index.', className='mb-3'),

        dbc.Row([
            dbc.Col(C.card('📅 Earnings ahead', html.Div(id='pf-earnings'),
                           subtitle='Your positions and watchlist, next 21 days'),
                    lg=6, className='mb-3'),
            dbc.Col(C.card('🌐 Macro & regime', html.Div(id='pf-macro'),
                           subtitle='FRED series; context only'),
                    lg=6, className='mb-3'),
        ]),

        C.card('📈 Live track record', html.Div(id='pf-track'),
               subtitle='What each strategy’s stored top-10 picks did next, '
                        'bought at the following close. Out of sample by '
                        'construction.', className='mb-3'),
    ])


# ─────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────

def _f(v):
    try:
        return float(v) if v not in (None, '') else None
    except (TypeError, ValueError):
        return None


@callback(
    Output('pf-status', 'children'), Output('pf-version', 'data'),
    Input('pf-add', 'n_clicks'), Input('pf-close', 'n_clicks'),
    Input('pf-levels', 'n_clicks'), Input('wl-add', 'n_clicks'),
    Input('wl-remove', 'n_clicks'),
    State('pf-ticker', 'value'), State('pf-shares', 'value'),
    State('pf-cost', 'value'), State('pf-date', 'value'),
    State('pf-stop', 'value'), State('pf-target', 'value'),
    State('pf-act-ticker', 'value'), State('pf-act-price', 'value'),
    State('pf-act-stop', 'value'), State('pf-act-target', 'value'),
    State('wl-tickers', 'value'), State('wl-note', 'value'),
    State('pf-version', 'data'),
    prevent_initial_call=True,
)
def _act(_a, _c, _l, _wa, _wr, ticker, shares, cost, opened, stop, target,
         act_ticker, act_price, act_stop, act_target, wl_tickers, wl_note,
         version):
    from quant import portfolio as PF
    trig = ctx.triggered_id
    try:
        if trig == 'pf-add':
            if not ticker or _f(shares) is None or _f(cost) is None:
                return '⚠️ Ticker, shares and cost per share are required.', no_update
            PF.add_lot(ticker, _f(shares), _f(cost), opened=opened or None,
                       stop=_f(stop), target=_f(target))
            msg = f'✅ Added {ticker.upper()}.'
        elif trig == 'pf-close':
            if not act_ticker:
                return '⚠️ Enter the ticker (or lot id) to close.', no_update
            n = PF.close(act_ticker, exit_price=_f(act_price))
            msg = f'✅ Closed {n} lot(s).' if n else '⚠️ No open lot matches.'
        elif trig == 'pf-levels':
            if not act_ticker:
                return '⚠️ Enter the ticker to update.', no_update
            n = PF.set_levels(act_ticker, stop=_f(act_stop), target=_f(act_target))
            msg = f'✅ Updated {n} lot(s).'
        elif trig == 'wl-add':
            names = PF.watch_add((wl_tickers or '').replace(' ', ',').split(','),
                                 note=wl_note or None)
            msg = f'✅ Watching {", ".join(names)}.' if names else '⚠️ No tickers given.'
        elif trig == 'wl-remove':
            n = PF.watch_remove((wl_tickers or '').replace(' ', ',').split(','))
            msg = f'🗑 Removed {n}.'
        else:
            return no_update, no_update
    except ValueError as exc:
        return f'⚠️ {exc}', no_update
    except Exception as exc:                       # noqa: BLE001
        log.exception('portfolio action failed')
        return f'❌ {type(exc).__name__}: {str(exc)[:160]}', no_update
    return msg, (version or 0) + 1


# ─────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────

def _pct(v):
    return f'{v:+.1%}' if v is not None and np.isfinite(v) else '—'


@callback(
    Output('pf-holdings', 'children'), Output('pf-alerts', 'children'),
    Output('wl-table', 'children'), Output('pf-earnings', 'children'),
    Output('pf-macro', 'children'), Output('pf-track', 'children'),
    Input('tabs', 'active_tab'), Input('pf-version', 'data'),
)
def _render(active_tab, _version):
    if active_tab != 'tab-portfolio':
        return (no_update,) * 6
    from data import earnings as ER
    from data import macro
    from quant import portfolio as PF
    from quant import track

    def safe(fn, empty='—'):
        try:
            return fn()
        except Exception as exc:                   # noqa: BLE001
            log.warning('portfolio panel failed: %s', exc)
            return C.note(f'Unavailable: {type(exc).__name__}: {str(exc)[:120]}', 'warn')

    def holdings_view():
        h = PF.holdings()
        if h.empty:
            return C.placeholder('No open positions. Add a lot above, or from the '
                                 'command line: python cli.py portfolio add AAPL 10 185.50')
        s = PF.summary(h)
        money = lambda v: f'${v:,.0f}' if v is not None and np.isfinite(v) else '—'  # noqa: E731
        tone = lambda v: TH.POS if v is not None and np.isfinite(v) and v >= 0 else TH.NEG  # noqa: E731
        t = h.reset_index()
        view = pd.DataFrame({
            'ticker': t['ticker'], 'shares': t['shares'].round(4),
            'avg_cost': t['avg_cost'].round(2), 'last': t['last'].round(2),
            'day': t['day_chg'].map(_pct), 'pnl_pct': t['pnl_pct'].map(_pct),
            'pnl': t['pnl'].map(money), 'weight': t['weight'].map(
                lambda v: f'{v:.0%}' if np.isfinite(v) else '—'),
            'stop': t['stop'].round(2), 'target': t['target'].round(2),
            'opened': t['opened'].astype(str), 'lots': t['lots'],
        })
        return html.Div([
            C.metric_row([
                (money(s['market_value']), 'market value', TH.INFO),
                (money(s['pnl']), 'unrealised P&L', tone(s['pnl'])),
                (_pct(s['pnl_pct']), 'on cost', tone(s['pnl_pct'])),
                (money(s['day_pnl']), 'last session', tone(s['day_pnl'])),
            ]),
            html.Div(C.data_table(view, page_size=25), className='mt-2'),
        ])

    def alerts_view():
        al = PF.alerts()
        if not al:
            return C.placeholder('Nothing needs attention.')
        icon = {'risk': '🔴', 'watch': '🟠', 'info': '🔵'}
        return html.Div([html.Div(f'{icon[a["level"]]} {a["ticker"]} — {a["text"]}',
                                  style={'marginBottom': '4px', 'fontSize': '0.86rem'})
                         for a in al])

    def watch_view():
        wl = PF.watchlist()
        if not wl:
            return C.placeholder('Watchlist is empty.')
        from data import yahoo
        today = date.today()
        px = yahoo.price_history(wl, start=today - pd.Timedelta(days=45), end=today,
                                 field='adj_close')
        notes = PF.watchlist_table().set_index('ticker')['note'].to_dict()
        rows = []
        for t in wl:
            col = px[t].dropna() if t in getattr(px, 'columns', []) else pd.Series(dtype=float)
            n = len(col)
            rows.append({'ticker': t,
                         'last': round(float(col.iloc[-1]), 2) if n else None,
                         '1D': _pct(col.iloc[-1] / col.iloc[-2] - 1) if n >= 2 else '—',
                         '1M': _pct(col.iloc[-1] / col.iloc[max(0, n - 22)] - 1) if n >= 2 else '—',
                         'note': notes.get(t) or ''})
        return C.data_table(pd.DataFrame(rows), page_size=25)

    def earnings_view():
        names = sorted(set(PF.watchlist()) | set(PF.lots('open')['ticker']))
        er = ER.upcoming(names, days=21)
        if er.empty:
            return C.placeholder('Nothing scheduled for your names in the next three weeks.')
        er = er.assign(report_date=er['report_date'].dt.date.astype(str),
                       source=er['confirmed'].map({True: 'confirmed', False: 'estimated'}))
        return C.data_table(er[['ticker', 'report_date', 'time', 'source',
                                'eps_estimate', 'days_until']], page_size=15)

    def macro_view():
        snap = macro.snapshot(date.today())
        if snap.empty:
            return C.placeholder('No macro data yet: python cli.py macro --refresh')
        icon = {'ok': '🟢', 'watch': '🟠', 'risk': '🔴'}
        flags = [html.Div(f'{icon[l]} {t}', style={'fontSize': '0.84rem'})
                 for l, t in macro.regime(date.today(), snap)]
        tbl = snap[['label', 'value', 'chg_1w', 'chg_1m']].round(2)
        return html.Div(flags + [html.Div(C.data_table(tbl, page_size=10),
                                          className='mt-2')])

    def track_view():
        tr = track.summary()
        if tr.empty:
            return C.placeholder('No completed cohorts yet. Each day the routine '
                                 'stores a screen; the first 1-week results appear '
                                 'a week later.')
        v = tr.assign(mean_pick_return=tr['mean_pick_return'].map(_pct),
                      mean_excess=tr['mean_excess'].map(_pct),
                      hit_rate=tr['hit_rate'].map(lambda x: f'{x:.0%}'),
                      mean_ic=tr['mean_ic'].round(3))
        return C.data_table(v[['name', 'horizon', 'cohorts', 'independent',
                               'mean_pick_return', 'mean_excess', 'hit_rate',
                               'mean_ic']], page_size=30)

    return (safe(holdings_view), safe(alerts_view), safe(watch_view),
            safe(earnings_view), safe(macro_view), safe(track_view))
