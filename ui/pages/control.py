"""
Control tab — run the routine from anywhere, phone first.

Status at a glance (market, last email, data freshness, what is configured),
one large button per action, and the live output of whatever is running.
Every action is a fixed command from core/jobs.py behind a confirm dialog,
so a mis-tap on a phone cannot send an email on its own.
"""

from __future__ import annotations

import logging
from datetime import date

import dash_bootstrap_components as dbc
import pandas as pd
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

import config
from core import jobs as JOBS
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)

PRE = {'background': TH.PANEL_ALT, 'color': TH.TEXT, 'fontSize': '0.72rem',
       'padding': '10px', 'borderRadius': '8px', 'maxHeight': '50vh',
       'overflow': 'auto', 'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word',
       'marginBottom': 0}


def _action_button(a: JOBS.Action):
    return html.Div([
        dcc.ConfirmDialogProvider(
            dbc.Button(a.label, color='success', outline=a.key != 'brief',
                       className='w-100', size='lg', style={'textAlign': 'left'}),
            id={'type': 'ctl-act', 'key': a.key},
            message=f'{a.label.split(" ", 1)[1]}?\n\n{a.blurb}'),
        html.Div(f'{a.blurb}' + (f'  ·  {a.minutes}' if a.minutes else ''),
                 style={'color': TH.MUTED, 'fontSize': '0.74rem',
                        'margin': '4px 2px 14px'}),
    ])


def layout() -> html.Div:
    return html.Div([
        dcc.Interval(id='ctl-tick', interval=3000, disabled=True),
        dcc.Store(id='ctl-version', data=0),
        dbc.Row([
            dbc.Col([
                C.card('📍 Status', html.Div(id='ctl-status'), className='mb-3'),
                C.card('⚡ Run', [
                    html.Div(id='ctl-msg', style={'fontSize': '0.82rem',
                                                  'marginBottom': '10px'}),
                    *[_action_button(a) for a in JOBS.ACTIONS.values()],
                ], className='mb-3'),
            ], lg=5),
            dbc.Col([
                C.card('📜 Current job', [
                    html.Div(id='ctl-job'),
                    dbc.Button('Stop it', id='ctl-cancel', color='danger',
                               outline=True, size='sm', className='mt-2'),
                ], className='mb-3'),
                C.card('🗓️ Scheduled routine — latest log',
                       html.Div(id='ctl-routine'), className='mb-3'),
                C.card('🕘 Recent jobs', html.Div(id='ctl-history'),
                       className='mb-3'),
            ], lg=7),
        ]),
    ])


# ─────────────────────────────────────────────
# ACTIONS
# ─────────────────────────────────────────────

@callback(
    Output('ctl-msg', 'children'), Output('ctl-version', 'data'),
    Output('ctl-tick', 'disabled', allow_duplicate=True),
    Input({'type': 'ctl-act', 'key': ALL}, 'submit_n_clicks'),
    Input('ctl-cancel', 'n_clicks'),
    State('ctl-version', 'data'),
    prevent_initial_call=True,
)
def _act(_submits, _cancel, version):
    trig = ctx.triggered_id
    if not ctx.triggered or ctx.triggered[0].get('value') in (None, 0):
        return no_update, no_update, no_update
    if trig == 'ctl-cancel':
        ok = JOBS.cancel()
        return ('⏹ Stop requested.' if ok else 'Nothing is running.'), \
            (version or 0) + 1, no_update
    try:
        job = JOBS.start(trig['key'])
    except RuntimeError as exc:
        return C.note(str(exc), 'warn'), no_update, False
    except KeyError:
        return C.note('Unknown action.', 'error'), no_update, no_update
    return f'▶ Started: {job.label}', (version or 0) + 1, False


# ─────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────

def _fmt_secs(s: float) -> str:
    s = int(s)
    return f'{s // 60}m {s % 60:02d}s' if s >= 60 else f'{s}s'


def _status_view():
    from core import maintenance as M
    from core import market_calendar as MC
    from core import db
    from nlp import providers
    from reports import builder as RB

    st = MC.market_status()
    phase = {'open': '🟢 Market open', 'pre-market': '🟡 Pre-market',
             'after-hours': '🟠 After hours'}.get(
        st['phase'], f'⚪ Market closed — {st["closed_reason"]}')
    rows = [
        (phase, f'{st["now_et"]:%a %b %d, %H:%M} ET'),
        ('Last completed session', RB._fmt_day(st['last_completed_session'])),
        ('Next session', RB._fmt_day(st['next_session'])
         + (' (1 p.m. close)' if MC.is_early_close(st['next_session']) else '')),
    ]
    from core import runlock
    busy = runlock.holder()
    rows.append(('Routine', '💤 idle' if busy is None else
                 f'⏳ running: {busy.get("what", "?")} · since '
                 f'{str(busy.get("started", "?"))[11:16] or "?"} '
                 f'(pid {busy.get("pid", "?")}) — a new run waits for it'))
    last = db.read_sql("SELECT kind, period_end, emailed_at FROM reports "
                       "WHERE emailed_at IS NOT NULL ORDER BY emailed_at DESC LIMIT 1",
                       parse_dates=['emailed_at'])
    rows.append(('Last email', '—' if last.empty else
                 f'{last["kind"].iloc[0]} · {config.local_fmt(last["emailed_at"].iloc[0])}'))
    warnings = RB.freshness_warnings(date.today(), None)
    newest = RB._newest_bar(config.BENCHMARK_TICKER)
    rows.append(('Prices through', RB._fmt_day(newest) if newest else 'none stored'))
    path = M.sqlite_path()
    if path and path.exists():
        use = M.usage()
        rows.append(('Storage', f'{use["total"] / 1000:,.1f} GB — database '
                     f'{use["database"]:,.0f} MB, downloads cache {use["cache"]:,.0f} MB, '
                     f'backups {use["backups"]:,.0f} MB (pruned automatically)'))
    rows.append(('Email (SMTP)', 'configured' if config.SMTP_USER and config.SMTP_PASS
                 else 'not set — add SMTP_USER / SMTP_PASS to .env'))
    rows.append(('Recipient', config.QUANT_EMAIL or 'not set — QUANT_EMAIL in .env'))
    rows.append(('AI layer', providers.status().get('reason', '—')))

    table = html.Table([html.Tbody([
        html.Tr([html.Td(k, style={'color': TH.MUTED, 'padding': '4px 10px 4px 0',
                                   'fontSize': '0.8rem', 'verticalAlign': 'top'}),
                 html.Td(v, style={'fontSize': '0.84rem', 'padding': '4px 0'})])
        for k, v in rows])], style={'width': '100%'})
    notes = [C.note(w, 'warn') for w in warnings]
    return html.Div(notes + [table])


def _job_view(job):
    if job is None:
        hist = JOBS.history()
        if not hist:
            return C.placeholder('Nothing has run from here yet.'), True
        job = hist[0]
    state = (f'⏳ running · {_fmt_secs(job.elapsed)}' if job.running else
             f'{"✅ finished" if job.returncode == 0 else f"❌ exit {job.returncode}"}'
             f' · {_fmt_secs(job.elapsed)}')
    return html.Div([
        html.Div([html.B(job.label), html.Span(f'  {state}',
                                               style={'color': TH.MUTED})],
                 style={'marginBottom': '8px', 'fontSize': '0.86rem'}),
        html.Pre(JOBS.tail(job.log_path) or '(no output yet)', style=PRE),
    ]), not job.running


@callback(
    Output('ctl-status', 'children'), Output('ctl-job', 'children'),
    Output('ctl-routine', 'children'), Output('ctl-history', 'children'),
    Output('ctl-tick', 'disabled'),
    Input('tabs', 'active_tab'), Input('ctl-tick', 'n_intervals'),
    Input('ctl-version', 'data'),
)
def _render(active_tab, _n, _v):
    if active_tab != 'tab-control':
        return no_update, no_update, no_update, no_update, True

    def safe(fn):
        try:
            return fn()
        except Exception as exc:                   # noqa: BLE001
            log.warning('control panel section failed: %s', exc)
            return C.note(f'Unavailable: {type(exc).__name__}: {str(exc)[:120]}', 'warn')

    job = JOBS.current()
    job_div, done = _job_view(job)

    def routine():
        path = JOBS.latest_routine_log()
        if path is None:
            return C.placeholder('No scheduled run has logged here yet. See '
                                 'docs/remote-access.md for the schedule.')
        stamp = config.local_fmt(pd.Timestamp(path.stat().st_mtime, unit='s'))
        return html.Div([html.Div(f'{path.name} · {stamp}',
                                  style={'color': TH.MUTED, 'fontSize': '0.74rem',
                                         'marginBottom': '6px'}),
                         html.Pre(JOBS.tail(path, 25), style=PRE)])

    def hist():
        rows = JOBS.history()[:8]
        if not rows:
            return C.placeholder('None yet.')
        return html.Div([html.Div(
            f'{"✅" if j.returncode == 0 else "❌"} {j.label} · '
            f'{config.local_fmt(pd.Timestamp(j.started, unit="s"), "%b %d %H:%M")} · '
            f'{_fmt_secs(j.elapsed)}', style={'fontSize': '0.8rem', 'marginBottom': '4px'})
            for j in rows])

    return safe(_status_view), job_div, safe(routine), safe(hist), done
