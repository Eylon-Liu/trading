"""Reports tab — generate, preview and email daily or monthly briefs."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

import config
from quant import strategies as ST
from reports import builder as RB
from reports import email as MAILER
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)

CRON_HELP = """\
# Nightly ingest at 6pm, daily brief at 7am (crontab -e)
0 18 * * 1-5  cd "{root}" && /usr/bin/python3 cli.py ingest --index SPY --full
0  7 * * 1-5  cd "{root}" && /usr/bin/python3 cli.py report --kind daily \\
                  --strategy quality_value --email you@example.com

# Monthly review on the 1st
0  8 1 * *    cd "{root}" && /usr/bin/python3 cli.py report --kind monthly \\
                  --strategy quality_value --email you@example.com
"""


def layout() -> html.Div:
    return html.Div([
        C.card('📄 Generate a report', [
            dbc.Row([
                dbc.Col([
                    C.label('Kind'),
                    dcc.Dropdown(id='rep-kind',
                                 options=[{'label': 'Daily brief', 'value': 'daily'},
                                          {'label': 'Monthly review', 'value': 'monthly'}],
                                 value='daily', clearable=False),
                ], lg=3, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Strategy'),
                    dcc.Dropdown(id='rep-strategy', options=ST.options(),
                                 value='quality_value', clearable=False),
                ], lg=3, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Email to (optional)'),
                    dbc.Input(id='rep-email', type='email',
                              placeholder='you@example.com'),
                ], lg=4, md=8, className='mb-3'),
                dbc.Col([
                    dbc.Button('Build', id='rep-button', color='success',
                               className='w-100'),
                ], lg=2, md=4, className='mb-3'),
            ], className='align-items-end'),
            C.note('Reports are rendered from stored runs rather than refetched, '
                   'so they are fast and reproducible. Run a screen first if the '
                   'strategy has no stored run yet.', 'info'),
            html.Div(id='rep-status', style={'color': TH.MUTED,
                                             'fontSize': '0.84rem'}),
        ], className='mb-3'),

        html.Div(id='rep-preview'),

        C.card('🗂️ Previous reports', html.Div(id='rep-history'),
               className='mb-3'),

        C.card('⏰ Run it on a schedule', [
            html.P('The CLI does everything this page does, so scheduling is a '
                   'cron line. On macOS, launchd works too.',
                   style={'fontSize': '0.84rem'}),
            html.Pre(CRON_HELP.format(root=config.ROOT),
                     style={'background': TH.PANEL_ALT, 'padding': '14px',
                            'borderRadius': '8px', 'fontSize': '0.74rem',
                            'color': TH.TEXT, 'overflowX': 'auto'}),
        ], className='mb-3'),
    ])


@callback(
    Output('rep-status', 'children'), Output('rep-preview', 'children'),
    Output('rep-history', 'children'),
    Input('rep-button', 'n_clicks'),
    State('rep-kind', 'value'), State('rep-strategy', 'value'),
    State('rep-email', 'value'),
    prevent_initial_call=True,
)
def _build(n_clicks, kind, strategy, email):
    if not n_clicks:
        return no_update, no_update, no_update

    try:
        path, html_doc = RB.build_report(strategy, kind=kind, as_of=date.today())
    except Exception as exc:                       # noqa: BLE001
        log.exception('report build failed')
        return f'❌ {type(exc).__name__}: {str(exc)[:200]}', None, _history()

    status = f'✅ Written to {path}'
    if email:
        try:
            MAILER.send_report(html_doc, email,
                               subject=f'{kind.title()} Brief — {strategy}')
            status += f'  ·  📧 emailed to {email}'
        except MAILER.SMTPNotConfigured as exc:
            status += f'  ·  ⚠️ email skipped: {exc}'
        except Exception as exc:                   # noqa: BLE001
            status += f'  ·  ⚠️ email failed: {str(exc)[:120]}'

    preview = C.card('👁️ Preview', html.Iframe(
        srcDoc=html_doc,
        style={'width': '100%', 'height': '720px', 'border': 'none',
               'borderRadius': '8px', 'background': TH.BG}),
        className='mb-3')

    return status, preview, _history()


@callback(Output('rep-history', 'children', allow_duplicate=True),
          Input('tabs', 'active_tab'), prevent_initial_call=True)
def _refresh_history(active_tab):
    if active_tab != 'tab-reports':
        return no_update
    return _history()


def _history():
    try:
        df = RB.list_reports(limit=25)
    except Exception:                              # noqa: BLE001
        return C.placeholder('No reports yet.')
    if df.empty:
        return C.placeholder('No reports generated yet.')

    df = df.copy()
    df['file'] = df['path'].apply(lambda p: Path(p).name)
    df['created'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
    df['emailed'] = df['emailed_at'].notna().map({True: '📧', False: '—'})
    return C.data_table(df[['kind', 'file', 'created', 'emailed']], page_size=10)
