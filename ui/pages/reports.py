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

        C.card('📧 Email setup', [
            html.Div(id='rep-smtp-status', className='mb-3'),
            dbc.Accordion([
                dbc.AccordionItem(_smtp_help(), title='How to set this up',
                                  item_id='smtp'),
            ], start_collapsed=True, flush=True),
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


SMTP_ENV = """\
# ── in .env, alongside the other settings ─────────────────────
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=abcdefghijklmnop        # 16-char App Password, NOT your login
"""

PROVIDERS = [
    ('Gmail', 'smtp.gmail.com', '587',
     'Requires 2-Step Verification, then an App Password.'),
    ('Outlook / Microsoft 365', 'smtp.office365.com', '587',
     'Use an app password if MFA is on.'),
    ('iCloud Mail', 'smtp.mail.me.com', '587',
     'Requires an app-specific password.'),
    ('Fastmail', 'smtp.fastmail.com', '587', 'Create an app password.'),
]


def _smtp_help():
    return html.Div([
        html.P('Reports are sent over SMTP from your own mailbox — there is no '
               'third-party mail service and nothing leaves this machine except '
               'the message itself.',
               style={'fontSize': '0.85rem'}),

        html.B('1. Put credentials in .env',
               style={'color': TH.ACCENT, 'fontSize': '0.86rem'}),
        html.Pre(SMTP_ENV, style={'background': TH.PANEL_ALT, 'padding': '13px',
                                  'borderRadius': '7px', 'fontSize': '0.75rem',
                                  'color': TH.TEXT, 'overflowX': 'auto',
                                  'margin': '6px 0 14px'}),

        html.B('2. For Gmail, create an App Password',
               style={'color': TH.ACCENT, 'fontSize': '0.86rem'}),
        html.Ol([
            html.Li('Turn on 2-Step Verification in your Google account.'),
            html.Li(['Go to ', html.A('myaccount.google.com/apppasswords',
                                      href='https://myaccount.google.com/apppasswords',
                                      target='_blank'),
                     ' and create a password for “Mail”.']),
            html.Li('Paste the 16-character value as SMTP_PASS.'),
        ], style={'fontSize': '0.83rem', 'lineHeight': '1.7'}),
        C.note('Your normal Google password will not work, and Google will '
               'reject it — an App Password is required whenever 2-Step '
               'Verification is on.', 'warn'),

        html.B('3. Restart the app, then test',
               style={'color': TH.ACCENT, 'fontSize': '0.86rem',
                      'display': 'block', 'marginTop': '14px'}),
        html.Pre('python cli.py test-email --to you@example.com',
                 style={'background': TH.PANEL_ALT, 'padding': '11px 13px',
                        'borderRadius': '7px', 'fontSize': '0.76rem',
                        'color': TH.TEXT, 'margin': '6px 0 14px'}),

        html.B('Common SMTP servers',
               style={'color': TH.ACCENT, 'fontSize': '0.86rem'}),
        C.data_table(pd.DataFrame([
            {'provider': p, 'SMTP_SERVER': s, 'SMTP_PORT': port, 'note': n}
            for p, s, port, n in PROVIDERS]), page_size=6),

        html.Div('.env is gitignored, so these credentials are never committed.',
                 style={'color': TH.MUTED, 'fontSize': '0.74rem',
                        'marginTop': '10px'}),
    ])


@callback(Output('rep-smtp-status', 'children'),
          Input('tabs', 'active_tab'),
          State('rep-smtp-status', 'children'))
def _smtp_status(active_tab, existing):
    if active_tab != 'tab-reports' or existing:
        return no_update

    configured = bool(config.SMTP_USER and config.SMTP_PASS)
    if configured:
        return html.Div([
            dbc.Badge('CONFIGURED', color='success', className='me-2'),
            html.Span(f'Sending as {config.SMTP_USER} via '
                      f'{config.SMTP_SERVER}:{config.SMTP_PORT}',
                      style={'fontSize': '0.84rem'}),
        ])

    missing = [n for n, v in (('SMTP_USER', config.SMTP_USER),
                              ('SMTP_PASS', config.SMTP_PASS)) if not v]
    return html.Div([
        dbc.Badge('NOT CONFIGURED', color='secondary', className='me-2'),
        html.Span(f'Missing {" and ".join(missing)} in .env — reports still '
                  f'generate and can be read here, only sending is disabled.',
                  style={'fontSize': '0.84rem'}),
    ])


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
    df['on disk'] = df['path'].apply(
        lambda p: '✓' if Path(p).exists() else 'missing')

    return html.Div([
        C.data_table(df[['kind', 'file', 'created', 'emailed', 'on disk']],
                     page_size=10, table_id='rep-history-table'),
        html.Div('Click any row to open that report below.',
                 style={'color': TH.MUTED, 'fontSize': '0.74rem',
                        'marginTop': '8px'}),
    ])


@callback(
    Output('rep-preview', 'children', allow_duplicate=True),
    Input('rep-history-table', 'active_cell'),
    prevent_initial_call=True,
)
def _open_past_report(active_cell):
    """
    Load a stored report into the preview pane.

    The history table used to be inert — clicking a row did nothing, because
    nothing listened for it. Reports are written to disk, so opening one is a
    file read rather than a rebuild.
    """
    if not active_cell:
        return no_update

    try:
        df = RB.list_reports(limit=25)
    except Exception as exc:                       # noqa: BLE001
        return C.note(f'Could not list reports: {exc}', 'error')

    row = active_cell.get('row')
    if row is None or row >= len(df):
        return no_update

    rec = df.iloc[row]
    path = Path(rec['path'])
    name = path.name

    if not path.exists():
        return C.note(
            f'“{name}” is recorded in the database but the file is gone from '
            f'{path.parent}. Regenerate it with the button above.', 'warn')

    try:
        html_doc = path.read_text(encoding='utf-8')
    except Exception as exc:                       # noqa: BLE001
        return C.note(f'Could not read {name}: {exc}', 'error')

    return C.card(f'👁️ {name}', [
        html.Div([
            dbc.Badge(rec['kind'], color='dark', className='me-2',
                      style={'border': f'1px solid {TH.BORDER}'}),
            html.Span(f"created {pd.to_datetime(rec['created_at']):%Y-%m-%d %H:%M}",
                      style={'color': TH.MUTED, 'fontSize': '0.76rem'}),
            html.A('⬇ open in a new tab', href=f'/reports/{name}',
                   target='_blank',
                   style={'marginLeft': '14px', 'fontSize': '0.78rem'}),
        ], className='mb-2'),
        html.Iframe(srcDoc=html_doc,
                    style={'width': '100%', 'height': '720px', 'border': 'none',
                           'borderRadius': '8px', 'background': TH.BG}),
    ], className='mb-3')
