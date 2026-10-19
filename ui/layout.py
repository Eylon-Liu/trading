"""Application shell — header and the eight tabs."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

import config
from ui import components as C
from ui import theme as TH
from ui.pages import (builder, compare, data, intel, methodology, reports,
                      research, screen)

TABS = [
    ('tab-screen', '🎯 Screen', screen),
    ('tab-intel', '🔎 Intel', intel),
    ('tab-compare', '🔄 Compare', compare),
    ('tab-reports', '📄 Reports', reports),
    ('tab-research', '📈 Backtest', research),
    ('tab-builder', '🧪 Builder', builder),
    ('tab-methodology', '📐 Methodology', methodology),
    ('tab-data', '🗄️ Data', data),
]


def header() -> dbc.Card:
    return dbc.Card(dbc.CardBody(html.Div([
        html.H1('📊 Quant Research Terminal',
                className='mb-1',
                style={'color': TH.ACCENT, 'fontWeight': '800',
                       'letterSpacing': '-0.7px', 'fontSize': '1.9rem'}),
        html.P('Point-in-time equity research — screening, backtesting, '
               'filings intelligence and policy tracking',
               className='mb-0',
               style={'color': TH.MUTED, 'fontSize': '0.9rem'}),
    ], className='text-center py-3')),
        className='mb-3',
        style={'background': f'linear-gradient(135deg,{TH.BG} 0%,{TH.PANEL} 100%)',
               'border': '1px solid rgba(17,153,142,0.32)'})


def demo_banner() -> html.Div | None:
    """Say plainly what a public deployment can and cannot do.

    A visitor who clicks Pull and gets nothing should not have to guess
    whether the app is broken or deliberately limited.
    """
    if not config.DEMO_MODE:
        return None

    return html.Div([
        html.Span('🔒 Read-only demo. ', style={'fontWeight': '700'}),
        html.Span(
            'Screening, backtesting, filings intelligence and reports all run '
            'on a point-in-time database snapshot shipped with this deployment. '
            'Live data ingestion, report emailing and new AI generation are '
            'disabled, because an unauthenticated page cannot spend the '
            "deployer's API quota and send mail on request. Everything works "
            'without limits when you run it yourself.'),
    ], style={
        'background': 'rgba(0,212,255,0.08)',
        'border': f'1px solid {TH.INFO}',
        'borderRadius': '8px',
        'color': TH.TEXT,
        'fontSize': '0.8rem',
        'lineHeight': '1.6',
        'padding': '10px 14px',
        'marginBottom': '12px',
    })


def serve_layout() -> dbc.Container:
    return dbc.Container([
        html.Br(),
        header(),
        demo_banner(),
        dbc.Tabs(
            [dbc.Tab(label=label, tab_id=tab_id) for tab_id, label, _mod in TABS],
            id='tabs', active_tab='tab-screen', className='mb-3'),

        # Every tab is mounted once and shown or hidden with CSS, rather than
        # swapped into a single container. Swapping destroyed the component
        # tree on each switch, so a screen, a backtest or a loaded ticker was
        # thrown away and had to be re-run on return. Building all eight costs
        # under 2ms, which is a trivial price for keeping results alive.
        html.Div([
            html.Div(module.layout(), id=f'panel-{tab_id}',
                     style={} if tab_id == TABS[0][0] else {'display': 'none'})
            for tab_id, _label, module in TABS
        ], id='tab-panels'),

        C.DISCLAIMER,
    ], fluid=True, style={'maxWidth': '1600px'})
