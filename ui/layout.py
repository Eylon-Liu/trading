"""Application shell — header and the six tabs."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ui import components as C
from ui import theme as TH
from ui.pages import (compare, data, intel, methodology, reports, research,
                      screen)

TABS = [
    ('tab-screen', '🎯 Screen', screen),
    ('tab-compare', '🔄 Compare', compare),
    ('tab-research', '🔬 Research', research),
    ('tab-intel', '🔎 Intel', intel),
    ('tab-methodology', '📐 Methodology', methodology),
    ('tab-reports', '📄 Reports', reports),
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


def serve_layout() -> dbc.Container:
    return dbc.Container([
        html.Br(),
        header(),
        dbc.Tabs(
            [dbc.Tab(label=label, tab_id=tab_id) for tab_id, label, _mod in TABS],
            id='tabs', active_tab='tab-screen', className='mb-3'),

        # Every tab is mounted once and shown or hidden with CSS, rather than
        # swapped into a single container. Swapping destroyed the component
        # tree on each switch, so a screen, a backtest or a loaded ticker was
        # thrown away and had to be re-run on return. Building all seven costs
        # under 2ms, which is a trivial price for keeping results alive.
        html.Div([
            html.Div(module.layout(), id=f'panel-{tab_id}',
                     style={} if tab_id == TABS[0][0] else {'display': 'none'})
            for tab_id, _label, module in TABS
        ], id='tab-panels'),

        C.DISCLAIMER,
    ], fluid=True, style={'maxWidth': '1600px'})
