"""
Methodology tab — how every number in this app is computed.

Renders quant/factor_docs.py verbatim: the shared scoring pipeline, the exact
formula behind each factor, the trade-plan arithmetic, and a per-strategy
weight breakdown. A score you cannot reproduce by hand is a score you should
not act on.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dcc, html

from quant import factor_docs as FD
from quant import strategies as ST
from ui import components as C
from ui import theme as TH

MONO = {'fontFamily': 'ui-monospace, SFMono-Regular, Menlo, monospace',
        'fontSize': '0.76rem', 'whiteSpace': 'pre', 'overflowX': 'auto',
        'background': TH.PANEL_ALT, 'padding': '12px 14px',
        'borderRadius': '6px', 'color': TH.TEXT, 'lineHeight': '1.5',
        'margin': '0'}

FAMILY_COLOR = {
    'Value': '#38ef7d', 'Quality': '#00d4ff', 'Growth': '#c77dff',
    'Momentum': '#ffd93d', 'Trend': '#ffb74d', 'Risk': '#ff6b9d',
    'Income': '#06d6a0', 'Alt Data': '#a29bfe', 'Technical': '#f7c59f',
}


def layout() -> html.Div:
    return html.Div([
        C.card('📐 How the scoring works', [
            C.note('Every strategy below is a weighted blend of the same factor '
                   'library, put through one shared pipeline. Nothing here is a '
                   'black box — the formulas are the implementation.', 'info'),
            html.Pre(FD.PIPELINE.strip(), style=MONO),
        ], className='mb-3'),

        C.card('🎯 Strategies', [
            html.Div([
                C.label('Show'),
                dcc.RadioItems(
                    id='meth-horizon',
                    options=[{'label': '  All', 'value': 'all'},
                             {'label': '  Long-term only', 'value': 'long'},
                             {'label': '  Mid-term only', 'value': 'mid'}],
                    value='all', inline=True,
                    inputStyle={'marginRight': '5px'},
                    labelStyle={'marginRight': '18px', 'fontSize': '0.84rem'}),
            ], className='mb-3'),
            html.Div(id='meth-strategies'),
        ], className='mb-3'),

        C.card('🧮 Trade-plan mathematics (mid-term)', [
            C.note('Long-term strategies emit no price levels — the exit is a '
                   'broken thesis. Everything below applies only to the mid-term '
                   'horizon.', 'info'),
            html.Pre(FD.TRADE_PLAN_MATH.strip(), style=MONO),
        ], className='mb-3'),

        C.card('📚 Factor reference', [
            # Built here rather than in a callback: the content is a pure
            # function of the factor docs, so rendering it once keeps any
            # accordion the reader opened from collapsing on a tab switch.
            _factor_reference(),
        ], className='mb-3'),
    ])


def _factor_body(d: FD.FactorDoc) -> html.Div:
    bits = [
        html.Pre(d.formula, style={**MONO, 'marginBottom': '10px'}),
        html.Div([
            html.B('Inputs: ', style={'color': TH.MUTED}),
            html.Span(d.inputs, style={'color': TH.TEXT}),
        ], style={'fontSize': '0.8rem', 'marginBottom': '6px'}),
        html.Div([
            html.B('Direction: ', style={'color': TH.MUTED}),
            html.Span(f'{d.direction} is better',
                      style={'color': TH.POS if d.direction == 'higher' else TH.WARN}),
        ], style={'fontSize': '0.8rem', 'marginBottom': '10px'}),
        html.Div(d.rationale, style={'fontSize': '0.84rem', 'lineHeight': '1.6',
                                     'marginBottom': '8px'}),
    ]
    if d.caveat:
        bits.append(html.Div([
            html.B('⚠️ Caveat: ', style={'color': TH.WARN}),
            html.Span(d.caveat, style={'color': TH.MUTED}),
        ], style={'fontSize': '0.8rem', 'lineHeight': '1.55'}))
    return html.Div(bits)


def _factor_reference() -> html.Div:
    sections = []
    for family, docs in FD.by_family().items():
        colour = FAMILY_COLOR.get(family, TH.ACCENT)
        items = [
            dbc.AccordionItem(
                _factor_body(d),
                title=f'{d.label}  ·  {d.name}',
                item_id=f'f-{d.name}',
            ) for d in sorted(docs, key=lambda x: x.label)
        ]
        sections.append(html.Div([
            html.Div(f'{family}  ({len(docs)})',
                     style={'color': colour, 'fontWeight': '800',
                            'fontSize': '0.9rem', 'margin': '18px 0 8px',
                            'letterSpacing': '0.3px'}),
            dbc.Accordion(items, start_collapsed=True, always_open=True,
                          flush=True),
        ]))
    return html.Div(sections)


@callback(Output('meth-strategies', 'children'),
          Input('meth-horizon', 'value'))
def _render_strategies(horizon):
    pool = (ST.ALL_STRATEGIES if horizon == 'all'
            else ST.by_horizon(horizon))

    items = []
    for key, strat in sorted(pool.items(), key=lambda kv: (kv[1].horizon, kv[0])):
        rows = FD.strategy_breakdown(strat)
        table = pd.DataFrame([{
            'Factor': r['label'],
            'Family': r['family'],
            'Weight': r['weight'],
            'Share %': r['weight_pct'],
            'Direction': r['direction'],
        } for r in rows])

        filters = strat.filters or {}
        filter_text = ', '.join(f'{k} = {v}' for k, v in filters.items()) or 'none'

        formula_lines = []
        for r in rows:
            formula_lines.append(f"  {r['weight_pct']:>5.1f}%  {r['label']}")
        composite = (
            'composite = weighted mean of sector-relative z-scores\n\n'
            + '\n'.join(formula_lines)
            + f"\n\n  neutralization : {strat.neutralize}"
            + f"\n  hard filters   : {filter_text}"
            + (f"\n  trade setup    : {strat.setup}" if strat.needs_trade_plan else '')
        )

        body = html.Div([
            html.Div(strat.description,
                     style={'fontSize': '0.9rem', 'marginBottom': '8px'}),
            html.Div([html.B('Thesis: ', style={'color': TH.ACCENT}),
                      html.Span(strat.thesis)],
                     style={'fontSize': '0.84rem', 'lineHeight': '1.6',
                            'marginBottom': '12px'}),
            html.Div([
                C.horizon_badge(strat.horizon),
                dbc.Badge(
                    'trade plan: entry / stop / target'
                    if strat.needs_trade_plan else 'exit: thesis-based, no levels',
                    color='dark', className='ms-2',
                    style={'border': f'1px solid {TH.BORDER}'}),
                dbc.Badge(f'{len(strat.weights)} factors', color='dark',
                          className='ms-2',
                          style={'border': f'1px solid {TH.BORDER}'}),
            ], className='mb-3'),
            html.Pre(composite, style=MONO),
            html.Div(style={'height': '10px'}),
            C.data_table(table, page_size=15),
        ])

        items.append(dbc.AccordionItem(
            body, title=f'{strat.name}   ({ST.HORIZONS[strat.horizon]["label"]})',
            item_id=f's-{key}'))

    return dbc.Accordion(items, start_collapsed=True, always_open=True, flush=True)
