"""Reusable UI building blocks."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from ui import theme as TH


def label(text: str) -> html.Label:
    return html.Label(text, style={'color': TH.MUTED, 'fontSize': '0.74rem',
                                   'fontWeight': '700', 'marginBottom': '5px',
                                   'textTransform': 'uppercase',
                                   'letterSpacing': '0.4px'})


def card(title: str, children, subtitle: str | None = None, **kw) -> dbc.Card:
    header = [html.H6(title, className='mb-0',
                      style={'color': TH.ACCENT, 'fontWeight': '700'})]
    if subtitle:
        header.append(html.Div(subtitle, style={'color': TH.MUTED,
                                                'fontSize': '0.72rem',
                                                'marginTop': '2px'}))
    return dbc.Card([dbc.CardHeader(header), dbc.CardBody(children)], **kw)


# Alias so helpers taking a `card=True/False` flag can still build one.
_card = card


def metric(value, label_text: str, color: str | None = None,
           hint: str | None = None) -> html.Div:
    """A single headline number with its caption."""
    body = [
        html.Div(value, className='metric-value',
                 style={'color': color or TH.TEXT}),
        html.Div(label_text, className='metric-label'),
    ]
    if hint:
        body.append(html.Div(hint, style={'color': TH.MUTED,
                                          'fontSize': '0.66rem',
                                          'marginTop': '3px'}))
    return html.Div(body, style={'textAlign': 'center', 'padding': '6px 4px'})


def metric_row(metrics: list[tuple]) -> dbc.Row:
    """metrics: list of (value, label, color) or (value, label, color, hint)."""
    cols = []
    for m in metrics:
        value, lbl, color = m[0], m[1], m[2]
        hint = m[3] if len(m) > 3 else None
        cols.append(dbc.Col(metric(value, lbl, color, hint)))
    return dbc.Row(cols, className='g-2')


def placeholder(msg: str = 'Run an analysis to populate this view.') -> html.Div:
    return html.Div(msg, style={'color': TH.MUTED, 'padding': '26px 4px',
                                'textAlign': 'center', 'fontSize': '0.86rem'})


def note(msg: str, kind: str = 'info') -> dbc.Alert:
    colors = {'info': 'info', 'warn': 'warning', 'error': 'danger',
              'ok': 'success'}
    return dbc.Alert(msg, color=colors.get(kind, 'info'),
                     className='py-2 px-3 mb-2',
                     style={'fontSize': '0.8rem'})


def data_table(df: pd.DataFrame, columns: list[dict] | None = None,
               page_size: int = 20, table_id: str | None = None,
               extra_conditional: list[dict] | None = None,
               tooltips: dict | None = None,
               cell_tooltips: list[str] | None = None) -> dash_table.DataTable:
    """Themed DataTable with sensible defaults."""
    if df is None or df.empty:
        return dash_table.DataTable(data=[], columns=[])

    if columns is None:
        columns = [{'name': c.replace('_', ' ').title(), 'id': c}
                   for c in df.columns]

    conditional = list(TH.TABLE_CONDITIONAL)
    for col in ('signal',):
        if col in df.columns:
            conditional += TH.signal_style(col)
    if extra_conditional:
        conditional += extra_conditional

    # Budget the width per column rather than letting the widest value decide.
    # Without this a "Information Technology" sector or a reason sentence
    # pushed the whole grid into horizontal scroll.
    cell_conditional = [
        {'if': {'column_id': col}, 'width': w, 'minWidth': w, 'maxWidth': w}
        for col, w in TH.COLUMN_WIDTHS.items() if col in df.columns
    ]
    numeric = [c for c in df.columns
               if c not in TH.COLUMN_WIDTHS and df[c].dtype.kind in 'if']
    cell_conditional += [
        {'if': {'column_id': c}, 'width': '80px', 'minWidth': '66px',
         'maxWidth': '90px', 'textAlign': 'right'} for c in numeric
    ]

    kwargs = {}
    if table_id:
        kwargs['id'] = table_id
    if tooltips:
        kwargs['tooltip_header'] = tooltips
        kwargs['tooltip_delay'] = 400
        kwargs['tooltip_duration'] = None

    if cell_tooltips:
        # Full text on hover for columns whose values are long prose. Keeps
        # the grid to one line per row instead of letting a sentence set the
        # row height for every other column.
        kwargs['tooltip_data'] = [
            {c: {'value': str(row[c]), 'type': 'markdown'}
             for c in cell_tooltips if c in df.columns and pd.notna(row[c])}
            for _i, row in df.iterrows()
        ]
        kwargs['tooltip_delay'] = 200
        kwargs['tooltip_duration'] = None

    return dash_table.DataTable(
        data=df.to_dict('records'), columns=columns,
        sort_action='native', filter_action='none',
        page_size=page_size, style_as_list_view=True,
        style_header=TH.TABLE_HEADER, style_cell=TH.TABLE_CELL,
        style_data_conditional=conditional,
        style_cell_conditional=cell_conditional,
        style_table={'overflowX': 'auto'},
        **kwargs)


def insight_strip(items: list[tuple]) -> dbc.Card:
    """
    The headline read on a result set, above the detail.

    Each item is (value, label, colour, hint). Kept to one row so the first
    thing on screen after a run is the conclusion, not a grid to parse.
    """
    cols = []
    for value, lbl, colour, hint in items:
        cols.append(dbc.Col(html.Div([
            html.Div(str(value), style={
                'fontSize': '1.45rem', 'fontWeight': '800',
                'letterSpacing': '-0.5px', 'color': colour or TH.TEXT,
                'lineHeight': '1.2'}),
            html.Div(lbl, style={
                'fontSize': '0.68rem', 'color': TH.MUTED,
                'textTransform': 'uppercase', 'letterSpacing': '0.6px',
                'marginTop': '3px'}),
            html.Div(hint, style={
                'fontSize': '0.72rem', 'color': TH.MUTED, 'marginTop': '4px',
                'lineHeight': '1.35'}) if hint else None,
        ], className='text-center px-2')))

    return dbc.Card(dbc.CardBody(dbc.Row(cols, className='g-0')),
                    className='mb-3',
                    style={'background': TH.PANEL,
                           'border': f'1px solid {TH.BORDER}'})


def gradient_button(text: str, button_id: str, **kw) -> dbc.Button:
    return dbc.Button(
        text, id=button_id, size='lg',
        style={'background': TH.GRADIENT, 'border': 'none',
               'borderRadius': '28px', 'color': '#000', 'fontWeight': '800',
               'fontSize': '1rem', 'padding': '12px 42px',
               'boxShadow': '0 4px 20px rgba(17,153,142,0.42)'},
        **kw)


def horizon_badge(horizon: str) -> dbc.Badge:
    from quant import strategies as ST
    meta = ST.HORIZONS.get(horizon, {})
    return dbc.Badge(meta.get('label', horizon),
                     color=None,
                     style={'background': TH.HORIZON_COLOR.get(horizon, TH.INFO),
                            'color': '#000', 'fontWeight': '700'})


def sector_bar(counts: pd.Series, title: str = '') -> go.Figure:
    """Horizontal sector distribution, colored by the sector palette."""
    if counts is None or counts.empty:
        return go.Figure(TH.empty_figure('No sector data.'))

    counts = counts.sort_values(ascending=True)
    colors = [TH.SECTOR_COLORS.get(s, '#636e72') for s in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.values, y=list(counts.index), orientation='h',
        marker_color=colors, text=counts.values, textposition='outside',
        textfont={'color': TH.TEXT, 'size': 11},
        hovertemplate='<b>%{y}</b><br>%{x} companies<extra></extra>'))
    fig.update_layout(**TH.plot_layout(
        title, height=max(240, 32 * len(counts) + 60), showlegend=False,
        xaxis={'showgrid': True, 'griddash': 'dash',
               'gridcolor': 'rgba(255,255,255,0.10)', 'title': ''},
        yaxis={'showgrid': False, 'title': ''},
        margin={'l': 150, 'r': 44, 't': 40 if title else 12, 'b': 30}))
    return fig


def equity_curve(strategy: pd.Series, benchmark: pd.Series | None = None,
                 title: str = '') -> go.Figure:
    fig = go.Figure()
    if strategy is not None and len(strategy):
        fig.add_trace(go.Scatter(
            x=strategy.index, y=strategy.values, name='Strategy',
            line={'color': TH.ACCENT, 'width': 2.2},
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.3f}<extra>Strategy</extra>'))
    if benchmark is not None and len(benchmark):
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values, name='Benchmark (SPY)',
            line={'color': TH.MUTED, 'width': 1.6, 'dash': 'dot'},
            hovertemplate='%{x|%Y-%m-%d}<br>%{y:.3f}<extra>Benchmark</extra>'))
    fig.update_layout(**TH.plot_layout(title, height=340,
                                       yaxis={'title': 'Growth of 1.0',
                                              'showgrid': True,
                                              'griddash': 'dash',
                                              'gridcolor': 'rgba(255,255,255,0.10)'}))
    return fig


def drawdown_chart(equity: pd.Series, title: str = '') -> go.Figure:
    if equity is None or len(equity) < 2:
        return go.Figure(TH.empty_figure('No equity curve yet.'))
    dd = (equity / equity.cummax() - 1.0) * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, fill='tozeroy', name='Drawdown',
        line={'color': TH.NEG, 'width': 1.2},
        fillcolor='rgba(255,107,107,0.22)',
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>'))
    fig.update_layout(**TH.plot_layout(title, height=200, showlegend=False,
                                       yaxis={'title': 'Drawdown %',
                                              'showgrid': True,
                                              'griddash': 'dash',
                                              'gridcolor': 'rgba(255,255,255,0.10)'}))
    return fig


def heatmap(matrix: pd.DataFrame, title: str = '') -> go.Figure:
    if matrix is None or matrix.empty:
        return go.Figure(TH.empty_figure('Not enough data for a matrix.'))
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=list(matrix.index),
        colorscale=[[0, TH.NEG], [0.5, '#1a1d2e'], [1, TH.ACCENT]],
        zmid=0, zmin=-1, zmax=1,
        hovertemplate='%{y} vs %{x}<br>corr %{z:.2f}<extra></extra>',
        colorbar={'thickness': 10, 'tickfont': {'size': 9}}))
    n = len(matrix)
    fig.update_layout(**TH.plot_layout(
        title, height=max(320, 22 * n + 130),
        xaxis={'tickangle': -45, 'tickfont': {'size': 9}, 'showgrid': False},
        yaxis={'tickfont': {'size': 9}, 'showgrid': False},
        margin={'l': 160, 'r': 30, 't': 44, 'b': 150}))
    return fig


def loading(children, component_id: str):
    return dcc.Loading(children, id=component_id, type='circle',
                       color=TH.ACCENT)


def ai_panel(generate, title: str = '🤖 AI analysis',
             footnote: str | None = None, card: bool = True, **card_kw):
    """
    Render AI commentary, or an honest explanation of its absence.

    `generate` is a zero-argument callable returning markdown or None. It is
    invoked only when a backend is configured, so pages pay nothing for this
    without a key.

    Every failure mode is visible rather than silent: no key, a key that did
    not produce output, and an exception each render a distinct message. A
    blank space where analysis should be would leave the reader unsure whether
    the model had nothing to say or the call broke.

    `card=False` returns the bare body, for embedding in an accordion.
    """
    from nlp import llm as LLM              # deferred: keeps import cost off page load

    def _wrap(children):
        return _card(title, children, **card_kw) if card else html.Div(children)

    status = LLM.status()
    if not status['enabled']:
        return _wrap([
            note(status['reason'] + '  Add GEMINI_API_KEY (or '
                 'ANTHROPIC_API_KEY) to .env and restart to enable it.', 'info'),
        ])

    try:
        text = generate()
    except Exception as exc:                       # noqa: BLE001
        return _wrap([
            note(f'AI analysis failed: {exc.__class__.__name__}. The '
                 'quantitative output is unaffected.', 'warn'),
        ])

    if not text:
        return _wrap([
            note(f"AI analysis is enabled ({status['model']}) but this request "
                 'returned nothing — check the server log. The quantitative '
                 'output is unaffected.', 'warn'),
        ])

    default_note = (
        f"Generated by {status['model']} from the computed values above — "
        f'interpretation only. Scores, ranks and trade levels come from the '
        f'formulas in the Methodology tab and are unaffected by this panel.')

    return _wrap([
        dcc.Markdown(text, style={'fontSize': '0.87rem', 'lineHeight': '1.65'}),
        html.Div(footnote or default_note,
                 style={'color': TH.MUTED, 'fontSize': '0.72rem',
                        'marginTop': '12px'}),
    ])


DISCLAIMER = html.Div([
    html.Hr(style={'borderColor': '#2a2e42', 'marginTop': '28px'}),
    html.P([
        html.B('⚠️ Disclaimer. '),
        'This is an automated quantitative research tool provided for '
        'informational and educational purposes only. Nothing here is '
        'investment advice or a recommendation to buy or sell any security. '
        'Scores, signals, and any entry, stop or target levels are mechanical '
        'outputs of published rules — not judgements about your circumstances, '
        'objectives or risk tolerance. Backtested results are hypothetical, do '
        'not reflect actual trading, and are no guarantee of future results. '
        'Data comes from free public sources and may be delayed, incomplete or '
        'wrong. Do your own research and consider consulting a licensed '
        'financial adviser.',
    ], style={'color': '#6b7280', 'fontSize': '0.7rem', 'lineHeight': '1.55',
              'paddingBottom': '24px'}),
])
