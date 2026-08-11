"""
Strategy builder — copy a preset, then tune it.

The built-in strategies are a starting set. Several are close cousins, and the
useful move is usually "that one, but weighted differently" rather than
picking from a fixed menu. So: copy any preset into an editable strategy,
rename it, drag the weights, add or drop factors.

Weights are edited as *shares of the score* rather than raw numbers. Raw
weights are unitless and only mean anything relative to their neighbours, so
changing one silently changes every other factor's influence — which makes the
controls feel broken. Editing shares keeps the change where the user put it.

Copies are snapshots. Editing a preset later does not reach into strategies
built from it, because a saved strategy that changes underneath you is not
worth saving.
"""

from __future__ import annotations

import logging

import dash_bootstrap_components as dbc
import pandas as pd
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update

from quant import custom as CU
from quant import factor_docs as FD
from quant import factors as FA
from quant import strategies as ST
from quant import strategy_notes as SN
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)


def _factor_options():
    """Every known factor, grouped by family in the label."""
    docs = {d.name: d for d in FD.FACTOR_DOCS.values()}
    opts = []
    for name in sorted(FA.FACTOR_DIRECTION):
        d = docs.get(name)
        label = f'{d.label}  ·  {d.family}' if d else name
        opts.append({'label': label, 'value': name})
    return opts


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id='bld-weights'),
        dcc.Store(id='bld-key'),

        C.card('🧪 Build your own strategy', [
            C.note('Copy a preset, then change it. The original is never '
                   'modified — built-ins stay as documented so the Methodology '
                   'tab keeps telling the truth.', 'info'),
            dbc.Row([
                dbc.Col([
                    C.label('Start from'),
                    dcc.Dropdown(id='bld-source', options=ST.options(),
                                 value='quality_value', clearable=False),
                ], lg=5, md=6, className='mb-2'),
                dbc.Col([
                    C.label('Call it'),
                    dbc.Input(id='bld-name', placeholder='My Quality Tilt',
                              type='text'),
                ], lg=4, md=6, className='mb-2'),
                dbc.Col([
                    dbc.Button('📋 Copy & edit', id='bld-copy', color='success',
                               className='w-100'),
                ], lg=3, md=12, className='mb-2'),
            ], className='align-items-end'),
            html.Div(id='bld-status',
                     style={'fontSize': '0.84rem', 'marginTop': '10px'}),
        ], className='mb-3'),

        html.Div(id='bld-editor'),

        C.card('⭐ Your strategies', html.Div(id='bld-list'), className='mb-3'),
    ])


# ─────────────────────────────────────────────
# COPY
# ─────────────────────────────────────────────

@callback(
    Output('bld-weights', 'data'), Output('bld-key', 'data'),
    Output('bld-status', 'children'), Output('bld-list', 'children'),
    Input('bld-copy', 'n_clicks'),
    State('bld-source', 'value'), State('bld-name', 'value'),
    prevent_initial_call=True,
)
def _copy(n_clicks, source, name):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    try:
        key = CU.copy_from(source, name or f'My {ST.get(source).name}')
    except CU.ValidationError as exc:
        return no_update, no_update, f'❌ {exc}', _saved_list()
    except Exception as exc:                       # noqa: BLE001
        log.exception('copy failed')
        return no_update, no_update, f'❌ {type(exc).__name__}: {exc}', _saved_list()

    s = CU.resolve(key)
    return (dict(s.weights), key,
            f'✅ Created “{s.name}”. Tune it below, then save.',
            _saved_list())


# ─────────────────────────────────────────────
# EDITOR
# ─────────────────────────────────────────────

@callback(Output('bld-editor', 'children'),
          Input('bld-weights', 'data'), State('bld-key', 'data'))
def _render_editor(weights, key):
    if not weights or not key:
        return C.placeholder('Copy a preset above to start editing.')

    try:
        strat = CU.resolve(key)
    except KeyError:
        return C.placeholder('That strategy no longer exists.')

    shares = CU.normalized(weights)
    docs = {d.name: d for d in FD.FACTOR_DOCS.values()}

    rows = []
    for factor, share in sorted(shares.items(), key=lambda kv: -kv[1]):
        d = docs.get(factor)
        direction = FA.FACTOR_DIRECTION.get(factor, True)
        rows.append(dbc.Row([
            dbc.Col([
                html.Div(d.label if d else factor,
                         style={'fontSize': '0.87rem', 'fontWeight': '600'}),
                html.Div([
                    html.Span(d.family if d else '—',
                              style={'color': TH.MUTED}),
                    html.Span('  ·  higher is better' if direction
                              else '  ·  lower is better',
                              style={'color': TH.POS if direction else TH.WARN}),
                ], style={'fontSize': '0.72rem'}),
            ], lg=4, md=5, xs=12),
            dbc.Col(
                dcc.Slider(id={'type': 'bld-w', 'factor': factor},
                           min=0, max=60, step=1, value=round(share),
                           marks=None,
                           tooltip={'placement': 'bottom', 'always_visible': True}),
                lg=6, md=5, xs=9),
            dbc.Col(
                dbc.Button('✕', id={'type': 'bld-rm', 'factor': factor},
                           size='sm', color='dark',
                           style={'border': f'1px solid {TH.BORDER}'}),
                lg=2, md=2, xs=3, className='text-end'),
        ], className='mb-3 align-items-center'))

    total = sum(shares.values())
    return html.Div([
        C.card(f'🎛️ {strat.name}', [
            dbc.Row([
                dbc.Col([
                    C.label('Add a factor'),
                    dcc.Dropdown(id='bld-add', options=_factor_options(),
                                 placeholder='Pick a factor to add…'),
                ], lg=8, md=8, className='mb-3'),
                dbc.Col([
                    C.label('Horizon'),
                    html.Div(C.horizon_badge(strat.horizon)),
                ], lg=4, md=4, className='mb-3'),
            ]),
            html.Hr(style={'borderColor': TH.BORDER}),
            html.Div(rows),
            html.Div(f'Shares total {total:.0f}% — they are renormalised on '
                     f'save, so only the proportions matter.',
                     style={'color': TH.MUTED, 'fontSize': '0.75rem',
                            'marginTop': '4px'}),
            html.Hr(style={'borderColor': TH.BORDER}),
            dbc.Row([
                dbc.Col(dbc.Button('💾 Save', id='bld-save', color='success',
                                   className='w-100'), lg=3, md=4),
                dbc.Col(dbc.Button('🗑️ Delete', id='bld-delete', color='danger',
                                   outline=True, className='w-100'), lg=3, md=4),
                dbc.Col(html.Div(id='bld-save-status',
                                 style={'fontSize': '0.83rem'}),
                        lg=6, md=4),
            ], className='g-2'),
        ], className='mb-3'),
        _inherited_notes(strat),
    ])


def _inherited_notes(strat):
    """Carry the parent's trade-offs across, clearly labelled as inherited."""
    base = None
    try:
        row = db_lookup(strat.key)
        base = row.get('based_on') if row else None
    except Exception:                              # noqa: BLE001
        base = None
    note = SN.get(base) if base else None
    if note is None:
        return None
    return C.card('⚠️ Inherited trade-offs', [
        C.note(f'Copied from {ST.get(base).name}. Changing weights changes '
               f'these — a heavier value tilt buys the value failure mode '
               f'along with the value premium.', 'warn'),
        html.Div([html.B('Drawback: ', style={'color': TH.WARN}),
                  html.Span(note.drawback)],
                 style={'fontSize': '0.88rem', 'lineHeight': '1.7',
                        'marginBottom': '8px'}),
        html.Div([html.B('How it fails: ', style={'color': TH.NEG}),
                  html.Span(note.risk)],
                 style={'fontSize': '0.88rem', 'lineHeight': '1.7'}),
    ], className='mb-3')


def db_lookup(key: str) -> dict | None:
    from core import db
    df = db.read_sql('SELECT * FROM custom_strategies WHERE key = :k', {'k': key})
    return None if df.empty else df.iloc[0].to_dict()


# ─────────────────────────────────────────────
# EDITS
# ─────────────────────────────────────────────

@callback(
    Output('bld-weights', 'data', allow_duplicate=True),
    Input({'type': 'bld-w', 'factor': ALL}, 'value'),
    Input({'type': 'bld-rm', 'factor': ALL}, 'n_clicks'),
    Input('bld-add', 'value'),
    State('bld-weights', 'data'),
    prevent_initial_call=True,
)
def _edit(slider_values, remove_clicks, add_factor, weights):
    if not weights:
        return no_update
    trigger = ctx.triggered_id
    if not trigger:
        return no_update

    if trigger == 'bld-add':
        if not add_factor or add_factor in weights:
            return no_update
        # New factors enter at roughly an equal share of the existing spread.
        avg = (sum(weights.values()) / len(weights)) if weights else 1.0
        return {**weights, add_factor: round(avg, 4)}

    if isinstance(trigger, dict) and trigger.get('type') == 'bld-rm':
        factor = trigger['factor']
        if factor not in weights or len(weights) <= 1:
            return no_update
        return {k: v for k, v in weights.items() if k != factor}

    if isinstance(trigger, dict) and trigger.get('type') == 'bld-w':
        factor = trigger['factor']
        for item, value in zip(ctx.inputs_list[0], slider_values):
            if item['id'].get('factor') == factor and value is not None:
                return CU.rebalanced(weights, factor, value)

    return no_update


@callback(
    Output('bld-save-status', 'children'),
    Output('bld-list', 'children', allow_duplicate=True),
    Input('bld-save', 'n_clicks'),
    State('bld-key', 'data'), State('bld-weights', 'data'),
    prevent_initial_call=True,
)
def _save(n_clicks, key, weights):
    if not n_clicks or not key or not weights:
        return no_update, no_update
    try:
        existing = db_lookup(key) or {}
        CU.save(existing.get('name') or key, existing.get('horizon', 'long'),
                weights, description=existing.get('description', ''),
                thesis=existing.get('thesis', ''),
                based_on=existing.get('based_on'),
                neutralize=existing.get('neutralize', 'sector_z'),
                setup=existing.get('setup', 'momentum'), key=key)
    except CU.ValidationError as exc:
        return f'❌ {exc}', no_update
    except Exception as exc:                       # noqa: BLE001
        log.exception('save failed')
        return f'❌ {type(exc).__name__}: {exc}', no_update
    return ('✅ Saved. It now appears in the Screen and Compare dropdowns.',
            _saved_list())


@callback(
    Output('bld-save-status', 'children', allow_duplicate=True),
    Output('bld-weights', 'data', allow_duplicate=True),
    Output('bld-list', 'children', allow_duplicate=True),
    Input('bld-delete', 'n_clicks'),
    State('bld-key', 'data'),
    prevent_initial_call=True,
)
def _delete(n_clicks, key):
    if not n_clicks or not key:
        return no_update, no_update, no_update
    try:
        CU.delete(key)
    except CU.ValidationError as exc:
        return f'❌ {exc}', no_update, no_update
    return '🗑️ Deleted.', None, _saved_list()


def _saved_list():
    df = CU.summary()
    if df.empty:
        return C.placeholder('None yet — copy a preset above.')
    return html.Div([
        C.data_table(df[['name', 'horizon', 'factors']], page_size=10),
        html.Div('Saved strategies appear alongside the built-ins on the '
                 'Screen, Compare and Research tabs, marked with a ★.',
                 style={'color': TH.MUTED, 'fontSize': '0.75rem',
                        'marginTop': '8px'}),
    ])


@callback(Output('bld-list', 'children', allow_duplicate=True),
          Input('tabs', 'active_tab'), prevent_initial_call=True)
def _refresh_list(active_tab):
    if active_tab != 'tab-builder':
        return no_update
    return _saved_list()
