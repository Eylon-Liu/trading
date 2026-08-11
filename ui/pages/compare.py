"""
Compare tab — what changed between two stored runs.

This is the payoff for persisting every run: "how does today differ from a
month ago" becomes a join rather than a recomputation, and the answer is
reproducible because both sides are snapshots rather than re-derived numbers.
"""

from __future__ import annotations

import logging

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from core import db
from nlp import llm as LLM
from quant import research as RS
from quant import strategies as ST
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)


def layout() -> html.Div:
    return html.Div([
        C.card('🔄 Compare two runs', [
            dbc.Row([
                dbc.Col([
                    C.label('Strategy'),
                    dcc.Dropdown(id='cmp-strategy', options=ST.options(),
                                 value='quality_value', clearable=False),
                ], lg=4, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Earlier run'),
                    dcc.Dropdown(id='cmp-run-a', placeholder='Select a run'),
                ], lg=4, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Later run'),
                    dcc.Dropdown(id='cmp-run-b', placeholder='Select a run'),
                ], lg=4, md=6, className='mb-3'),
            ]),
            dbc.Button('Compare', id='cmp-button', color='success',
                       className='mt-1'),
            html.Div(id='cmp-status', style={'color': TH.MUTED,
                                             'fontSize': '0.8rem',
                                             'marginTop': '8px'}),
        ], className='mb-3'),
        html.Div(id='cmp-results'),
    ])


def _run_options(strategy: str) -> list[dict]:
    df = db.read_sql("""
        SELECT run_id, as_of, universe_n, created_at FROM runs
        WHERE strategy = :s ORDER BY as_of DESC, created_at DESC LIMIT 60
    """, {'s': strategy})
    if df.empty:
        return []
    return [{'label': f'{r["as_of"]}  ({r["universe_n"]} names)',
             'value': r['run_id']} for _i, r in df.iterrows()]


@callback(
    Output('cmp-run-a', 'options'), Output('cmp-run-b', 'options'),
    Output('cmp-run-a', 'value'), Output('cmp-run-b', 'value'),
    Input('cmp-strategy', 'value'), Input('tabs', 'active_tab'),
)
def _populate_runs(strategy, active_tab):
    if not strategy:
        return [], [], None, None
    opts = _run_options(strategy)
    if len(opts) < 2:
        return opts, opts, None, (opts[0]['value'] if opts else None)
    # Default to comparing the newest against the next-oldest available.
    return opts, opts, opts[-1]['value'], opts[0]['value']


@callback(
    Output('cmp-results', 'children'), Output('cmp-status', 'children'),
    Input('cmp-button', 'n_clicks'),
    State('cmp-run-a', 'value'), State('cmp-run-b', 'value'),
    prevent_initial_call=True,
)
def _compare(n_clicks, run_a, run_b):
    if not n_clicks:
        return no_update, ''
    if not run_a or not run_b:
        return None, '⚠️ Pick two runs. Run the same strategy on two dates first.'
    if run_a == run_b:
        return None, '⚠️ Those are the same run.'

    try:
        comp = RS.compare_runs(run_a, run_b)
    except Exception as exc:                       # noqa: BLE001
        log.exception('compare failed')
        return None, f'❌ {exc}'

    if comp.empty:
        return None, '⚠️ No overlapping data between those runs.'

    meta = _run_meta(run_a, run_b)
    comp = comp.reset_index()
    entered = comp[comp['status'] == 'entered']
    dropped = comp[comp['status'] == 'dropped']
    held = comp[comp['status'] == 'held']
    changed = held[held['signal_changed']] if 'signal_changed' in held else pd.DataFrame()

    metrics = C.metric_row([
        (len(held), 'held', TH.TEXT),
        (len(entered), 'entered', TH.ACCENT),
        (len(dropped), 'dropped', TH.NEG),
        (len(changed), 'signal changed', TH.WARN),
    ])

    def _tbl(df, cols):
        if df is None or df.empty:
            return C.placeholder('None.')
        sub = df[[c for c in cols if c in df.columns]].copy()
        for c in sub.columns:
            if sub[c].dtype.kind == 'f':
                sub[c] = sub[c].round(3)
        return C.data_table(sub, page_size=10)

    move_cols = ['ticker', 'rank_then', 'rank_now', 'rank_change',
                 'score_change', 'signal_now']

    return html.Div([
        C.card('📊 Summary', metrics, className='mb-3'),
        dbc.Row([
            dbc.Col(C.card('📈 Biggest rank gains',
                           _tbl(held.nlargest(10, 'rank_change'), move_cols)),
                    lg=6, className='mb-3'),
            dbc.Col(C.card('📉 Biggest rank falls',
                           _tbl(held.nsmallest(10, 'rank_change'), move_cols)),
                    lg=6, className='mb-3'),
        ]),
        dbc.Row([
            dbc.Col(C.card('🆕 New entrants',
                           _tbl(entered, ['ticker', 'rank_now', 'composite_now',
                                          'signal_now'])),
                    lg=6, className='mb-3'),
            dbc.Col(C.card('👋 Dropped out',
                           _tbl(dropped, ['ticker', 'rank_then',
                                          'composite_then', 'signal_then'])),
                    lg=6, className='mb-3'),
        ]),
        C.card('🔔 Signal changes',
               _tbl(changed, ['ticker', 'signal_then', 'signal_now',
                              'rank_change', 'score_change']),
               className='mb-3'),
        C.ai_panel(lambda: LLM.compare_runs_narrative(
            strategy_name=meta['strategy'], then=meta['then'], now=meta['now'],
            entered=entered['ticker'].tolist(),
            dropped=dropped['ticker'].tolist(),
            risers=list(held.nlargest(10, 'rank_change')[
                ['ticker', 'rank_change']].itertuples(index=False, name=None)),
            fallers=list(held.nsmallest(10, 'rank_change')[
                ['ticker', 'rank_change']].itertuples(index=False, name=None)),
        ), title='🤖 AI read on this rotation', className='mb-3'),
    ]), f'✅ Compared {len(comp)} names.'


def _run_meta(run_a: str, run_b: str) -> dict:
    """Strategy and the two as-of dates, for the narrative prompt."""
    df = db.read_sql(
        'SELECT run_id, strategy, as_of FROM runs WHERE run_id IN (:a, :b)',
        {'a': run_a, 'b': run_b})
    by_id = {r['run_id']: r for _i, r in df.iterrows()}
    a, b = by_id.get(run_a), by_id.get(run_b)
    return {
        'strategy': (a or b or {}).get('strategy', 'this strategy'),
        'then': (a or {}).get('as_of', '?'),
        'now': (b or {}).get('as_of', '?'),
    }
