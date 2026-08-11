"""
Compare tab — two questions a single screen cannot answer.

**Over time**: has my ranking changed, and how much would I be trading?
**Across strategies**: where do two independent methods agree, and where do
they contradict each other?

Both are computed from stored data at two point-in-time views, so neither
depends on having run the app before. The previous design diffed two *saved*
runs, which meant a day of clicking produced a list of identical runs on the
same date and comparing them showed nothing.
"""

from __future__ import annotations

import logging
from datetime import date

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from data.universe import PRESETS, UniverseSpec
from nlp import llm as LLM
from quant import compare as CMP
from quant import strategies as ST
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)


def layout() -> html.Div:
    return html.Div([
        C.card('🔄 Compare', [
            dcc.RadioItems(
                id='cmp-mode',
                options=[
                    {'label': '  Over time — one strategy, two dates',
                     'value': 'time'},
                    {'label': '  Across strategies — two strategies, one date',
                     'value': 'cross'},
                ],
                value='cross', inline=True,
                inputStyle={'marginRight': '6px'},
                labelStyle={'marginRight': '26px', 'fontSize': '0.86rem'},
                className='mb-3'),

            html.Div(id='cmp-blurb',
                     style={'color': TH.MUTED, 'fontSize': '0.78rem',
                            'marginBottom': '14px'}),

            dbc.Row([
                dbc.Col([
                    C.label('Universe'),
                    dcc.Dropdown(
                        id='cmp-preset',
                        options=[{'label': s.label, 'value': k}
                                 for k, s in PRESETS.items()],
                        value='dia', clearable=False),
                ], lg=3, md=6, className='mb-2'),

                dbc.Col([
                    C.label('Strategy'),
                    dcc.Dropdown(id='cmp-strategy-a', options=ST.options(),
                                 value='compounder', clearable=False),
                ], lg=3, md=6, className='mb-2'),

                dbc.Col([
                    html.Div(id='cmp-second-label'),
                    html.Div(id='cmp-second-control'),
                ], lg=3, md=6, className='mb-2'),

                dbc.Col([
                    C.label('As of'),
                    dbc.Input(id='cmp-asof', type='date',
                              value=date.today().isoformat(),
                              max=date.today().isoformat()),
                ], lg=2, md=4, className='mb-2'),

                dbc.Col([
                    C.label('Depth'),
                    dcc.Dropdown(id='cmp-topn',
                                 options=[{'label': f'Top {n}', 'value': n}
                                          for n in (10, 15, 20, 30)],
                                 value=15, clearable=False),
                ], lg=1, md=4, className='mb-2'),
            ], className='align-items-end'),

            html.Div(C.gradient_button('🔄  Compare', 'cmp-button'),
                     className='text-center mt-3'),
            C.loading(html.Div(id='cmp-status',
                               style={'color': TH.MUTED, 'fontSize': '0.84rem',
                                      'minHeight': '26px', 'marginTop': '10px',
                                      'textAlign': 'center'}),
                      'cmp-loading'),
        ], className='mb-3'),

        html.Div(id='cmp-results'),
    ])


BLURBS = {
    'time': ('Recomputes the same screen at two dates using only what had been '
             'filed by each. Shows what entered, what left, and how much you '
             'would have traded — a stable strategy should barely move.'),
    'cross': ('Runs two strategies over the same universe on the same date. '
              'Names in both lists have support from independent evidence; '
              'names each ranks oppositely are where the real research is.'),
}


@callback(
    Output('cmp-blurb', 'children'),
    Output('cmp-second-label', 'children'),
    Output('cmp-second-control', 'children'),
    Input('cmp-mode', 'value'),
)
def _mode_controls(mode):
    if mode == 'time':
        return (BLURBS['time'], C.label('Compare against'),
                dcc.Dropdown(
                    id='cmp-second',
                    options=[{'label': f'{k} ago', 'value': k}
                             for k in CMP.LOOKBACKS],
                    value='3M', clearable=False))
    return (BLURBS['cross'], C.label('Second strategy'),
            dcc.Dropdown(id='cmp-second', options=ST.options(),
                         value='deep_value', clearable=False))


@callback(
    Output('cmp-results', 'children'), Output('cmp-status', 'children'),
    Input('cmp-button', 'n_clicks'),
    State('cmp-mode', 'value'), State('cmp-preset', 'value'),
    State('cmp-strategy-a', 'value'), State('cmp-second', 'value'),
    State('cmp-asof', 'value'), State('cmp-topn', 'value'),
    prevent_initial_call=True,
)
def _compare(n_clicks, mode, preset, strat_a, second, as_of, top_n):
    if not n_clicks or not strat_a or not second:
        return no_update, ''

    spec = PRESETS.get(preset or 'dia', UniverseSpec())
    try:
        if mode == 'time':
            res = CMP.drift(spec, strat_a, as_of, lookback=second, top_n=top_n)
            return _render_drift(res), f'✅ {res.summary()}'
        res = CMP.crossover(spec, strat_a, second, as_of, top_n=top_n)
        return _render_crossover(res), f'✅ {res.summary()}'
    except Exception as exc:                       # noqa: BLE001
        log.exception('compare failed')
        return None, f'❌ {type(exc).__name__}: {str(exc)[:200]}'


# ─────────────────────────────────────────────
# RENDER — TIME DRIFT
# ─────────────────────────────────────────────

def _render_drift(r: CMP.DriftResult):
    if r.table.empty:
        return C.note(r.note or 'Nothing to compare.', 'warn')

    stability = ('very stable' if r.rank_corr > 0.9 else
                 'stable' if r.rank_corr > 0.7 else
                 'shifting' if r.rank_corr > 0.4 else 'reordered')
    turn_colour = (TH.POS if r.turnover < 0.2 else
                   TH.WARN if r.turnover < 0.4 else TH.NEG)

    strip = C.insight_strip([
        (f'{r.turnover*100:.0f}%', 'turnover', turn_colour,
         f'{len(r.entered)} of {len(r.entered) + len(r.held)} names replaced'),
        (f'{r.rank_corr:+.2f}', 'rank correlation',
         TH.POS if r.rank_corr > 0.7 else TH.WARN, stability),
        (len(r.entered), 'entered', TH.ACCENT, 'newly qualified'),
        (len(r.dropped), 'dropped', TH.NEG, 'no longer qualify'),
        (str(r.then), 'from', TH.MUTED, ''),
        (str(r.now), 'to', TH.MUTED, ''),
    ])

    held = r.held.copy()
    movers = (held.reindex(held['rank_change'].abs()
                           .sort_values(ascending=False).index)
              if not held.empty else held)

    def _tbl(df, cols):
        if df is None or df.empty:
            return C.placeholder('None.')
        out = df.reset_index().rename(columns={'index': 'ticker'})
        out = out[[c for c in cols if c in out.columns]]
        for c in out.columns:
            if out[c].dtype.kind == 'f':
                out[c] = out[c].round(3)
        return C.data_table(out, page_size=10)

    reading = _drift_reading(r)

    return html.Div([
        strip,
        C.note(reading, 'info'),
        dbc.Row([
            dbc.Col(C.card('📈 Climbed', _tbl(
                movers[movers['rank_change'] > 0],
                ['ticker', 'rank_then', 'rank_now', 'rank_change',
                 'score_change', 'signal_now'])), lg=6, className='mb-3'),
            dbc.Col(C.card('📉 Fell', _tbl(
                movers[movers['rank_change'] < 0],
                ['ticker', 'rank_then', 'rank_now', 'rank_change',
                 'score_change', 'signal_now'])), lg=6, className='mb-3'),
        ]),
        dbc.Row([
            dbc.Col(C.card('🆕 Entered the list', _tbl(
                r.entered, ['ticker', 'rank_now', 'composite_now',
                            'signal_now', 'sector_now'])),
                lg=6, className='mb-3'),
            dbc.Col(C.card('👋 Dropped out', _tbl(
                r.dropped, ['ticker', 'rank_then', 'composite_then',
                            'signal_then', 'sector_then'])),
                lg=6, className='mb-3'),
        ]),
        C.ai_panel(lambda: LLM.compare_runs_narrative(
            strategy_name=r.strategy, then=r.then, now=r.now,
            entered=list(r.entered.index), dropped=list(r.dropped.index),
            risers=list(movers[movers['rank_change'] > 0]['rank_change']
                        .head(8).items()),
            fallers=list(movers[movers['rank_change'] < 0]['rank_change']
                         .head(8).items())),
            title='🤖 AI read on this rotation', className='mb-3'),
    ])


def _drift_reading(r: CMP.DriftResult) -> str:
    if r.turnover == 0:
        return (f'Nothing changed hands between {r.then} and {r.now}. For a '
                f'quality or low-volatility strategy that is the expected '
                f'result — the businesses it selects do not change character '
                f'in a quarter. High turnover here would be the warning sign.')
    if r.turnover >= 0.4:
        return (f'{r.turnover*100:.0f}% of the list turned over. At that rate '
                f'trading costs matter: check the backtest turnover figure '
                f'before assuming the paper returns survive execution.')
    return (f'{r.turnover*100:.0f}% turnover with a rank correlation of '
            f'{r.rank_corr:+.2f}. The ordering is broadly intact; the names '
            f'that entered and left are where to look for a changed thesis.')


# ─────────────────────────────────────────────
# RENDER — CROSSOVER
# ─────────────────────────────────────────────

def _render_crossover(r: CMP.CrossoverResult):
    if r.table.empty:
        return C.note(r.note or 'Nothing to compare.', 'warn')

    label, note = CMP.diversification_note(r.rank_corr)
    corr_colour = (TH.NEG if label == 'redundant' else
                   TH.WARN if label == 'related' else TH.POS)

    strip = C.insight_strip([
        (len(r.both), 'in both lists', TH.POS if len(r.both) else TH.MUTED,
         'independent support'),
        (f'{r.rank_corr:+.2f}', 'rank correlation', corr_colour, label),
        (len(r.only_a), f'only {_short(r.name_a)}', TH.TEXT, ''),
        (len(r.only_b), f'only {_short(r.name_b)}', TH.TEXT, ''),
        (len(r.conflict), 'in conflict', TH.WARN, 'top on one, bottom on other'),
        (str(r.as_of), 'as of', TH.MUTED, ''),
    ])

    def _tbl(df, cols, n=10):
        if df is None or df.empty:
            return C.placeholder('None.')
        out = df.head(n).reset_index().rename(columns={'index': 'ticker'})
        out = out[[c for c in cols if c in out.columns]]
        for c in out.columns:
            if out[c].dtype.kind == 'f':
                out[c] = out[c].round(3)
        return C.data_table(out, page_size=10)

    agree_cols = ['ticker', 'rank_a', 'rank_b', 'composite_a', 'composite_b',
                  'sector_a']

    return html.Div([
        strip,
        C.note(note, 'info'),

        C.card(f'🎯 Ranked highly by both — {r.name_a} and {r.name_b}', [
            _tbl(r.both, agree_cols),
            html.Div('Two methods built on different evidence reaching the '
                     'same name is a stronger case than either alone. These '
                     'are the natural candidates for a larger position.',
                     style={'color': TH.MUTED, 'fontSize': '0.76rem',
                            'marginTop': '10px'}),
        ], className='mb-3'),

        C.card('⚔️ Direct conflicts', [
            _tbl(r.conflict, ['ticker', 'rank_a', 'rank_b', 'rank_gap',
                              'signal_a', 'signal_b', 'sector_a']),
            html.Div(f'Top quintile on one method, bottom quintile on the '
                     f'other. A name {r.name_a} loves and {r.name_b} rejects '
                     f'is either mispriced or a trap, and neither screen can '
                     f'tell you which — that is the research these lists are '
                     f'for.',
                     style={'color': TH.MUTED, 'fontSize': '0.76rem',
                            'marginTop': '10px'}),
        ], className='mb-3'),

        dbc.Row([
            dbc.Col(C.card(f'Only {r.name_a}',
                           _tbl(r.only_a, ['ticker', 'rank_a', 'rank_b',
                                           'signal_a'])),
                    lg=6, className='mb-3'),
            dbc.Col(C.card(f'Only {r.name_b}',
                           _tbl(r.only_b, ['ticker', 'rank_b', 'rank_a',
                                           'signal_b'])),
                    lg=6, className='mb-3'),
        ]),
    ])


def _short(name: str, n: int = 14) -> str:
    return name if len(name) <= n else name[:n - 1] + '…'
