"""
Screen tab — universe builder, strategy selection, ranked results.

The horizon toggle changes what the page is *for*, not just which rows appear:
long-term shows business quality and reasoning with no price levels, mid-term
adds entry, stop, target, size and a portfolio-risk rollup. Picking a horizon
re-populates the strategy list so the two never mix.
"""

from __future__ import annotations

import logging
from datetime import date

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

import config
from data.universe import PRESETS, UniverseSpec
from nlp import llm as LLM
from quant import engine as EN
from quant import factor_docs as FD
from quant import strategies as ST
from quant import tradeplan as TP
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([
        dcc.Store(id='screen-store'),
        dcc.Store(id='screen-meta'),

        C.card('⚙️ Build a screen', [
            dbc.Row([
                dbc.Col([
                    C.label('Horizon'),
                    dbc.RadioItems(
                        id='horizon-toggle',
                        options=[{'label': v['label'], 'value': k}
                                 for k, v in ST.HORIZONS.items()],
                        value='long', inline=True,
                        inputStyle={'marginRight': '5px'},
                        labelStyle={'marginRight': '16px', 'fontSize': '0.84rem'},
                    ),
                    html.Div(id='horizon-blurb',
                             style={'color': TH.MUTED, 'fontSize': '0.74rem',
                                    'marginTop': '4px'}),
                ], lg=5, md=12, className='mb-3'),

                dbc.Col([
                    C.label('Strategy'),
                    dcc.Dropdown(id='strategy-dd', clearable=False),
                ], lg=4, md=6, className='mb-3'),

                dbc.Col([
                    C.label('As of date'),
                    dbc.Input(id='asof-date', type='date',
                              value=date.today().isoformat(),
                              max=date.today().isoformat()),
                ], lg=3, md=6, className='mb-3'),
            ]),

            dbc.Row([
                dbc.Col([
                    C.label('Universe preset'),
                    dcc.Dropdown(
                        id='preset-dd',
                        options=[{'label': s.label, 'value': k}
                                 for k, s in PRESETS.items()],
                        value='dia', clearable=False),
                ], lg=6, md=6, className='mb-2'),

                dbc.Col([
                    C.label('Positions to show'),
                    dcc.Slider(id='topn-slider', min=5, max=50, step=5, value=15,
                               marks={5: '5', 20: '20', 35: '35', 50: '50'},
                               tooltip={'placement': 'bottom'}),
                ], lg=6, md=6, className='mb-2'),
            ], className='align-items-end'),

            # Secondary filters start folded. Four more controls on screen at
            # all times pushed the results below the fold for a setting most
            # runs never change.
            dbc.Button('⚙️ More filters', id='more-filters-btn', size='sm',
                       color='dark', className='mt-1',
                       style={'border': f'1px solid {TH.BORDER}',
                              'fontSize': '0.78rem'}),
            dbc.Collapse(dbc.Row([
                dbc.Col([
                    C.label('Sector filter'),
                    dcc.Dropdown(
                        id='sector-dd',
                        options=[{'label': s, 'value': s}
                                 for s in config.GICS_SECTORS if s != 'Unknown'],
                        multi=True, placeholder='All sectors'),
                ], lg=6, md=6, className='mb-2'),

                dbc.Col([
                    C.label('Min market cap ($B)'),
                    dcc.Slider(id='mcap-slider', min=0, max=50, step=1, value=0,
                               marks={0: '0', 10: '10', 25: '25', 50: '50'},
                               tooltip={'placement': 'bottom'}),
                ], lg=6, md=6, className='mb-2'),
            ], className='align-items-end mt-3'),
                id='more-filters', is_open=False),
        ], className='mb-3'),

        dbc.Row(dbc.Col(html.Div([
            C.gradient_button('🚀  Run screen', 'run-button'),
            C.loading(html.Div(id='screen-status',
                               style={'color': TH.MUTED, 'fontSize': '0.84rem',
                                      'minHeight': '26px', 'marginTop': '10px'}),
                      'screen-loading'),
        ], className='text-center')), className='mb-4'),

        html.Div(id='screen-results'),
        html.Div(id='screen-plans'),
        html.Div(id='screen-explain'),
    ])


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

@callback(
    Output('strategy-dd', 'options'),
    Output('strategy-dd', 'value'),
    Output('horizon-blurb', 'children'),
    Input('horizon-toggle', 'value'),
)
def _strategies_for_horizon(horizon):
    opts = ST.options(horizon)
    default = opts[0]['value'] if opts else None
    blurb = ST.HORIZONS.get(horizon, {}).get('blurb', '')
    return opts, default, blurb


@callback(
    Output('more-filters', 'is_open'),
    Input('more-filters-btn', 'n_clicks'),
    State('more-filters', 'is_open'),
    prevent_initial_call=True,
)
def _toggle_filters(_n, is_open):
    return not is_open


@callback(
    Output('screen-store', 'data'),
    Output('screen-meta', 'data'),
    Output('screen-status', 'children'),
    Output('run-button', 'disabled'),
    Input('run-button', 'n_clicks'),
    State('strategy-dd', 'value'),
    State('preset-dd', 'value'),
    State('sector-dd', 'value'),
    State('mcap-slider', 'value'),
    State('topn-slider', 'value'),
    State('asof-date', 'value'),
    prevent_initial_call=True,
)
def _run_screen(n_clicks, strategy, preset, sectors, mcap_b, top_n, as_of):
    if not n_clicks or not strategy:
        return no_update, no_update, '', False

    try:
        base = PRESETS.get(preset or 'dia', UniverseSpec())
        spec = UniverseSpec(
            preset=base.preset,
            sectors=list(sectors) if sectors else list(base.sectors),
            min_market_cap=(mcap_b * 1e9) if mcap_b else None,
            min_dollar_adv=base.min_dollar_adv,
            min_price=base.min_price,
            label=base.label,
        )

        result = EN.run(spec, strategy, as_of=as_of, top_n=top_n)
        if result.scores.empty:
            return None, None, ('❌ No results. Check that data has been ingested '
                                'for this universe (see the Data tab).'), False

        scores = result.scores.head(top_n).reset_index().rename(
            columns={'index': 'ticker'})
        if 'ticker' not in scores.columns:
            scores.insert(0, 'ticker', result.scores.head(top_n).index)

        plans = (result.plans.reset_index().to_dict('records')
                 if not result.plans.empty else [])

        meta = {
            'run_id': result.run_id, 'strategy': strategy,
            'strategy_name': result.strategy.name,
            'horizon': result.horizon,
            'as_of': str(result.as_of),
            'universe_n': len(result.universe),
            'shown': len(scores),
            'spec': spec.describe(),
            'plans': plans,
        }
        if result.sync:
            meta['sync'] = result.sync.summary()
            meta['sync_changed'] = result.sync.changed

        status = (f'✅ {result.strategy.name} — ranked {len(scores)} of '
                  f'{len(result.universe)} names as of {result.as_of}')
        if result.sync:
            status += f'  ·  data: {result.sync.summary()}'
        return scores.to_dict('records'), meta, status, False

    except Exception as exc:                       # noqa: BLE001
        log.exception('screen failed')
        return None, None, f'❌ {type(exc).__name__}: {str(exc)[:220]}', False


@callback(
    Output('screen-results', 'children'),
    Input('screen-store', 'data'),
    State('screen-meta', 'data'),
    prevent_initial_call=True,
)
def _render_results(data, meta):
    if not data or not meta:
        return C.placeholder()

    df = pd.DataFrame(data)
    horizon = meta.get('horizon', 'long')

    display = pd.DataFrame({
        'rank': df.get('rank'),
        'ticker': df.get('ticker'),
        'sector': df.get('sector'),
        'score': df.get('composite', pd.Series(dtype=float)).round(3),
        'signal': df.get('signal'),
    })

    if horizon == 'long':
        for src, dst, mult, dp in (('ROIC', 'roic %', 100, 1),
                                   ('EARNINGS_YIELD', 'earn yld %', 100, 2),
                                   ('PIOTROSKI_F', 'F-score', 1, 0)):
            if src in df.columns:
                display[dst] = (pd.to_numeric(df[src], errors='coerce')
                                * mult).round(dp)
    else:
        for src, dst, dp in (('PRICE', 'price', 2), ('entry', 'entry', 2),
                             ('stop', 'stop', 2), ('target', 'target', 2),
                             ('weight_pct', 'size %', 1),
                             ('RSI_14', 'rsi', 0)):
            if src in df.columns:
                display[dst] = pd.to_numeric(df[src], errors='coerce').round(dp)

    # Long prose belongs in a tooltip, not a grid cell: left inline it wraps
    # to three or four lines and sets the row height for every column.
    reasons = df.get('reasons', pd.Series('', index=df.index)).fillna('')
    display['why'] = reasons.map(
        lambda s: (s[:52] + '…') if isinstance(s, str) and len(s) > 53 else s)
    display = display.dropna(axis=1, how='all')

    counts = df['signal'].value_counts() if 'signal' in df else pd.Series(dtype=int)
    sector_counts = (df['sector'].value_counts() if 'sector' in df
                     else pd.Series(dtype=int))

    return html.Div([
        html.Div([
            C.horizon_badge(horizon),
            html.Span(meta['strategy_name'],
                      style={'marginLeft': '10px', 'fontWeight': '700',
                             'fontSize': '1.02rem'}),
            html.Span(f'  ·  {meta["spec"]}',
                      style={'color': TH.MUTED, 'fontSize': '0.76rem'}),
        ], className='mb-2'),
        _insight_strip(df, meta, counts, sector_counts),
        dbc.Row([
            dbc.Col(C.card('🏆 Ranked results', [
                C.data_table(display, page_size=25, table_id='results-table',
                             cell_tooltips=['why']),
                html.Div('Hover any “why” cell for the full reasoning. '
                         'Click a column header to sort.',
                         style={'color': TH.MUTED, 'fontSize': '0.72rem',
                                'marginTop': '8px'}),
            ]), lg=9, className='mb-3'),

            dbc.Col([
                C.card('📊 Sector mix',
                       dcc.Graph(figure=C.sector_bar(sector_counts),
                                 config={'displayModeBar': False},
                                 style={'height': '250px'}),
                       className='mb-3'),
                C.card('🚦 Signal mix', _signal_breakdown(counts)),
            ], lg=3, className='mb-3'),
        ]),
    ])


def _insight_strip(df, meta, counts, sector_counts):
    """The conclusion, before the detail."""
    horizon = meta.get('horizon', 'long')

    top = df.iloc[0] if len(df) else None
    top_txt = f"{top['ticker']}" if top is not None else '—'
    top_hint = (f"score {top['composite']:+.2f}"
                if top is not None and pd.notna(top.get('composite')) else '')

    # "Actionable" means the signal is a call to do something now, which is
    # the number a reader actually wants off this screen.
    actionable = int(sum(v for k, v in counts.items()
                         if any(w in str(k) for w in
                                ('Buy', 'Strong', 'Enter', 'Accumulate'))))

    conc, conc_colour, conc_hint = '—', TH.TEXT, ''
    if len(sector_counts):
        share = sector_counts.iloc[0] / max(sector_counts.sum(), 1)
        conc = f'{share*100:.0f}%'
        conc_hint = str(sector_counts.index[0])
        conc_colour = TH.NEG if share > 0.5 else TH.WARN if share > 0.35 else TH.POS

    med = df['composite'].median() if 'composite' in df else float('nan')

    return C.insight_strip([
        (top_txt, 'top ranked', TH.ACCENT, top_hint),
        (actionable, 'actionable now', TH.POS if actionable else TH.MUTED,
         'buy / enter signals'),
        (len(df), 'names shown', TH.TEXT, f'of {meta.get("universe_n", "?")} screened'),
        (f'{med:+.2f}' if pd.notna(med) else '—', 'median score', TH.TEXT,
         'z-score vs sector peers'),
        (conc, 'top sector weight', conc_colour, conc_hint),
        (meta.get('as_of', '—'), 'as of', TH.MUTED,
         ST.HORIZONS.get(horizon, {}).get('label', '')),
    ])


def _signal_breakdown(counts):
    """Signal counts as proportional bars — readable at a glance."""
    if not len(counts):
        return C.placeholder('No signals.')
    total = max(int(counts.sum()), 1)
    rows = []
    for label, n in counts.items():
        pct = n / total * 100
        colour = (TH.POS if any(w in str(label) for w in ('Strong', 'Buy', 'Enter'))
                  else TH.WARN if any(w in str(label) for w in ('Hold', 'Watch'))
                  else TH.MUTED)
        rows.append(html.Div([
            html.Div([
                html.Span(str(label), style={'fontSize': '0.79rem'}),
                html.Span(f'{n}', style={'float': 'right', 'fontWeight': '700',
                                         'fontSize': '0.79rem'}),
            ]),
            html.Div(html.Div(style={
                'width': f'{pct:.0f}%', 'height': '5px', 'background': colour,
                'borderRadius': '3px'}),
                style={'background': TH.PANEL_ALT, 'borderRadius': '3px',
                       'height': '5px', 'marginTop': '3px'}),
        ], className='mb-2'))
    return html.Div(rows)


@callback(
    Output('screen-plans', 'children'),
    Input('screen-store', 'data'),
    State('screen-meta', 'data'),
    prevent_initial_call=True,
)
def _render_plans(data, meta):
    """Mid-term horizon only: the trade plans and their aggregate risk."""
    if not data or not meta or meta.get('horizon') != 'mid':
        return None

    plans = meta.get('plans') or []
    if not plans:
        return C.note('No tradable setups: every candidate needed a stop wider '
                      'than the risk limit allows, or lacked price history.',
                      'warn')

    pf = pd.DataFrame(plans)
    risk = TP.portfolio_risk(pf.set_index('ticker'))

    cols = ['ticker', 'price', 'entry', 'entry_type', 'stop', 'stop_pct',
            'stop_basis', 'target', 'target_pct', 'r_multiple', 'weight_pct',
            'time_stop_date']
    table = pf[[c for c in cols if c in pf.columns]].copy()
    for c in ('price', 'entry', 'stop', 'target', 'stop_pct', 'target_pct',
              'weight_pct'):
        if c in table:
            table[c] = pd.to_numeric(table[c], errors='coerce').round(2)

    metrics = C.metric_row([
        (risk.get('n_positions', 0), 'positions', TH.TEXT),
        (f'{risk.get("total_weight_pct", 0):.0f}%', 'invested', TH.ACCENT),
        (f'{risk.get("cash_pct", 0):.0f}%', 'cash', TH.MUTED),
        (f'{risk.get("portfolio_risk_pct", 0):.2f}%', 'risk if all stopped', TH.NEG),
        (f'{risk.get("avg_stop_pct", 0):.1f}%', 'avg stop distance', TH.WARN),
    ])

    return C.card('🎯 Trade plans', [
        C.note('Position sizes are percentages of portfolio implied by a 1% '
               'risk budget and each name\'s stop distance — a wider stop earns '
               'a smaller position. Levels are mechanical outputs, not advice.',
               'info'),
        metrics,
        html.Hr(style={'borderColor': TH.BORDER}),
        C.data_table(table, page_size=20),
        html.Div([
            html.B('Invalidation: ', style={'color': TH.MUTED,
                                            'fontSize': '0.76rem'}),
            html.Span(pf.iloc[0].get('invalidation', ''),
                      style={'color': TH.MUTED, 'fontSize': '0.76rem'}),
        ], className='mt-2') if 'invalidation' in pf.columns else None,
    ], className='mb-3')


@callback(
    Output('screen-explain', 'children'),
    Input('screen-store', 'data'),
    State('screen-meta', 'data'),
    prevent_initial_call=True,
)
def _render_explanation(data, meta):
    """
    Why this strategy picked these names — formulas first, model second.

    The formula breakdown always renders. The AI panel is additive and absent
    without a key, so the explanation never depends on the network.
    """
    if not data or not meta:
        return None

    try:
        strategy = ST.get(meta['strategy'])
    except KeyError:
        return None

    rows = FD.strategy_breakdown(strategy)
    weights_table = pd.DataFrame([{
        'Factor': r['label'], 'Family': r['family'],
        'Share %': r['weight_pct'], 'Direction': r['direction'],
    } for r in rows])

    detail = [
        dbc.AccordionItem([
            html.Pre(r['formula'], style={
                'fontFamily': 'ui-monospace, Menlo, monospace',
                'fontSize': '0.75rem', 'whiteSpace': 'pre',
                'overflowX': 'auto', 'background': TH.PANEL_ALT,
                'padding': '11px 13px', 'borderRadius': '6px',
                'color': TH.TEXT, 'margin': '0 0 10px'}),
            html.Div(r['rationale'],
                     style={'fontSize': '0.83rem', 'lineHeight': '1.6'}),
            html.Div([html.B('⚠️ ', style={'color': TH.WARN}),
                      html.Span(r['caveat'], style={'color': TH.MUTED})],
                     style={'fontSize': '0.78rem', 'marginTop': '7px'})
            if r['caveat'] else None,
        ], title=f"{r['label']}  —  {r['weight_pct']}% of the score",
            item_id=f"sf-{r['factor']}")
        for r in rows
    ]

    methodology_body = html.Div([
        html.Div([html.B('Thesis: ', style={'color': TH.ACCENT}),
                  html.Span(strategy.thesis)],
                 style={'fontSize': '0.85rem', 'lineHeight': '1.6',
                        'marginBottom': '12px'}),
        C.data_table(weights_table, page_size=12),
        html.Div('Expand any factor for its formula and caveats:',
                 style={'color': TH.MUTED, 'fontSize': '0.76rem',
                        'margin': '14px 0 6px'}),
        dbc.Accordion(detail, start_collapsed=True, always_open=True,
                      flush=True),
        html.Div([
            'Full pipeline, trade-plan arithmetic and every factor formula '
            'live in the ', html.B('📐 Methodology'), ' tab.',
        ], style={'color': TH.MUTED, 'fontSize': '0.76rem',
                  'marginTop': '12px'}),
    ])

    scores = pd.DataFrame(data).set_index('ticker', drop=False)
    ai_body = C.ai_panel(
        lambda: LLM.explain_screen(
            strategy_name=strategy.name, horizon=strategy.horizon,
            thesis=strategy.thesis, scores=scores,
            universe_desc=meta['spec'],
            as_of=pd.to_datetime(meta['as_of']).date()),
        title='', card=False)

    # Both were full-width cards stacked under the table, so reaching the AI
    # read meant scrolling past every formula. Collapsed and side by side,
    # the reader opens the one they want.
    return dbc.Accordion([
        dbc.AccordionItem(ai_body, title='🤖 AI analysis of this screen',
                          item_id='ai'),
        dbc.AccordionItem(methodology_body,
                          title=f'📐 How "{strategy.name}" scores — '
                                f'{len(rows)} factors, with formulas',
                          item_id='how'),
    ], start_collapsed=True, always_open=True, className='mb-3')
