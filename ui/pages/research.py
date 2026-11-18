"""
Research tab — backtest a strategy and check whether its ranking predicts.

The equity curve is the least interesting panel here. Rank IC is what
distinguishes a strategy that worked from a strategy that got lucky, so it is
reported alongside the returns and flagged explicitly when it is near zero.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from data.universe import PRESETS, UniverseSpec
from nlp import llm as LLM
from quant import backtest as BT
from quant import custom as CU
from quant import strategies as ST
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)


def layout() -> html.Div:
    return html.Div([
        C.card('🔬 Backtest', [
            dbc.Row([
                dbc.Col([
                    C.label('Strategy'),
                    dcc.Dropdown(id='bt-strategy', options=CU.options(),
                                 value='buffett', clearable=False),
                ], lg=3, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Universe'),
                    dcc.Dropdown(id='bt-preset',
                                 options=[{'label': s.label, 'value': k}
                                          for k, s in PRESETS.items()],
                                 value='dia', clearable=False),
                ], lg=3, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Start'),
                    dbc.Input(id='bt-start', type='date',
                              value=(date.today() - timedelta(days=1460)).isoformat(),
                              max=(date.today() - timedelta(days=200)).isoformat()),
                ], lg=2, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Holdings'),
                    dcc.Slider(id='bt-holdings', min=5, max=30, step=5, value=10,
                               marks={5: '5', 15: '15', 30: '30'},
                               tooltip={'placement': 'bottom'}),
                ], lg=2, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Rebalance'),
                    dcc.Dropdown(
                        id='bt-rebalance',
                        options=[{'label': v, 'value': k}
                                 for k, v in {'M': 'Monthly', 'Q': 'Quarterly',
                                              'SA': 'Semi-annual',
                                              'A': 'Annual'}.items()],
                        value='Q', clearable=False),
                ], lg=2, md=6, className='mb-3'),
            ], className='align-items-end'),

            html.Div(className='text-center', children=[
                C.gradient_button('📈  Run backtest', 'bt-button'),
                C.loading(html.Div(id='bt-status',
                                   style={'color': TH.MUTED,
                                          'fontSize': '0.84rem',
                                          'minHeight': '26px',
                                          'marginTop': '10px'}),
                          'bt-loading'),
            ]),
            C.note('A walk-forward backtest re-scores the universe at every '
                   'rebalance using only data filed by that date. Runs take a '
                   'while because each rebalance is a full screen.', 'info'),
        ], className='mb-3'),

        html.Div(id='bt-results'),
    ])


@callback(
    Output('bt-results', 'children'), Output('bt-status', 'children'),
    Output('bt-button', 'disabled'),
    Input('bt-button', 'n_clicks'),
    State('bt-strategy', 'value'), State('bt-preset', 'value'),
    State('bt-start', 'value'), State('bt-holdings', 'value'),
    State('bt-rebalance', 'value'),
    prevent_initial_call=True,
)
def _run_backtest(n_clicks, strategy, preset, start, holdings, rebalance):
    if not n_clicks or not strategy:
        return no_update, '', False

    try:
        base = PRESETS.get(preset or 'dia', UniverseSpec())
        spec = UniverseSpec(preset=base.preset, sectors=list(base.sectors),
                            min_market_cap=None, min_dollar_adv=None,
                            label=base.label)
        strat = ST.get(strategy)

        if strat.horizon == 'mid':
            res = BT.run_trade_backtest(spec, strategy, start,
                                        n_positions=holdings)
        else:
            res = BT.run_backtest(spec, strategy, start, n_holdings=holdings,
                                  rebalance=rebalance)
    except Exception as exc:                       # noqa: BLE001
        log.exception('backtest failed')
        return None, f'❌ {type(exc).__name__}: {str(exc)[:220]}', False

    if not res.metrics:
        return None, '❌ No results — try a longer window or a bigger universe.', False

    m = res.metrics
    panels = []

    # The verdict first. Beating the benchmark is the headline a reader looks
    # for, and whether the ranking actually predicted is the one that decides
    # if the headline means anything — so both sit at the top, not buried.
    excess = m.get('cagr', 0) - m.get('benchmark_cagr', 0)
    ic = m.get('mean_ic')
    if ic is None:
        ic_val, ic_colour, ic_hint = '—', TH.MUTED, 'not measured'
    elif abs(ic) < 0.02:
        ic_val, ic_colour, ic_hint = f'{ic:+.3f}', TH.NEG, 'no predictive power'
    else:
        ic_val = f'{ic:+.3f}'
        ic_colour = TH.POS if ic > 0 else TH.NEG
        ic_hint = 'signal works' if ic > 0 else 'inverted signal'

    panels.append(C.insight_strip([
        (f'{m.get("cagr", 0)*100:.1f}%', 'CAGR',
         TH.POS if m.get('cagr', 0) > 0 else TH.NEG, 'annualised'),
        (f'{excess*100:+.1f}%', 'vs benchmark',
         TH.POS if excess > 0 else TH.NEG,
         f'benchmark {m.get("benchmark_cagr", 0)*100:.1f}%'),
        (f'{m.get("sharpe", 0):.2f}', 'Sharpe',
         TH.POS if m.get('sharpe', 0) > 1 else TH.WARN, 'return per unit risk'),
        (f'{m.get("max_drawdown", 0)*100:.1f}%', 'max drawdown', TH.NEG,
         'worst peak-to-trough'),
        (ic_val, 'mean rank IC', ic_colour, ic_hint),
        (f'{m.get("turnover", 0)*100:.0f}%', 'turnover', TH.MUTED, 'per year'),
    ]))

    panels.append(C.card('📊 Full metrics', C.metric_row([
        (f'{m.get("sortino", 0):.2f}', 'Sortino', TH.TEXT),
        (f'{m.get("volatility", 0)*100:.1f}%', 'Volatility', TH.WARN),
        (f'{m.get("benchmark_cagr", 0)*100:.1f}%', 'Benchmark CAGR', TH.MUTED),
        (f'{m.get("ic_ir", 0):+.2f}', 'IC info ratio', TH.TEXT),
        (f'{m.get("ic_hit_rate", 0)*100:.0f}%', 'periods IC > 0', TH.TEXT),
    ]), className='mb-3'))

    if len(res.equity) > 1:
        panels.append(C.card('📈 Growth of 1.0', [
            dcc.Graph(figure=C.equity_curve(res.equity, res.benchmark),
                      config={'displayModeBar': False}),
            dcc.Graph(figure=C.drawdown_chart(res.equity),
                      config={'displayModeBar': False}),
        ], className='mb-3'))

    # Predictive-power panel — the honest read on whether the ranking worked.
    if 'mean_ic' in m:
        ic = m['mean_ic']
        weak = abs(ic) < 0.02
        ic_children = [C.metric_row([
            (f'{ic:+.3f}', 'mean rank IC',
             TH.POS if ic > 0.02 else TH.NEG if ic < -0.02 else TH.WARN),
            (f'{m.get("ic_ir", 0):+.2f}', 'IC info ratio', TH.TEXT),
            (f'{m.get("ic_hit_rate", 0)*100:.0f}%', 'periods IC > 0', TH.TEXT),
        ])]
        if weak:
            ic_children.append(C.note(
                'Mean rank IC is near zero: over this window the ranking had '
                'essentially no predictive power. Any outperformance above is '
                'more likely a few individual positions than a working signal. '
                'Cross-sectional scores need a wide universe — a 30-name index '
                'is usually too small to rank meaningfully.', 'warn'))
        panels.append(C.card('🎯 Does the ranking predict?', ic_children,
                             className='mb-3'))

    if strat.horizon == 'mid' and 'n_trades' in m:
        panels.append(C.card('🎲 Trade statistics', [
            C.metric_row([
                (m.get('n_trades', 0), 'trades', TH.TEXT),
                (f'{m.get("win_rate", 0)*100:.0f}%', 'win rate',
                 TH.POS if m.get('win_rate', 0) > 0.5 else TH.NEG),
                (f'{m.get("avg_r", 0):+.2f}', 'average R', TH.TEXT),
                (f'{m.get("profit_factor", 0):.2f}', 'profit factor', TH.TEXT),
                (f'{m.get("avg_days_held", 0):.0f}', 'avg days held', TH.MUTED),
            ]),
            html.Div(f'exits — {m.get("exit_mix", {})}',
                     style={'color': TH.MUTED, 'fontSize': '0.78rem',
                            'marginTop': '10px', 'textAlign': 'center'}),
        ], className='mb-3'))

        if not res.trades.empty:
            t = res.trades.copy()
            for c in ('entry', 'exit', 'net_return', 'r_multiple'):
                if c in t:
                    t[c] = pd.to_numeric(t[c], errors='coerce').round(3)
            panels.append(C.card('📋 Trades', C.data_table(
                t[['entry_date', 'exit_date', 'ticker', 'entry', 'exit',
                   'reason', 'net_return', 'r_multiple', 'days_held']].tail(40),
                page_size=15), className='mb-3'))

    # Interpretation last: the reader should meet the numbers before the prose.
    panels.append(C.ai_panel(
        lambda: LLM.explain_backtest(strat.name, m),
        title='🤖 AI read on this backtest', className='mb-3'))

    return html.Div(panels), f'✅ {res.summary()}', False
