"""
Intel tab — company direction, filings, insiders and government policy.

Sentiment is one column here, not the headline. What matters for a thesis is
what changed: guidance raised or cut, capital returned, management replaced,
a restructuring booked, insiders buying, or a rule landing on the sector.
"""

from __future__ import annotations

import logging
from datetime import date

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from core import db
from data import news as NEWS
from data import policy as POL
from data import sec as SEC
from nlp import eightk as EK
from nlp import llm as LLM
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)


def layout() -> html.Div:
    return html.Div([
        C.card('🔎 Company intelligence', [
            dbc.Row([
                dbc.Col([
                    C.label('Ticker'),
                    dcc.Dropdown(id='intel-ticker', placeholder='Pick a ticker'),
                ], lg=4, md=6, className='mb-3'),
                dbc.Col([
                    C.label('Lookback (days)'),
                    dcc.Slider(id='intel-days', min=7, max=180, step=7, value=30,
                               marks={7: '7', 30: '30', 90: '90', 180: '180'},
                               tooltip={'placement': 'bottom'}),
                ], lg=5, md=6, className='mb-3'),
                dbc.Col([
                    dbc.Button('Load', id='intel-button', color='success',
                               className='w-100'),
                ], lg=3, md=12, className='mb-3'),
            ], className='align-items-end'),
        ], className='mb-3'),

        html.Div(id='intel-company'),
        html.Hr(style={'borderColor': TH.BORDER}),
        C.card('🏛️ Policy & regulation', [
            C.note('Federal Register rules, proposed rules and executive '
                   'actions, mapped from the issuing agency to the sectors they '
                   'touch. Counts show activity, not direction — regulation can '
                   'be a tailwind or a headwind.', 'info'),
            html.Div(id='intel-policy'),
        ], className='mb-3'),
    ])


@callback(
    Output('intel-ticker', 'options'),
    Input('tabs', 'active_tab'),
    Input('intel-ticker', 'search_value'),
)
def _ticker_options(active_tab, search):
    df = db.read_sql('SELECT ticker, name FROM securities ORDER BY ticker')
    if df.empty:
        return []
    opts = [{'label': f'{r["ticker"]} — {(r["name"] or "")[:34]}',
             'value': r['ticker']} for _i, r in df.iterrows()]
    if not search:
        return opts
    q = search.upper().strip()
    if not q:
        return opts
    exact  = [o for o in opts if o['value'] == q]
    prefix = [o for o in opts if o['value'].startswith(q) and o['value'] != q]
    rest   = [o for o in opts if not o['value'].startswith(q)]
    return exact + prefix + rest


@callback(
    Output('intel-company', 'children'),
    Input('intel-button', 'n_clicks'),
    State('intel-ticker', 'value'), State('intel-days', 'value'),
    prevent_initial_call=True,
)
def _load_company(n_clicks, ticker, days):
    if not n_clicks or not ticker:
        return C.placeholder('Pick a ticker and press Load.')

    today = date.today()
    panels = []

    # ── forward direction from news ───────────────────────────────
    try:
        summary = NEWS.direction_summary(ticker, today, lookback_days=days)
    except Exception as exc:                       # noqa: BLE001
        log.debug('direction summary failed: %s', exc)
        summary = {}

    notes = summary.get('direction_notes') or []
    if notes:
        panels.append(C.card(f'🧭 What recent coverage implies — {ticker}', [
            html.Ul([html.Li(n, style={'marginBottom': '5px'}) for n in notes],
                    style={'fontSize': '0.88rem', 'paddingLeft': '18px'}),
            html.Div(
                f'{summary.get("material_count", 0)} material of '
                f'{summary.get("n_articles", 0)} articles  ·  '
                f'mean sentiment {summary.get("mean_sentiment", 0):+.2f}',
                style={'color': TH.MUTED, 'fontSize': '0.76rem',
                       'marginTop': '8px'}),
        ], className='mb-3'))
    else:
        panels.append(C.note(
            f'No recent news stored for {ticker}. Run '
            f'"python cli.py ingest --index DIA --full" to populate the news '
            f'stream.', 'warn'))

    # ── 8-K corporate events (structured, no NLP needed) ──────────
    try:
        events = SEC.events_asof([ticker], today, lookback_days=max(days, 90))
    except Exception:                              # noqa: BLE001
        events = pd.DataFrame()

    if not events.empty:
        ev = events.copy()
        ev['filed'] = pd.to_datetime(ev['filed']).dt.date
        ev['significance'] = ev['item_code'].map(EK.significance)
        ek = EK.summarize(ev)

        # Only the codes actually present, so the reference stays short.
        present = [c for c in ev['item_code'].drop_duplicates() if EK.get(c)]
        present.sort(key=lambda c: -EK.WEIGHT_ORDER[EK.significance(c)])
        explain = [
            dbc.AccordionItem([
                html.Div([html.B('What it is: ', style={'color': TH.MUTED}),
                          html.Span(EK.get(c).means)],
                         style={'fontSize': '0.88rem', 'lineHeight': '1.7',
                                'marginBottom': '8px'}),
                html.Div([html.B('How to read it: ', style={'color': TH.ACCENT}),
                          html.Span(EK.get(c).read)],
                         style={'fontSize': '0.88rem', 'lineHeight': '1.7',
                                'marginBottom': '8px'}),
                html.Div([html.B('Check: ', style={'color': TH.WARN}),
                          html.Span(EK.get(c).check)],
                         style={'fontSize': '0.88rem', 'lineHeight': '1.7'}),
            ], title=f'{c} — {EK.get(c).means[:58]}…'
                     f'   [{EK.get(c).significance}]',
                item_id=f'ek-{c}')
            for c in present
        ]

        panels.append(C.card('📁 8-K corporate events', [
            C.metric_row([
                (ek['material'], 'material',
                 TH.NEG if ek['material'] else TH.MUTED),
                (ek['notable'], 'notable',
                 TH.WARN if ek['notable'] else TH.MUTED),
                (ek['routine'], 'routine', TH.MUTED),
                (ek['total'], 'filings', TH.TEXT),
            ]),
            C.note(ek['headline'], 'info'),

            html.Div([
                html.B('What this pattern says',
                       style={'color': TH.ACCENT, 'fontSize': '0.86rem'}),
                html.Ul([html.Li(o, style={'marginBottom': '6px'})
                         for o in ek['observations']],
                        style={'fontSize': '0.88rem', 'lineHeight': '1.7',
                               'paddingLeft': '18px', 'marginTop': '6px'}),
            ], className='mb-3'),

            C.data_table(
                ev[['filed', 'item_code', 'significance', 'subtype']].head(20),
                page_size=10,
                extra_conditional=[
                    {'if': {'filter_query': '{significance} = "material"',
                            'column_id': 'significance'},
                     'color': TH.NEG, 'fontWeight': '700'},
                    {'if': {'filter_query': '{significance} = "notable"',
                            'column_id': 'significance'},
                     'color': TH.WARN},
                ]),

            html.Div('Item codes are assigned by the filer, so the '
                     'classification needs no text parsing. Everything below '
                     'is interpretation — expand a code for what it means.',
                     style={'color': TH.MUTED, 'fontSize': '0.75rem',
                            'margin': '10px 0 6px'}),
            dbc.Accordion(explain, start_collapsed=True, always_open=True,
                          flush=True),
        ], className='mb-3'))

    # ── insider activity ──────────────────────────────────────────
    try:
        ins = SEC.insiders_asof([ticker], today, lookback_days=max(days, 180))
    except Exception:                              # noqa: BLE001
        ins = pd.DataFrame()

    if not ins.empty:
        signal = ins[ins['txn_code'].isin(['P', 'S'])].copy()
        if not signal.empty:
            signal['txn_date'] = pd.to_datetime(signal['txn_date']).dt.date
            signal['side'] = signal['shares'].apply(
                lambda s: 'BUY' if (s or 0) > 0 else 'SELL')
            signal['shares'] = signal['shares'].abs().round(0)
            signal['value'] = signal['value'].round(0)
            buys = signal[signal['side'] == 'BUY']['value'].sum()
            sells = signal[signal['side'] == 'SELL']['value'].sum()
            panels.append(C.card('👤 Insider activity (Form 4, open market)', [
                C.metric_row([
                    (f'${buys/1e6:.1f}M', 'bought', TH.POS),
                    (f'${sells/1e6:.1f}M', 'sold', TH.NEG),
                    (signal[signal['side'] == 'BUY']['insider'].nunique(),
                     'distinct buyers', TH.TEXT),
                ]),
                html.Hr(style={'borderColor': TH.BORDER}),
                C.data_table(signal[['txn_date', 'insider', 'role', 'side',
                                     'shares', 'price', 'value']].head(20),
                             page_size=10),
            ], className='mb-3'))

    # ── headlines ─────────────────────────────────────────────────
    try:
        articles = NEWS.news_asof([ticker], today, lookback_days=days)
    except Exception:                              # noqa: BLE001
        articles = pd.DataFrame()

    if not articles.empty:
        rows = []
        for _i, r in articles.head(15).iterrows():
            rows.append(html.Div([
                html.Span(r.get('sentiment_label', ''),
                          style={'marginRight': '8px'}),
                html.A(r['title'][:130], href=r.get('url') or '#',
                       target='_blank',
                       style={'color': TH.TEXT, 'textDecoration': 'none'}),
                html.Div([
                    html.Span(pd.to_datetime(r['published']).strftime('%Y-%m-%d')),
                    html.Span(f'  ·  {r.get("source", "")}'),
                    html.Span(f'  ·  {r["events"]}' if r.get('events') else ''),
                ], style={'color': TH.MUTED, 'fontSize': '0.72rem',
                          'marginTop': '2px'}),
            ], style={'marginBottom': '12px', 'paddingBottom': '10px',
                      'borderBottom': f'1px solid {TH.BORDER}'}))
        panels.append(C.card('📰 Recent coverage', rows, className='mb-3'))

    # ── AI synthesis across the whole coverage set ────────────────
    # The rule-based extraction above already found the events; this
    # reconciles them across articles into a single read.
    if not articles.empty:
        company = _company_name(ticker)
        panels.append(C.ai_panel(
            lambda: LLM.synthesize_news(ticker, company, articles, summary),
            title=f'🤖 AI briefing — {ticker}',
            footnote=None, className='mb-3'))

    return html.Div(panels)


def _company_name(ticker: str) -> str:
    """Registered name for the prompt, falling back to the ticker."""
    try:
        df = db.read_sql('SELECT name FROM securities WHERE ticker = :t',
                         params={'t': ticker})
        if not df.empty and df.iloc[0]['name']:
            return str(df.iloc[0]['name'])
    except Exception:                              # noqa: BLE001
        pass
    return ticker


@callback(
    Output('intel-policy', 'children'),
    Input('tabs', 'active_tab'),
    State('intel-policy', 'children'),
)
def _load_policy(active_tab, existing):
    if active_tab != 'tab-intel':
        return no_update
    # Panels stay mounted, so a second visit already has this rendered.
    # Re-reading would cost a DB round trip and an LLM call for nothing.
    if existing:
        return no_update
    try:
        docs = POL.policy_asof(date.today(), lookback_days=30)
        heat = POL.sector_heat(date.today(), lookback_days=30)
    except Exception as exc:                       # noqa: BLE001
        log.debug('policy load failed: %s', exc)
        return C.placeholder('Policy data unavailable.')

    if docs.empty:
        return C.note('No policy data stored yet. Run '
                      '"python cli.py ingest --index DIA --full" to populate '
                      'the Federal Register stream.', 'warn')

    children = []
    if not heat.empty:
        children.append(dcc.Graph(
            figure=C.sector_bar(heat.set_index('sector')['documents'],
                                'Policy documents touching each sector (30d)'),
            config={'displayModeBar': False}))

    rows = []
    for _i, r in docs.head(20).iterrows():
        rows.append(html.Div([
            dbc.Badge(r['doc_type'], color='dark', className='me-2',
                      style={'border': f'1px solid {TH.BORDER}'}),
            html.A(r['title'][:140], href=r.get('url') or '#', target='_blank',
                   style={'color': TH.TEXT, 'textDecoration': 'none'}),
            html.Div([
                html.Span(str(pd.to_datetime(r['published']).date())),
                html.Span(f'  ·  {(r.get("agencies") or "")[:70]}'),
                html.Span(f'  →  {r.get("affected_sectors") or "—"}',
                          style={'color': TH.INFO}),
                html.Span(f'  ·  themes: {r["themes"]}' if r.get('themes') else ''),
            ], style={'color': TH.MUTED, 'fontSize': '0.72rem',
                      'marginTop': '3px'}),
        ], style={'marginBottom': '12px', 'paddingBottom': '10px',
                  'borderBottom': f'1px solid {TH.BORDER}'}))

    children.append(html.Div(rows))

    sectors = (heat['sector'].head(8).tolist()
               if not heat.empty and 'sector' in heat else [])
    children.append(C.ai_panel(
        lambda: LLM.policy_impact(docs, sectors),
        title='🤖 AI read on policy exposure', className='mt-3'))

    return html.Div(children)
