"""
Intel tab — company direction, filings, insiders and government policy.

Sentiment is one column here, not the headline. What matters for a thesis is
what changed: guidance raised or cut, capital returned, management replaced,
a restructuring booked, insiders buying, or a rule landing on the sector.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


# ─────────────────────────────────────────────
# COMPANY OVERVIEW & PRICE CHART
# ─────────────────────────────────────────────

def _company_overview(ticker: str) -> html.Div | None:
    """Company header with name, sector, industry, and key financial metrics."""
    try:
        sec_row = db.read_sql(
            'SELECT name, sector, industry, exchange, country '
            'FROM securities WHERE ticker = :t', {'t': ticker})
    except Exception:  # noqa: BLE001
        return None
    if sec_row.empty:
        return None

    r = sec_row.iloc[0]
    name = r['name'] or ticker
    sector = r['sector'] or '—'
    industry = r['industry'] or '—'
    exchange = r['exchange'] or '—'

    metrics = []
    try:
        prof = db.read_sql(
            'SELECT * FROM profile_snapshots WHERE ticker = :t '
            'ORDER BY snapshot_date DESC LIMIT 1', {'t': ticker})
        if not prof.empty:
            p = prof.iloc[0]
            mcap = p.get('market_cap')
            if mcap and pd.notna(mcap) and mcap > 0:
                if mcap >= 1e12:
                    metrics.append((f'${mcap/1e12:.2f}T', 'Mkt Cap', TH.TEXT))
                elif mcap >= 1e9:
                    metrics.append((f'${mcap/1e9:.1f}B', 'Mkt Cap', TH.TEXT))
                else:
                    metrics.append((f'${mcap/1e6:.0f}M', 'Mkt Cap', TH.TEXT))

            for label, col, fmt in [
                ('Fwd P/E', 'forward_pe', '.1f'),
                ('Trail P/E', 'trailing_pe', '.1f'),
                ('P/B', 'price_to_book', '.1f'),
                ('ROE', 'roe', '%'),
                ('Div Yield', 'dividend_yield', 'pct'),
                ('Beta', 'beta', '.2f'),
            ]:
                val = p.get(col)
                if val and pd.notna(val) and val != 0:
                    if fmt == '%':
                        metrics.append((f'{val:.1%}', label, TH.MUTED))
                    elif fmt == 'pct':
                        metrics.append((f'{val:.2f}%', label, TH.MUTED))
                    else:
                        metrics.append((f'{val:{fmt}}', label, TH.MUTED))
    except Exception:  # noqa: BLE001
        pass

    badges = [
        dbc.Badge(sector, color='dark', className='me-2',
                  style={'border': f'1px solid {TH.BORDER}',
                         'fontSize': '0.78rem'}),
        dbc.Badge(industry, color='dark', className='me-2',
                  style={'border': f'1px solid {TH.BORDER}',
                         'fontSize': '0.74rem'}),
        html.Span(f'{exchange}', style={'color': TH.MUTED,
                                        'fontSize': '0.76rem',
                                        'marginLeft': '4px'}),
    ]

    children = [
        html.Div([
            html.Span(name, style={'fontSize': '1.2rem', 'fontWeight': '700',
                                    'marginRight': '10px'}),
            html.Span(ticker, style={'fontSize': '1.0rem', 'color': TH.ACCENT,
                                      'fontWeight': '600'}),
        ], style={'marginBottom': '6px'}),
        html.Div(badges, style={'marginBottom': '10px'}),
    ]
    if metrics:
        children.append(C.metric_row(metrics))

    return C.card(f'🏢 Company overview — {ticker}', children, className='mb-3')


def _price_chart(ticker: str, days: int = 180) -> html.Div | None:
    """Interactive Plotly price chart for the given ticker."""
    try:
        start = date.today() - pd.Timedelta(days=max(days, 180))
        prices = db.read_sql(
            'SELECT date, open, high, low, close, adj_close, volume '
            'FROM prices WHERE ticker = :t AND date >= :s ORDER BY date',
            {'t': ticker, 's': str(start)},
            parse_dates=['date'])
    except Exception:  # noqa: BLE001
        return None
    if prices.empty or len(prices) < 3:
        return None

    close = prices['adj_close'].fillna(prices['close'])
    current = close.iloc[-1]
    prev = close.iloc[0]
    change_pct = (current / prev - 1) * 100
    colour = TH.POS if change_pct >= 0 else TH.NEG

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=prices['date'],
        open=prices['open'],
        high=prices['high'],
        low=prices['low'],
        close=prices['close'],
        increasing={'line': {'color': TH.POS, 'width': 1}},
        decreasing={'line': {'color': TH.NEG, 'width': 1}},
        name=ticker,
    ))

    # 50-day MA
    if len(close) >= 50:
        ma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=prices['date'], y=ma50, name='50d MA',
            line={'color': TH.ACCENT, 'width': 1.2, 'dash': 'dot'},
            hovertemplate='%{x|%Y-%m-%d}<br>50d MA: $%{y:.2f}<extra></extra>',
        ))

    fig.update_layout(
        **TH.plot_layout(f'{ticker}  ${current:.2f}', height=340,
                         showlegend=True,
                         xaxis={'rangeslider': {'visible': False}},
                         yaxis={'title': 'Price ($)',
                                'showgrid': True,
                                'griddash': 'dash',
                                'gridcolor': 'rgba(255,255,255,0.10)'}))

    header = html.Div([
        html.Span(f'${current:.2f}', style={'fontSize': '1.1rem',
                                             'fontWeight': '700',
                                             'marginRight': '10px'}),
        html.Span(f'{change_pct:+.1f}% ({len(prices)} bars)',
                  style={'color': colour, 'fontSize': '0.88rem',
                         'fontWeight': '600'}),
    ], style={'marginBottom': '4px'})

    return C.card(f'📈 Price — {ticker}', [
        header,
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
    ], className='mb-3')


# ─────────────────────────────────────────────
# SCREEN CONTEXT
# ─────────────────────────────────────────────

def _latest_screen_context(ticker: str) -> dict | None:
    """Fetch the latest screen result for this ticker across all strategies."""
    try:
        df = db.read_sql("""
            SELECT r.strategy, r.as_of, s.composite, s.rank, s.signal,
                   s.reasons, r.universe_n, r.created_at
            FROM runs r
            JOIN scores s ON r.run_id = s.run_id
            WHERE s.ticker = :t
            ORDER BY r.created_at DESC
            LIMIT 5
        """, {'t': ticker})
    except Exception:
        return None
    if df.empty:
        return None

    results = []
    seen = set()
    for _, row in df.iterrows():
        strat = row['strategy']
        if strat in seen:
            continue
        seen.add(strat)
        results.append({
            'strategy': strat,
            'as_of': str(row['as_of']),
            'composite': row['composite'],
            'rank': int(row['rank']),
            'signal': row['signal'],
            'reasons': row['reasons'] or '',
            'universe_n': int(row['universe_n'] or 0),
        })
    return {'ticker': ticker, 'results': results} if results else None


def _latest_factor_scores(ticker: str) -> pd.DataFrame | None:
    """Key factor scores from the most recent run."""
    try:
        df = db.read_sql("""
            SELECT fs.factor, fs.raw, fs.sector_z, fs.pct_rank
            FROM factor_scores fs
            JOIN (
                SELECT run_id FROM scores
                WHERE ticker = :t
                ORDER BY rowid DESC LIMIT 1
            ) latest ON fs.run_id = latest.run_id
            WHERE fs.ticker = :t
        """, {'t': ticker})
    except Exception:
        return None
    return df if not df.empty else None


def _screen_context_panel(ticker: str) -> html.Div | None:
    ctx = _latest_screen_context(ticker)
    if not ctx:
        return None

    panels = []
    for r in ctx['results']:
        score = r['composite']
        rank = r['rank']
        signal = r['signal']
        colour = TH.POS if 'Buy' in signal or 'Strong' in signal else (
            TH.WARN if 'Hold' in signal or 'Watch' in signal else TH.MUTED)

        panels.append(html.Div([
            html.Div([
                dbc.Badge(r['strategy'], color='dark', className='me-2',
                          style={'border': f'1px solid {TH.BORDER}',
                                 'fontSize': '0.78rem'}),
                html.Span(signal, style={'fontWeight': '700', 'color': colour,
                                         'fontSize': '0.92rem'}),
                html.Span(f'  ·  rank #{rank} of {r["universe_n"]}',
                          style={'color': TH.MUTED, 'fontSize': '0.82rem',
                                 'marginLeft': '10px'}),
                html.Span(f'  ·  composite {score:+.2f}',
                          style={'color': TH.MUTED, 'fontSize': '0.82rem'}),
            ]),
            html.Div(r['reasons'],
                     style={'color': TH.MUTED, 'fontSize': '0.78rem',
                            'marginTop': '2px', 'marginLeft': '4px'}),
        ], style={'marginBottom': '10px'}))

    # Check for recent M&A to flag inorganic growth
    had_ma = False
    try:
        ma_events = db.read_sql(
            "SELECT 1 FROM corporate_events WHERE ticker = :t "
            "AND item_code = '2.01' AND filed >= date(:cutoff) LIMIT 1",
            {'t': ticker, 'cutoff': str(date.today().replace(year=date.today().year - 1))})
        had_ma = not ma_events.empty
    except Exception:  # noqa: BLE001
        pass
    GROWTH_FACTORS = {'Revenue Growth 1Y', 'Earnings Growth 1Y',
                      'Revenue Cagr 3Y', 'Equity Cagr 3Y'}

    factors = _latest_factor_scores(ticker)
    factor_panel = None
    if factors is not None and not factors.empty:
        key_factors = factors.sort_values('pct_rank', ascending=False).head(5)
        badges = []
        for _, f in key_factors.iterrows():
            pct = f['pct_rank']
            colour = TH.POS if pct > 0.8 else TH.NEG if pct < 0.2 else TH.MUTED
            raw_val = f['raw']
            raw_str = (f'{raw_val:.2f}' if isinstance(raw_val, float)
                       and abs(raw_val) < 100 else f'{raw_val:.0f}')
            label = f['factor'].replace('_', ' ').title()
            ma_tag = (' M&A' if had_ma and label in GROWTH_FACTORS else '')
            badges.append(html.Span([
                html.Span(label,
                          style={'fontSize': '0.72rem', 'color': TH.MUTED}),
                html.Span(f' {raw_str}',
                          style={'fontWeight': '700', 'color': colour,
                                 'fontSize': '0.78rem'}),
                html.Span(f' (p{pct*100:.0f})',
                          style={'fontSize': '0.68rem', 'color': TH.MUTED}),
                *([] if not ma_tag else [html.Span(
                    ma_tag, style={'fontSize': '0.62rem', 'color': TH.WARN,
                                   'fontWeight': '700', 'marginLeft': '2px'})]),
            ], style={'marginRight': '16px', 'display': 'inline-block'}))
        factor_panel = html.Div([
            html.Div('Top factor scores (percentile within sector):',
                     style={'color': TH.MUTED, 'fontSize': '0.74rem',
                            'marginBottom': '4px'}),
            html.Div(badges),
        ], style={'marginTop': '8px'})

    return C.card(f'📊 Screen recommendation — {ticker}', [
        html.Div(panels),
        factor_panel,
        html.Div(f'as of {ctx["results"][0]["as_of"]}',
                 style={'color': TH.MUTED, 'fontSize': '0.7rem',
                        'marginTop': '4px'}),
    ], className='mb-3')


# ─────────────────────────────────────────────
# DIRECTION NOTES (enriched)
# ─────────────────────────────────────────────

def _enriched_direction(summary: dict) -> html.Div:
    """Direction notes with positive/negative grouping and counts."""
    notes = summary.get('direction_notes') or []
    if not notes:
        return C.note('No directional signals extracted from recent coverage.',
                      'info')

    analyst = summary.get('analyst_actions') or {}
    guidance = summary.get('guidance') or {}

    positive_keywords = {'raised', 'upgrade', 'beat', 'expansion', 'buyers',
                         'increased', 'buyback', 'Dividend increased',
                         'initiated'}
    negative_keywords = {'cut', 'downgrade', 'missed', 'deteriorated',
                         'withdrawn', 'selling', 'overhang', 'restructuring',
                         'pressure'}

    pos_notes, neg_notes, neutral_notes = [], [], []
    for n in notes:
        low = n.lower()
        if any(k in low for k in positive_keywords):
            pos_notes.append(n)
        elif any(k in low for k in negative_keywords):
            neg_notes.append(n)
        else:
            neutral_notes.append(n)

    rows = []
    if pos_notes:
        rows.append(html.Div([
            html.Span('🟢 Supports: ', style={'fontWeight': '700',
                                               'color': TH.POS,
                                               'fontSize': '0.84rem'}),
            html.Span(' · '.join(pos_notes),
                      style={'fontSize': '0.84rem'}),
        ], style={'marginBottom': '8px'}))
    if neg_notes:
        rows.append(html.Div([
            html.Span('🔴 Concerns: ', style={'fontWeight': '700',
                                               'color': TH.NEG,
                                               'fontSize': '0.84rem'}),
            html.Span(' · '.join(neg_notes),
                      style={'fontSize': '0.84rem'}),
        ], style={'marginBottom': '8px'}))
    if neutral_notes:
        rows.append(html.Div([
            html.Span('⚪ Other: ', style={'fontWeight': '700',
                                            'color': TH.MUTED,
                                            'fontSize': '0.84rem'}),
            html.Span(' · '.join(neutral_notes),
                      style={'fontSize': '0.84rem'}),
        ], style={'marginBottom': '8px'}))

    if analyst:
        n_up = analyst.get('upgrade', 0)
        n_dn = analyst.get('downgrade', 0)
        net = n_up - n_dn
        net_str = (f'net +{net} upgrade' if net > 0
                   else f'net {net} downgrade' if net < 0
                   else 'mixed')
        net_colour = TH.POS if net > 0 else TH.NEG if net < 0 else TH.WARN
        rows.append(html.Div([
            html.Span('Analyst actions: ', style={'color': TH.MUTED,
                                                   'fontSize': '0.76rem'}),
            html.Span(f'{n_up} up, {n_dn} down',
                      style={'fontSize': '0.76rem', 'fontWeight': '700'}),
            html.Span(f'  ({net_str})',
                      style={'fontSize': '0.76rem', 'color': net_colour,
                             'fontWeight': '600'}),
        ]))
    if guidance:
        parts = [f'{v}× {k}' for k, v in guidance.items()]
        rows.append(html.Div([
            html.Span('Guidance signals: ', style={'color': TH.MUTED,
                                                    'fontSize': '0.76rem'}),
            html.Span(', '.join(parts),
                      style={'fontSize': '0.76rem'}),
        ]))

    return html.Div([
        html.Div(rows),
        html.Div(
            f'{summary.get("material_count", 0)} material of '
            f'{summary.get("n_articles", 0)} articles  ·  '
            f'mean sentiment {summary.get("mean_sentiment", 0):+.2f}',
            style={'color': TH.MUTED, 'fontSize': '0.76rem',
                   'marginTop': '8px'}),
    ])


# ─────────────────────────────────────────────
# CONFIRMATION SCORECARD
# ─────────────────────────────────────────────

def _confirmation_scorecard(ticker: str, summary: dict, events: pd.DataFrame,
                            insiders: pd.DataFrame) -> html.Div | None:
    """Synthesize all intel into a buy/avoid checklist."""
    supports, concerns, checks = [], [], []

    # News direction
    notes = summary.get('direction_notes') or []
    analyst = summary.get('analyst_actions') or {}
    guidance = summary.get('guidance') or {}

    if guidance.get('raised') or guidance.get('beat'):
        supports.append('Management raised guidance or beat expectations')
    if guidance.get('lowered') or guidance.get('missed'):
        concerns.append('Management cut guidance or missed expectations')
    if guidance.get('withdrawn'):
        concerns.append('Guidance withdrawn — visibility deteriorated')

    n_upgrades = analyst.get('upgrade', 0)
    n_downgrades = analyst.get('downgrade', 0)
    if n_upgrades > n_downgrades:
        supports.append(f'Net analyst upgrades ({n_upgrades} up, {n_downgrades} down)')
    elif n_downgrades > n_upgrades:
        concerns.append(f'Net analyst downgrades ({n_downgrades} down, {n_upgrades} up)')
    elif n_upgrades and n_downgrades:
        checks.append(f'Analyst disagreement ({n_upgrades} upgrade, {n_downgrades} downgrade) — read the reasoning')

    # Insider activity — buys are voluntary and meaningful; sells need
    # holdings context because most are routine compensation events.
    if not insiders.empty:
        signal = insiders[insiders['txn_code'].isin(['P', 'S'])].copy()
        if not signal.empty:
            buys = signal[signal['shares'] > 0]['value'].sum()
            sells = signal[signal['shares'] < 0]['value'].sum()
            n_buyers = signal[signal['shares'] > 0]['insider'].nunique()
            if buys > sells * 2 and n_buyers >= 2:
                supports.append(f'Cluster insider buying ({n_buyers} buyers, ${buys/1e6:.1f}M)')
            elif buys > sells:
                supports.append(f'Insiders net buyers (${buys/1e6:.1f}M bought)')
            elif sells > buys:
                # Check if we have holdings context to qualify the selling
                has_pct = 'post_txn_shares' in signal.columns
                big_sales = pd.DataFrame()
                if has_pct:
                    sell_rows = signal[signal['shares'] < 0].copy()
                    sell_rows['pct'] = sell_rows.apply(
                        lambda r: abs(r['shares']) /
                        (r['post_txn_shares'] + abs(r['shares'])) * 100
                        if pd.notna(r.get('post_txn_shares'))
                        and r['post_txn_shares'] + abs(r['shares']) > 0
                        else np.nan, axis=1)
                    big_sales = sell_rows[sell_rows['pct'] > 20]
                if not big_sales.empty:
                    concerns.append(
                        f'Insider selling with large position reductions '
                        f'(>20% of holdings, ${sells/1e6:.1f}M)')
                elif sells > buys * 3:
                    checks.append(
                        f'Heavy insider selling (${sells/1e6:.1f}M) — '
                        f'check % of holdings in insider panel to assess conviction')
                else:
                    checks.append(
                        f'Insiders net sellers — likely routine '
                        f'(check insider panel for position context)')

    # Filing pattern
    if not events.empty:
        codes = events['item_code'].value_counts().to_dict()
        if codes.get('4.02'):
            concerns.append('Prior financials declared unreliable (restatement)')
        if codes.get('2.06'):
            concerns.append('Material asset impairment booked')
        if codes.get('2.05'):
            checks.append('Restructuring charges filed — turnaround or decline?')
        if codes.get('2.01'):
            checks.append('M&A completed — trailing ratios now blend old and new company')
        if codes.get('3.01'):
            concerns.append('Listing-rule problem disclosed')

    sentiment = summary.get('mean_sentiment', 0)
    if sentiment > 0.2:
        supports.append(f'Positive coverage sentiment ({sentiment:+.2f})')
    elif sentiment < -0.2:
        concerns.append(f'Negative coverage sentiment ({sentiment:+.2f})')

    # Valuation / technical extension guardrail
    try:
        prices = db.read_sql(
            'SELECT close FROM prices WHERE ticker = :t '
            'ORDER BY date DESC LIMIT 200', {'t': ticker})
        if len(prices) >= 200:
            current = prices.iloc[0]['close']
            ma200 = prices['close'].mean()
            pct_above = (current / ma200 - 1) * 100
            if pct_above > 15:
                concerns.append(
                    f'Technically extended — {pct_above:+.0f}% above 200-day MA '
                    f'(${current:.0f} vs ${ma200:.0f})')
            elif pct_above > 10:
                checks.append(
                    f'Price elevated — {pct_above:+.0f}% above 200-day MA '
                    f'(${current:.0f} vs ${ma200:.0f})')
            elif pct_above < -15:
                checks.append(
                    f'Deep below 200-day MA ({pct_above:+.0f}%) — '
                    f'check if value or value trap')
    except Exception:  # noqa: BLE001
        pass

    if not supports and not concerns and not checks:
        return None

    def _list(items, colour):
        return html.Ul([
            html.Li(item, style={'marginBottom': '4px', 'fontSize': '0.84rem'})
            for item in items
        ], style={'paddingLeft': '18px', 'color': colour, 'marginBottom': '6px'})

    panels = []
    if supports:
        panels.append(html.Div([
            html.B('✅ Supports the recommendation',
                   style={'color': TH.POS, 'fontSize': '0.86rem'}),
            _list(supports, TH.TEXT),
        ]))
    if concerns:
        panels.append(html.Div([
            html.B('⚠️ Concerns',
                   style={'color': TH.WARN, 'fontSize': '0.86rem'}),
            _list(concerns, TH.TEXT),
        ]))
    if checks:
        panels.append(html.Div([
            html.B('🔍 Check before acting',
                   style={'color': TH.INFO,
                          'fontSize': '0.86rem'}),
            _list(checks, TH.TEXT),
        ]))

    return C.card(f'📋 Confirmation scorecard — {ticker}', panels,
                  className='mb-3')


# ─────────────────────────────────────────────
# INSIDER INTERPRETATION
# ─────────────────────────────────────────────

def _insider_interpretation(signal: pd.DataFrame, buys: float, sells: float,
                            n_buyers: int, n_sellers: int) -> html.Div:
    """Context-aware read of insider transactions."""
    notes = []
    colour = TH.MUTED

    sellers = signal[signal['side'] == 'SELL']
    buyers = signal[signal['side'] == 'BUY']

    # Check if we have holdings data to assess % sold
    has_pct = 'pct_of_holdings' in signal.columns
    if has_pct and not sellers.empty:
        big_sales = sellers[sellers['pct_of_holdings'] > 20]
        small_sales = sellers[sellers['pct_of_holdings'] <= 5]
        if not big_sales.empty:
            names = big_sales['insider'].unique()[:3]
            notes.append(f'Large position reduction (>20% of holdings) by '
                         f'{", ".join(names)} — may signal conviction')
            colour = TH.WARN
        elif len(small_sales) == len(sellers.dropna(subset=['pct_of_holdings'])):
            notes.append('All sales are small (<5% of holdings) — '
                         'likely routine diversification or tax management')

    # Cluster detection: multiple C-suite selling in same week
    if not sellers.empty and n_sellers >= 2:
        csuite_roles = {'CEO', 'CFO', 'COO', 'CTO', 'President', 'Chief'}
        csuite_sellers = sellers[sellers['role'].str.contains(
            '|'.join(csuite_roles), case=False, na=False)]
        if len(csuite_sellers['insider'].unique()) >= 2:
            dates = pd.to_datetime(sellers['txn_date'])
            span = (dates.max() - dates.min()).days
            if span <= 14:
                notes.append(f'{len(csuite_sellers["insider"].unique())} '
                             f'C-suite officers sold within {span} days — '
                             f'clustered selling is a stronger signal')
                colour = TH.NEG

    # Buys are rarer and more meaningful
    if n_buyers >= 2 and buys > 100_000:
        notes.append(f'Cluster buying by {n_buyers} insiders '
                     f'(${buys/1e6:.1f}M) — open-market purchases are '
                     f'voluntary and historically predictive')
        colour = TH.POS
    elif n_buyers == 1 and buys > 500_000:
        buyer = buyers.iloc[0]['insider'] if not buyers.empty else 'insider'
        notes.append(f'{buyer} bought ${buys/1e6:.1f}M — '
                     f'single large purchase suggests conviction')
        colour = TH.POS

    # No buys, only sells — add context
    if n_buyers == 0 and n_sellers > 0 and not notes:
        notes.append('All transactions are sales — common for '
                     'executives with equity compensation. '
                     'Check "% held" column for position context')

    if not notes:
        notes.append('Mixed activity with no clear directional pattern')

    return html.Div([
        html.Div([
            html.Span('Reading: ', style={'color': TH.ACCENT,
                                           'fontWeight': '700',
                                           'fontSize': '0.84rem'}),
            html.Span(' · '.join(notes),
                      style={'fontSize': '0.84rem', 'color': colour}),
        ], style={'margin': '8px 0'}),
    ])


# ─────────────────────────────────────────────
# AI BRIEFING CACHE
# ─────────────────────────────────────────────

def _article_hash(articles: pd.DataFrame) -> str:
    """Short hash of article IDs to detect when new articles arrive."""
    import hashlib
    ids = sorted(articles.index.astype(str).tolist()[:30])
    return hashlib.md5('|'.join(ids).encode()).hexdigest()[:12]


def _cached_briefing(ticker: str, articles: pd.DataFrame,
                     summary: dict) -> str | None:
    """Return cached AI briefing if fresh, otherwise generate and cache."""
    brief_type = 'news_synthesis'
    current_hash = _article_hash(articles)

    try:
        cached = db.read_sql(
            'SELECT content, article_hash, generated_at FROM company_briefs '
            'WHERE ticker = :t AND brief_type = :bt',
            {'t': ticker, 'bt': brief_type})
        if not cached.empty:
            row = cached.iloc[0]
            if row['article_hash'] == current_hash:
                return row['content']
    except Exception:  # noqa: BLE001
        pass

    company = _company_name(ticker)
    text = LLM.synthesize_news(ticker, company, articles, summary)

    if text:
        try:
            import config
            db.upsert(db.company_briefs, [{
                'ticker': ticker,
                'brief_type': brief_type,
                'content': text,
                'generated_at': config.utc_now(),
                'article_hash': current_hash,
            }])
        except Exception:  # noqa: BLE001
            pass

    return text


# ─────────────────────────────────────────────
# MAIN CALLBACK
# ─────────────────────────────────────────────

EDGAR_8K_URL = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&dateb=&owner=include&count=10'


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

    # Fetch independent data sources in parallel.
    def _fetch_screen():
        return _screen_context_panel(ticker)

    def _fetch_summary():
        try:
            return NEWS.direction_summary(ticker, today, lookback_days=days)
        except Exception as exc:                   # noqa: BLE001
            log.debug('direction summary failed: %s', exc)
            return {}

    def _fetch_events():
        try:
            return SEC.events_asof([ticker], today, lookback_days=max(days, 90))
        except Exception:                          # noqa: BLE001
            return pd.DataFrame()

    def _fetch_insiders():
        try:
            return SEC.insiders_asof([ticker], today, lookback_days=max(days, 180))
        except Exception:                          # noqa: BLE001
            return pd.DataFrame()

    def _fetch_articles():
        try:
            return NEWS.news_asof([ticker], today, lookback_days=days)
        except Exception:                          # noqa: BLE001
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=7) as pool:
        f_overview = pool.submit(_company_overview, ticker)
        f_chart = pool.submit(_price_chart, ticker, days)
        f_screen = pool.submit(_fetch_screen)
        f_summary = pool.submit(_fetch_summary)
        f_events = pool.submit(_fetch_events)
        f_insiders = pool.submit(_fetch_insiders)
        f_articles = pool.submit(_fetch_articles)

        overview = f_overview.result()
        chart = f_chart.result()
        screen_panel = f_screen.result()
        summary = f_summary.result()
        events = f_events.result()
        ins = f_insiders.result()
        articles = f_articles.result()

    # ── company overview & price chart ───────────────────────────
    if overview:
        panels.append(overview)
    if chart:
        panels.append(chart)

    # ── screen recommendation context ────────────────────────────
    if screen_panel:
        panels.append(screen_panel)

    notes = summary.get('direction_notes') or []
    if notes:
        panels.append(C.card(
            f'🧭 What recent coverage implies — {ticker}',
            _enriched_direction(summary), className='mb-3'))
    else:
        panels.append(C.note(
            f'No recent news stored for {ticker}. Run '
            f'"python cli.py ingest --index DIA --full" to populate the news '
            f'stream.', 'warn'))

    # ── 8-K corporate events (structured, no NLP needed) ──────────
    if not events.empty:
        ev = events.copy()
        ev['filed'] = pd.to_datetime(ev['filed']).dt.date
        ev['significance'] = ev['item_code'].map(EK.significance)
        ek = EK.summarize(ev)

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

            html.Div([
                html.Span('Item codes are from the filer, not parsed. ',
                          style={'color': TH.MUTED, 'fontSize': '0.75rem'}),
                html.A('View actual filings on EDGAR →',
                       href=EDGAR_8K_URL.format(ticker=ticker),
                       target='_blank',
                       style={'fontSize': '0.75rem'}),
            ], style={'margin': '10px 0 6px'}),

            dbc.Accordion(explain, start_collapsed=True, always_open=True,
                          flush=True),
        ], className='mb-3'))

    # ── insider activity ──────────────────────────────────────────
    if not ins.empty:
        TXN_LABELS = {'P': 'Purchase', 'S': 'Sale', 'A': 'Award',
                      'M': 'Exercise', 'F': 'Tax w/h', 'G': 'Gift',
                      'D': 'Disposition'}
        signal = ins[ins['txn_code'].isin(['P', 'S'])].copy()
        other = ins[~ins['txn_code'].isin(['P', 'S'])].copy()
        if not signal.empty:
            signal['txn_date'] = pd.to_datetime(signal['txn_date']).dt.date
            signal['side'] = signal['shares'].apply(
                lambda s: 'BUY' if (s or 0) > 0 else 'SELL')
            signal['shares_raw'] = signal['shares']
            signal['shares'] = signal['shares'].abs().round(0)
            signal['value'] = signal['value'].round(0)

            # Holdings context: % of position sold/bought
            has_holdings = 'post_txn_shares' in signal.columns
            if has_holdings:
                signal['pct_of_holdings'] = signal.apply(
                    lambda r: (abs(r['shares_raw']) /
                               (r['post_txn_shares'] + abs(r['shares_raw'])) * 100)
                    if pd.notna(r.get('post_txn_shares'))
                    and r['post_txn_shares'] + abs(r['shares_raw']) > 0
                    else np.nan, axis=1)

            buys = signal[signal['side'] == 'BUY']['value'].sum()
            sells = signal[signal['side'] == 'SELL']['value'].sum()
            n_buyers = signal[signal['side'] == 'BUY']['insider'].nunique()
            n_sellers = signal[signal['side'] == 'SELL']['insider'].nunique()

            # Interpret the insider activity
            interpretation = _insider_interpretation(signal, buys, sells,
                                                     n_buyers, n_sellers)

            show_cols = ['txn_date', 'insider', 'role', 'side',
                         'shares', 'price', 'value']
            if has_holdings and signal['pct_of_holdings'].notna().any():
                signal['% held'] = signal['pct_of_holdings'].apply(
                    lambda v: f'{v:.0f}%' if pd.notna(v) else '—')
                show_cols.append('% held')

            panels.append(C.card('👤 Insider activity (Form 4, open market)', [
                C.metric_row([
                    (f'${buys/1e6:.1f}M', 'bought', TH.POS),
                    (f'${sells/1e6:.1f}M', 'sold', TH.NEG),
                    (n_buyers, 'distinct buyers', TH.TEXT),
                    (n_sellers, 'distinct sellers', TH.TEXT),
                ]),
                interpretation,
                html.Hr(style={'borderColor': TH.BORDER}),
                C.data_table(signal[show_cols].head(20), page_size=10),
            ], className='mb-3'))
        if not other.empty and len(other) >= 2:
            other = other.copy()
            other['txn_date'] = pd.to_datetime(other['txn_date']).dt.date
            other['type'] = other['txn_code'].map(TXN_LABELS).fillna(other['txn_code'])
            other['shares'] = other['shares'].abs().round(0)
            other['value'] = other['value'].fillna(0).round(0)
            n_awards = len(other[other['txn_code'] == 'A'])
            n_exercises = len(other[other['txn_code'] == 'M'])
            label = (f'📋 Other insider transactions ({n_awards} award'
                     f'{"s" if n_awards != 1 else ""}, '
                     f'{n_exercises} exercise'
                     f'{"s" if n_exercises != 1 else ""}, '
                     f'{len(other) - n_awards - n_exercises} other)')
            panels.append(C.card(label, [
                C.note('Awards (A), option exercises (M), and tax withholdings (F) '
                       'are routine compensation events — not conviction signals. '
                       'Most insider selling follows a vest→exercise→sell cycle and '
                       'does not indicate bearish conviction.',
                       'info'),
                C.data_table(other[['txn_date', 'insider', 'role', 'type',
                                    'shares', 'value']].head(15),
                             page_size=8),
            ], className='mb-3'))

    # ── headlines ─────────────────────────────────────────────────
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

    # ── confirmation scorecard ────────────────────────────────────
    scorecard = _confirmation_scorecard(ticker, summary, events, ins)
    if scorecard:
        panels.append(scorecard)

    # ── AI synthesis across the whole coverage set ────────────────
    if not articles.empty:
        panels.append(C.ai_panel(
            lambda: _cached_briefing(ticker, articles, summary),
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
                          style={'color': TH.INFO
                                 if hasattr(TH, 'INFO') else TH.ACCENT}),
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
