"""
Data tab — coverage, freshness, and where the sources disagree.

Deliberately surfaces the membership disagreement between the crowd-sourced
Wikipedia spine and the ETF's own N-PORT filing rather than resolving it
silently. When two sources differ, that is information about data quality, and
hiding it behind a single reconciled number would be the wrong call.
"""

from __future__ import annotations

import logging
from datetime import date

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

import config
from core import db, http
from data import members
from data import sync as SY
from nlp import llm as LLM
from ui import components as C
from ui import theme as TH

log = logging.getLogger(__name__)

INGEST_HELP = """\
# One-time setup for a full S&P 500 universe (takes a while)
python cli.py init
python cli.py ingest --index SPY --full --members-from 2015-01-01

# Faster: the Dow's 30 names
python cli.py ingest --index DIA --full

# Then, nightly
python cli.py ingest --index SPY
"""


def layout() -> html.Div:
    return html.Div([
        C.card('🗄️ Data health', [
            html.Div(id='data-health',
                     style={'minHeight': '40px'}),
        ], className='mb-3'),

        C.card('⬇️ Pull new data', [
            dbc.Row([
                dbc.Col([
                    C.label('Universe to refresh'),
                    dcc.Dropdown(
                        id='pull-index',
                        options=[{'label': '🗄️ All stored tickers', 'value': 'ALL'}]
                                + [{'label': f'{k} — {v}', 'value': k}
                                   for k, v in config.INDEX_OPTIONS.items()],
                        value='DIA', clearable=False),
                ], lg=4, md=6, className='mb-2'),
                dbc.Col([
                    C.label('Scope'),
                    dcc.RadioItems(
                        id='pull-scope',
                        options=[
                            {'label': '  Stale only (recommended)', 'value': 'stale'},
                            {'label': '  Everything (slow)', 'value': 'force'},
                        ],
                        value='stale', inline=False,
                        inputStyle={'marginRight': '6px'},
                        labelStyle={'display': 'block', 'fontSize': '0.83rem'}),
                ], lg=4, md=6, className='mb-2'),
                dbc.Col([
                    C.gradient_button('⬇️  Pull new data', 'pull-button'),
                ], lg=4, md=12, className='mb-2 d-flex align-items-end'),
            ], className='align-items-end'),

            dbc.Row([
                dbc.Col([
                    dbc.Button('🔄 Refresh all index memberships',
                               id='sync-members-button', size='sm',
                               color='dark', className='mt-2',
                               style={'border': f'1px solid {TH.BORDER}',
                                      'fontSize': '0.78rem'}),
                    html.Span(' Checks SPY / QQQ / DIA / IWM for '
                              'constituent changes in one click.',
                              style={'color': TH.MUTED, 'fontSize': '0.72rem',
                                     'marginLeft': '8px'}),
                ]),
            ]),
            C.loading(html.Div(id='pull-status',
                               style={'fontSize': '0.85rem', 'minHeight': '26px',
                                      'marginTop': '12px'}),
                      'pull-loading'),
            html.Div(id='pull-report'),
        ], className='mb-3'),

        C.card('🔬 Sanity checks', [
            dbc.Row([
                dbc.Col([
                    C.label('Universe to validate'),
                    dcc.Dropdown(
                        id='validate-universe',
                        options=[
                            {'label': '🗄️ All stored tickers', 'value': 'ALL'},
                            {'label': 'DIA — Dow 30 (fast)', 'value': 'DIA'},
                            {'label': 'SPY — S&P 500 (thorough)', 'value': 'SPY'},
                            {'label': 'QQQ — Nasdaq 100', 'value': 'QQQ'},
                        ],
                        value='ALL', clearable=False),
                ], lg=4, md=6, className='mb-2'),
                dbc.Col([
                    C.label('Include survivorship check'),
                    dcc.Checklist(
                        id='validate-survivorship',
                        options=[{'label': '  Test historical membership (slower)',
                                  'value': 'yes'}],
                        value=[],
                        inputStyle={'marginRight': '6px'},
                        labelStyle={'fontSize': '0.83rem'}),
                ], lg=4, md=6, className='mb-2'),
                dbc.Col([
                    C.gradient_button('🔬  Run sanity checks', 'validate-button'),
                ], lg=4, md=12, className='mb-2 d-flex align-items-end'),
            ], className='align-items-end'),
            C.loading(html.Div(id='validate-results',
                               style={'minHeight': '26px', 'marginTop': '12px'}),
                      'validate-loading'),
        ], subtitle='Builds the factor frame for the chosen universe and runs '
                    'every integrity check — bounds, infinities, coverage, '
                    'market-cap agreement, stale shares, split discontinuities.',
           className='mb-3'),

        dbc.Button('📋 Details', id='data-details-btn', size='sm',
                   color='dark', className='mb-3',
                   style={'border': f'1px solid {TH.BORDER}',
                          'fontSize': '0.78rem'}),
        dbc.Collapse([
            dbc.Row([
                dbc.Col(C.card('🗄️ Stored data', html.Div(id='data-tables')),
                        lg=6, className='mb-3'),
                dbc.Col(C.card('🕐 Ingest freshness',
                               html.Div(id='data-freshness')),
                        lg=6, className='mb-3'),
            ]),
            dbc.Row([
                dbc.Col(C.card('🔍 Index membership sources',
                               html.Div(id='data-members')),
                        lg=7, className='mb-3'),
                dbc.Col(C.card('💾 HTTP cache', html.Div(id='data-cache')),
                        lg=5, className='mb-3'),
            ]),
            C.card('🔄 Source freshness',
                   html.Div(id='data-sync'),
                   subtitle='Every run syncs first, then reads. A source inside '
                            'its refresh window is served from the database and '
                            'never re-fetched.',
                   className='mb-3'),

            C.card('🔬 Data validity', html.Div(id='data-validity'),
                   subtitle='Looks for values that are present and wrong, not '
                            'just missing — a wrong number ranks a company on '
                            'a fiction, a missing one only costs breadth.',
                   className='mb-3'),

            C.card('🏦 Market cap coverage', html.Div(id='data-mcap'),
                   className='mb-3'),

            C.card('🤖 AI analysis layer', html.Div(id='data-ai'),
                   className='mb-3'),

            C.card('🚀 Populating data', [
                html.P('All ingest runs from the command line, which is what '
                       'makes scheduling possible. Data is stored locally in '
                       'SQLite and refreshed incrementally.',
                       style={'fontSize': '0.84rem'}),
                html.Pre(INGEST_HELP,
                         style={'background': TH.PANEL_ALT, 'padding': '14px',
                                'borderRadius': '8px', 'fontSize': '0.74rem',
                                'color': TH.TEXT, 'overflowX': 'auto'}),
                html.Div(f'Database: {config.DATABASE_URL}',
                         style={'color': TH.MUTED, 'fontSize': '0.72rem'}),
            ], className='mb-3'),
        ], id='data-details', is_open=False),
    ])


def _mcap_block() -> html.Div:
    """How many names have a market cap, and where it came from."""
    from data import marketdata as MD
    from data.universe import PRESETS

    st = MD.status()
    try:
        tickers = PRESETS['spy'].resolve()
        missing = MD.gaps(tickers)
        covered = len(tickers) - len(missing)
    except Exception:                              # noqa: BLE001
        tickers, missing, covered = [], [], 0

    return html.Div([
        C.metric_row([
            (covered, 'from SEC filings', TH.POS, 'auditable'),
            (len(missing), 'need external fill',
             TH.WARN if missing else TH.MUTED, 'multi-class filers'),
        ]),
        html.Div([
            dbc.Badge('ENABLED' if st['enabled'] else 'SEC ONLY',
                      color='success' if st['enabled'] else 'secondary',
                      className='me-2'),
            html.Span(st['reason'], style={'fontSize': '0.84rem'}),
        ], className='my-2'),
        html.Div(f"Not covered by SEC: {', '.join(sorted(missing)[:14])}"
                 f"{'…' if len(missing) > 14 else ''}"
                 if missing else 'Every name has a filed share count.',
                 style={'color': TH.MUTED, 'fontSize': '0.76rem'}),
        html.Div('A filed share count always wins. The external source only '
                 'fills gaps, and its values are stamped with the run date so '
                 'a historical backtest can never see a quote fetched today.',
                 style={'color': TH.MUTED, 'fontSize': '0.75rem',
                        'marginTop': '8px'}),
    ])


def _validity_block() -> html.Div:
    """Placeholder directing users to the manual sanity-checks button."""
    return C.note(
        'Use the \U0001f52c Run sanity checks button above to validate '
        'data integrity on demand.',
        'info',
    )


@callback(
    Output('validate-results', 'children'),
    Input('validate-button', 'n_clicks'),
    State('validate-universe', 'value'),
    State('validate-survivorship', 'value'),
    prevent_initial_call=True,
)
def _run_validation(n_clicks, universe, survivorship):
    if not n_clicks:
        return no_update

    import time
    from quant import factors as FA
    from quant import validate as V

    t0 = time.perf_counter()
    try:
        if universe == 'ALL':
            tickers = db.read_sql(
                'SELECT ticker FROM securities ORDER BY ticker'
            )['ticker'].tolist()
            universe_label = 'all stored'
        else:
            from data.universe import PRESETS
            preset = universe.lower()
            tickers = PRESETS.get(preset, PRESETS['dia']).resolve()
            universe_label = universe
    except Exception as exc:                         # noqa: BLE001
        return C.note(f'Could not resolve universe: {exc}', 'error')

    if not tickers:
        return C.note(f'No tickers found for {universe}.', 'error')

    try:
        factors = FA.build_all(tickers, date.today())
    except Exception as exc:                         # noqa: BLE001
        log.exception('validation factor build failed')
        return C.note(f'Factor build failed: {exc}', 'error')

    findings = V.run_all(factors)

    if survivorship and 'yes' in survivorship and universe != 'ALL':
        for yr_back in (1, 3, 5):
            try:
                as_of = date(date.today().year - yr_back, date.today().month,
                             date.today().day)
                findings += V.check_survivorship(universe, as_of)
            except Exception as exc:                 # noqa: BLE001
                log.debug('survivorship %dY failed: %s', yr_back, exc)

    elapsed = time.perf_counter() - t0
    errors = [f for f in findings if f.severity == 'error']
    warnings_ = [f for f in findings if f.severity == 'warning']
    infos = [f for f in findings if f.severity == 'info']

    header = C.metric_row([
        (len(errors), 'errors', TH.NEG if errors else TH.POS),
        (len(warnings_), 'warnings', TH.WARN if warnings_ else TH.MUTED),
        (len(infos), 'info', TH.MUTED),
        (f'{elapsed:.1f}s', 'elapsed', TH.TEXT),
        (f'{len(tickers)}', 'tickers checked', TH.TEXT),
    ])

    if not findings:
        body = C.note('Every check passed: no impossible ratios, no '
                      'infinities, no future-dated bars, no facts filed '
                      'before the period they describe.', 'info')
    else:
        body = C.data_table(V.to_frame(findings), page_size=12,
                            extra_conditional=[
            {'if': {'filter_query': '{severity} = "error"',
                    'column_id': 'severity'},
             'color': TH.NEG, 'fontWeight': '700'},
            {'if': {'filter_query': '{severity} = "warning"',
                    'column_id': 'severity'}, 'color': TH.WARN},
        ])

    verdict = 'pass' if not errors else 'FAIL'
    verdict_color = TH.POS if not errors else TH.NEG

    return html.Div([
        header,
        html.Div([
            dbc.Badge(verdict, color='success' if not errors else 'danger',
                      className='me-2',
                      style={'fontSize': '0.82rem', 'padding': '6px 12px'}),
            html.Span(f'Checked {len(tickers)} tickers ({universe_label})',
                      style={'color': TH.MUTED, 'fontSize': '0.82rem'}),
        ], className='my-2'),
        html.Hr(style={'borderColor': TH.BORDER}),
        body,
        html.Div('Checks report rather than repair. Silently patching data is '
                 'how a store stops being trustworthy.',
                 style={'color': TH.MUTED, 'fontSize': '0.75rem',
                        'marginTop': '10px'}),
    ])


@callback(
    Output('data-details', 'is_open'),
    Input('data-details-btn', 'n_clicks'),
    State('data-details', 'is_open'),
    prevent_initial_call=True,
)
def _toggle_details(n, is_open):
    return not is_open


@callback(
    Output('data-health', 'children'),
    Input('tabs', 'active_tab'),
)
def _health_badge(active_tab):
    if active_tab != 'tab-data':
        return no_update
    try:
        fresh = SY.freshness()
        n_fresh = int((fresh['status'] == 'fresh').sum())
        n_stale = int((fresh['status'] == 'stale').sum())
        n_never = int((fresh['status'] == 'never fetched').sum())
        total = n_fresh + n_stale + n_never

        if n_never > 0:
            never_names = list(fresh.loc[fresh['status'] == 'never fetched', 'source'])
            suffix = f' ({", ".join(never_names)})'
            if n_fresh > 0:
                color, verdict = TH.WARN, f'{n_never} source never fetched'
            else:
                color, verdict = TH.WARN, 'Not initialised'
        elif n_stale > 0:
            color, verdict = TH.WARN, 'Stale — pull recommended'
        else:
            color, verdict = TH.POS, 'All sources fresh'
            suffix = ''

        last = SY.last_session()
        last_pulled = fresh['last refreshed'].replace('never', pd.NaT)
        last_pulled = pd.to_datetime(last_pulled, errors='coerce')
        newest_pull = last_pulled.max()
        pull_str = (config.local_fmt(newest_pull)
                    if pd.notna(newest_pull) else 'never')
    except Exception:                                # noqa: BLE001
        return C.note('Could not read freshness.', 'warn')

    return C.metric_row([
        (verdict + (suffix if n_never > 0 else ''), 'status', color),
        (f'{n_fresh}/{total}', 'sources fresh', TH.POS if n_fresh == total else TH.MUTED),
        (pull_str, 'last pulled', TH.TEXT),
    ])


@callback(
    Output('pull-status', 'children'), Output('pull-report', 'children'),
    Output('data-sync', 'children', allow_duplicate=True),
    Output('data-tables', 'children', allow_duplicate=True),
    Input('pull-button', 'n_clicks'),
    State('pull-index', 'value'), State('pull-scope', 'value'),
    prevent_initial_call=True,
)
def _pull(n_clicks, index, scope):
    """Refresh stale sources. The single network entry point in the UI."""
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    try:
        if index == 'ALL':
            tickers = db.read_sql(
                'SELECT ticker FROM securities ORDER BY ticker'
            )['ticker'].tolist()
            if not tickers:
                return ('❌ No tickers stored yet. Pull an index first to seed '
                        'the database.', None, no_update, no_update)
            report = SY.sync(tickers, index=None, force=(scope == 'force'))
        else:
            tickers = members.latest_members(index)
            if not tickers:
                return (f'❌ Could not resolve members for {index}. Run '
                        f'"python cli.py ingest --index {index}" once to seed it.',
                        None, no_update, no_update)
            report = SY.sync(tickers, index=index, force=(scope == 'force'))
    except Exception as exc:                       # noqa: BLE001
        log.exception('pull failed')
        return (f'❌ {type(exc).__name__}: {str(exc)[:220]}',
                None, no_update, no_update)

    icon = '✅' if report.changed else 'ℹ️'
    msg = f'{icon} {report.summary()}'
    if not report.changed:
        msg += '  ·  nothing was stale, so nothing was fetched'
    else:
        msg += '  ·  re-run your screen to pick up the new data'

    detail = C.card('📋 What was pulled',
                    C.data_table(report.to_frame(), page_size=12),
                    className='mt-3 mb-0')

    try:
        counts = db.table_counts()
        counts = counts[counts['rows'] > 0]
        tables = C.data_table(counts, page_size=14)
    except Exception:                              # noqa: BLE001
        tables = no_update

    return msg, detail, _sync_status_block(), tables


@callback(
    Output('pull-status', 'children', allow_duplicate=True),
    Output('pull-report', 'children', allow_duplicate=True),
    Input('sync-members-button', 'n_clicks'),
    prevent_initial_call=True,
)
def _sync_all_members(n_clicks):
    if not n_clicks:
        return no_update, no_update

    results = []
    for idx in config.INDEX_OPTIONS:
        try:
            rows = members.backfill(idx, date.today())
            results.append(f'{idx}: {rows} rows')
        except Exception as exc:                     # noqa: BLE001
            results.append(f'{idx}: failed ({exc})')
            log.debug('membership sync %s failed: %s', idx, exc)

    return (f'✅ Membership refresh complete — {", ".join(results)}',
            None)


def _sync_status_block() -> html.Div:
    """Per-source age against its refresh window."""
    try:
        df = SY.freshness()
    except Exception as exc:                       # noqa: BLE001
        return C.note(f'Could not read freshness: {exc}', 'error')

    fresh_n = int((df['status'] == 'fresh').sum())
    return html.Div([
        C.metric_row([
            (fresh_n, 'fresh', TH.POS),
            (int((df['status'] == 'stale').sum()), 'stale', TH.WARN),
            (int((df['status'] == 'never fetched').sum()), 'never fetched',
             TH.MUTED),
            (str(SY.last_session()), 'last session', TH.TEXT),
        ]),
        html.Hr(style={'borderColor': TH.BORDER}),
        C.data_table(df, page_size=12),
        html.Div('Stale simply means the next run will refresh it. Nothing is '
                 'fetched while a source is inside its window, which is what '
                 'makes a repeat run fast.',
                 style={'color': TH.MUTED, 'fontSize': '0.74rem',
                        'marginTop': '10px'}),
    ])


def _ai_status_block() -> html.Div:
    """Whether the optional narrative layer is live, and on which backend."""
    st = LLM.status()
    ok = st['enabled']
    return html.Div([
        html.Div([
            dbc.Badge('ENABLED' if ok else 'LOCAL ONLY',
                      color='success' if ok else 'secondary',
                      className='me-2'),
            html.Span(st['reason'], style={'fontSize': '0.84rem'}),
        ], className='mb-2'),
        html.Div(
            'This layer only interprets numbers the quant engine already '
            'computed. It never contributes to a score, a rank or a trade '
            'level — those come from the formulas in the Methodology tab and '
            'are identical with or without a key.',
            style={'color': TH.MUTED, 'fontSize': '0.76rem',
                   'lineHeight': '1.55'}),
        html.Div(
            'Configure with GEMINI_API_KEY or ANTHROPIC_API_KEY in .env; '
            'LLM_PROVIDER accepts auto | gemini | anthropic | off.',
            style={'color': TH.MUTED, 'fontSize': '0.72rem',
                   'marginTop': '8px'}),
    ])


@callback(
    Output('data-tables', 'children'), Output('data-freshness', 'children'),
    Output('data-members', 'children'), Output('data-cache', 'children'),
    Output('data-ai', 'children'), Output('data-sync', 'children'),
    Output('data-validity', 'children'), Output('data-mcap', 'children'),
    Input('tabs', 'active_tab'),
)
def _refresh(active_tab):
    if active_tab != 'tab-data':
        return (no_update,) * 8

    # ── row counts ────────────────────────────────────────────────
    try:
        counts = db.table_counts()
        counts = counts[counts['rows'] > 0]
        tables = C.data_table(counts, page_size=14) if not counts.empty \
            else C.placeholder('Database is empty — run an ingest.')
    except Exception as exc:                       # noqa: BLE001
        tables = C.note(f'Could not read the database: {exc}', 'error')

    # ── freshness ─────────────────────────────────────────────────
    try:
        fresh = db.read_sql("""
            SELECT source,
                   COUNT(*) AS keys,
                   MAX(last_success) AS newest,
                   SUM(CASE WHEN last_error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                   SUM(rows) AS rows
            FROM ingest_log GROUP BY source ORDER BY source
        """)
        if fresh.empty:
            freshness = C.placeholder('Nothing ingested yet.')
        else:
            fresh['newest'] = pd.to_datetime(fresh['newest']).apply(
                lambda dt: config.local_fmt(dt, '%m-%d %H:%M'))
            freshness = C.data_table(fresh, page_size=14, extra_conditional=[
                {'if': {'filter_query': '{errors} > 0', 'column_id': 'errors'},
                 'color': TH.NEG, 'fontWeight': '700'}])
    except Exception as exc:                       # noqa: BLE001
        freshness = C.note(f'{exc}', 'error')

    # ── membership sources & disagreement ─────────────────────────
    member_panels = []
    try:
        for idx in ('SPY', 'QQQ', 'DIA'):
            cov = members.coverage(idx)
            if cov.empty:
                continue
            member_panels.append(html.Div([
                html.B(f'{idx} — {config.INDEX_OPTIONS.get(idx, "")}',
                       style={'color': TH.ACCENT, 'fontSize': '0.86rem'}),
                C.data_table(cov, page_size=5),
            ], className='mb-3'))

            cmp_ = members.compare_sources(idx, date.today())
            if cmp_.get('comparable'):
                agree = cmp_['agreement_pct']
                member_panels.append(html.Div([
                    C.metric_row([
                        (cmp_['wiki_n'], 'wiki spine', TH.TEXT),
                        (cmp_['nport_n'], 'N-PORT filing', TH.TEXT),
                        (f'{agree:.0f}%', 'agreement',
                         TH.POS if agree > 90 else TH.WARN),
                    ]),
                    html.Div(
                        f'wiki only: {", ".join(cmp_["wiki_only"][:8])}  |  '
                        f'N-PORT only: {", ".join(cmp_["nport_only"][:8])}',
                        style={'color': TH.MUTED, 'fontSize': '0.7rem',
                               'marginTop': '6px'}),
                    C.note('Disagreement is shown rather than reconciled. Most '
                           'of it is share classes (BF-A vs BF-B) and the '
                           '~2-month N-PORT filing lag, both legitimate.',
                           'info'),
                ], className='mb-3'))
    except Exception as exc:                       # noqa: BLE001
        log.debug('membership panel failed: %s', exc)

    membership = html.Div(member_panels) if member_panels else \
        C.placeholder('No membership data — run an ingest with --members-from.')

    # ── cache ─────────────────────────────────────────────────────
    try:
        stats = http.cache_stats()
        if stats:
            cdf = pd.DataFrame([{'category': k, 'files': v['files'], 'MB': v['mb']}
                                for k, v in sorted(stats.items())])
            cache = html.Div([
                C.data_table(cdf, page_size=10),
                html.Div(f'total {cdf["MB"].sum():.1f} MB',
                         style={'color': TH.MUTED, 'fontSize': '0.76rem',
                                'marginTop': '8px'}),
            ])
        else:
            cache = C.placeholder('Cache is empty.')
    except Exception as exc:                       # noqa: BLE001
        cache = C.note(f'{exc}', 'error')

    return (tables, freshness, membership, cache,
            _ai_status_block(), _sync_status_block(), _validity_block(),
            _mcap_block())
