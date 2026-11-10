#!/usr/bin/env python3
"""
Quant Research Terminal — web application.

Runs a Dash server on your machine; open http://localhost:8050 in a browser.
Nothing is hosted and no data leaves the machine except the API calls that
fetch public market and filing data.

    python app.py

It listens on this computer only (127.0.0.1), never on the Wi-Fi. To use it
from a phone, set a login with `python cli.py set-password` and reach it
through Tailscale (docs/remote-access.md); see core/auth.py for the rules.

Everything the UI does is also available headlessly through cli.py, which is
what makes nightly ingest and scheduled reports possible.
"""

from __future__ import annotations

import logging
import os
import warnings

warnings.filterwarnings('ignore')

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback

pd.options.mode.chained_assignment = None

import config
from core import auth as AUTH
from core import db
from ui import theme as TH
from ui.layout import TABS, serve_layout

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s %(name)s: %(message)s',
    datefmt='%H:%M:%S')
log = logging.getLogger('app')


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title='Quant Research Terminal',
    suppress_callback_exceptions=True,
    update_title=None,
)
server = app.server
app.index_string = TH.INDEX_STRING
app.layout = serve_layout
AUTH_ENABLED = AUTH.install(server)


@server.route('/reports/<path:name>')
def _serve_report(name: str):
    """
    Serve a generated report so it can be opened in its own tab.

    Only files directly inside the report directory are served, and the
    resolved path is checked to be inside it — otherwise a crafted name like
    `../../.env` would turn this into an arbitrary file read.
    """
    from flask import abort, send_file

    root = config.REPORT_DIR.resolve()
    target = (root / name).resolve()
    if root not in target.parents or not target.is_file():
        abort(404)
    if target.suffix.lower() not in ('.html', '.pdf'):
        abort(404)
    return send_file(target)


@callback(
    [Output(f'panel-{tab_id}', 'style') for tab_id, _l, _m in TABS],
    Input('tabs', 'active_tab'),
)
def _switch_tab(active_tab):
    """
    Show one panel, hide the rest.

    Toggling visibility rather than re-rendering is what makes results
    survive a tab switch: the browser keeps every table, chart and selection
    exactly as it was, and no callback re-fires on return.
    """
    return [{} if tab_id == active_tab else {'display': 'none'}
            for tab_id, _l, _m in TABS]


def _startup_checks() -> None:
    """Create the schema if absent and report what data is present."""
    db.init_db()
    try:
        counts = db.table_counts()
        populated = counts[counts['rows'] > 0]
        if populated.empty:
            log.warning(
                'Database is empty. Populate it first:\n'
                '    python cli.py ingest --index DIA --full')
        else:
            top = ', '.join(f'{r["table"]}={r["rows"]:,}'
                            for _i, r in populated.head(5).iterrows())
            log.info('data present: %s', top)
    except Exception as exc:              # noqa: BLE001
        log.warning('could not inspect the database: %s', exc)


def main() -> int:
    _startup_checks()
    port = int(os.environ.get('PORT', 8050))
    host = AUTH.bind_host()
    problem = AUTH.check_bind(host)
    if problem:
        print(f'\n  ❌ {problem}\n')
        return 2

    print(f'\n  📊 Quant Research Terminal')
    print(f'  ➜  http://localhost:{port}')
    if not AUTH.is_loopback(host) and os.path.exists('/.dockerenv'):
        print(f'  🔒 in a container: the published port decides who can reach '
              f'it (127.0.0.1 in docker-compose.yml); login required '
              f'(user "{config.QUANT_USER}")')
    elif not AUTH.is_loopback(host):
        print(f'  ⚠️  listening on {host}:{port} — reachable by other devices '
              f'on this network; login required (user "{config.QUANT_USER}")')
    elif AUTH.configured():
        print(f'  🔒 this computer only; login required (user "{config.QUANT_USER}")')
        print(f'     phone: `tailscale serve --bg {port}` — docs/remote-access.md')
    else:
        print('  🔒 this computer only, no login — `python cli.py set-password` '
              'before using it from a phone')
    print(f'  ➜  database: {config.DATABASE_URL}')
    print(f'  ➜  headless equivalent: python cli.py --help\n')

    try:
        from waitress import serve
    except ImportError:                   # dev server: fine locally, not on a network
        log.warning('waitress not installed — using the development server')
        app.run(debug=False, port=port, host=host)
        return 0
    # A production server: the Flask development server is single-purpose
    # and not meant to face a network. Threads let a long backtest run
    # without freezing every other tab.
    # waitress strips X-Forwarded-* by default; pass them through so
    # core/auth.py decides whom to trust (a tunnel on this computer only).
    # Without them every phone looks like 127.0.0.1: one shared login limit
    # and no Secure flag on the session cookie.
    serve(server, host=host, port=port, threads=8, ident='qrt',
          clear_untrusted_proxy_headers=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
