#!/usr/bin/env python3
"""
Quant Research Terminal — web application.

Runs a Dash server on your machine; open http://localhost:8050 in a browser.
Nothing is hosted and no data leaves the machine except the API calls that
fetch public market and filing data.

    python app.py

Without a password it answers on this machine only. To use it from a phone,
run `python cli.py set-password`: it then listens on the network and every
page requires a login (core/auth.py). docs/remote-access.md covers reaching
it safely from outside your home network.

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
    if host not in AUTH.LOCAL_HOSTS:
        import socket
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            ip = '?'
        name = socket.gethostname()
        if not name.endswith('.local'):
            name += '.local'
        print(f'  ➜  http://{name}:{port}  (stable hostname)')
        print(f'  ➜  http://{ip}:{port}  (LAN IP / Tailscale)')
        print(f'  🔒 login required — user "{config.QUANT_USER}"')
    else:
        print('  🔒 this machine only — `python cli.py set-password` to allow '
              'phone access')
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
    serve(server, host=host, port=port, threads=8, ident='qrt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
