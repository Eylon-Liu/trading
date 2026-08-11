#!/usr/bin/env python3
"""
Headless command line for the Quant Research Terminal.

Everything the web UI does is available here, which is what makes the whole
thing cron-able: a nightly ingest followed by a morning report needs no browser.

    python cli.py init
    python cli.py ingest --index DIA --full
    python cli.py screen --universe "SPY,sector=Information Technology" --strategy quality_value
    python cli.py backtest --strategy momentum_trend --start 2023-01-01
    python cli.py report --kind daily --strategy quality_value --email you@example.com
    python cli.py strategies
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

import pandas as pd

import config
from core import db


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s  %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S')


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────

def cmd_init(args) -> int:
    db.init_db()
    print(f'✅  Schema ready at {config.DATABASE_URL}')
    print(db.table_counts().to_string(index=False))
    return 0


def cmd_ingest(args) -> int:
    """
    The pull phase, run explicitly.

    By default this only fetches sources that are outside their refresh
    window, so a nightly cron over an already-current store is nearly free.
    `--force` refetches regardless, for a first build or after a schema change.
    """
    from data import members, sync, yahoo

    db.init_db()
    index = args.index.upper()

    print(f'▸ resolving {index} membership…')
    if args.members_from:
        n = members.backfill(index, pd.to_datetime(args.members_from).date(),
                             freq='QE')
        print(f'  backfilled {n} membership rows')

    tickers = members.latest_members(index)
    if not tickers:
        print(f'❌  could not resolve members for {index}')
        return 1
    if args.limit:
        tickers = tickers[:args.limit]
    print(f'  {len(tickers)} constituents')

    sources = None if args.full else sync.ESSENTIAL + ['filings', 'profiles']
    report = sync.sync(tickers, index=index, sources=sources,
                       force=args.force,
                       progress=lambda m: print(f'  ▸ {m}'))

    # `period` only matters on a first build; incremental updates extend from
    # the newest stored bar regardless.
    if args.force and args.period:
        yahoo.update_prices(tickers + [config.BENCHMARK_TICKER],
                            period=args.period)

    print('\n' + report.to_frame().to_string(index=False))
    print(f'\n✅  {report.summary()}')
    if not report.changed:
        print('   Nothing was stale — the stored data is already current.')
    print()
    print(db.table_counts().head(12).to_string(index=False))
    return 0


def cmd_test_email(args) -> int:
    """Send a one-line test message, and explain clearly when it fails."""
    from reports import email as mailer

    print(f'SMTP_SERVER : {config.SMTP_SERVER}:{config.SMTP_PORT}')
    print(f'SMTP_USER   : {config.SMTP_USER or "(not set)"}')
    print(f'SMTP_PASS   : {"set" if config.SMTP_PASS else "(not set)"}')

    if not (config.SMTP_USER and config.SMTP_PASS):
        print('\n❌  Credentials missing. Add SMTP_USER and SMTP_PASS to .env.')
        print('    Gmail needs an App Password, not your account password:')
        print('    https://myaccount.google.com/apppasswords')
        return 1

    body = ('<h2>Quant Research Terminal</h2>'
            '<p>SMTP is configured correctly. Scheduled reports will send.</p>')
    try:
        mailer.send_report(body, args.to, subject='Test — Quant Research Terminal')
    except mailer.SMTPNotConfigured as exc:
        print(f'\n❌  {exc}')
        return 1
    except Exception as exc:                       # noqa: BLE001
        msg = str(exc)
        print(f'\n❌  Send failed: {type(exc).__name__}: {msg[:200]}')
        if 'Username and Password not accepted' in msg or '535' in msg:
            print('    Google rejects normal account passwords. Create an App '
                  'Password at https://myaccount.google.com/apppasswords')
        elif 'Connection refused' in msg or 'timed out' in msg:
            print(f'    Could not reach {config.SMTP_SERVER}:{config.SMTP_PORT}. '
                  f'Check the server/port, or whether a firewall blocks it.')
        return 1

    print(f'\n✅  Test email sent to {args.to}. Check the inbox (and spam).')
    return 0


def cmd_freshness(args) -> int:
    """Show what is fresh, what is stale, and when each was last pulled."""
    from data import sync

    db.init_db()
    df = sync.freshness()
    print(df.to_string(index=False))
    print(f'\nlast completed session: {sync.last_session()}')
    stale = df[df['status'] != 'fresh']
    if stale.empty:
        print('all sources fresh — a run now would fetch nothing')
    else:
        print(f'{len(stale)} source(s) would be refreshed on the next run')
    return 0


def cmd_screen(args) -> int:
    from data.universe import UniverseSpec, parse_spec
    from quant import engine

    db.init_db()
    spec = parse_spec(args.universe) if args.universe else UniverseSpec(preset='SPY')
    result = engine.run(spec, args.strategy, as_of=args.as_of, top_n=args.top)

    if result.scores.empty:
        print('❌  no results — check the universe resolves and data is ingested')
        return 1

    print(f'\n{result.strategy.name}  ·  {result.as_of}  ·  '
          f'{len(result.universe)} screened  ·  horizon: {result.horizon}')
    print(f'universe: {spec.describe()}\n')

    cols = ['rank', 'composite', 'signal', 'sector']
    if result.horizon == 'mid' and not result.plans.empty:
        cols += ['entry', 'stop', 'target', 'weight_pct']
    shown = result.top(args.top)[[c for c in cols if c in result.scores.columns]]
    print(shown.round(3).to_string())

    if result.horizon == 'mid' and not result.plans.empty:
        from quant.tradeplan import portfolio_risk
        print('\nportfolio if all taken:', portfolio_risk(result.plans))

    print(f'\nrun_id: {result.run_id}')
    return 0


def cmd_backtest(args) -> int:
    from data.universe import UniverseSpec, parse_spec
    from quant import backtest as BT
    from quant import strategies as ST

    db.init_db()
    spec = parse_spec(args.universe) if args.universe else UniverseSpec(preset='SPY')
    strat = ST.get(args.strategy)

    if strat.horizon == 'mid':
        print(f'▸ mid-horizon backtest (stop/target aware): {strat.name}')
        res = BT.run_trade_backtest(spec, args.strategy, args.start, args.end,
                                    n_positions=args.top)
    else:
        print(f'▸ long-horizon backtest: {strat.name}')
        res = BT.run_backtest(spec, args.strategy, args.start, args.end,
                              n_holdings=args.top, rebalance=args.rebalance)

    if not res.metrics:
        print('❌  backtest produced no results')
        return 1

    print(f'\n{res.summary()}\n')
    for k, v in sorted(res.metrics.items()):
        print(f'  {k:22} {v:,.4f}' if isinstance(v, float) else f'  {k:22} {v}')

    if res.metrics.get('mean_ic') is not None and abs(res.metrics['mean_ic']) < 0.02:
        print('\n⚠️  Mean rank IC is near zero: the ranking showed little '
              'predictive power over this window. Any outperformance is more '
              'likely a few positions than a working signal.')
    return 0


def cmd_report(args) -> int:
    from reports import builder
    from reports import email as mailer

    db.init_db()
    path, html = builder.build_report(args.strategy, kind=args.kind,
                                      as_of=args.as_of)
    print(f'✅  report written to {path}')

    if args.email:
        try:
            mailer.send_report(html, args.email,
                               subject=f'{args.kind.title()} Brief — {args.strategy}')
            print(f'📧  emailed to {args.email}')
        except mailer.SMTPNotConfigured as exc:
            print(f'⚠️  email skipped: {exc}')
        except Exception as exc:                   # noqa: BLE001
            print(f'⚠️  email failed: {exc}')
    return 0


def cmd_strategies(args) -> int:
    from quant import strategies as ST
    rows = ST.summary_table()
    print(f'\n{"KEY":20} {"HORIZON":22} {"TRADE PLAN":26} NAME')
    print('─' * 100)
    for r in rows:
        print(f'{r["key"]:20} {r["horizon"]:22} {r["trade_plan"]:26} {r["name"]}')
    print('\nUse --strategy <KEY> with the screen and backtest commands.')
    return 0


def cmd_status(args) -> int:
    from core import http
    db.init_db()
    print('TABLES'); print(db.table_counts().to_string(index=False))
    print('\nCACHE')
    for cat, s in sorted(http.cache_stats().items()):
        print(f'  {cat:14} {s["files"]:5} files  {s["mb"]:8.2f} MB')
    print('\nINGEST FRESHNESS')
    fresh = db.read_sql("""
        SELECT source, COUNT(*) AS keys, MAX(last_success) AS newest,
               SUM(CASE WHEN last_error IS NOT NULL THEN 1 ELSE 0 END) AS errors
        FROM ingest_log GROUP BY source ORDER BY source
    """)
    print(fresh.to_string(index=False) if not fresh.empty else '  (nothing yet)')
    return 0


# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='cli.py', description='Quant Research Terminal — headless interface')
    p.add_argument('-v', '--verbose', action='store_true')
    sub = p.add_subparsers(dest='command', required=True)

    sub.add_parser('init', help='create the database schema').set_defaults(func=cmd_init)

    ing = sub.add_parser('ingest', help='download and store market + filing data')
    ing.add_argument('--index', default='DIA', help='SPY | QQQ | DIA | IWM')
    ing.add_argument('--period', default='10y', help='price history depth')
    ing.add_argument('--full', action='store_true',
                     help='also fetch insiders, news, policy and attention')
    ing.add_argument('--members-from', help='backfill membership from this date')
    ing.add_argument('--insider-days', type=int, default=730)
    ing.add_argument('--force', action='store_true',
                     help='refetch every source, ignoring freshness windows')
    ing.add_argument('--limit', type=int, help='cap constituents (for testing)')
    ing.set_defaults(func=cmd_ingest)

    scr = sub.add_parser('screen', help='rank a universe under a strategy')
    scr.add_argument('--universe', help='e.g. "SPY,sector=Health Care,mcap>5B"')
    scr.add_argument('--strategy', default='quality_value')
    scr.add_argument('--as-of', help='YYYY-MM-DD (defaults to today)')
    scr.add_argument('--top', type=int, default=20)
    scr.set_defaults(func=cmd_screen)

    bt = sub.add_parser('backtest', help='walk-forward test a strategy')
    bt.add_argument('--universe')
    bt.add_argument('--strategy', default='quality_value')
    bt.add_argument('--start', default=str(date.today() - timedelta(days=1460)))
    bt.add_argument('--end')
    bt.add_argument('--top', type=int, default=15)
    bt.add_argument('--rebalance', default=None, help='M | Q | SA | A')
    bt.set_defaults(func=cmd_backtest)

    rep = sub.add_parser('report', help='render a daily or monthly report')
    rep.add_argument('--kind', default='daily', choices=['daily', 'monthly'])
    rep.add_argument('--strategy', default='quality_value')
    rep.add_argument('--as-of')
    rep.add_argument('--email', help='send the rendered report to this address')
    rep.set_defaults(func=cmd_report)

    sub.add_parser('strategies', help='list available strategies by horizon') \
        .set_defaults(func=cmd_strategies)
    sub.add_parser('status', help='show data coverage and freshness') \
        .set_defaults(func=cmd_status)
    sub.add_parser('freshness', help='show which sources are stale') \
        .set_defaults(func=cmd_freshness)

    te = sub.add_parser('test-email', help='verify SMTP settings')
    te.add_argument('--to', required=True, help='recipient address')
    te.set_defaults(func=cmd_test_email)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print('\ninterrupted')
        return 130


if __name__ == '__main__':
    sys.exit(main())
