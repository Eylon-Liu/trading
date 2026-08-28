#!/usr/bin/env python3
"""
Headless command line for the Quant Research Terminal.

Everything the web UI does is available here, which is what makes the whole
thing cron-able: a nightly ingest followed by a morning report needs no browser.

    python cli.py init
    python cli.py ingest --index DIA --full
    python cli.py screen --universe "SPY,sector=Information Technology" --strategy buffett
    python cli.py backtest --strategy momentum_trend --start 2023-01-01
    python cli.py report --kind daily --strategy buffett --email you@example.com
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

    if args.sources:
        wanted = [s.strip() for s in args.sources.split(',') if s.strip()]
        unknown = [s for s in wanted if s not in sync.SOURCES]
        if unknown:
            print(f'❌  unknown source(s): {", ".join(unknown)}')
            print(f'    available: {", ".join(sync.SOURCES)}')
            return 1
        sources = wanted
    elif args.full:
        sources = None
    else:
        sources = sync.ESSENTIAL + ['filings', 'profiles']
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


def cmd_schedule(args) -> int:
    """
    Print an ingest schedule matched to how fast each source actually changes.

    Fetching faster than the source updates spends rate limit and returns the
    same bytes; fetching slower leaves the screen reading yesterday. The
    cadence below comes from `sync.SOURCES`, where each entry already declares
    the age past which it is considered stale — so this reads the schedule off
    the data model rather than inventing a second one that can drift from it.

    `ingest` is idempotent and freshness-aware: a run over an already-current
    store makes no network calls, so overlapping schedules cost nothing.
    """
    import sys as _sys

    from data import sync

    root = config.ROOT
    py = _sys.executable

    # Grouped by declared max age. Market data settles once a day; news moves
    # within hours; index membership changes on scheduled reviews.
    buckets: dict[str, list[str]] = {}
    for key, src in sync.SOURCES.items():
        if src.max_age_hours <= 6:
            buckets.setdefault('hourly', []).append(key)
        elif src.max_age_hours <= 24:
            buckets.setdefault('daily', []).append(key)
        elif src.max_age_hours <= 24 * 7:
            buckets.setdefault('weekly', []).append(key)
        else:
            buckets.setdefault('monthly', []).append(key)

    index = args.index.upper()
    print(f'Ingest cadence for {index}, derived from sync.SOURCES\n')
    for when in ('hourly', 'daily', 'weekly', 'monthly'):
        names = sorted(buckets.get(when, []))
        if not names:
            continue
        ages = {k: sync.SOURCES[k].max_age_hours for k in names}
        print(f'  {when:8s} {", ".join(names)}')
        print(f'           (stale after {min(ages.values())}-{max(ages.values())}h)')
    print()

    def line(sched: str, sources: list[str], comment: str) -> str:
        return (f'# {comment}\n{sched}  cd "{root}" && {py} cli.py ingest '
                f'--index {index} --sources {",".join(sorted(sources))} '
                f'>> "{root}/logs/ingest.log" 2>&1')

    print('─' * 70)
    print('crontab -e   (macOS and Linux)')
    print('─' * 70)
    print(f'# Market data lands after the close; the rest follow the filing day.')
    if buckets.get('hourly'):
        print(line('0 * * * *', buckets['hourly'],
                   'News moves intraday — hourly during and after market hours.'))
    if buckets.get('daily'):
        print(line('30 18 * * 1-5', buckets['daily'],
                   'Weekday evening, after bars settle and filings post.'))
    if buckets.get('weekly'):
        print(line('0 7 * * 6', buckets['weekly'],
                   'Saturday morning — membership changes on scheduled reviews.'))
    if buckets.get('monthly'):
        print(line('0 7 1 * *', buckets['monthly'],
                   'Monthly — effectively static reference data.'))
    print()
    print('Create the log directory first:')
    print(f'  mkdir -p "{root}/logs"')
    print()
    print('Notes')
    print('  · ingest is freshness-aware, so a run with nothing stale is free.')
    print('  · cron does not load your shell profile; the absolute python path')
    print('    above is deliberate, and .env is read from the project root.')
    print('  · on macOS, cron needs Full Disk Access for the calling terminal,')
    print('    or use launchd instead (launchctl load ~/Library/LaunchAgents).')
    return 0


def _explain_smtp_error(exc: Exception) -> None:
    msg = str(exc)
    print(f'\n❌  {type(exc).__name__}: {msg[:200]}')
    if 'Username and Password not accepted' in msg or '535' in msg:
        print('    Google rejects normal account passwords here. Create an App '
              'Password at https://myaccount.google.com/apppasswords, and make '
              'sure 2-Step Verification is on.')
    elif 'Connection refused' in msg or 'timed out' in msg or 'Name or service' in msg:
        print(f'    Could not reach {config.SMTP_SERVER}:{config.SMTP_PORT}. '
              f'Check the server and port, or whether a firewall blocks it.')


def cmd_test_email(args) -> int:
    """
    Verify SMTP settings.

    By default this authenticates and disconnects without sending anything —
    enough to prove the credentials work, without putting mail in someone's
    inbox as a side effect of a config check. Pass --send to actually deliver
    a short test message.
    """
    import smtplib

    from reports import email as mailer

    print(f'SMTP_SERVER : {config.SMTP_SERVER}:{config.SMTP_PORT}')
    print(f'SMTP_USER   : {config.SMTP_USER or "(not set)"}')
    print(f'SMTP_PASS   : '
          f'{f"set, {len(config.SMTP_PASS)} chars" if config.SMTP_PASS else "(not set)"}')

    if not (config.SMTP_USER and config.SMTP_PASS):
        print('\n❌  Credentials missing. Add SMTP_USER and SMTP_PASS to .env.')
        print('    Gmail needs an App Password, not your account password:')
        print('    https://myaccount.google.com/apppasswords')
        return 1

    # ── authentication check, no message ─────────────────────────
    try:
        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.login(config.SMTP_USER, config.SMTP_PASS)
    except Exception as exc:                       # noqa: BLE001
        _explain_smtp_error(exc)
        return 1

    print('\n✅  Authenticated successfully — reports can be emailed.')

    if not args.send:
        print('    No message was sent. Add --send to deliver a test email.')
        return 0

    if not args.to:
        print('    --send needs --to <address>.')
        return 1

    body = ('<h2>Quant Research Terminal</h2>'
            '<p>SMTP is configured correctly. Scheduled reports will send.</p>')
    try:
        mailer.send_report(body, args.to, subject='Test — Quant Research Terminal')
    except Exception as exc:                       # noqa: BLE001
        _explain_smtp_error(exc)
        return 1

    print(f'📧  Test email sent to {args.to}. Check the inbox, and spam.')
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
    path, html = builder.build_report(
        strategy=getattr(args, 'strategy', None),
        kind=args.kind, as_of=args.as_of)
    print(f'✅  report written to {path}')

    if args.email:
        try:
            mailer.send_report(html, args.email,
                               subject=f'{args.kind.title()} Market Brief')
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
    ing.add_argument('--sources',
                     help='comma-separated subset to refresh, e.g. "news" or '
                          '"prices,facts,splits". Lets a schedule run each '
                          'source at the rate its data actually changes.')
    ing.add_argument('--members-from', help='backfill membership from this date')
    ing.add_argument('--insider-days', type=int, default=730)
    ing.add_argument('--force', action='store_true',
                     help='refetch every source, ignoring freshness windows')
    ing.add_argument('--limit', type=int, help='cap constituents (for testing)')
    ing.set_defaults(func=cmd_ingest)

    scr = sub.add_parser('screen', help='rank a universe under a strategy')
    scr.add_argument('--universe', help='e.g. "SPY,sector=Health Care,mcap>5B"')
    scr.add_argument('--strategy', default='buffett')
    scr.add_argument('--as-of', help='YYYY-MM-DD (defaults to today)')
    scr.add_argument('--top', type=int, default=20)
    scr.set_defaults(func=cmd_screen)

    bt = sub.add_parser('backtest', help='walk-forward test a strategy')
    bt.add_argument('--universe')
    bt.add_argument('--strategy', default='buffett')
    bt.add_argument('--start', default=str(date.today() - timedelta(days=1460)))
    bt.add_argument('--end')
    bt.add_argument('--top', type=int, default=15)
    bt.add_argument('--rebalance', default=None, help='M | Q | SA | A')
    bt.set_defaults(func=cmd_backtest)

    rep = sub.add_parser('report', help='render a daily or monthly report')
    rep.add_argument('--kind', default='daily', choices=['daily', 'monthly'])
    rep.add_argument('--strategy', default='buffett')
    rep.add_argument('--as-of')
    rep.add_argument('--email', help='send the rendered report to this address')
    rep.set_defaults(func=cmd_report)

    sch = sub.add_parser('schedule',
                         help='print an ingest schedule matched to each source')
    sch.add_argument('--index', default='SPY', help='SPY | QQQ | DIA | IWM')
    sch.set_defaults(func=cmd_schedule)

    sub.add_parser('strategies', help='list available strategies by horizon') \
        .set_defaults(func=cmd_strategies)
    sub.add_parser('status', help='show data coverage and freshness') \
        .set_defaults(func=cmd_status)
    sub.add_parser('freshness', help='show which sources are stale') \
        .set_defaults(func=cmd_freshness)

    te = sub.add_parser('test-email', help='verify SMTP settings')
    te.add_argument('--to', help='recipient address (only needed with --send)')
    te.add_argument('--send', action='store_true',
                    help='actually deliver a test message (default: auth only)')
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
