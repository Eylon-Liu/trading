#!/usr/bin/env python3
"""
Headless command line for the Quant Research Terminal.

Everything the web UI does is available here, which is what makes the whole
thing cron-able: a nightly ingest followed by a morning report needs no browser.

    python cli.py init
    python cli.py ingest --index DIA --full
    python cli.py screen --universe "SPY,sector=Information Technology" --strategy buffett
    python cli.py backtest --strategy momentum_trend --start 2023-01-01
    python cli.py report --kind daily --email you@example.com
    python cli.py daily                      # the scheduled routine, end to end
    python cli.py db status                  # size, backups, row counts
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


def _resolve_as_of(value: str | None):
    """'session' / 'last-session' -> the last completed trading session."""
    if value and value.lower() in ('session', 'last-session', 'last_session'):
        from core import market_calendar as MC
        return MC.last_completed_session()
    return value


def cmd_screen(args) -> int:
    from data.universe import UniverseSpec, parse_spec
    from quant import engine

    db.init_db()
    spec = parse_spec(args.universe) if args.universe else UniverseSpec(preset='SPY')
    result = engine.run(spec, args.strategy, as_of=_resolve_as_of(args.as_of),
                        top_n=args.top)

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

    try:
        if strat.horizon == 'mid':
            print(f'▸ mid-horizon backtest (stop/target aware): {strat.name}')
            res = BT.run_trade_backtest(spec, args.strategy, args.start, args.end,
                                        n_positions=args.top)
        else:
            print(f'▸ long-horizon backtest: {strat.name}')
            res = BT.run_backtest(spec, args.strategy, args.start, args.end,
                                  n_holdings=args.top, rebalance=args.rebalance)
    except (ValueError, BT.DegradedBacktest) as exc:
        print(f'❌  {exc}')
        return 1

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
    as_of = pd.to_datetime(args.as_of or date.today()).date()
    kind = builder.auto_kind(as_of) if args.kind == 'auto' else args.kind
    path, html = builder.build_report(kind=kind, as_of=as_of)
    print(f'✅  {kind} report written to {path}')

    if args.email:
        try:
            mailer.send_report(html, args.email,
                               subject=builder.subject_for(kind, as_of),
                               report_id=builder.report_id_for(kind, as_of))
            print(f'📧  emailed to {args.email}')
        except mailer.SMTPNotConfigured as exc:
            print(f'⚠️  email skipped: {exc}')
            return 1
        except Exception as exc:                   # noqa: BLE001
            _explain_smtp_error(exc)
            return 1
    return 0


def _all_strategy_keys() -> list[str]:
    from quant import custom as CU
    from quant import strategies as ST
    keys = list(ST.ALL_STRATEGIES)
    try:
        extra = CU.summary()
        if not extra.empty:
            keys += [k for k in extra['key'] if k not in keys]
    except Exception:                              # noqa: BLE001
        pass
    return keys


def cmd_daily(args) -> int:
    """
    The scheduled routine, end to end: pull → screen → build → email.

    Which email goes out is decided by the NYSE calendar for the day it runs:

      trading day      the full brief — screens for every strategy, dated
                       to the last completed session, compared with the
                       session before it
      weekend/holiday  a news-only digest of what was published since the
                       last close; nothing is screened, because nothing
                       traded

    Safe to run more than once a day: a report already emailed for this date
    is not sent again (unless --resend), and a re-run replaces the day's
    screens instead of adding to them. A failed source is reported and the
    brief is still built from stored data, with a warning in the email.
    """
    from core import maintenance
    from core import market_calendar as MC
    from data import members, sync
    from data.universe import parse_spec
    from quant import engine
    from reports import builder
    from reports import email as mailer

    db.init_db()
    today = pd.to_datetime(args.date).date() if args.date else date.today()
    kind = builder.auto_kind(today) if args.kind == 'auto' else args.kind
    index = (args.index or config.QUANT_INDEX).upper()
    to = [] if args.no_email else mailer.recipients(args.email or config.QUANT_EMAIL)
    report_id = builder.report_id_for(kind, today)
    session = MC.last_completed_session()
    spec = parse_spec(index)
    from quant import portfolio as PF
    watch = PF.watchlist()
    held = PF.lots('open')['ticker'].unique().tolist()

    reason = MC.closed_reason(today)
    print(f'▸ {today:%a %Y-%m-%d} · '
          f'{"market open" if reason is None else f"market closed ({reason})"}'
          f' · last completed session {session} · report: {kind}')
    print(f'  index={index}  mode={args.mode}  '
          f'email={", ".join(to) if to else "(none)"}'
          + (f'  watchlist={",".join(watch)}' if watch else ''))

    if not to and not args.no_email:
        print('⚠️  no recipient: set QUANT_EMAIL in .env or pass --email '
              '(or --no-email to only prepare data).')
        return 1

    if kind == 'news' and not to and not args.force:
        print('  market closed and no email requested — nothing to prepare.')
        return 0

    if to and not args.resend and builder.emailed_at(report_id):
        print(f'✓ the {kind} report for {today} was already emailed — '
              f'nothing to do (pass --resend to send again).')
        return 0

    if kind == 'news':
        sources = ['news', 'policy', 'earnings']
    elif args.mode == 'full':
        sources = None                          # every source
    else:
        sources = ['prices', 'news', 'policy', 'macro', 'earnings']

    if args.dry_run:
        print(f'  dry run: would ingest {sources or "all sources"}, '
              f'{"skip screens" if kind == "news" else f"screen as of {session}"}, '
              f'build "{builder.subject_for(kind, today)}"'
              + (f', email {", ".join(to)}' if to else ''))
        return 0

    # ── 1. pull ──────────────────────────────────────────────────
    if not args.skip_ingest:
        try:
            tickers = members.latest_members(index)
        except Exception as exc:                   # noqa: BLE001
            print(f'⚠️  could not resolve {index} members: {exc}')
            tickers = []
        tickers = sorted(set(tickers) | set(watch) | set(held))
        print(f'▸ ingest {", ".join(sources) if sources else "all sources"} '
              f'for {len(tickers)} names')
        try:
            rep = sync.sync(tickers, index=index, sources=sources,
                            progress=lambda m: print(f'  ▸ {m}'))
            print(f'  {rep.summary()}')
        except Exception as exc:                   # noqa: BLE001
            print(f'⚠️  ingest failed ({type(exc).__name__}: {exc}) — '
                  f'continuing with stored data')

    # ── 1b. AI earnings reviews of new results filings ───────────
    if not args.no_llm:
        _run_reviews(index, watch, held, spec)

    # ── 2. screen ────────────────────────────────────────────────
    failed: list[str] = []
    if kind != 'news' and not args.skip_screens:
        keys = _all_strategy_keys()
        print(f'▸ screening {len(keys)} strategies on {index} as of {session}')
        for key in keys:
            try:
                res = engine.run(spec, key, as_of=session)
                note = f'{len(res.scores)} ranked' if not res.scores.empty else 'no results'
                if res.warnings:
                    note += f' ({len(res.warnings)} warning(s))'
                print(f'  ▸ {key:22} {note}')
            except Exception as exc:               # noqa: BLE001
                failed.append(key)
                print(f'  ⚠️ {key:22} failed: {type(exc).__name__}: {str(exc)[:120]}')

    # ── 3. build ─────────────────────────────────────────────────
    if not to and builder.emailed_at(report_id):
        # The evening job: this morning's email is the record of what was
        # sent, so leave its file alone. Tomorrow's brief reads these screens.
        print(f'✓ screens ready; the {kind} report for {today} was already '
              f'emailed, so its file is left as sent.')
        return 1 if failed else 0
    path, html = builder.build_report(kind=kind, as_of=today, universe=spec)
    print(f'✅  {kind} report written to {path}')

    # ── 4. email ─────────────────────────────────────────────────
    rc = 0
    if to:
        try:
            mailer.send_report(html, to, subject=builder.subject_for(kind, today),
                               report_id=report_id)
            print(f'📧  emailed to {", ".join(to)}')
        except mailer.SMTPNotConfigured as exc:
            print(f'❌  email not sent: {exc}')
            rc = 1
        except Exception as exc:                   # noqa: BLE001
            _explain_smtp_error(exc)
            rc = 1

    if args.backup:
        try:
            out = maintenance.backup()
            if out:
                print(f'💾  database backed up to {out}')
        except Exception as exc:                   # noqa: BLE001
            print(f'⚠️  backup failed: {exc}')

    if failed:
        print(f'⚠️  {len(failed)} strategy screen(s) failed: {", ".join(failed)}')
    return rc


def cmd_db(args) -> int:
    """Database upkeep: status, backup, integrity check, clean-up."""
    from core import maintenance as M

    db.init_db()
    if args.action == 'status':
        info = M.status()
        print(f'database : {info["url"]}')
        if info.get('size_mb') is not None:
            print(f'size     : {info["size_mb"]} MB (+{info["wal_mb"]} MB WAL)')
        print(f'backups  : {info["backups"]}'
              + (f' — newest {info["latest_backup"]}' if info['latest_backup'] else ''))
        print()
        print(info['tables'].to_string(index=False))
        problems = {k: v for k, v in M.consistency_report().items() if v}
        print()
        if problems:
            print('cleanable: ' + ', '.join(f'{k}={v}' for k, v in problems.items()))
            print('           run `python cli.py db clean` to remove them')
        else:
            print('no duplicate or orphaned rows')
        return 0

    if args.action == 'backup':
        out = M.backup(keep=args.keep)
        if out is None:
            print('❌  backups are only supported for a local SQLite database')
            return 1
        print(f'✅  backup written to {out}')
        return 0

    if args.action == 'check':
        res = M.integrity_check(full=args.full)
        if res == ['ok']:
            print('✅  integrity check passed')
        else:
            print('❌  integrity problems:')
            for line in res[:50]:
                print(f'   {line}')
        for k, v in M.consistency_report().items():
            print(f'  {k:22} {v}')
        return 0 if res == ['ok'] else 1

    if args.action == 'clean':
        if not args.no_backup:
            out = M.backup(keep=args.keep)
            if out:
                print(f'💾  backup first: {out}')
        res = M.clean(vacuum=args.vacuum, factor_detail_days=args.prune_detail_days)
        for k, v in res.items():
            print(f'  {k:22} {v}')
        print('✅  clean-up done' + (' (vacuumed)' if args.vacuum else ''))
        return 0
    return 1


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
# PORTFOLIO, WATCHLIST, TRACK RECORD, CALENDARS
# ─────────────────────────────────────────────

def _fmt_pct(v) -> str:
    import numpy as np
    return f'{v:+.1%}' if v is not None and np.isfinite(v) else '—'


def cmd_watch(args) -> int:
    from quant import portfolio as PF
    db.init_db()
    if args.action == 'add':
        names = PF.watch_add(args.tickers, note=args.note)
        print(f'✅  watching {", ".join(names)}')
    elif args.action == 'remove':
        n = PF.watch_remove(args.tickers)
        print(f'🗑  removed {n}')
    wl = PF.watchlist_table()
    env_only = sorted(set(config.WATCHLIST) - set(wl['ticker']))
    if wl.empty and not env_only:
        print('watchlist is empty — `python cli.py watch add AAPL MSFT`')
        return 0
    for _i, r in wl.iterrows():
        print(f'  {r["ticker"]:8} {r["note"] or ""}')
    for t in env_only:
        print(f'  {t:8} (from QUANT_WATCHLIST in .env)')
    return 0


def cmd_portfolio(args) -> int:
    from quant import portfolio as PF
    db.init_db()
    a = args.action
    if a == 'add':
        if args.shares is None or args.cost is None:
            print('usage: portfolio add TICKER SHARES COST [--date --stop --target]')
            return 1
        lot = PF.add_lot(args.ticker, args.shares, args.cost, opened=args.date,
                         stop=args.stop, target=args.target,
                         strategy=args.strategy, note=args.note)
        print(f'✅  added {args.ticker.upper()} lot {lot}')
    elif a == 'close':
        n = PF.close(args.ticker, exit_price=args.price, closed=args.date)
        print(f'✅  closed {n} lot(s)' if n else '❌  no open lot matches')
        if not n:
            return 1
    elif a == 'set':
        n = PF.set_levels(args.ticker, stop=args.stop, target=args.target)
        print(f'✅  updated {n} lot(s)')
    elif a == 'import':
        n = PF.import_csv(args.ticker)
        print(f'✅  imported {n} lot(s) from {args.ticker}')
    elif a == 'remove':
        n = PF.delete_lot(args.ticker)
        print(f'🗑  deleted {n} lot(s)')

    if a in ('show', 'add', 'close', 'set', 'import', 'remove'):
        h = PF.holdings()
        if h.empty:
            print('no open positions — `python cli.py portfolio add AAPL 10 185.50`')
        else:
            view = h[['shares', 'avg_cost', 'last', 'day_chg', 'pnl', 'pnl_pct',
                      'weight', 'stop', 'target']].copy()
            for c in ('day_chg', 'pnl_pct', 'weight'):
                view[c] = view[c].map(_fmt_pct)
            print(view.round(2).to_string())
            sm = PF.summary(h)
            money = lambda v: f'${v:,.0f}' if v == v else '—'      # noqa: E731
            print(f'\nvalue {money(sm["market_value"])}  ·  P&L {money(sm["pnl"])} '
                  f'({_fmt_pct(sm["pnl_pct"])})  ·  last session {money(sm["day_pnl"])}'
                  + (f'  ·  {sm["unpriced"]} without a stored price — run ingest'
                     if sm['unpriced'] else ''))
        if args.lots:
            print('\nlots:')
            print(PF.lots('open')[['lot_id', 'ticker', 'shares', 'cost_basis', 'opened',
                                   'stop', 'target', 'note']].to_string(index=False))
    if a in ('show', 'alerts'):
        al = PF.alerts()
        print('\nalerts:' if al else '\nno alerts')
        icon = {'risk': '🔴', 'watch': '🟠', 'info': '🔵'}
        for x in al:
            print(f'  {icon[x["level"]]} {x["ticker"]:6} {x["text"]}')
    return 0


def cmd_track(args) -> int:
    from data.universe import parse_spec
    from quant import track
    db.init_db()
    spec_json = parse_spec(args.index).to_json() if args.index else None
    tr = track.summary(lookback_days=args.days, top_n=args.top, spec_json=spec_json)
    if tr.empty:
        print('no completed cohorts yet — the routine needs to have stored '
              'screens at least a week ago')
        return 0
    view = tr[['name', 'horizon', 'cohorts', 'independent', 'mean_pick_return',
               'mean_excess', 'hit_rate', 'mean_ic']].copy()
    for c in ('mean_pick_return', 'mean_excess', 'hit_rate'):
        view[c] = view[c].map(_fmt_pct)
    view['mean_ic'] = view['mean_ic'].round(3)
    print(view.to_string(index=False))
    print('\nPicks bought at the close after each stored screen. Daily cohorts '
          'overlap; trust "independent", not "cohorts".')
    return 0


def cmd_earnings(args) -> int:
    from data import earnings as ER
    from data import members
    from quant import portfolio as PF
    db.init_db()
    names = set(PF.watchlist()) | set(PF.lots('open')['ticker'])
    if args.index:
        names |= set(members.latest_members(args.index.upper()))
    if args.refresh:
        print(f'▸ refreshing calendar for {len(names)} names')
        n = ER.update_earnings(sorted(names))
        for src, st in ER.LAST_STATUS.items():
            print(f'  {src:8} {st}')
        print(f'  {n} date(s) stored')
    er = ER.upcoming(sorted(names), days=args.days)
    if er.empty:
        print('nothing scheduled — add names with `watch add`, or pass --index SPY')
        return 0
    er['report_date'] = er['report_date'].dt.date
    er['status'] = er['confirmed'].map({True: 'confirmed', False: 'estimated'})
    print(er[['ticker', 'report_date', 'time', 'source', 'status', 'eps_estimate',
              'days_until']].to_string(index=False))
    return 0


def _run_reviews(index: str, watch: list[str], held: list[str], spec) -> int:
    """Review new results filings; your names first, then current picks."""
    from data import members
    from nlp import earnings_review as ERV
    from reports import builder
    if not ERV.enabled():
        return 0
    try:
        universe = sorted(set(members.latest_members(index)) | set(watch) | set(held))
        picks = builder._current_picks(date.today(), spec.to_json())
        done = ERV.review_recent(tickers=universe,
                                 priority=list(held) + list(watch) + picks,
                                 progress=lambda m: print(f'  ▸ {m}'))
    except Exception as exc:                       # noqa: BLE001
        print(f'⚠️  AI reviews skipped: {type(exc).__name__}: {exc}')
        return 0
    if done:
        print(f'🧠  {len(done)} new AI earnings review(s): '
              + ', '.join(f'{r["ticker"]} {r["verdict"]}' for r in done))
    return len(done)


def cmd_review(args) -> int:
    """Run (or list) AI earnings reviews of recent results filings."""
    from data import members
    from nlp import earnings_review as ERV
    from nlp import providers
    from quant import portfolio as PF
    db.init_db()
    if args.run:
        if not ERV.enabled():
            print('❌  no LLM configured — set GEMINI_API_KEY or ANTHROPIC_API_KEY '
                  'in .env (or QUANT_LLM_REVIEWS is off)')
            return 1
        names = [t.upper() for t in args.tickers] or sorted(
            set(members.latest_members(args.index.upper())) | set(PF.watchlist())
            | set(PF.lots('open')['ticker']))
        todo = ERV.pending(tickers=names, lookback_days=args.days)
        print(f'▸ {len(todo)} results filing(s) without a review in the last '
              f'{args.days} days · model {providers.model_name()}')
        done = ERV.review_recent(tickers=names, lookback_days=args.days,
                                 max_reviews=args.max,
                                 priority=list(PF.lots('open')['ticker']) + PF.watchlist(),
                                 progress=lambda m: print(f'  ▸ {m}'))
        print(f'✅  {len(done)} review(s) written')
    rv = ERV.latest_reviews([t.upper() for t in args.tickers] or None,
                            days=args.days if not args.run else 120)
    if rv.empty:
        print('no reviews stored yet' + ('' if args.run else
              ' — run `python cli.py review --run`'))
        return 0
    for _i, r in rv.head(40).iterrows():
        print(f'{r["ticker"]:6} {r["verdict"]:10} score {r["score"]:+.1f} '
              f'({r["conviction"]})  guidance {r["guidance"]:9}  '
              f'filed {r["filed"].date()}  · {r["summary"][:110]}')
    return 0


def cmd_consensus_test(args) -> int:
    """
    Does agreement between strategy families predict returns? Two answers:
    the agreement study (forward excess return by number of agreeing
    families) and a walk-forward backtest of the consensus, optionally next
    to each of its member strategies.
    """
    import numpy as np

    from data.universe import parse_spec
    from quant import availability as AV
    from quant import backtest as BT
    from quant import consensus as CONS
    from quant import strategies as ST

    db.init_db()
    spec = parse_spec(args.universe or config.QUANT_INDEX)
    start = pd.to_datetime(args.start).date()
    strat, notes = AV.meta_members_at('consensus', start, spec.preset or 'SPY')
    print(f'▸ consensus on {spec.describe()} from {start}')
    print(f'  families: {", ".join(strat.components) or "none"}')
    for n in notes:
        print(f'  note: {n}')
    if len(strat.components) < 2:
        print('❌  fewer than two strategy families have data from this start '
              'date — pick a later --start or ingest more history')
        return 1

    print(f'\n▸ agreement study: {args.horizon}-month forward return vs SPY, '
          f'sampled {"monthly" if args.freq == "ME" else "quarterly"}')
    study = CONS.agreement_study(spec, start, args.end, args.horizon,
                                 freq=args.freq, strategy_key=strat,
                                 progress=(lambda m: print(f'  ▸ {m}'))
                                 if args.verbose else None)
    b = study['buckets']
    if b.empty:
        print('  no dates with enough data')
    else:
        view = b[['label', 'mean_excess', 'median_excess', 'hit_rate',
                  'names', 'periods']].copy()
        for c in ('mean_excess', 'median_excess', 'hit_rate'):
            view[c] = view[c].map(lambda v: f'{v:+.2%}' if c != 'hit_rate' else f'{v:.0%}')
        print(view.to_string(index=False))
        ic = study['agreement_ic']
        print(f'\n  agreement IC (Spearman, votes vs forward excess): {ic:+.3f} '
              f'over {study["dates"]} dates, positive on '
              f'{study["agreement_ic_hit"]:.0%} of them')
        # A bucket holding a couple of names per date is noise; it is shown
        # above but does not decide the verdict.
        solid = b[(b['bucket'] >= 0) & (b['names'] >= CONS.MIN_BUCKET_NAMES)]
        thin = b[(b['bucket'] >= 0) & (b['names'] < CONS.MIN_BUCKET_NAMES)]
        if not thin.empty:
            print(f'  ({", ".join(thin["label"])}: under {CONS.MIN_BUCKET_NAMES} '
                  f'observations — too few to judge, left out of the verdict)')
        spread = CONS.agreement_spread(study['by_date'])
        if spread:
            print(f'  2+ families vs none: {spread["mean"]:+.2%} per period on '
                  f'average, ahead on {spread["wins"]} of {spread["dates"]} dates')
        ordered = solid.sort_values('bucket')['mean_excess'].tolist()
        rising = len(ordered) > 2 and all(x <= y for x, y in zip(ordered, ordered[1:]))
        # Averages can be carried by a few big dates; ask for consistency too.
        steady = bool(spread) and spread['wins'] >= 0.6 * spread['dates']
        if np.isfinite(ic) and ic > 0.02 and rising and steady:
            print('  ✅ more agreement meant higher returns, step by step, on most dates')
        elif np.isfinite(ic) and ic > 0 and (rising or (spread and spread['mean'] > 0)):
            print('  🟡 positive on average but not consistent from date to date — '
                  'a few strong periods carry it; treat agreement as a tilt, '
                  'not a rule')
        else:
            print('  ❌ agreement did not predict returns on this universe/window')

    print('\n▸ walk-forward backtest (quarterly rebalance, next-close fills)')
    keys = [strat] + ([k for fam in strat.components.values() for k in fam]
                      if args.compare else [])
    rows = []
    for k in keys:
        name = k.name if hasattr(k, 'name') else ST.get(k).name
        try:
            if (ST.get(k).horizon == 'mid'):
                res = BT.run_trade_backtest(spec, k, start, args.end, n_positions=10)
            else:
                res = BT.run_backtest(spec, k, start, args.end, n_holdings=args.top,
                                      rebalance='Q')
        except (ValueError, BT.DegradedBacktest) as exc:
            rows.append({'strategy': name, 'note': str(exc)[:70]})
            continue
        m = res.metrics
        rows.append({'strategy': name, 'cagr': m.get('cagr'),
                     'vs_spy': m.get('excess_cagr'), 'sharpe': m.get('sharpe'),
                     'max_dd': m.get('max_drawdown'), 'mean_ic': m.get('mean_ic')})
    out = pd.DataFrame(rows)
    for c in ('cagr', 'vs_spy', 'max_dd'):
        if c in out:
            out[c] = out[c].map(lambda v: f'{v:+.1%}' if isinstance(v, float) and v == v else '—')
    for c in ('sharpe', 'mean_ic'):
        if c in out:
            out[c] = out[c].map(lambda v: f'{v:+.2f}' if isinstance(v, float) and v == v else '—')
    print(out.fillna('').to_string(index=False))
    print('\nThe AI review vote is forward-only and is not part of any backtest; '
          'its value shows up in `python cli.py track`.')
    return 0


def cmd_macro(args) -> int:
    from data import macro
    db.init_db()
    if args.refresh:
        print(f'▸ fetched {macro.update_macro()} observations')
    snap = macro.snapshot(date.today())
    if snap.empty:
        print('no macro data yet — run `python cli.py macro --refresh`')
        return 0
    print(snap[['label', 'value', 'chg_1w', 'chg_1m', 'date']].round(2).to_string(index=False))
    print()
    icon = {'ok': '🟢', 'watch': '🟠', 'risk': '🔴'}
    for lvl, txt in macro.regime(date.today(), snap):
        print(f'{icon[lvl]} {txt}')
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
    scr.add_argument('--as-of', help='YYYY-MM-DD, or "session" for the last '
                                     'completed trading session (default: today)')
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

    rep = sub.add_parser('report', help='render a daily, news or monthly report')
    rep.add_argument('--kind', default='daily',
                     choices=['daily', 'news', 'monthly', 'auto'],
                     help='news = closed-market digest; auto = daily on '
                          'trading days, news otherwise')
    rep.add_argument('--strategy', default='buffett',
                     help='ignored; kept for old cron lines')
    rep.add_argument('--as-of')
    rep.add_argument('--email', help='send the rendered report to this address '
                                     '(comma-separate several)')
    rep.set_defaults(func=cmd_report)

    dly = sub.add_parser(
        'daily', help='scheduled routine: ingest, screen, build and email — '
                      'full brief on trading days, news-only when closed')
    dly.add_argument('--mode', choices=['quick', 'full'], default='quick',
                     help='quick: prices+news+policy; full: every source')
    dly.add_argument('--index', help=f'universe (default QUANT_INDEX={config.QUANT_INDEX})')
    dly.add_argument('--email', help='recipient(s); default QUANT_EMAIL from .env')
    dly.add_argument('--no-email', action='store_true',
                     help='prepare data and screens only (the evening job)')
    dly.add_argument('--kind', choices=['auto', 'daily', 'news'], default='auto')
    dly.add_argument('--date', help='pretend today is YYYY-MM-DD (for testing)')
    dly.add_argument('--resend', action='store_true',
                     help='send even if today\'s report was already emailed')
    dly.add_argument('--force', action='store_true',
                     help='run even when the market is closed and no email is due')
    dly.add_argument('--skip-ingest', action='store_true')
    dly.add_argument('--skip-screens', action='store_true')
    dly.add_argument('--backup', action='store_true',
                     help='back up the database afterwards (keeps 5)')
    dly.add_argument('--no-llm', action='store_true',
                     help='skip the AI earnings reviews this run')
    dly.add_argument('--dry-run', action='store_true',
                     help='print what would happen and exit')
    dly.set_defaults(func=cmd_daily)

    dbp = sub.add_parser('db', help='database upkeep: status | backup | check | clean')
    dbp.add_argument('action', choices=['status', 'backup', 'check', 'clean'])
    dbp.add_argument('--keep', type=int, default=5, help='backups to keep')
    dbp.add_argument('--full', action='store_true',
                     help='check: full integrity_check instead of quick_check')
    dbp.add_argument('--vacuum', action='store_true',
                     help='clean: VACUUM afterwards to return space to disk')
    dbp.add_argument('--prune-detail-days', type=int,
                     help='clean: drop per-factor detail of runs older than N days')
    dbp.add_argument('--no-backup', action='store_true',
                     help='clean: skip the safety backup')
    dbp.set_defaults(func=cmd_db)

    wt = sub.add_parser('watch', help='watchlist: add | remove | list')
    wt.add_argument('action', choices=['add', 'remove', 'list'])
    wt.add_argument('tickers', nargs='*')
    wt.add_argument('--note')
    wt.set_defaults(func=cmd_watch)

    pf = sub.add_parser(
        'portfolio', help='your positions: show | add | close | set | alerts | '
                          'import | remove')
    pf.add_argument('action', choices=['show', 'add', 'close', 'set', 'alerts',
                                       'import', 'remove'], nargs='?', default='show')
    pf.add_argument('ticker', nargs='?',
                    help='ticker (add/close/set), lot id (close/remove), or CSV path (import)')
    pf.add_argument('shares', nargs='?', type=float)
    pf.add_argument('cost', nargs='?', type=float, help='cost per share')
    pf.add_argument('--date', help='open/close date, YYYY-MM-DD')
    pf.add_argument('--stop', type=float, help='alert level (set -1 to clear)')
    pf.add_argument('--target', type=float, help='alert level (set -1 to clear)')
    pf.add_argument('--price', type=float, help='close: exit price (default last close)')
    pf.add_argument('--strategy', help='which screen it came from')
    pf.add_argument('--note')
    pf.add_argument('--lots', action='store_true', help='also list individual lots')
    pf.set_defaults(func=cmd_portfolio)

    tr = sub.add_parser('track', help='live track record of stored screens')
    tr.add_argument('--index', default=config.QUANT_INDEX,
                    help='only runs on this universe (default QUANT_INDEX)')
    tr.add_argument('--days', type=int, default=400, help='lookback in days')
    tr.add_argument('--top', type=int, default=10, help='picks per run')
    tr.set_defaults(func=cmd_track)

    er = sub.add_parser('earnings', help='upcoming earnings for your names')
    er.add_argument('--days', type=int, default=21)
    er.add_argument('--index', help='also include an index, e.g. SPY')
    er.add_argument('--refresh', action='store_true', help='fetch the calendar first')
    er.set_defaults(func=cmd_earnings)

    mc = sub.add_parser('macro', help='rates, credit, VIX and regime flags (FRED)')
    mc.add_argument('--refresh', action='store_true', help='fetch from FRED first')
    mc.set_defaults(func=cmd_macro)

    rv = sub.add_parser('review', help='AI earnings reviews of recent results filings')
    rv.add_argument('tickers', nargs='*', help='limit to these tickers')
    rv.add_argument('--run', action='store_true',
                    help='review filings that have none yet (otherwise just list)')
    rv.add_argument('--days', type=int, default=10, help='filings from the last N days')
    rv.add_argument('--max', type=int, default=None,
                    help='cap on new reviews (default QUANT_LLM_MAX_REVIEWS)')
    rv.add_argument('--index', default=config.QUANT_INDEX)
    rv.set_defaults(func=cmd_review)

    ct = sub.add_parser('consensus-test',
                        help='does multi-strategy agreement predict returns? '
                             '(agreement study + backtest)')
    ct.add_argument('--universe', help=f'default QUANT_INDEX ({config.QUANT_INDEX})')
    ct.add_argument('--start', default=str(date.today() - timedelta(days=365 * 4)))
    ct.add_argument('--end')
    ct.add_argument('--horizon', type=int, default=3, help='forward months')
    ct.add_argument('--freq', default='QE', choices=['ME', 'QE'],
                    help='sampling: ME monthly, QE quarterly')
    ct.add_argument('--top', type=int, default=20, help='holdings in the backtest')
    ct.add_argument('--compare', action='store_true',
                    help='also backtest every member strategy for comparison')
    ct.set_defaults(func=cmd_consensus_test)

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
