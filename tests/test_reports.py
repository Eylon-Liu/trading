"""
The daily email: full brief on trading days, news-only digest when the
market is closed, one history row per report, and no double sends.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

import cli
import config
from data.universe import UniverseSpec
from quant import engine as EN
from quant import strategies as ST
from reports import builder as RB
from reports import email as MAIL

FRI, SAT, MON = date(2026, 11, 6), date(2026, 11, 7), date(2026, 11, 9)
THU = date(2026, 11, 5)
SPEC = UniverseSpec(preset='SPY')


@pytest.fixture
def store(fresh_db, monkeypatch, tmp_path):
    """Two sessions of runs for two strategies, prices, news and filings."""
    db = fresh_db
    monkeypatch.setattr(config, 'REPORT_DIR', tmp_path / 'out')
    monkeypatch.setattr(config, 'WATCHLIST', ['NVDA'])
    # No network: the benchmark refresh and any auto-run must not fire.
    monkeypatch.setattr(RB.yahoo, 'update_prices', lambda *a, **k: 0)
    monkeypatch.setattr(RB, '_all_strategy_keys', lambda: ['buffett', 'momentum_trend'])
    monkeypatch.setattr(RB, '_auto_run_prior',
                        lambda *a, **k: (None, 'auto-run disabled in tests'))

    def run(rid, as_of, strategy, order):
        n = len(order)
        scores = pd.DataFrame({
            'composite': [1.0 - i / n for i in range(n)], 'rank': range(1, n + 1),
            'signal': ['🟢 Buy'] * n, 'signal_score': 3.0, 'reasons': '',
            'coverage': 1.0, 'sector': 'Information Technology'}, index=order)
        EN._persist(rid, as_of, SPEC, ST.get(strategy), order, scores, pd.DataFrame())

    run('b-thu', THU, 'buffett', ['AAPL', 'MSFT', 'NVDA'])
    run('m-thu', THU, 'momentum_trend', ['AAPL', 'AMZN', 'NVDA'])
    run('b-fri', FRI, 'buffett', ['MSFT', 'AAPL', 'NVDA'])
    run('m-fri', FRI, 'momentum_trend', ['MSFT', 'NVDA', 'AAPL'])

    bars = []
    for i in range(30):
        d = FRI - timedelta(days=i)
        if d.weekday() < 5:
            for t, base in (('SPY', 600), ('NVDA', 180), ('MSFT', 500), ('AAPL', 270)):
                px = base * (1 + 0.001 * (30 - i))
                bars.append({'ticker': t, 'date': d, 'open': px, 'high': px,
                             'low': px, 'close': px, 'adj_close': px, 'volume': 1e6})
    db.upsert(db.prices, bars)

    close_fri = datetime(2026, 11, 6, 21, 0)          # 16:00 ET, naive UTC
    news = [
        ('NVDA', close_fri + timedelta(hours=3), 'Nvidia raises guidance & expands buyback',
         'guidance_raised,buyback', 'positive'),
        ('MSFT', close_fri + timedelta(hours=10), 'Microsoft names new CFO',
         'management_change', 'neutral'),
        ('XOM', close_fri + timedelta(hours=12), 'Exxon to acquire rival in $20B deal',
         'acquisition', 'positive'),
        ('AAPL', close_fri - timedelta(hours=5), 'Apple shares edge higher before close',
         '', 'neutral'),
    ]
    from data.news import _mk_id
    db.upsert(db.news, [{
        'id': _mk_id(t, title), 'ticker': t, 'published': when, 'source': 'Test',
        'title': title, 'url': f'https://example.com/{t}', 'body_summary': '',
        'sentiment': 0.4 if lab == 'positive' else 0.0, 'sentiment_label': lab,
        'events': ev, 'extracted_json': '{}', 'fetched_at': when,
    } for t, when, title, ev, lab in news])
    db.upsert(db.corporate_events, [{
        'ticker': 'XOM', 'filed': SAT, 'accession': 'acc-1', 'item_code': '1.01',
        'category': 'strategy', 'subtype': 'Material definitive agreement',
        'detail': ''}])
    return db


# ── which email ─────────────────────────────────────────────────

@pytest.mark.parametrize('day,kind', [
    (date(2026, 11, 6), 'daily'),      # Friday
    (date(2026, 11, 7), 'news'),       # Saturday
    (date(2026, 11, 8), 'news'),       # Sunday
    (date(2026, 11, 26), 'news'),      # Thanksgiving
    (date(2026, 11, 27), 'daily'),     # half day is still a session
    (date(2026, 7, 3), 'news'),        # Independence Day observed
])
def test_kind_follows_the_exchange_calendar(day, kind):
    assert RB.auto_kind(day) == kind


def test_subjects_say_what_the_email_is():
    assert 'Weekend News' in RB.subject_for('news', SAT)
    assert 'Thanksgiving' in RB.subject_for('news', date(2026, 11, 26))
    assert 'Market Brief' in RB.subject_for('daily', MON)


# ── weekend digest ───────────────────────────────────────────────

def test_weekend_digest_is_news_only(store):
    path, html = RB.build_report(kind='news', as_of=SAT,
                                 now=datetime(2026, 11, 7, 12, 0), universe=SPEC)
    assert 'Weekend News Digest' in html
    assert 'Nvidia raises guidance &amp; expands buyback' in html     # escaped
    assert 'Exxon to acquire rival' in html                            # material event
    assert 'Material definitive agreement' in html                     # 8-K
    assert 'Apple shares edge higher before close' not in html         # pre-close
    # Nothing priced or ranked on a closed day.
    for absent in ('Market snapshot', 'Consensus picks', '1D return'):
        assert absent not in html
    assert 'Your portfolio &amp; watchlist' in html
    assert path.endswith('news_digest_2026-11-07.html')


def test_sunday_digest_does_not_repeat_saturday(store):
    start, _end, last = RB.digest_window(date(2026, 11, 8),
                                         now=datetime(2026, 11, 8, 12, 0))
    assert last == FRI
    assert start == datetime(2026, 11, 7, 12, 0)


def test_missed_email_widens_the_window_back_to_the_last_send(store):
    store.upsert(store.reports, [{'report_id': 'daily-2026-11-06', 'kind': 'daily',
                                  'emailed_at': datetime(2026, 11, 6, 12, 0)}])
    start, _end, _last = RB.digest_window(date(2026, 11, 8),
                                          now=datetime(2026, 11, 8, 12, 0))
    assert start == datetime(2026, 11, 6, 21, 0)       # Friday close, not Sat noon


# ── trading-day brief ────────────────────────────────────────────

def test_daily_brief_compares_consecutive_sessions(store):
    _path, html = RB.build_report(kind='daily', as_of=MON, universe=SPEC)
    assert 'Market Intelligence Brief' in html
    assert 'data through the Fri Nov 6 close' in html
    assert 'previous session (Thu Nov 5)' in html
    assert 'Your watchlist' in html and 'NVDA' in html
    # Data is current through Friday, so no staleness banner.
    assert 'class="alert"' not in html


def test_stale_prices_are_flagged(store):
    store.execute("DELETE FROM prices WHERE date = '2026-11-06'")
    _path, html = RB.build_report(kind='daily', as_of=MON, universe=SPEC)
    assert 'Prices are current only to the Thu Nov 5 close' in html


def test_adhoc_ui_run_does_not_hijack_the_scheduled_brief(store):
    narrow = UniverseSpec(preset='SPY', sectors=['Energy'])
    scores = pd.DataFrame({'composite': [1.0], 'rank': [1], 'signal': ['🟢 Buy'],
                           'signal_score': 3.0, 'reasons': '', 'coverage': 1.0,
                           'sector': 'Energy'}, index=['XOM'])
    EN._persist('adhoc', MON, narrow, ST.get('buffett'), ['XOM'], scores, pd.DataFrame())
    run = RB._latest_run('buffett', MON, SPEC.to_json())
    assert run['run_id'] == 'b-fri'


def test_rebuild_keeps_one_row_and_the_emailed_stamp(store):
    RB.build_report(kind='daily', as_of=MON, universe=SPEC)
    store.upsert(store.reports, [{'report_id': 'daily-2026-11-09',
                                  'emailed_at': datetime(2026, 11, 9, 12)}])
    RB.build_report(kind='daily', as_of=MON, universe=SPEC)
    rows = store.read_sql("SELECT report_id, emailed_at FROM reports")
    assert len(rows) == 1
    assert RB.emailed_at('daily-2026-11-09') is not None


# ── the routine ──────────────────────────────────────────────────

def test_daily_command_sends_once_per_day(store, monkeypatch):
    sent = []
    monkeypatch.setattr(MAIL, 'send_report',
                        lambda html, to, subject=None, report_id=None, **k:
                        (sent.append((to, subject)),
                         store.upsert(store.reports, [{'report_id': report_id,
                                                       'emailed_at': datetime.utcnow()}])))
    args = ['daily', '--date', '2026-11-07', '--email', 'me@example.com',
            '--skip-ingest', '--index', 'SPY']
    assert cli.main(args) == 0
    assert cli.main(args) == 0                     # e.g. launchd re-fires on wake
    assert len(sent) == 1
    assert 'Weekend News' in sent[0][1]
    assert cli.main(args + ['--resend']) == 0
    assert len(sent) == 2


def test_evening_job_does_nothing_when_closed(store, capsys):
    assert cli.main(['daily', '--date', '2026-11-07', '--no-email',
                     '--skip-ingest']) == 0
    assert 'nothing to prepare' in capsys.readouterr().out


def test_missing_recipient_is_an_error(store, monkeypatch):
    monkeypatch.setattr(config, 'QUANT_EMAIL', None)
    assert cli.main(['daily', '--date', '2026-11-07', '--skip-ingest']) == 1


# ── mail formatting ──────────────────────────────────────────────

def test_plain_text_part_is_readable():
    text = MAIL.html_to_text('<html><head><style>.x{}</style></head><body>'
                             '<h1>Brief</h1><table><tr><td>AAPL</td><td>+1%</td>'
                             '</tr></table></body></html>')
    assert 'Brief' in text and 'AAPL +1%' in text and '.x{}' not in text


def test_several_recipients_are_split():
    assert MAIL.recipients('a@x.com, b@y.com;c@z.com') == ['a@x.com', 'b@y.com', 'c@z.com']
