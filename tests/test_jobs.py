"""Background jobs started from the Control tab."""

from __future__ import annotations

import time

import pytest

from core import jobs as JOBS


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(JOBS, 'LOG_DIR', tmp_path / 'jobs')
    monkeypatch.setattr(JOBS, 'ACTIONS', {
        'quick': JOBS.Action('quick', '⚡ Quick', ('strategies',), 'lists strategies'),
        'slow': JOBS.Action('slow', '⏳ Slow', ('-v', 'freshness'), 'checks freshness'),
    })
    monkeypatch.setattr(JOBS, '_current', None)
    monkeypatch.setattr(JOBS, '_history', [])
    yield
    job = JOBS.current()
    if job and job.proc:
        job.proc.kill()


def _wait(timeout=60):
    end = time.time() + timeout
    while JOBS.current() is not None and time.time() < end:
        time.sleep(0.2)


def test_runs_a_whitelisted_command_and_logs_it(runner):
    job = JOBS.start('quick')
    assert job.running
    _wait()
    done = JOBS.history()[0]
    assert done.returncode == 0 and not done.running
    out = JOBS.tail(done.log_path)
    assert '$ python cli.py strategies' in out
    assert 'buffett' in out


def test_one_job_at_a_time(runner):
    JOBS.start('slow')
    with pytest.raises(RuntimeError, match='still running'):
        JOBS.start('quick')
    _wait()


def test_only_listed_actions_can_run(runner):
    with pytest.raises(KeyError):
        JOBS.start('rm -rf /')


def test_cancel(runner):
    JOBS.start('slow')
    assert JOBS.cancel() is True
    _wait()
    assert JOBS.history()[0].returncode != 0 or JOBS.history()[0].finished


def test_tail_reads_only_the_end(tmp_path):
    f = tmp_path / 'big.log'
    f.write_text('\n'.join(f'line {i}' for i in range(100_000)))
    out = JOBS.tail(f, 3)
    assert out.splitlines() == ['line 99997', 'line 99998', 'line 99999']


def test_every_real_action_is_a_known_cli_command():
    import cli
    parser = cli.build_parser()
    for a in JOBS.ACTIONS.values():
        args = [x for x in a.args if not x.startswith('-v')]
        parser.parse_args(args)                    # raises SystemExit if invalid


# ── built-in scheduler (Docker / servers) ────────────────────────

from datetime import datetime, timezone   # noqa: E402


def _utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_nothing_due_before_the_morning_run(monkeypatch):
    import cli
    import config
    monkeypatch.setattr(config, 'QUANT_MORNING_TIME', '07:00')
    monkeypatch.setattr(config, 'QUANT_EVENING_TIME', '18:30')
    due, nxt = cli.scheduled_runs(_utc('2026-11-10T11:30:00'), 'America/New_York')
    assert due == [] and (nxt.hour, nxt.minute) == (7, 0)


def test_a_restart_shortly_after_still_sends_the_email(monkeypatch):
    import cli
    import config
    monkeypatch.setattr(config, 'QUANT_MORNING_TIME', '07:00')
    due, _ = cli.scheduled_runs(_utc('2026-11-10T13:20:00'), 'America/New_York')
    assert due == ['morning']                          # 08:20 ET, within 3h
    due, _ = cli.scheduled_runs(_utc('2026-11-10T13:20:00'), 'America/New_York',
                                last_done={'morning': '2026-11-10'})
    assert due == []                                   # already ran today


def test_evening_prep_is_weekdays_only(monkeypatch):
    import cli
    import config
    monkeypatch.setattr(config, 'QUANT_EVENING_TIME', '18:30')
    due, nxt = cli.scheduled_runs(_utc('2026-11-14T23:40:00'), 'America/New_York')
    assert 'evening' not in due                        # Saturday 18:40 ET
    assert nxt.weekday() == 6 and nxt.hour == 7        # next: Sunday morning


def test_times_follow_the_configured_zone_across_dst(monkeypatch):
    import cli
    import config
    monkeypatch.setattr(config, 'QUANT_MORNING_TIME', '07:00')
    # 7:00 New York is 11:00 UTC in summer and 12:00 UTC in winter.
    assert cli.scheduled_runs(_utc('2026-07-14T11:05:00'), 'America/New_York')[0] == ['morning']
    assert cli.scheduled_runs(_utc('2026-11-10T11:05:00'), 'America/New_York')[0] == []
