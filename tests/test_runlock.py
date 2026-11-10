"""One routine at a time across processes: no double emails, no two ingests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

import config
from core import runlock

pytestmark = pytest.mark.skipif(runlock.fcntl is None, reason='needs fcntl')


@pytest.fixture
def lockdir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DB_DIR', tmp_path)
    monkeypatch.setattr(runlock, 'POLL_SECONDS', 0.05)
    return tmp_path


def test_free_lock_has_no_holder(lockdir):
    assert runlock.holder() is None
    with runlock.hold('daily --mode quick', say=lambda m: None) as ok:
        assert ok
    assert runlock.holder() is None


def test_another_process_sees_the_holder_and_waits(lockdir):
    # A real second process, like the 7:00 job versus a Control-tab button.
    code = textwrap.dedent(f'''
        import sys, time
        sys.path.insert(0, {str(config.ROOT)!r})
        import config; config.DB_DIR = {str(lockdir)!r}
        from core import runlock
        with runlock.hold('daily --mode quick'):
            print('locked', flush=True)
            time.sleep(1.5)
    ''')
    proc = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == 'locked'
    info = runlock.holder()
    assert info and info['what'] == 'daily --mode quick' and info['pid'] == proc.pid

    said = []
    with runlock.hold('daily', wait_seconds=0.2, say=said.append) as ok:
        assert ok is False                       # gave up: still held
    assert any('waiting for another run' in m for m in said)

    t0 = time.time()
    with runlock.hold('daily', say=said.append) as ok:
        assert ok and time.time() - t0 > 0.3     # waited for it to finish
    proc.wait(5)
    assert any('finished' in m for m in said)


def test_a_crashed_run_does_not_leave_it_stuck(lockdir):
    code = textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {str(config.ROOT)!r})
        import config; config.DB_DIR = {str(lockdir)!r}
        from core import runlock
        with runlock.hold('daily'):
            os._exit(9)
    ''')
    assert subprocess.call([sys.executable, '-c', code]) == 9
    with runlock.hold('daily', wait_seconds=1, say=lambda m: None) as ok:
        assert ok


def test_two_daily_runs_send_one_email(lockdir, monkeypatch):
    """The scheduled job and a button press overlap: one brief, not two."""
    import cli
    sent = []
    emailed = {'today': False}

    def fake_daily(args):
        if emailed['today'] and not args.resend:      # the real "already emailed?" check
            return 0
        time.sleep(0.3)                                # ingest + screens
        sent.append(args.mode)
        emailed['today'] = True
        return 0

    monkeypatch.setattr(cli, '_daily', fake_daily)
    args = SimpleNamespace(dry_run=False, mode='quick', no_email=False, resend=False)
    runs = [threading.Thread(target=cli.cmd_daily, args=(args,)) for _ in range(2)]
    for t in runs:
        t.start()
    for t in runs:
        t.join(10)
    assert sent == ['quick']
