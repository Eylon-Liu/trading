"""
Background jobs started from the web UI — the "remote control".

Only the actions listed in ACTIONS can run, each a fixed `cli.py` command
line; nothing a browser sends becomes part of a command. One job runs at a
time, as a subprocess, so a long ingest neither blocks the web server nor
takes it down if it fails. Output goes to logs/jobs/, which is what the
Control tab tails while a job runs.

State lives in this process. The production server (waitress) is a single
process with threads, so every request sees the same job.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import config

log = logging.getLogger(__name__)

LOG_DIR = config.ROOT / 'logs' / 'jobs'
KEEP_LOGS = 40


@dataclass(frozen=True)
class Action:
    key: str
    label: str
    args: tuple[str, ...]
    blurb: str
    minutes: str = ''


ACTIONS: dict[str, Action] = {a.key: a for a in (
    Action('brief', '📧 Send today\'s email', ('daily',),
           'Refresh news and prices, run the screens and email the brief — '
           'the same as the 7:00 job. Skipped if today\'s already went out.',
           '5–20 min'),
    Action('resend', '🔁 Re-send today\'s email', ('daily', '--resend', '--skip-ingest'),
           'Rebuild today\'s brief from stored data and send it again.', '1–3 min'),
    Action('preview', '👁️ Build today\'s brief (no email)', ('report', '--kind', 'auto'),
           'From stored data only; open it on the Reports tab.', '< 1 min'),
    Action('refresh', '🔄 Full refresh (no email)',
           ('daily', '--mode', 'full', '--no-email', '--force'),
           'Every source — SEC filings, fundamentals, prices — then all screens. '
           'What the 18:30 job does. The very first one also loads two years '
           'of insider filings and takes a few hours.', '10–40 min'),
    Action('reviews', '🧠 Run AI earnings reviews', ('review', '--run'),
           'Review results filings from the last 10 days that have none yet.',
           '1–5 min'),
    Action('calendars', '📅 Refresh earnings & macro',
           ('earnings', '--refresh'), 'Upcoming report dates for your names.',
           '< 2 min'),
    Action('backup', '💾 Back up the database', ('db', 'backup'),
           'Consistent, compressed copy to db/backups/ (keeps 3).', '1–2 min'),
)}


@dataclass
class Job:
    key: str
    label: str
    log_path: Path
    started: float
    proc: subprocess.Popen | None = field(default=None, repr=False)
    finished: float | None = None
    returncode: int | None = None

    @property
    def running(self) -> bool:
        return self.finished is None

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


_lock = threading.Lock()
_current: Job | None = None
_history: list[Job] = []


def _reap() -> None:
    """Record the exit of the running job, if it has exited."""
    global _current
    if _current is not None and _current.proc is not None and _current.running:
        rc = _current.proc.poll()
        if rc is not None:
            _current.returncode = rc
            _current.finished = time.time()
            _history.insert(0, _current)
            del _history[20:]
            _current = None


def current() -> Job | None:
    with _lock:
        _reap()
        return _current


def history() -> list[Job]:
    with _lock:
        _reap()
        return list(_history)


def start(key: str) -> Job:
    """Start a whitelisted action. Raises KeyError / RuntimeError."""
    global _current
    action = ACTIONS[key]
    with _lock:
        _reap()
        if _current is not None:
            raise RuntimeError(f'“{_current.label}” is still running')
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f'{datetime.now():%Y%m%d-%H%M%S}-{key}.log'
        fh = path.open('w')
        fh.write(f'$ python cli.py {" ".join(action.args)}\n')
        fh.flush()
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        proc = subprocess.Popen([sys.executable, 'cli.py', *action.args],
                                cwd=str(config.ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        fh.close()
        _current = Job(key, action.label, path, time.time(), proc)
        log.info('job started: %s (pid %s)', key, proc.pid)
    _prune_logs()
    return _current


def cancel() -> bool:
    with _lock:
        _reap()
        if _current is None or _current.proc is None:
            return False
        _current.proc.terminate()
        return True


def tail(path: Path, lines: int = 40) -> str:
    """The last `lines` of a log, without reading a huge file into memory."""
    try:
        with path.open('rb') as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 16_000))
            text = fh.read().decode('utf-8', errors='replace')
    except OSError:
        return ''
    return '\n'.join(text.splitlines()[-lines:])


def latest_routine_log() -> Path | None:
    """Newest log written by the scheduled routine (launchd, cron or Docker)."""
    logs = sorted((config.ROOT / 'logs').glob('routine_*.log'),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    daily = config.ROOT / 'logs' / 'daily.log'
    cands = logs[:1] + ([daily] if daily.exists() else [])
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _prune_logs() -> None:
    try:
        files = sorted(LOG_DIR.glob('*.log'), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        for old in files[KEEP_LOGS:]:
            old.unlink(missing_ok=True)
    except OSError:
        pass
