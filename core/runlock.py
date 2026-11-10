"""
One routine at a time, across processes.

The 7:00 job (launchd, cron or `cli.py scheduler`), a button on the Control
tab and a command typed in a terminal are separate processes. Without a
shared lock two of them could ingest into the same database at once, and —
worse — both pass the "already emailed today?" check before either has
sent, so the brief arrives twice.

`hold()` takes an exclusive OS file lock (flock) on db/.routine.lock. A
second run waits for the first to finish and then does its own check, which
by then sees the email as sent. The lock is released by the kernel if the
process dies, so a crash never leaves it stuck. On a system without fcntl
(Windows) it degrades to no locking.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import config

try:
    import fcntl
except ImportError:                       # Windows
    fcntl = None

WAIT_SECONDS = 3 * 3600
POLL_SECONDS = 5


def lock_path() -> Path:
    return Path(config.DB_DIR) / '.routine.lock'


def _try_lock(fh) -> bool:
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text() or 'null')
    except (OSError, ValueError):
        return None


@contextmanager
def hold(what: str, wait_seconds: float = WAIT_SECONDS,
         say: Callable[[str], None] = print):
    """
    Yield True once this process holds the routine lock, or False if another
    run still held it after `wait_seconds`. Callers return early on False.
    """
    if fcntl is None:
        yield True
        return
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, 'a+')
    try:
        deadline = time.time() + wait_seconds
        told = False
        while not _try_lock(fh):
            if not told:
                other = _read(path) or {}
                say(f'⏳ waiting for another run to finish: '
                    f'{other.get("what", "?")} (pid {other.get("pid", "?")}, '
                    f'started {other.get("started", "?")})')
                told = True
            if time.time() >= deadline:
                say('❌ gave up waiting for the other run')
                yield False
                return
            time.sleep(POLL_SECONDS)
        if told:
            say('▸ the other run finished — continuing')
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({'pid': os.getpid(), 'what': what,
                             'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                             'since': time.time()}))
        fh.flush()
        try:
            yield True
        finally:
            fh.seek(0)
            fh.truncate()
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def holder() -> dict | None:
    """Who holds the lock right now (pid, what, started), or None if free."""
    if fcntl is None:
        return None
    path = lock_path()
    if not path.exists():
        return None
    with open(path, 'a+') as fh:
        if _try_lock(fh):
            fcntl.flock(fh, fcntl.LOCK_UN)
            return None
    return _read(path) or {'what': 'unknown', 'pid': '?'}
