"""
Shared test setup.

Tests run against a throwaway SQLite file, never the real store. Several
code paths write as a side effect of reading (membership resolution stores
what it fetches, a screen persists its run), so pointing the suite at
db/quant.sqlite meant `pytest` could quietly add rows to the database the
daily email is built from — and on a fresh clone the suite failed outright
because that database had no tables yet.

Set QUANT_TEST_REAL_DB=1 to run the data-dependent tests against the
configured database instead; they skip themselves when it holds no data.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

import pytest

if os.environ.get('QUANT_TEST_REAL_DB') != '1':
    _TMP = tempfile.mkdtemp(prefix='quant-tests-')
    # Must happen before `config` is imported anywhere: it reads the URL at
    # import time, and load_dotenv() does not override an existing variable.
    os.environ['DATABASE_URL'] = f'sqlite:///{_TMP}/session.sqlite'
    atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


@pytest.fixture(scope='session', autouse=True)
def _session_schema():
    from core import db
    db.init_db()
    yield


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """An empty database for one test, with the full schema."""
    import config
    from core import db

    monkeypatch.setattr(config, 'DATABASE_URL',
                        f'sqlite:///{tmp_path}/fresh.sqlite')
    previous = db._engine
    db._engine = None
    db.init_db()
    yield db
    db.get_engine().dispose()
    db._engine = previous
