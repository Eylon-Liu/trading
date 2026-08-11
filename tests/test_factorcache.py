"""
Factor-cache tests.

The cache is only safe because its key folds in a data version that any
ingest moves. These tests pin that property down, plus the rule that a broken
cache degrades to a miss rather than an exception.
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from quant import factorcache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / 'factors'
    d.mkdir()
    monkeypatch.setattr(factorcache, 'CACHE_DIR', d)
    return d


@pytest.fixture
def frame():
    return pd.DataFrame({'ROIC': [0.3, 0.2], 'MOM_12_1': [0.1, -0.05]},
                        index=['AAPL', 'MSFT'])


# ── keying ────────────────────────────────────────────────────────

def test_same_inputs_give_same_key():
    a = factorcache.key(['AAPL', 'MSFT'], '2026-08-10', version='v1')
    b = factorcache.key(['AAPL', 'MSFT'], '2026-08-10', version='v1')
    assert a == b


def test_ticker_order_does_not_change_the_key():
    """The universe is a set; a reordered list is the same request."""
    a = factorcache.key(['AAPL', 'MSFT'], '2026-08-10', version='v1')
    b = factorcache.key(['MSFT', 'AAPL'], '2026-08-10', version='v1')
    assert a == b


def test_different_universe_gives_different_key():
    a = factorcache.key(['AAPL'], '2026-08-10', version='v1')
    b = factorcache.key(['AAPL', 'MSFT'], '2026-08-10', version='v1')
    assert a != b


def test_different_as_of_gives_different_key():
    """Point-in-time: the same universe on another date is another answer."""
    a = factorcache.key(['AAPL'], '2026-08-10', version='v1')
    b = factorcache.key(['AAPL'], '2026-08-09', version='v1')
    assert a != b


def test_new_data_invalidates_the_key():
    """The property the whole design rests on: ingest must bust the cache."""
    a = factorcache.key(['AAPL'], '2026-08-10', version='facts=100')
    b = factorcache.key(['AAPL'], '2026-08-10', version='facts=101')
    assert a != b


# ── round trip ────────────────────────────────────────────────────

def test_save_then_load_round_trips(cache_dir, frame):
    factorcache.save('k1', frame)
    out = factorcache.load('k1')
    pd.testing.assert_frame_equal(out, frame)


def test_missing_key_is_a_miss(cache_dir):
    assert factorcache.load('nope') is None


def test_empty_frames_are_not_cached(cache_dir):
    """Caching an empty result would pin a transient failure in place."""
    factorcache.save('k2', pd.DataFrame())
    assert factorcache.load('k2') is None


def test_corrupt_entry_is_a_miss_not_a_crash(cache_dir, frame):
    factorcache.save('k3', frame)
    (cache_dir / 'k3.pkl').write_bytes(b'not a pickle at all')
    assert factorcache.load('k3') is None


def test_corrupt_entry_is_removed(cache_dir, frame):
    factorcache.save('k4', frame)
    (cache_dir / 'k4.pkl').write_bytes(b'garbage')
    factorcache.load('k4')
    assert not (cache_dir / 'k4.pkl').exists()


def test_non_dataframe_payload_is_a_miss(cache_dir):
    with (cache_dir / 'k5.pkl').open('wb') as fh:
        pickle.dump({'not': 'a frame'}, fh)
    assert factorcache.load('k5') is None


def test_no_temp_files_survive_a_save(cache_dir, frame):
    """Writes are atomic, so a reader never sees a partial entry."""
    factorcache.save('k6', frame)
    assert list(cache_dir.glob('*.tmp')) == []


# ── housekeeping ──────────────────────────────────────────────────

def test_prune_enforces_the_entry_cap(cache_dir, frame, monkeypatch):
    monkeypatch.setattr(factorcache, 'MAX_ENTRIES', 5)
    for i in range(12):
        factorcache.save(f'e{i}', frame)
    assert len(list(cache_dir.glob('*.pkl'))) <= 5


def test_clear_empties_the_cache(cache_dir, frame):
    for i in range(3):
        factorcache.save(f'c{i}', frame)
    assert factorcache.clear() == 3
    assert factorcache.stats()['entries'] == 0


def test_stats_reports_entries_and_size(cache_dir, frame):
    factorcache.save('s1', frame)
    st = factorcache.stats()
    assert st['entries'] == 1 and st['bytes'] > 0


def test_version_failure_bypasses_rather_than_serving_stale(monkeypatch):
    """If freshness cannot be proven, miss — never guess."""
    def boom(*a, **kw):
        raise RuntimeError('db down')

    monkeypatch.setattr(factorcache.db, 'read_sql', boom)
    v1 = factorcache.data_version()
    v2 = factorcache.data_version()
    assert v1.startswith('nover-') and v1 != v2
