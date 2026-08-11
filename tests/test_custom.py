"""
Custom-strategy tests.

The properties that matter: a copy is independent of its parent, a built-in
can never be overwritten, and a malformed definition is rejected at the door
with a message a user can act on — rather than failing somewhere inside the
scoring pipeline with an error that names no strategy.
"""

from __future__ import annotations

import pytest

from quant import custom as CU
from quant import strategies as ST


# ── slugs ─────────────────────────────────────────────────────────

def test_slug_is_prefixed_so_custom_keys_are_recognisable():
    assert CU.slugify('Quality Tilt').startswith(CU.CUSTOM_PREFIX)
    assert CU.is_custom(CU.slugify('Quality Tilt'))


def test_slug_does_not_double_prefix():
    """A name starting with "My" must not become my_my_..."""
    assert CU.slugify('My Quality Tilt') == 'my_quality_tilt'


def test_slug_survives_punctuation_and_blanks():
    assert CU.slugify('Deep  Value!! (v2)') == 'my_deep_value_v2'
    assert CU.slugify('   ').startswith(CU.CUSTOM_PREFIX)


# ── validation ────────────────────────────────────────────────────

def test_unknown_factor_is_rejected_by_name():
    with pytest.raises(CU.ValidationError, match='NOT_A_FACTOR'):
        CU.validate('X', 'long', {'NOT_A_FACTOR': 1.0})


def test_empty_weights_rejected():
    with pytest.raises(CU.ValidationError, match='at least one factor'):
        CU.validate('X', 'long', {})


def test_all_zero_weights_rejected():
    """Would divide by zero and rank nothing."""
    with pytest.raises(CU.ValidationError, match='nothing would be scored'):
        CU.validate('X', 'long', {'ROIC': 0})


def test_negative_weight_rejected_with_an_explanation():
    with pytest.raises(CU.ValidationError, match='negative'):
        CU.validate('X', 'long', {'ROIC': -1})


def test_blank_name_rejected():
    with pytest.raises(CU.ValidationError):
        CU.validate('  ', 'long', {'ROIC': 1})


def test_unknown_horizon_rejected():
    with pytest.raises(CU.ValidationError, match='horizon'):
        CU.validate('X', 'sideways', {'ROIC': 1})


def test_too_many_factors_rejected():
    from quant import factors as FA
    many = {f: 1.0 for f in list(FA.FACTOR_DIRECTION)[:CU.MAX_FACTORS + 1]}
    with pytest.raises(CU.ValidationError, match='limit'):
        CU.validate('X', 'long', many)


def test_zero_weights_are_dropped_not_kept():
    cleaned = CU.validate('X', 'long', {'ROIC': 1.0, 'ROE': 0.0})
    assert cleaned == {'ROIC': 1.0}


# ── share arithmetic ──────────────────────────────────────────────

def test_normalized_sums_to_100():
    out = CU.normalized({'ROIC': 3, 'ROE': 1})
    assert out == {'ROIC': 75.0, 'ROE': 25.0}


def test_rebalance_hits_the_target_share():
    w = {'ROIC': 1, 'ROE': 1, 'FCF_YIELD': 2}
    out = CU.normalized(CU.rebalanced(w, 'ROIC', 50))
    assert out['ROIC'] == pytest.approx(50, abs=0.6)


def test_rebalance_keeps_other_factors_in_proportion():
    """Dragging one slider must not reshuffle the rest."""
    w = {'ROIC': 1, 'ROE': 1, 'FCF_YIELD': 2}
    out = CU.rebalanced(w, 'ROIC', 50)
    assert out['FCF_YIELD'] == pytest.approx(out['ROE'] * 2, rel=1e-6)


def test_rebalance_is_clamped():
    w = {'ROIC': 1, 'ROE': 1}
    assert CU.rebalanced(w, 'ROIC', 500)['ROIC'] <= 1.0
    assert CU.rebalanced(w, 'ROIC', -10)['ROIC'] >= 0.0


# ── persistence & isolation ───────────────────────────────────────

@pytest.fixture
def store(monkeypatch):
    """In-memory stand-in for the custom_strategies table."""
    rows: dict[str, dict] = {}

    def fake_upsert(table, records):
        for r in records:
            rows[r['key']] = dict(r)
        return len(records)

    def fake_read_sql(sql, params=None, **kw):
        import pandas as pd
        if params and 'k' in params:
            r = rows.get(params['k'])
            return pd.DataFrame([r] if r else [])
        return pd.DataFrame(list(rows.values()))

    def fake_execute(sql, params=None):
        if params and 'k' in params and params['k'] in rows:
            del rows[params['k']]
            return 1
        return 0

    monkeypatch.setattr(CU.db, 'upsert', fake_upsert)
    monkeypatch.setattr(CU.db, 'read_sql', fake_read_sql)
    monkeypatch.setattr(CU.db, 'execute', fake_execute)
    return rows


def test_copy_is_independent_of_its_parent(store):
    """A saved strategy that changes underneath you is not worth saving."""
    key = CU.copy_from('compounder', 'Tilted')
    parent_weights = dict(ST.ALL_STRATEGIES['compounder'].weights)

    copy = CU.resolve(key)
    copy_weights = dict(copy.weights)
    copy_weights['ROIC'] = 99.0
    CU.save('Tilted', copy.horizon, copy_weights, key=key)

    assert ST.ALL_STRATEGIES['compounder'].weights == parent_weights
    assert CU.resolve(key).weights['ROIC'] == 99.0


def test_copy_records_its_parent(store):
    key = CU.copy_from('deep_value', 'My Value')
    assert store[key]['based_on'] == 'deep_value'


def test_custom_cannot_shadow_a_builtin(store):
    with pytest.raises(CU.ValidationError, match='collides'):
        CU.save('x', 'long', {'ROIC': 1}, key='compounder')


def test_builtins_cannot_be_deleted(store):
    with pytest.raises(CU.ValidationError, match='cannot be deleted'):
        CU.delete('compounder')


def test_registry_contains_both_kinds(store):
    key = CU.copy_from('garp', 'Mine')
    reg = CU.registry()
    assert 'garp' in reg and key in reg


def test_options_mark_custom_entries(store):
    key = CU.copy_from('garp', 'Mine')
    labels = {o['value']: o['label'] for o in CU.options()}
    assert labels[key].startswith('★')
    assert not labels['garp'].startswith('★')


def test_a_corrupt_row_does_not_hide_the_others(store, monkeypatch):
    good = CU.copy_from('garp', 'Good One')
    store['my_broken'] = {**store[good], 'key': 'my_broken',
                          'weights_json': '{not json'}
    loaded = CU.load_all()
    assert good in loaded and 'my_broken' not in loaded
