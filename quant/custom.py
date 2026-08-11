"""
User-defined strategies.

The built-in strategies are a starting set, not a menu to pick from. Several
are close cousins — "Quality at a Reasonable Price" and "Quality Compounder"
differ mainly in whether valuation is weighted at all — and the honest answer
to "which should I use?" is usually "one of these with the weights moved".

So a preset can be copied into an editable strategy: rename it, reweight it,
add or drop factors. The copy is independent of its parent, because otherwise
tuning a preset would silently change every saved variant built from it.

Custom strategies are stored as rows rather than code so they survive
restarts, and they are validated on the way in: an unknown factor name or a
weight vector summing to zero would fail deep inside the scoring pipeline with
an error that says nothing about which strategy caused it.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import replace
from datetime import datetime

import pandas as pd

from core import db
from quant import factors as FA
from quant import strategies as ST

log = logging.getLogger(__name__)

CUSTOM_PREFIX = 'my_'
MAX_FACTORS = 25


class ValidationError(ValueError):
    """Raised with a message meant to be shown directly to the user."""


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def slugify(name: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', (name or '').strip().lower()).strip('_')
    if not s:
        return f'{CUSTOM_PREFIX}{uuid.uuid4().hex[:8]}'
    # A name that already begins with "my" would otherwise become my_my_...
    return s if s.startswith(CUSTOM_PREFIX) else f'{CUSTOM_PREFIX}{s}'


def validate(name: str, horizon: str, weights: dict) -> dict:
    """Check a definition and return cleaned weights, or raise ValidationError."""
    if not (name or '').strip():
        raise ValidationError('Give the strategy a name.')
    if horizon not in ST.HORIZONS:
        raise ValidationError(f'Unknown horizon {horizon!r}.')
    if not weights:
        raise ValidationError('Add at least one factor.')
    if len(weights) > MAX_FACTORS:
        raise ValidationError(
            f'{len(weights)} factors is more than the {MAX_FACTORS} limit. '
            f'Blending too many dilutes every one of them.')

    known = set(FA.FACTOR_DIRECTION)
    cleaned: dict[str, float] = {}
    for factor, weight in weights.items():
        if factor not in known:
            raise ValidationError(
                f'{factor!r} is not a known factor. See the Methodology tab '
                f'for the full list.')
        try:
            w = float(weight)
        except (TypeError, ValueError):
            raise ValidationError(f'Weight for {factor} is not a number.')
        if w < 0:
            raise ValidationError(
                f'Weight for {factor} is negative. Direction is handled by the '
                f'factor itself — use the factor whose direction you want '
                f'rather than a negative weight.')
        if w > 0:
            cleaned[factor] = w

    if not cleaned:
        raise ValidationError('Every weight is zero, so nothing would be scored.')
    return cleaned


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def save(name: str, horizon: str, weights: dict, *, description: str = '',
         thesis: str = '', based_on: str | None = None,
         filters: dict | None = None, neutralize: str = 'sector_z',
         setup: str = 'momentum', key: str | None = None) -> str:
    """Create or overwrite a custom strategy. Returns its key."""
    cleaned = validate(name, horizon, weights)
    key = key or slugify(name)

    if key in ST.ALL_STRATEGIES and not key.startswith(CUSTOM_PREFIX):
        raise ValidationError(
            f'“{name}” collides with the built-in strategy {key!r}. '
            f'Pick another name — built-ins are read-only by design, so that '
            f'the documented behaviour stays documented.')

    db.upsert(db.custom_strategies, [{
        'key': key,
        'name': name.strip(),
        'horizon': horizon,
        'description': description.strip() or f'Custom strategy: {name.strip()}.',
        'thesis': thesis.strip(),
        'based_on': based_on,
        'weights_json': json.dumps(cleaned),
        'filters_json': json.dumps(filters or {}),
        'neutralize': neutralize,
        'setup': setup,
        'updated_at': datetime.utcnow(),
    }])
    log.info('saved custom strategy %s (%d factors)', key, len(cleaned))
    return key


def delete(key: str) -> bool:
    if not key.startswith(CUSTOM_PREFIX):
        raise ValidationError('Built-in strategies cannot be deleted.')
    n = db.execute('DELETE FROM custom_strategies WHERE key = :k', {'k': key})
    return bool(n)


def load_all() -> dict[str, ST.Strategy]:
    """Every stored custom strategy, as Strategy objects."""
    try:
        rows = db.read_sql('SELECT * FROM custom_strategies ORDER BY name')
    except Exception as exc:                       # noqa: BLE001
        log.debug('could not read custom strategies: %s', exc)
        return {}

    out: dict[str, ST.Strategy] = {}
    for _i, r in rows.iterrows():
        try:
            out[r['key']] = ST.Strategy(
                key=r['key'], name=r['name'], horizon=r['horizon'],
                description=r['description'] or '',
                thesis=r['thesis'] or '',
                weights=json.loads(r['weights_json']),
                filters=json.loads(r['filters_json'] or '{}'),
                neutralize=r['neutralize'] or 'sector_z',
                setup=r['setup'] or 'momentum',
            )
        except Exception as exc:                   # noqa: BLE001
            # One malformed row must not hide the rest.
            log.warning('skipping unreadable custom strategy %s: %s',
                        r.get('key'), exc)
    return out


def copy_from(source_key: str, new_name: str) -> str:
    """
    Duplicate a strategy into an editable one.

    The copy takes a snapshot of the parent's weights rather than referencing
    it, so later edits to either are independent.
    """
    base = resolve(source_key)
    return save(new_name, base.horizon, dict(base.weights),
                description=f'Copied from {base.name}. {base.description}',
                thesis=base.thesis, based_on=source_key,
                filters=dict(base.filters), neutralize=base.neutralize,
                setup=base.setup)


# ─────────────────────────────────────────────
# REGISTRY VIEW  (built-ins + custom)
# ─────────────────────────────────────────────

def registry() -> dict[str, ST.Strategy]:
    """Built-ins plus custom. Custom cannot shadow a built-in key."""
    merged = dict(ST.ALL_STRATEGIES)
    for k, s in load_all().items():
        if k in merged:
            log.warning('custom strategy %s shadows a built-in; ignoring', k)
            continue
        merged[k] = s
    return merged


def resolve(key: str) -> ST.Strategy:
    """`strategies.get`, extended to custom strategies."""
    reg = registry()
    if key not in reg:
        raise KeyError(f'unknown strategy {key!r}')
    return reg[key]


def options(horizon: str | None = None) -> list[dict]:
    """Dropdown options, with custom strategies grouped under a marker."""
    reg = registry()
    built_in, custom = [], []
    for k, s in sorted(reg.items(), key=lambda kv: kv[1].name):
        if horizon and s.horizon != horizon:
            continue
        (custom if k.startswith(CUSTOM_PREFIX) else built_in).append(
            {'label': (f'★ {s.name}' if k.startswith(CUSTOM_PREFIX) else s.name),
             'value': k})
    return built_in + custom


def is_custom(key: str) -> bool:
    return key.startswith(CUSTOM_PREFIX)


def summary() -> pd.DataFrame:
    """Table of saved custom strategies, for the editor list."""
    reg = load_all()
    if not reg:
        return pd.DataFrame()
    return pd.DataFrame([{
        'name': s.name,
        'horizon': ST.HORIZONS[s.horizon]['label'],
        'factors': len(s.weights),
        'key': k,
    } for k, s in sorted(reg.items(), key=lambda kv: kv[1].name)])


def normalized(weights: dict) -> dict:
    """Weights as percentage shares, for display."""
    total = sum(weights.values()) or 1.0
    return {k: round(v / total * 100, 1) for k, v in weights.items()}


def rebalanced(weights: dict, factor: str, share_pct: float) -> dict:
    """
    Set one factor to a target share, scaling the rest to fit.

    Editing a raw weight changes every other factor's share as a side effect,
    which makes the sliders feel broken. Working in shares keeps the edit
    local to the factor being dragged.
    """
    share_pct = max(0.0, min(float(share_pct), 99.0))
    others = {k: v for k, v in weights.items() if k != factor}
    other_total = sum(others.values())
    if other_total <= 0:
        return {**{k: 0.0 for k in others}, factor: 1.0}

    target = share_pct / 100.0
    scale = (1.0 - target) / other_total
    # Rounded generously: weights are unitless and only their ratios matter,
    # so coarse rounding would visibly distort the untouched factors.
    out = {k: round(v * scale, 8) for k, v in others.items()}
    out[factor] = round(target, 8)
    return out
