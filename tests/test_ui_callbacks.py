"""
Every UI callback must reference properties its components actually have.

Dash only reports a bad property in the browser console, at click time: the
Reports tab's Build button read `value` from a dcc.Store (which only has
`data`) and silently did nothing. This walks the real layout and checks
every Input and State against the component's declared properties.
"""

from __future__ import annotations

from dash.development.base_component import Component


def _components(node, out):
    if isinstance(node, Component):
        cid = getattr(node, 'id', None)
        if isinstance(cid, str):
            out[cid] = node
        children = getattr(node, 'children', None)
        if isinstance(children, (list, tuple)):
            for c in children:
                _components(c, out)
        elif children is not None:
            _components(children, out)
    return out


def test_callback_inputs_and_states_use_real_properties():
    import app
    from dash._callback import GLOBAL_CALLBACK_LIST

    layout = app.app.layout() if callable(app.app.layout) else app.app.layout
    comps = _components(layout, {})
    assert len(comps) > 50, 'layout walk found too few components'

    bad = []
    for cb in GLOBAL_CALLBACK_LIST + list(app.app.callback_map.values()):
        for dep in cb.get('inputs', []) + cb.get('state', []):
            cid, prop = dep['id'], dep['property']
            if isinstance(cid, str) and cid in comps \
                    and prop not in comps[cid].available_properties:
                bad.append(f'{cid}.{prop} ({type(comps[cid]).__name__})')
    assert not bad, f'callbacks reference missing properties: {sorted(set(bad))}'


def test_every_tab_has_a_panel():
    from ui.layout import TABS
    ids = [t for t, _l, _m in TABS]
    assert 'tab-portfolio' in ids
    assert len(ids) == len(set(ids))
