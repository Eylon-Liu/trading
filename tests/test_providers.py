"""
Transport-layer tests for nlp/providers.py.

No network. Every test either exercises pure selection logic or feeds a fake
response through the parser, because the property that matters is that a
failing backend degrades to None rather than raising into a Dash callback.
"""

from __future__ import annotations

import pytest

import config
from nlp import providers


# ─────────────────────────────────────────────
# PROVIDER SELECTION
# ─────────────────────────────────────────────

@pytest.fixture
def keys(monkeypatch):
    """Control both keys and the provider switch."""
    def _set(gemini=None, anthropic=None, choice='auto'):
        monkeypatch.setattr(config, 'GEMINI_API_KEY', gemini)
        monkeypatch.setattr(config, 'ANTHROPIC_API_KEY', anthropic)
        monkeypatch.setattr(config, 'LLM_PROVIDER', choice)
    return _set


def test_auto_prefers_gemini_when_both_present(keys):
    keys(gemini='g', anthropic='a')
    assert providers.resolve() == 'gemini'


def test_auto_falls_back_to_anthropic(keys):
    keys(gemini=None, anthropic='a')
    assert providers.resolve() == 'anthropic'


def test_no_key_resolves_to_nothing(keys):
    keys()
    assert providers.resolve() == ''
    assert providers.available() is False


def test_off_disables_even_with_keys(keys):
    keys(gemini='g', anthropic='a', choice='off')
    assert providers.resolve() == ''
    assert providers.status()['enabled'] is False


def test_explicit_choice_does_not_silently_fall_back(keys):
    """LLM_PROVIDER=gemini with no Gemini key must fail loudly, not use Claude.

    Silently switching backends would bill an unexpected account and change
    the model named in the UI footnote.
    """
    keys(gemini=None, anthropic='a', choice='gemini')
    assert providers.resolve() == ''
    assert 'GEMINI_API_KEY' in providers.status()['reason']


def test_complete_returns_none_without_a_backend(keys):
    keys()
    assert providers.complete('prompt', 'system', 100) is None


# ─────────────────────────────────────────────
# GEMINI RESPONSE PARSING
# ─────────────────────────────────────────────

class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def _patch_post(monkeypatch, response):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured['url'] = url
        captured['headers'] = headers or {}
        captured['body'] = json
        return response

    monkeypatch.setattr(providers.requests, 'post', fake_post)
    return captured


def _ok(text, finish='STOP'):
    return FakeResponse({'candidates': [
        {'finishReason': finish, 'content': {'parts': [{'text': text}]}}]})


def test_gemini_happy_path(monkeypatch, keys):
    keys(gemini='g')
    _patch_post(monkeypatch, _ok('  Rank IC is near zero.  '))
    assert providers.complete('p', 's', 500) == 'Rank IC is near zero.'


def test_api_key_travels_in_header_not_query_string(monkeypatch, keys):
    """A credential in a URL ends up in proxy logs and browser history."""
    keys(gemini='secret-key')
    cap = _patch_post(monkeypatch, _ok('text'))
    providers.complete('p', 's', 500)
    assert cap['headers']['x-goog-api-key'] == 'secret-key'
    assert 'secret-key' not in cap['url']
    assert 'key=' not in cap['url']


def test_thinking_parts_are_not_shown_to_the_reader(monkeypatch, keys):
    keys(gemini='g')
    _patch_post(monkeypatch, FakeResponse({'candidates': [{
        'finishReason': 'STOP',
        'content': {'parts': [
            {'text': 'internal reasoning', 'thought': True},
            {'text': 'the answer'},
        ]}}]}))
    assert providers.complete('p', 's', 500) == 'the answer'


def test_truncated_output_is_dropped_rather_than_shown(monkeypatch, keys):
    """Gemini charges thinking against maxOutputTokens.

    Measured on gemini-3.5-flash: a 40-token cap spent 35 on thinking and
    emitted one character. Half a sentence is worse than falling back to the
    local engine, so a short MAX_TOKENS response returns None.
    """
    keys(gemini='g')
    _patch_post(monkeypatch, _ok('A', finish='MAX_TOKENS'))
    assert providers.complete('p', 's', 500) is None


def test_long_max_tokens_response_is_kept(monkeypatch, keys):
    """A response that ran long is still useful — only a stub is discarded."""
    keys(gemini='g')
    _patch_post(monkeypatch, _ok('x' * 400, finish='MAX_TOKENS'))
    assert providers.complete('p', 's', 500) is not None


def test_thinking_budget_floor_is_applied(monkeypatch, keys):
    """A small caller budget must still leave room for thinking."""
    keys(gemini='g')
    cap = _patch_post(monkeypatch, _ok('text'))
    providers.complete('p', 's', 10)
    budget = cap['body']['generationConfig']['maxOutputTokens']
    assert budget >= providers.GEMINI_MIN_BUDGET


def test_safety_finish_returns_none(monkeypatch, keys):
    keys(gemini='g')
    _patch_post(monkeypatch, _ok('partial', finish='SAFETY'))
    assert providers.complete('p', 's', 500) is None


def test_prompt_block_returns_none(monkeypatch, keys):
    """A prompt-level block carries no candidates at all."""
    keys(gemini='g')
    _patch_post(monkeypatch,
                FakeResponse({'promptFeedback': {'blockReason': 'SAFETY'}}))
    assert providers.complete('p', 's', 500) is None


def test_empty_candidates_returns_none(monkeypatch, keys):
    keys(gemini='g')
    _patch_post(monkeypatch, FakeResponse({'candidates': []}))
    assert providers.complete('p', 's', 500) is None


@pytest.mark.parametrize('code', [400, 401, 403, 404, 429, 500, 503])
def test_http_errors_degrade_quietly(monkeypatch, keys, code):
    keys(gemini='g')
    _patch_post(monkeypatch, FakeResponse(
        {'error': {'message': 'boom', 'status': 'FAILED'}}, status_code=code))
    assert providers.complete('p', 's', 500) is None


def test_network_failure_degrades_quietly(monkeypatch, keys):
    keys(gemini='g')

    def boom(*a, **kw):
        raise providers.requests.exceptions.ConnectionError('no route')

    monkeypatch.setattr(providers.requests, 'post', boom)
    assert providers.complete('p', 's', 500) is None


def test_timeout_degrades_quietly(monkeypatch, keys):
    keys(gemini='g')

    def boom(*a, **kw):
        raise providers.requests.exceptions.Timeout('slow')

    monkeypatch.setattr(providers.requests, 'post', boom)
    assert providers.complete('p', 's', 500) is None


def test_non_json_body_degrades_quietly(monkeypatch, keys):
    keys(gemini='g')

    class Garbage:
        status_code = 200
        text = '<html>502</html>'

        def json(self):
            raise ValueError('not json')

    _patch_post(monkeypatch, Garbage())
    assert providers.complete('p', 's', 500) is None


# ─────────────────────────────────────────────
# THE CONTRACT THE UI DEPENDS ON
# ─────────────────────────────────────────────

def test_llm_module_never_raises_without_a_key(keys):
    """Every public entry point degrades to None, so panels stay renderable."""
    import pandas as pd

    from nlp import llm
    keys()

    assert llm.available() is False
    assert llm.explain_backtest('s', {'cagr': 0.1}) is None
    assert llm.explain_screen('s', 'long', 't', pd.DataFrame({'a': [1]}),
                              'u', __import__('datetime').date(2026, 1, 1)) is None
    assert llm.synthesize_news('AAPL', 'Apple', pd.DataFrame(
        {'published': ['2026-01-01'], 'title': ['x'], 'sentiment': [0.0]})) is None
    assert llm.compare_runs_narrative('s', '2026-01-01', '2026-02-01',
                                      ['A'], ['B'], [], []) is None
    assert llm.policy_impact(pd.DataFrame(
        {'published': ['2026-01-01'], 'doc_type': ['Rule'],
         'title': ['x']}), ['Energy']) is None


@pytest.mark.parametrize('shape', ['ui', 'engine', 'range'])
def test_explain_screen_accepts_every_frame_shape(keys, monkeypatch, shape):
    """The UI and the engine hand over differently-indexed frames.

    quant.engine returns tickers in an unnamed index; the Screen tab
    round-trips through a dcc.Store and sets `ticker` as both index and
    column. reset_index() on the latter raises "cannot insert ticker, already
    exists", which reached the UI as a bare ValueError.
    """
    import pandas as pd

    from nlp import llm
    keys(gemini='g')
    _patch_post(monkeypatch, _ok('commentary'))

    base = pd.DataFrame({'ticker': ['NVDA', 'AAPL'], 'composite': [1.2, 0.3],
                         'sector': ['IT', 'IT'], 'rank': [1, 2]})
    frame = {'ui': base.set_index('ticker', drop=False),
             'engine': base.set_index('ticker').rename_axis(None),
             'range': base}[shape]

    import datetime
    assert llm.explain_screen('S', 'long', 't', frame, 'u',
                              datetime.date(2026, 1, 1)) == 'commentary'


def test_empty_inputs_short_circuit_before_any_call(keys, monkeypatch):
    """With a key present, empty data must not produce a billable call."""
    import pandas as pd

    from nlp import llm
    keys(gemini='g')

    def explode(*a, **kw):
        raise AssertionError('should not have called the API')

    monkeypatch.setattr(providers.requests, 'post', explode)

    assert llm.synthesize_news('AAPL', 'Apple', pd.DataFrame()) is None
    assert llm.explain_screen('s', 'long', 't', pd.DataFrame(), 'u',
                              __import__('datetime').date(2026, 1, 1)) is None
    assert llm.explain_backtest('s', {}) is None
    assert llm.policy_impact(pd.DataFrame(), []) is None
    assert llm.compare_runs_narrative('s', 'a', 'b', [], [], [], []) is None
