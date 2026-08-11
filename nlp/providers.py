"""
LLM transport. One function — `complete()` — over two backends.

nlp/llm.py owns the prompts; this module owns the wire. Keeping them apart
means adding a provider never touches a prompt, and tuning a prompt never
risks the transport.

Both backends return `str | None` and never raise: an unreachable API, a bad
key, a safety refusal and a truncated response all degrade to None, and the
caller renders the local Loughran-McDonald output instead. A missing key must
never produce a blank panel or a stack trace in the UI.

Provider selection is `LLM_PROVIDER` in .env — auto | gemini | anthropic | off.
Under 'auto' Gemini wins when its key is present, because it is the cheaper of
the two for this workload (a few short calls per screen).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import requests

import config

log = logging.getLogger(__name__)

GEMINI_ENDPOINT = ('https://generativelanguage.googleapis.com/v1beta/'
                   'models/{model}:generateContent')
TIMEOUT = 180.0

# Gemini counts *thinking* tokens against maxOutputTokens, which Claude does
# not. Measured on gemini-3.5-flash: a one-sentence answer burned 323-459
# thinking tokens before emitting 27 visible ones, and a 40-token cap returned
# a single character. So every budget is floored well above the prose we want,
# or the model thinks itself out of an answer.
GEMINI_MIN_BUDGET = 4000
GEMINI_THINKING_RESERVE = 3000


# ─────────────────────────────────────────────
# PROVIDER RESOLUTION
# ─────────────────────────────────────────────

def resolve() -> str:
    """Which backend is actually usable right now: 'gemini' | 'anthropic' | ''."""
    choice = (config.LLM_PROVIDER or 'auto').strip().lower()

    if choice == 'off':
        return ''
    if choice == 'gemini':
        return 'gemini' if config.GEMINI_API_KEY else ''
    if choice == 'anthropic':
        return 'anthropic' if config.ANTHROPIC_API_KEY else ''

    # auto
    if config.GEMINI_API_KEY:
        return 'gemini'
    if config.ANTHROPIC_API_KEY:
        return 'anthropic'
    return ''


def model_name(provider: str | None = None) -> str:
    provider = provider or resolve()
    if provider == 'gemini':
        return config.GEMINI_MODEL
    if provider == 'anthropic':
        return config.LLM_MODEL
    return ''


def status() -> dict:
    """Human-readable availability for the UI."""
    choice = (config.LLM_PROVIDER or 'auto').strip().lower()
    provider = resolve()

    if choice == 'off':
        return {'enabled': False, 'provider': None,
                'reason': 'LLM_PROVIDER=off — local analysis only.'}

    if not provider:
        if choice in ('gemini', 'anthropic'):
            key = 'GEMINI_API_KEY' if choice == 'gemini' else 'ANTHROPIC_API_KEY'
            return {'enabled': False, 'provider': None,
                    'reason': f'LLM_PROVIDER={choice} but no {key} in .env.'}
        return {'enabled': False, 'provider': None,
                'reason': 'No API key in .env — using the local '
                          'Loughran-McDonald engine.'}

    if provider == 'anthropic' and _anthropic_client() is None:
        return {'enabled': False, 'provider': None,
                'reason': 'ANTHROPIC_API_KEY is set but the anthropic SDK is '
                          'missing. Run: pip install anthropic'}

    label = 'Gemini' if provider == 'gemini' else 'Claude'
    model = model_name(provider)
    return {'enabled': True, 'provider': provider, 'model': model,
            'reason': f'{label} analysis enabled ({model}).'}


def available() -> bool:
    return status()['enabled']


# ─────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────

def complete(user_prompt: str, system: str, max_tokens: int) -> str | None:
    """Run one completion on whichever backend is configured."""
    provider = resolve()
    if provider == 'gemini':
        return _complete_gemini(user_prompt, system, max_tokens)
    if provider == 'anthropic':
        return _complete_anthropic(user_prompt, system, max_tokens)
    return None


# ─────────────────────────────────────────────
# GEMINI (REST)
# ─────────────────────────────────────────────

def _complete_gemini(user_prompt: str, system: str,
                     max_tokens: int) -> str | None:
    """
    One Gemini generateContent call.

    The key travels in the `x-goog-api-key` header rather than the `?key=`
    query parameter the docs lead with — query strings end up in proxy logs
    and browser history, and a credential does not belong there.
    """
    budget = max(max_tokens + GEMINI_THINKING_RESERVE, GEMINI_MIN_BUDGET)
    body = {
        'system_instruction': {'parts': [{'text': system}]},
        'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
        'generationConfig': {
            'temperature': 0.2,          # commentary on fixed numbers; low variance
            'maxOutputTokens': budget,
            'thinkingConfig': {'thinkingLevel': config.GEMINI_THINKING_LEVEL},
        },
    }

    try:
        resp = requests.post(
            GEMINI_ENDPOINT.format(model=config.GEMINI_MODEL),
            headers={'x-goog-api-key': config.GEMINI_API_KEY,
                     'Content-Type': 'application/json'},
            json=body, timeout=TIMEOUT)
    except requests.exceptions.Timeout:
        log.warning('Gemini timed out after %ss — falling back to local analysis',
                    TIMEOUT)
        return None
    except requests.exceptions.RequestException as exc:
        log.warning('Could not reach Gemini (%s) — falling back to local analysis',
                    exc.__class__.__name__)
        return None

    if resp.status_code != 200:
        _log_gemini_error(resp)
        return None

    try:
        data = resp.json()
    except ValueError:
        log.warning('Gemini returned non-JSON (HTTP %s)', resp.status_code)
        return None

    # A prompt-level safety block carries no candidates at all, so this has to
    # be checked before indexing into them.
    blocked = (data.get('promptFeedback') or {}).get('blockReason')
    if blocked:
        log.warning('Gemini blocked the prompt (reason=%s)', blocked)
        return None

    candidates = data.get('candidates') or []
    if not candidates:
        log.warning('Gemini returned no candidates')
        return None

    cand = candidates[0]
    finish = cand.get('finishReason')
    parts = (cand.get('content') or {}).get('parts') or []

    # Reasoning parts are flagged `thought` and are not for the reader.
    text = ''.join(p.get('text', '') for p in parts if not p.get('thought')).strip()

    if finish in ('SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'RECITATION'):
        log.warning('Gemini stopped early (finishReason=%s)', finish)
        return None

    if finish == 'MAX_TOKENS' and len(text) < 200:
        # Thinking consumed the budget before the answer started. Returning the
        # fragment would put a truncated half-sentence in front of the user, so
        # fall back to the local engine instead.
        thoughts = (data.get('usageMetadata') or {}).get('thoughtsTokenCount')
        log.warning('Gemini hit maxOutputTokens with %d chars of output '
                    '(thinking used %s tokens) — raise GEMINI_MAX_TOKENS',
                    len(text), thoughts)
        return None

    return text or None


def _log_gemini_error(resp) -> None:
    """Map an HTTP status onto an actionable log line."""
    try:
        detail = (resp.json().get('error') or {}).get('message', '')[:200]
    except ValueError:
        detail = resp.text[:200]

    code = resp.status_code
    if code in (401, 403):
        log.error('GEMINI_API_KEY rejected (HTTP %s) — falling back to local '
                  'analysis. %s', code, detail)
    elif code == 404:
        log.error('Unknown model %r — check GEMINI_MODEL in .env. %s',
                  config.GEMINI_MODEL, detail)
    elif code == 429:
        log.warning('Gemini rate limit / quota hit — falling back to local '
                    'analysis. %s', detail)
    elif 500 <= code < 600:
        log.warning('Gemini server error %s — falling back to local analysis', code)
    else:
        log.warning('Gemini API error %s: %s', code, detail)


# ─────────────────────────────────────────────
# ANTHROPIC (SDK)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _anthropic_client():
    if not config.ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic
    except ImportError:
        log.info('anthropic SDK not installed; run: pip install anthropic')
        return None
    try:
        return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=120.0)
    except Exception as exc:                       # noqa: BLE001
        log.warning('could not build Anthropic client: %s', exc)
        return None


def _complete_anthropic(user_prompt: str, system: str,
                        max_tokens: int) -> str | None:
    """
    One Claude call.

    Server-side refusal fallback is enabled: if the model's safety classifiers
    decline, the API transparently re-runs on the recommended fallback model
    rather than handing back an empty response.
    """
    client = _anthropic_client()
    if client is None:
        return None

    try:
        response = client.beta.messages.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            thinking={'type': 'adaptive'},
            output_config={'effort': 'medium'},
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
            system=[{
                'type': 'text', 'text': system,
                # Identical across every call in a run, so caching it makes the
                # per-ticker calls markedly cheaper.
                'cache_control': {'type': 'ephemeral'},
            }],
            messages=[{'role': 'user', 'content': user_prompt}],
        )
    except Exception as exc:                       # noqa: BLE001
        return _log_anthropic_error(exc)

    # Check the stop reason before touching content: on a refusal the content
    # list is empty or partial, and indexing it blindly raises.
    if response.stop_reason == 'refusal':
        category = getattr(getattr(response, 'stop_details', None), 'category', None)
        log.warning('Claude declined this request (category=%s)', category)
        return None

    text = ''.join(b.text for b in response.content if b.type == 'text')
    return text.strip() or None


def _log_anthropic_error(exc: Exception) -> None:
    """Log an SDK error at the right level, most specific first."""
    try:
        import anthropic
    except ImportError:
        log.warning('LLM call failed: %s', exc)
        return None

    if isinstance(exc, anthropic.AuthenticationError):
        log.error('ANTHROPIC_API_KEY is invalid — falling back to local analysis')
    elif isinstance(exc, anthropic.RateLimitError):
        log.warning('Claude rate limit hit — falling back to local analysis')
    elif isinstance(exc, anthropic.NotFoundError):
        log.error('Unknown model %r — check LLM_MODEL in .env', config.LLM_MODEL)
    elif isinstance(exc, anthropic.APIStatusError):
        log.warning('Claude API error %s: %s', exc.status_code, exc.message)
    elif isinstance(exc, anthropic.APIConnectionError):
        log.warning('Could not reach the Claude API — falling back to local analysis')
    else:
        log.warning('LLM call failed: %s', exc)
    return None
