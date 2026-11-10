"""
HTTP layer — per-host rate limiting, disk cache, and retry with backoff.

This exists because the original app fired 50 tickers x 3 endpoints on every
button click with no cache. During development probing, a mere handful of
uncached yfinance calls was enough to trigger `YFRateLimitError`. Every
outbound request in the project goes through `fetch`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TOKEN-BUCKET RATE LIMITER (per host, thread-safe)
# ─────────────────────────────────────────────

class _TokenBucket:
    def __init__(self, rate: float, burst: float | None = None):
        self.rate = rate
        self.capacity = burst if burst is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self.rate
            time.sleep(min(deficit, 1.0))


_buckets: dict[str, _TokenBucket] = {}
_buckets_lock = threading.Lock()


def _bucket_for(host: str) -> _TokenBucket:
    with _buckets_lock:
        if host not in _buckets:
            rate = config.RATE_LIMITS.get(host, config.RATE_LIMITS['_default'])
            _buckets[host] = _TokenBucket(rate)
        return _buckets[host]


# ─────────────────────────────────────────────
# DISK CACHE
# ─────────────────────────────────────────────

def _cache_path(url: str, params: dict | None, category: str) -> Path:
    key = hashlib.sha256(
        f'{url}|{json.dumps(params or {}, sort_keys=True)}'.encode()
    ).hexdigest()[:32]
    d = config.CACHE_DIR / category
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{key}.cache'


# Responses are stored gzip-compressed. SEC's companyfacts JSON is 4-9 MB per
# company and compresses ~15x; uncompressed, that one category was 1.9 GB for
# the S&P 500 — more than twice the database built from it.
_GZIP_MAGIC = b'\x1f\x8b'


def _decode(raw: bytes) -> bytes:
    """Stored bytes → payload; files written before compression pass through."""
    if raw[:2] == _GZIP_MAGIC:
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError):
            return raw
    return raw


def _cache_read(path: Path, ttl: int) -> bytes | None:
    if not path.exists():
        return None
    if ttl >= 0 and (time.time() - path.stat().st_mtime) > ttl:
        return None
    try:
        return _decode(path.read_bytes())
    except OSError:
        return None


def _cache_write(path: Path, payload: bytes) -> None:
    try:
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(gzip.compress(payload, compresslevel=4))
        tmp.replace(path)
    except OSError as exc:
        log.debug('cache write failed for %s: %s', path, exc)


def sweep_cache(max_age_days: int = 30) -> dict[str, int]:
    """
    Keep the cache bounded: delete entries not refreshed for `max_age_days`
    (every TTL is far shorter, so these are only ever a last-resort fallback
    for a source that has been down that long) and compress any entry
    written before compression existed. Returns counts and bytes freed.
    """
    out = {'removed': 0, 'compressed': 0, 'bytes_freed': 0}
    root = config.CACHE_DIR
    if not root.exists():
        return out
    cutoff = time.time() - max_age_days * 86400
    for path in list(root.rglob('*.cache')) + list(root.rglob('*.tmp')):
        try:
            st = path.stat()
            # A .tmp is a write in progress — possibly another process's —
            # unless it has been sitting there for an hour.
            if path.suffix == '.tmp' and time.time() - st.st_mtime < 3600:
                continue
            if st.st_mtime < cutoff or path.suffix == '.tmp':
                path.unlink()
                out['removed'] += 1
                out['bytes_freed'] += st.st_size
                continue
            with open(path, 'rb') as fh:
                head = fh.read(2)
            if head == _GZIP_MAGIC or st.st_size < 4096:
                continue
            packed = gzip.compress(path.read_bytes(), compresslevel=4)
            if len(packed) < st.st_size:
                tmp = path.with_suffix('.tmp')
                tmp.write_bytes(packed)
                os.utime(tmp, (st.st_atime, st.st_mtime))    # age is unchanged
                tmp.replace(path)
                out['compressed'] += 1
                out['bytes_freed'] += st.st_size - len(packed)
        except OSError as exc:
            log.debug('cache sweep skipped %s: %s', path, exc)
    return out


# ─────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────

_session_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_session_local, 'session', None)
    if s is None:
        s = requests.Session()
        _session_local.session = s
    return s


def _headers_for(host: str, extra: dict | None) -> dict:
    # The SEC requires a descriptive UA with contact details and will 403
    # a browser-style one; everyone else wants the opposite.
    if host.endswith('sec.gov'):
        h = {'User-Agent': config.SEC_USER_AGENT,
             'Accept-Encoding': 'gzip, deflate'}
    else:
        h = {'User-Agent': config.BROWSER_USER_AGENT,
             'Accept-Encoding': 'gzip, deflate'}
    if extra:
        h.update(extra)
    return h


class RateLimited(Exception):
    """Raised when a host keeps refusing after the retry budget is spent."""


def fetch(
    url: str,
    params: dict | None = None,
    category: str = '_default',
    headers: dict | None = None,
    use_cache: bool = True,
    ttl: int | None = None,
    timeout: int | None = None,
    retries: int | None = None,
) -> bytes | None:
    """
    Fetch a URL with rate limiting, caching, and retry.

    Returns raw bytes, or None when the resource is genuinely unavailable
    (404, or the retry budget was exhausted). Never raises for transport
    problems — callers treat None as missing data.
    """
    host = urlparse(url).netloc
    ttl = config.CACHE_TTL_SECONDS.get(category, config.CACHE_TTL_SECONDS['_default']) \
        if ttl is None else ttl
    path = _cache_path(url, params, category)

    if use_cache:
        cached = _cache_read(path, ttl)
        if cached is not None:
            return cached

    bucket = _bucket_for(host)
    last_error: str | None = None

    # `retries` lets a caller that polls many URLs of one fragile host cap the
    # cost of a host that is down: 4 attempts x 30s each, across 15 calendar
    # days, would stall a morning run for half an hour.
    for attempt in range(config.HTTP_MAX_RETRIES if retries is None else max(1, retries)):
        bucket.acquire()
        try:
            resp = _session().get(
                url,
                params=params,
                headers=_headers_for(host, headers),
                timeout=timeout or config.HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            _sleep_backoff(attempt)
            continue

        if resp.status_code == 200:
            payload = resp.content
            if use_cache:
                _cache_write(path, payload)
            return payload

        if resp.status_code in (404, 403) and not host.endswith('sec.gov'):
            log.debug('%s -> %s (treating as missing)', url, resp.status_code)
            return None

        if resp.status_code in (429, 500, 502, 503, 504):
            retry_after = resp.headers.get('Retry-After')
            if retry_after:
                try:
                    time.sleep(min(float(retry_after), 60))
                except ValueError:
                    _sleep_backoff(attempt)
            else:
                _sleep_backoff(attempt)
            last_error = f'HTTP {resp.status_code}'
            continue

        last_error = f'HTTP {resp.status_code}'
        break

    log.warning('giving up on %s (%s)', url, last_error)

    # Prefer stale cache over nothing — a day-old price series still beats a
    # blank screen when Yahoo is throttling.
    if use_cache:
        stale = _cache_read(path, ttl=-1)
        if stale is not None:
            log.info('serving stale cache for %s', url)
            return stale

    return None


def _sleep_backoff(attempt: int) -> None:
    delay = config.HTTP_BACKOFF_BASE ** (attempt + 1)
    time.sleep(delay + random.uniform(0, 0.4 * delay))


def fetch_json(url: str, params: dict | None = None, category: str = '_default',
               **kw: Any) -> Any | None:
    """Fetch and parse JSON. Returns None on failure or malformed payload."""
    raw = fetch(url, params=params, category=category, **kw)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning('bad JSON from %s: %s', url, exc)
        return None


def fetch_text(url: str, params: dict | None = None, category: str = '_default',
               **kw: Any) -> str | None:
    """Fetch and decode text."""
    raw = fetch(url, params=params, category=category, **kw)
    if raw is None:
        return None
    return raw.decode('utf-8', errors='replace')


def clear_cache(category: str | None = None) -> int:
    """Delete cached responses. Returns the number of files removed."""
    root = config.CACHE_DIR / category if category else config.CACHE_DIR
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob('*.cache'):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def cache_stats() -> dict[str, dict[str, float]]:
    """Per-category cache size and file count, for the Data tab."""
    out: dict[str, dict[str, float]] = {}
    if not config.CACHE_DIR.exists():
        return out
    for d in config.CACHE_DIR.iterdir():
        if not d.is_dir():
            continue
        files = list(d.glob('*.cache'))
        out[d.name] = {
            'files': len(files),
            'mb': round(sum(f.stat().st_size for f in files) / 1e6, 2),
        }
    return out
