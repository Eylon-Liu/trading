"""
Login for the web UI.

The terminal holds your positions, can send email from your account and can
start long jobs, so it must not answer anyone who can reach the port. The
rules:

  * The server listens on 127.0.0.1 — this computer only — whether or not a
    password is set. Nothing on the same Wi-Fi can even see a login page.
    A phone reaches it through a tunnel that runs on this computer
    (Tailscale Serve, Cloudflare Tunnel), never through an open port.
  * No password configured  → no login, and only direct requests from this
    computer are answered. A request that came through a tunnel is refused,
    so starting a tunnel before setting a password exposes nothing.
  * Password configured     → every page, every Dash callback and every
    report file requires a signed session cookie from /login.
  * Listening on a network interface (QUANT_HOST=0.0.0.0) without a
    password is refused at startup.

Set the password with `python cli.py set-password`; it is stored in .env as a
scrypt hash, never in plain text. Changing it signs out every session.

Details that matter on a phone: the session lasts 30 days, so you log in
once per device; failed logins are rate-limited per address (5 per 15
minutes) so a password cannot be guessed at speed; the form carries a CSRF
token; cookies are HttpOnly, SameSite=Lax, and Secure whenever the page was
loaded over HTTPS. Forwarding headers (the phone's address, https) are
trusted only from a proxy on this computer, or from any peer when
QUANT_BEHIND_PROXY=1 (Docker, where the host's proxy is the only way in).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import logging
import os
import secrets
import threading
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import config

log = logging.getLogger(__name__)

SESSION_DAYS = 30
MAX_FAILURES = 5
WINDOW_SECONDS = 15 * 60
OPEN_PATHS = frozenset({'/login', '/healthz', '/favicon.ico'})
LOCAL_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})
# Headers a reverse proxy or tunnel adds. Their presence means the request
# did not come straight from a browser on this computer.
PROXY_HEADERS = ('HTTP_X_FORWARDED_FOR', 'HTTP_FORWARDED', 'HTTP_X_REAL_IP',
                 'HTTP_CF_CONNECTING_IP', 'HTTP_TAILSCALE_USER_LOGIN')

_failures: dict[str, list[float]] = {}
_lock = threading.Lock()


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

def configured() -> bool:
    return bool(config.QUANT_PASSWORD_HASH)


def bind_host() -> str:
    """Explicit QUANT_HOST, else this computer only — with or without a password."""
    return config.QUANT_HOST or '127.0.0.1'


def is_local_host_header(host: str | None) -> bool:
    """Host header names this computer (localhost / 127.x / [::1], any port)."""
    if not host:
        return False
    host = host.strip().lower()
    if host.startswith('['):
        name = host[1:host.find(']')] if ']' in host else host[1:]
    else:
        name = host.rsplit(':', 1)[0] if host.count(':') == 1 else host
    return is_loopback(name) or name.endswith('.localhost')


def is_loopback(addr: str | None) -> bool:
    """127.0.0.0/8, ::1 (also IPv4-mapped) or the name localhost — parsed as
    an address, so a public name like 127.0.0.1.nip.io does not qualify."""
    import ipaddress
    if not addr:
        return False
    addr = addr.strip().split('%', 1)[0]
    if addr.lower() == 'localhost':
        return True
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def check_bind(host: str) -> str | None:
    """An error message when `host` would expose an unprotected server."""
    if host in LOCAL_HOSTS or configured():
        return None
    return (f'Refusing to listen on {host} without a password: anyone on the '
            f'network could read your portfolio and send email from your '
            f'account. Run `python cli.py set-password` first, or bind '
            f'127.0.0.1.')


def hash_password(password: str) -> str:
    from werkzeug.security import generate_password_hash
    return generate_password_hash(password, method='scrypt')


def verify(username: str, password: str) -> bool:
    from werkzeug.security import check_password_hash
    if not configured():
        return False
    user_ok = hmac.compare_digest((username or '').strip().lower(),
                                  config.QUANT_USER.lower())
    try:
        pass_ok = check_password_hash(config.QUANT_PASSWORD_HASH, password or '')
    except (ValueError, TypeError):
        pass_ok = False
    return user_ok and pass_ok


def _session_version() -> str:
    """Changes whenever the password does, which signs every session out."""
    return hashlib.sha256((config.QUANT_PASSWORD_HASH or '').encode()).hexdigest()[:16]


def secret_key() -> str:
    """QUANT_SECRET_KEY, else a key generated once and kept in db/.secret_key."""
    if config.QUANT_SECRET_KEY:
        return config.QUANT_SECRET_KEY
    path = Path(config.DB_DIR) / '.secret_key'
    try:
        if path.exists():
            return path.read_text().strip()
        key = secrets.token_hex(32)
        path.write_text(key)
        os.chmod(path, 0o600)
        return key
    except OSError as exc:
        log.warning('could not persist the session key (%s); sessions will '
                    'not survive a restart', exc)
        return secrets.token_hex(32)


# ─────────────────────────────────────────────
# RATE LIMITING
# ─────────────────────────────────────────────

def _recent_failures(ip: str, now: float) -> list[float]:
    return [t for t in _failures.get(ip, []) if now - t < WINDOW_SECONDS]


def locked_out(ip: str, now: float | None = None) -> bool:
    now = now or time.time()
    with _lock:
        return len(_recent_failures(ip, now)) >= MAX_FAILURES


def record_failure(ip: str, now: float | None = None) -> None:
    now = now or time.time()
    with _lock:
        _failures[ip] = _recent_failures(ip, now) + [now]


def clear_failures(ip: str) -> None:
    with _lock:
        _failures.pop(ip, None)


# ─────────────────────────────────────────────
# FLASK WIRING
# ─────────────────────────────────────────────

class TrustLocalProxy:
    """
    WSGI middleware: honour X-Forwarded-For/-Proto/-Host only when the peer
    is this computer (a tunnel such as Tailscale Serve connects from
    127.0.0.1), or from anyone when `trust_all`. Otherwise those headers are
    ignored, so a client cannot fake its address to dodge the login limit.
    Marks proxied requests in `qrt.via_proxy` before the headers are consumed.
    """

    def __init__(self, app, trust_all: bool = False):
        from werkzeug.middleware.proxy_fix import ProxyFix
        self.app = app
        self.fixed = ProxyFix(app, x_for=1, x_proto=1, x_host=1)
        self.trust_all = trust_all

    def __call__(self, environ, start_response):
        environ['qrt.via_proxy'] = any(environ.get(h) for h in PROXY_HEADERS)
        environ['qrt.raw_host'] = environ.get('HTTP_HOST', '')
        if self.trust_all or is_loopback(environ.get('REMOTE_ADDR')):
            return self.fixed(environ, start_response)
        return self.app(environ, start_response)


def _session_interface():
    """Flask sessions whose cookie is Secure whenever the request was HTTPS."""
    from flask import request
    from flask.sessions import SecureCookieSessionInterface

    class _Sessions(SecureCookieSessionInterface):
        def get_cookie_secure(self, app):
            return bool(app.config.get('SESSION_COOKIE_SECURE') or request.is_secure)

    return _Sessions()


def direct_local(request) -> bool:
    """
    A browser on this computer, not a request relayed by a tunnel.

    Three checks, any one of which a tunnel fails: the peer is loopback, no
    forwarding header is present, and the Host header is localhost/127.x —
    a tunnel passes its own public name (your-mac.ts.net, quant.example.com).
    The Host check also stops DNS-rebinding: a malicious web page that points
    its own name at 127.0.0.1 still sends that name as Host.
    """
    env = request.environ
    return is_loopback(env.get('REMOTE_ADDR')) and not env.get('qrt.via_proxy') \
        and is_local_host_header(env.get('qrt.raw_host', env.get('HTTP_HOST')))


NO_PASSWORD_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Set a password first</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0f1117;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}
div{width:min(92vw,420px);background:#1a1d2e;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:24px}
h1{font-size:1.15rem;color:#ffd166;margin:0 0 10px}p{color:#b8bfd0;font-size:.9rem;line-height:1.5}
code{background:#0f1117;padding:2px 6px;border-radius:5px;color:#38ef7d}</style></head><body><div>
<h1>🔒 This terminal has no password yet</h1>
<p>Without one it only answers on the computer it runs on, so it will not
open from another device. On that computer run</p>
<p><code>python3 cli.py set-password</code></p>
<p>then restart the web UI and reload this page.</p></div></body></html>"""

def _safe_next(target: str | None) -> str:
    """Only same-site relative paths; anything else goes home."""
    if target and target.startswith('/') and not target.startswith('//') \
            and '\\' not in target:
        return target
    return '/'


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#11998e"><title>Sign in — Quant Research Terminal</title>
<style>
body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0f1117;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}}
form{{width:min(92vw,360px);background:#1a1d2e;border:1px solid rgba(255,255,255,.08);
border-radius:14px;padding:26px 22px}}
h1{{font-size:1.25rem;color:#38ef7d;margin:0 0 4px}} p{{color:#8b93a7;font-size:.85rem;margin:0 0 18px}}
label{{display:block;font-size:.72rem;color:#8b93a7;text-transform:uppercase;letter-spacing:.4px;margin:12px 0 6px}}
input{{width:100%;box-sizing:border-box;padding:12px;border-radius:9px;border:1px solid #2a2e42;
background:#0f1117;color:#e0e0e0;font-size:16px}}
button{{width:100%;margin-top:20px;padding:13px;border:0;border-radius:9px;font-weight:700;font-size:1rem;
background:linear-gradient(135deg,#11998e,#38ef7d);color:#000}}
.err{{background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.4);color:#ff9b9b;
padding:9px 12px;border-radius:8px;font-size:.84rem;margin-bottom:6px}}
</style></head><body>
<form method="post" action="/login" autocomplete="on">
<h1>📊 Quant Research Terminal</h1><p>Sign in to continue.</p>{error}
<input type="hidden" name="csrf" value="{csrf}"><input type="hidden" name="next" value="{next}">
<label for="u">Username</label><input id="u" name="username" autocomplete="username" autocapitalize="none" required>
<label for="p">Password</label><input id="p" name="password" type="password" autocomplete="current-password" required>
<button type="submit">Sign in</button></form></body></html>"""


def install(server, enabled: bool | None = None) -> bool:
    """
    Wire sessions, /login, /logout and /healthz into the Flask server, and
    guard every other route when `enabled` (default: a password is set).
    Returns whether the guard is active.
    """
    from flask import redirect, request, session

    enabled = configured() if enabled is None else enabled
    server.secret_key = secret_key()
    server.config.update(
        SESSION_COOKIE_NAME='qrt_session',
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_SECURE=False,          # per request: see _session_interface
        PERMANENT_SESSION_LIFETIME=timedelta(days=SESSION_DAYS),
    )
    server.session_interface = _session_interface()
    server.wsgi_app = TrustLocalProxy(server.wsgi_app,
                                      trust_all=config.QUANT_BEHIND_PROXY)

    @server.route('/healthz')
    def _healthz():
        return 'ok', 200, {'Content-Type': 'text/plain', 'Cache-Control': 'no-store'}

    def _page(error: str = '', status: int = 200, nxt: str = '/'):
        token = session.get('csrf') or secrets.token_urlsafe(24)
        session['csrf'] = token
        body = LOGIN_PAGE.format(
            error=f'<div class="err">{html.escape(error)}</div>' if error else '',
            csrf=token, next=html.escape(_safe_next(nxt), quote=True))
        return body, status, {'Cache-Control': 'no-store',
                              'X-Frame-Options': 'DENY'}

    @server.route('/login', methods=['GET', 'POST'])
    def _login():
        if not enabled:
            return redirect('/')
        ip = request.remote_addr or '?'
        if request.method == 'GET':
            return _page(nxt=request.args.get('next', '/'))
        nxt = request.form.get('next', '/')
        sent = request.form.get('csrf', '')
        if not hmac.compare_digest(sent, session.get('csrf') or '-'):
            return _page('Your session expired — please try again.', 400, nxt)
        if locked_out(ip):
            log.warning('login locked out for %s', ip)
            return _page('Too many failed attempts. Try again in 15 minutes.', 429, nxt)
        if verify(request.form.get('username', ''), request.form.get('password', '')):
            clear_failures(ip)
            session.clear()
            session.permanent = True
            session['user'] = config.QUANT_USER
            session['v'] = _session_version()
            log.info('login from %s', ip)
            return redirect(_safe_next(nxt))
        record_failure(ip)
        log.warning('failed login from %s', ip)
        time.sleep(0.5)                           # blunts rapid guessing further
        return _page('Wrong username or password.', 401, nxt)

    @server.route('/logout')
    def _logout():
        session.clear()
        return redirect('/login' if enabled else '/')

    if not enabled:
        @server.before_request
        def _local_only():
            # No password: answer this computer's own browser and nothing
            # else — not a tunnel, not the LAN — whatever the bind address.
            if request.path == '/healthz' or direct_local(request):
                return None
            log.warning('refused %s from %s: no password is set',
                        request.path, request.remote_addr)
            if request.path.startswith('/_dash-') or request.path.startswith('/_reload'):
                return 'Set a password first: python cli.py set-password', 403
            return NO_PASSWORD_PAGE, 403, {'Cache-Control': 'no-store'}

    @server.after_request
    def _headers(resp):
        resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
        resp.headers.setdefault('Referrer-Policy', 'same-origin')
        return resp

    if enabled:
        @server.before_request
        def _guard():
            if request.path in OPEN_PATHS:
                return None
            if session.get('user') == config.QUANT_USER and \
                    session.get('v') == _session_version():
                return None
            # Dash's own requests (callbacks, layout, bundles) get a plain 401;
            # a full page load is sent to the login form and back afterwards.
            if request.path.startswith('/_dash-') or request.path.startswith('/_reload'):
                return 'Sign in again', 401
            return redirect('/login?next=' + quote(request.full_path.rstrip('?')))

    return enabled
