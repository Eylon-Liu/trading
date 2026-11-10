"""Login wall for network / phone access."""

from __future__ import annotations

import re

import pytest
from flask import Flask

import config
from core import auth as AUTH

PASSWORD = 'correct horse battery'


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setattr(config, 'QUANT_USER', 'xl')
    monkeypatch.setattr(config, 'QUANT_PASSWORD_HASH', AUTH.hash_password(PASSWORD))
    monkeypatch.setattr(config, 'QUANT_SECRET_KEY', 'test-key')
    monkeypatch.setattr(config, 'QUANT_HOST', None)
    monkeypatch.setattr(config, 'QUANT_BEHIND_PROXY', False)
    AUTH._failures.clear()
    monkeypatch.setattr(AUTH.time, 'sleep', lambda s: None)
    app = Flask(__name__)

    @app.route('/')
    def home():
        return 'portfolio'

    @app.route('/_dash-update-component', methods=['POST'])
    def dash_cb():
        return '{}'

    @app.route('/reports/<path:name>')
    def report(name):
        return 'report'

    assert AUTH.install(app) is True
    return app.test_client()


def _csrf(client, path='/login'):
    page = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)


def _login(client, password=PASSWORD, user='xl', nxt='/'):
    return client.post('/login', data={'csrf': _csrf(client), 'username': user,
                                       'password': password, 'next': nxt})


# ── where it listens ─────────────────────────────────────────────

def test_without_a_password_it_stays_on_this_machine(monkeypatch):
    monkeypatch.setattr(config, 'QUANT_PASSWORD_HASH', None)
    monkeypatch.setattr(config, 'QUANT_HOST', None)
    assert AUTH.bind_host() == '127.0.0.1'
    assert AUTH.check_bind('127.0.0.1') is None
    assert 'Refusing' in AUTH.check_bind('0.0.0.0')


def test_a_password_does_not_open_it_to_the_wifi(secured):
    # The phone comes in through a tunnel on this computer, never the LAN.
    assert AUTH.bind_host() == '127.0.0.1'
    assert AUTH.check_bind('0.0.0.0') is None          # allowed only if asked


def test_loopback_detection():
    for a in ('127.0.0.1', '127.0.0.53', '::1', '::ffff:127.0.0.1', 'localhost'):
        assert AUTH.is_loopback(a), a
    for a in ('10.0.0.5', '192.168.1.20', '100.101.102.103', '', None, '::ffff:10.0.0.1'):
        assert not AUTH.is_loopback(a), a


def _open_app():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return 'portfolio'

    @app.route('/_dash-update-component', methods=['POST'])
    def dash_cb():
        return '{}'
    assert AUTH.install(app, enabled=False) is False
    return app.test_client()


def test_without_a_password_a_tunnel_gets_nothing(monkeypatch):
    # Starting `tailscale serve` (or a Cloudflare tunnel) before setting a
    # password must not publish an open terminal.
    monkeypatch.setattr(config, 'QUANT_SECRET_KEY', 'test-key')
    monkeypatch.setattr(config, 'QUANT_BEHIND_PROXY', False)
    c = _open_app()
    assert c.get('/').status_code == 200                               # this computer
    via_tunnel = {'X-Forwarded-For': '100.101.102.103', 'X-Forwarded-Proto': 'https'}
    r = c.get('/', headers=via_tunnel)
    assert r.status_code == 403 and 'set-password' in r.get_data(as_text=True)
    assert c.post('/_dash-update-component', json={}, headers=via_tunnel).status_code == 403
    assert c.get('/', headers={'Tailscale-User-Login': 'me@example.com'}).status_code == 403
    lan = c.get('/', environ_base={'REMOTE_ADDR': '192.168.1.20'})
    assert lan.status_code == 403
    assert c.get('/healthz', headers=via_tunnel).status_code == 200


def test_without_a_password_a_bare_tunnel_or_rebinding_page_gets_nothing(monkeypatch):
    # A tunnel that adds no forwarding header at all still passes its own
    # name as Host; so does a web page using DNS rebinding to reach 127.0.0.1.
    monkeypatch.setattr(config, 'QUANT_SECRET_KEY', 'test-key')
    monkeypatch.setattr(config, 'QUANT_BEHIND_PROXY', False)
    c = _open_app()
    for host in ('mymac.tail1234.ts.net', 'quant.example.com', 'evil.example:8050'):
        assert c.get('/', headers={'Host': host}).status_code == 403, host
        assert c.post('/_dash-update-component', json={},
                      headers={'Host': host}).status_code == 403, host
    for host in ('localhost:8050', '127.0.0.1:8050', '[::1]:8050', 'localhost'):
        assert c.get('/', headers={'Host': host}).status_code == 200, host


def test_host_header_parsing():
    assert AUTH.is_local_host_header('localhost:8050')
    assert AUTH.is_local_host_header('[::1]:8050')
    assert AUTH.is_local_host_header('qrt.localhost')
    assert not AUTH.is_local_host_header('localhost.evil.example')
    assert not AUTH.is_local_host_header('127.0.0.1.nip.io')
    assert not AUTH.is_local_host_header('')


# ── the guard ────────────────────────────────────────────────────

def test_pages_redirect_to_login_and_back(secured):
    r = secured.get('/reports/daily.html')
    assert r.status_code == 302
    assert r.headers['Location'].startswith('/login?next=/reports/daily.html')


def test_dash_callbacks_get_401_not_a_redirect(secured):
    assert secured.post('/_dash-update-component', json={}).status_code == 401


def test_health_check_is_open_and_says_nothing(secured):
    r = secured.get('/healthz')
    assert r.status_code == 200 and r.get_data(as_text=True) == 'ok'


def test_successful_login(secured):
    r = _login(secured, nxt='/reports/daily.html')
    assert r.status_code == 302 and r.headers['Location'] == '/reports/daily.html'
    assert secured.get('/').get_data(as_text=True) == 'portfolio'
    assert secured.post('/_dash-update-component', json={}).status_code == 200


def test_username_is_case_insensitive(secured):
    assert _login(secured, user='XL').status_code == 302


def test_wrong_password(secured):
    r = _login(secured, password='nope')
    assert r.status_code == 401 and 'Wrong username or password' in r.get_data(as_text=True)
    assert secured.get('/').status_code == 302


def test_missing_csrf_token_is_rejected(secured):
    secured.get('/login')
    r = secured.post('/login', data={'username': 'xl', 'password': PASSWORD})
    assert r.status_code == 400
    assert secured.get('/').status_code == 302


def test_lockout_after_repeated_failures(secured):
    for _ in range(AUTH.MAX_FAILURES):
        _login(secured, password='wrong')
    r = _login(secured)                                  # even the right one
    assert r.status_code == 429
    assert secured.get('/').status_code == 302


def test_next_cannot_leave_the_site(secured):
    for target in ('//evil.example', 'https://evil.example', '\\\\evil'):
        assert _login(secured, nxt=target).headers['Location'] == '/'
        secured.get('/logout')


def test_changing_the_password_signs_everyone_out(secured, monkeypatch):
    _login(secured)
    assert secured.get('/').status_code == 200
    monkeypatch.setattr(config, 'QUANT_PASSWORD_HASH', AUTH.hash_password('another one!!'))
    assert secured.get('/').status_code == 302


def test_logout(secured):
    _login(secured)
    secured.get('/logout')
    assert secured.get('/').status_code == 302


def test_cookie_flags(secured):
    r = _login(secured)
    cookie = r.headers.get('Set-Cookie', '')
    assert 'HttpOnly' in cookie and 'SameSite=Lax' in cookie
    assert 'Expires' in cookie                            # persists on the phone


def test_cookie_is_secure_when_the_tunnel_says_https(secured):
    r = secured.get('/login', headers={'X-Forwarded-For': '100.101.102.103',
                                       'X-Forwarded-Proto': 'https'})
    assert 'Secure' in r.headers.get('Set-Cookie', '')
    r = secured.get('/login')                              # plain http on this computer
    assert 'Secure' not in r.headers.get('Set-Cookie', '')


def test_forwarded_headers_from_the_lan_are_ignored(secured):
    # Someone not on this computer cannot claim https or a fresh address
    # per attempt to dodge the lockout.
    lan = {'REMOTE_ADDR': '192.168.1.20'}
    r = secured.get('/login', environ_base=lan, headers={'X-Forwarded-Proto': 'https'})
    assert 'Secure' not in r.headers.get('Set-Cookie', '')
    for i in range(AUTH.MAX_FAILURES):
        secured.post('/login', environ_base=lan,
                     headers={'X-Forwarded-For': f'10.9.9.{i}'},
                     data={'csrf': _csrf(secured), 'username': 'xl', 'password': 'x'})
    assert AUTH.locked_out('192.168.1.20')


def test_each_phone_behind_the_tunnel_has_its_own_limit(secured):
    for _ in range(AUTH.MAX_FAILURES):
        secured.post('/login', headers={'X-Forwarded-For': '100.64.0.9'},
                     data={'csrf': _csrf(secured), 'username': 'xl', 'password': 'x'})
    assert AUTH.locked_out('100.64.0.9')
    r = secured.post('/login', headers={'X-Forwarded-For': '100.64.0.2'},
                     data={'csrf': _csrf(secured), 'username': 'xl', 'password': PASSWORD})
    assert r.status_code == 302                            # your phone still gets in


def test_unsecured_install_does_not_guard():
    app = Flask(__name__)

    @app.route('/')
    def home():
        return 'portfolio'
    assert AUTH.install(app, enabled=False) is False
    assert app.test_client().get('/').status_code == 200


# ── storing the password ─────────────────────────────────────────

def test_password_is_stored_as_a_hash_dotenv_reads_back(tmp_path, monkeypatch):
    import cli
    from dotenv import dotenv_values
    env = tmp_path / '.env'
    env.write_text('SEC_USER_AGENT=Me me@example.com\nQUANT_USER=old\n')
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.stdin', __import__('io').StringIO(PASSWORD + '\n'))
    assert cli.main(['set-password', '--stdin', '--user', 'xl']) == 0
    text = env.read_text()
    assert PASSWORD not in text
    vals = dotenv_values(env)
    assert vals['SEC_USER_AGENT'] == 'Me me@example.com'
    assert vals['QUANT_USER'] == 'xl'
    from werkzeug.security import check_password_hash
    assert check_password_hash(vals['QUANT_PASSWORD_HASH'], PASSWORD)
    assert oct(env.stat().st_mode)[-3:] == '600'


def test_short_passwords_are_refused(tmp_path, monkeypatch):
    import cli
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr('sys.stdin', __import__('io').StringIO('short\n'))
    assert cli.main(['set-password', '--stdin']) == 1
    assert not (tmp_path / '.env').exists()
