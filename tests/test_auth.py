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


def test_with_a_password_it_listens_on_the_network(secured):
    assert AUTH.bind_host() == '0.0.0.0'
    assert AUTH.check_bind('0.0.0.0') is None


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
