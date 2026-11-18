"""
Which Form 4 rows are insider *decisions*.

Most of what Form 4 reports is compensation mechanics. Grants (A), exercises
(M) and tax withholding (F) are excluded by code; qualified ESPP purchases are
exempt from reporting. What is left under P and S still mixes decisions with
plan trades, exercise sales, fund trades and habitual sellers. These tests pin
each filter, the parser fixes they depend on, and the re-parse that carries a
parser fix into history already stored.
"""

from __future__ import annotations

import importlib
from datetime import date

import pandas as pd
import pytest

import config
from data import sec
from quant import factors as FA


# ─────────────────────────────────────────────
# SYNTHETIC FORM 4
# ─────────────────────────────────────────────

def _txn(code, shares, price, ad='A', day='2025-03-03', post=10_000, notes=()):
    refs = ''.join(f'<footnoteId id="{n}"/>' for n in notes)
    return f"""
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>{day}</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType>
        <transactionCode>{code}</transactionCode>{refs}</transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value>{refs}</transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{ad}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction>
        <value>{post}</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
    </nonDerivativeTransaction>"""


def _form4(txns, issuer_cik='0000320193', aff='0', director='0', officer='1',
           ten_pct='0', title='Chief Executive Officer', footnotes=None, remarks=''):
    fns = ''.join(f'<footnote id="{k}">{v}</footnote>' for k, v in (footnotes or {}).items())
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <schemaVersion>X0508</schemaVersion>
  <documentType>4</documentType>
  <periodOfReport>2025-03-03</periodOfReport>
  <aff10b5One>{aff}</aff10b5One>
  <issuer><issuerCik>{issuer_cik}</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001</rptOwnerCik><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>{director}</isDirector><isOfficer>{officer}</isOfficer>
      <isTenPercentOwner>{ten_pct}</isTenPercentOwner>
      <officerTitle>{title}</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>{''.join(txns)}</nonDerivativeTable>
  <footnotes>{fns}</footnotes>
  <remarks>{remarks}</remarks>
</ownershipDocument>""".encode()


def _parse(xml, cik='320193'):
    return sec._parse_form4(xml, 'AAPL', cik, '0001-25-000001', date(2025, 3, 5))


# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────

def test_lots_are_summed_not_overwritten():
    """Three lots of one sale share the table key; only one used to survive."""
    rows = _parse(_form4([_txn('S', 100, 10.0, 'D', post=900),
                          _txn('S', 200, 11.0, 'D', post=700),
                          _txn('S', 300, 12.0, 'D', post=400)]))
    assert len(rows) == 1
    r = rows[0]
    assert r['shares'] == -600
    assert r['value'] == pytest.approx(100 * 10 + 200 * 11 + 300 * 12)
    assert r['price'] == pytest.approx(r['value'] / 600)
    assert r['post_txn_shares'] == 400                 # after the last lot
    assert r['parse_version'] == sec.INSIDER_PARSE_VERSION


@pytest.mark.parametrize('aff,flagged', [('1', True), ('true', True),
                                         ('0', False), ('false', False)])
def test_10b5_1_checkbox_in_either_spelling(aff, flagged):
    rows = _parse(_form4([_txn('S', 100, 10.0, 'D')], aff=aff))
    assert ('10b5-1' in (rows[0]['flags'] or '')) is flagged


def test_10b5_1_stated_only_in_text_is_caught():
    by_remarks = _parse(_form4([_txn('S', 100, 10.0, 'D')],
                               remarks='Sales effected under a Rule 10b5-1 trading plan.'))
    by_note = _parse(_form4([_txn('S', 100, 10.0, 'D', notes=['F1'])],
                            footnotes={'F1': 'Effected pursuant to a Rule 10b5 - 1 plan '
                                             'adopted on May 1, 2024.'}))
    assert '10b5-1' in by_remarks[0]['flags'] and '10b5-1' in by_note[0]['flags']


@pytest.mark.parametrize('note', [
    'Shares acquired under the Employee Stock Purchase Plan.',
    'Purchased through the ESPP.',
    'Acquired through the dividend reinvestment feature of the plan.',
    'Shares purchased in the 401(k) plan.',
    'Sale to cover tax withholding obligations upon vesting.',
    'Shares sold to cover taxes on RSU vesting (sell-to-cover).',
])
def test_plan_mechanics_in_footnotes_are_flagged(note):
    rows = _parse(_form4([_txn('P', 50, 20.0, notes=['F1'])], footnotes={'F1': note}))
    assert 'plan' in rows[0]['flags']


def test_ordinary_footnote_is_not_a_flag():
    rows = _parse(_form4([_txn('P', 50, 20.0, notes=['F1'])],
                         footnotes={'F1': 'The price reported is a weighted average; '
                                          'shares were purchased in multiple trades.'}))
    assert rows[0]['flags'] is None


def test_sale_on_an_exercise_day_is_an_exercise_sale():
    rows = _parse(_form4([_txn('M', 1000, 5.0, 'A', day='2025-03-03'),
                          _txn('S', 1000, 50.0, 'D', day='2025-03-03'),
                          _txn('S', 500, 51.0, 'D', day='2025-03-04')]))
    sale = {r['txn_date']: r for r in rows if r['txn_code'] == 'S'}
    assert 'exercise' in sale[date(2025, 3, 3)]['flags']
    assert sale[date(2025, 3, 4)]['flags'] is None


def test_filings_about_another_issuer_are_dropped():
    """A company's submissions include Form 4s it filed as someone's 10% owner."""
    assert _parse(_form4([_txn('P', 100, 10.0)], issuer_cik='0000034088')) == []
    assert len(_parse(_form4([_txn('P', 100, 10.0)], issuer_cik='0000320193'))) == 1


@pytest.mark.parametrize('director,officer', [('true', 'false'), ('1', '0')])
def test_role_flags_in_either_spelling(director, officer):
    rows = _parse(_form4([_txn('P', 100, 10.0)], director=director,
                         officer=officer, title=''))
    assert rows[0]['role'] == 'Director'


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

AS_OF = date(2025, 6, 30)


def _row(insider, day, shares, value, code=None, role='Officer, CEO', flags=None,
         ticker='X', filed=None):
    code = code or ('P' if shares > 0 else 'S')
    d = pd.Timestamp(day)
    return {'ticker': ticker, 'insider': insider, 'txn_date': d,
            'filed': pd.Timestamp(filed) if filed else d + pd.Timedelta(days=2),
            'txn_code': code, 'shares': float(shares), 'value': float(value),
            'price': value / abs(shares), 'role': role, 'flags': flags,
            'post_txn_shares': 100_000.0}


def test_flagged_and_fund_trades_do_not_count():
    ins = pd.DataFrame([
        _row('A', '2025-05-01', 1000, 50_000),                        # decision
        _row('B', '2025-05-02', -5000, 250_000, flags='10b5-1'),      # plan sale
        _row('C', '2025-05-03', -5000, 250_000, flags='exercise'),    # exercise sale
        _row('D', '2025-05-04', 2000, 100_000, flags='plan'),         # ESPP / DRIP
        _row('E', '2025-05-05', -9000, 900_000, role='10% Owner'),    # a fund
    ])
    sig = FA.insider_signal(ins, AS_OF)
    assert sig.loc['X', 'INSIDER_NET_BUY'] == pytest.approx(1.0)
    assert sig.loc['X', 'INSIDER_CLUSTER'] == 1


def test_a_director_who_also_holds_10pct_still_counts():
    ins = pd.DataFrame([_row('F', '2025-05-01', 1000, 50_000,
                             role='Director, 10% Owner')])
    assert FA.insider_signal(ins, AS_OF).loc['X', 'INSIDER_CLUSTER'] == 1


def test_routine_traders_are_removed():
    """Selling every May for three years is a habit, not information."""
    hist = [_row('R', f'{y}-05-10', -1000, 50_000) for y in (2022, 2023, 2024)]
    now = [_row('R', '2025-05-10', -1000, 50_000),
           _row('O', '2025-05-12', 2000, 60_000)]
    sig = FA.insider_signal(pd.DataFrame(hist + now), AS_OF)
    assert sig.loc['X', 'INSIDER_NET_BUY'] == pytest.approx(1.0)   # only O's buy


def test_two_years_of_habit_is_not_yet_routine():
    hist = [_row('R', f'{y}-05-10', -1000, 50_000) for y in (2023, 2024)]
    now = [_row('R', '2025-05-10', -1000, 50_000)]
    sig = FA.insider_signal(pd.DataFrame(hist + now), AS_OF)
    assert sig.loc['X', 'INSIDER_NET_BUY'] < 0              # the sale still counts


def test_history_classifies_but_is_not_itself_scored():
    old = [_row('Z', '2024-06-01', 50_000, 5_000_000)]       # outside 180 days
    now = [_row('A', '2025-05-01', -1000, 50_000)]
    sig = FA.insider_signal(pd.DataFrame(old + now), AS_OF)
    assert sig.loc['X', 'INSIDER_NET_BUY'] < 0              # the old $5M buy is not in it
    assert sig.loc['X', 'INSIDER_CLUSTER'] == 0


def test_token_purchases_do_not_make_a_cluster():
    ins = pd.DataFrame([_row('A', '2025-05-01', 100, 5_000),
                        _row('B', '2025-05-02', 1000, 60_000),
                        _row('C', '2025-05-03', 1000, 60_000)])
    assert FA.insider_signal(ins, AS_OF).loc['X', 'INSIDER_CLUSTER'] == 2


def test_trades_filed_after_as_of_are_invisible():
    ins = pd.DataFrame([_row('A', '2025-06-28', 1000, 50_000, filed='2025-07-02')])
    assert FA.insider_signal(ins, AS_OF).empty


def test_ui_note_explains_each_excluded_row():
    df = pd.DataFrame([_row('A', '2025-05-01', 1, 1),
                       _row('B', '2025-05-01', 1, 1, flags='10b5-1'),
                       _row('C', '2025-05-01', 1, 1, role='10% Owner')])
    assert list(FA.insider_note(df)) == ['', '10b5-1 plan', '10% holder only']


# ─────────────────────────────────────────────
# STORAGE — MIGRATION AND RE-PARSE
# ─────────────────────────────────────────────

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATABASE_URL', f'sqlite:///{tmp_path}/t.sqlite')
    from core import db as _db
    importlib.reload(_db)
    yield _db
    _db.get_engine().dispose()
    monkeypatch.undo()
    importlib.reload(_db)


def test_init_adds_new_columns_to_an_existing_table(store):
    from sqlalchemy import inspect, text
    with store.get_engine().begin() as conn:
        conn.execute(text(
            'CREATE TABLE insider_txns (accession VARCHAR, ticker VARCHAR, '
            'insider VARCHAR, txn_date DATE, txn_code VARCHAR, cik VARCHAR, '
            'filed DATE, role VARCHAR, shares FLOAT, price FLOAT, value FLOAT, '
            'post_txn_shares FLOAT, PRIMARY KEY (accession, ticker, insider, '
            'txn_date, txn_code))'))
    store.init_db()
    cols = {c['name'] for c in inspect(store.get_engine()).get_columns('insider_txns')}
    assert {'flags', 'parse_version'} <= cols


def test_old_rows_are_reparsed_and_replaced(store, monkeypatch):
    store.init_db()
    today = date.today()
    store.upsert(store.sec_filings, [
        {'accession': 'own', 'cik': '320193', 'ticker': 'AAPL', 'form': '4',
         'filing_date': today, 'primary_doc': 'xslF345X05/own.xml'},
        {'accession': 'other', 'cik': '320193', 'ticker': 'AAPL', 'form': '4',
         'filing_date': today, 'primary_doc': 'xslF345X05/other.xml'}])
    # What an older parser stored: one lot of a split sale, and a filing that
    # was really about another issuer.
    store.upsert(store.insider_txns, [
        {'accession': 'own', 'ticker': 'AAPL', 'insider': 'Doe Jane',
         'txn_date': date(2025, 3, 3), 'txn_code': 'S', 'filed': today,
         'shares': -300.0, 'value': 3600.0},
        {'accession': 'other', 'ticker': 'AAPL', 'insider': 'Doe Jane',
         'txn_date': date(2025, 3, 3), 'txn_code': 'P', 'filed': today,
         'shares': 100.0, 'value': 1000.0}])
    docs = {'own.xml': _form4([_txn('S', 100, 10.0, 'D'), _txn('S', 300, 12.0, 'D')],
                              aff='1'),
            'other.xml': _form4([_txn('P', 100, 10.0)], issuer_cik='0000034088')}
    monkeypatch.setattr(sec, 'cik_lineage', lambda t: ['320193'])
    monkeypatch.setattr(sec.http, 'fetch',
                        lambda url, **k: docs[url.rsplit('/', 1)[-1]])
    monkeypatch.setattr(store, 'record_ingest', lambda *a, **k: None)

    sec.update_insiders(['AAPL'])
    got = store.read_sql('SELECT * FROM insider_txns ORDER BY accession')
    assert list(got['accession']) == ['own']                  # other issuer gone
    r = got.iloc[0]
    assert r['shares'] == -400 and r['flags'] == '10b5-1'
    assert r['parse_version'] == sec.INSIDER_PARSE_VERSION

    calls = []
    monkeypatch.setattr(sec.http, 'fetch', lambda url, **k: calls.append(url) or None)
    sec.update_insiders(['AAPL'])
    assert not any(u.endswith('own.xml') for u in calls), 'current rows re-fetched'


def test_insider_reads_carry_flags(store):
    store.init_db()
    store.upsert(store.insider_txns, [{
        'accession': 'a', 'ticker': 'X', 'insider': 'I', 'txn_date': date(2025, 1, 2),
        'txn_code': 'S', 'filed': date(2025, 1, 3), 'flags': '10b5-1'}])
    assert sec.insiders_asof(['X'], date(2025, 1, 5))['flags'].iloc[0] == '10b5-1'
