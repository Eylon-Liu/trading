"""Who is in the universe on a past date — and only them."""

from __future__ import annotations

from datetime import date

import pandas as pd

from data import members as MEM
from data import sec
from quant import factors as FA


# ── ticker changes ───────────────────────────────────────────────

def test_old_symbols_resolve_to_todays():
    assert MEM.current_symbol('FB') == 'META'
    assert MEM.current_symbol('AAPL') == 'AAPL'
    assert MEM.current_symbols(['DISCA', 'DISCK', 'AAPL']) == ['WBD', 'AAPL']   # two classes, one company
    # A rename never points at a symbol that is itself renamed away.
    for old, new in MEM.TICKER_RENAMES.items():
        assert MEM.current_symbol(old) == MEM.current_symbol(new)


def test_acquired_companies_are_not_mapped_to_their_acquirer():
    # XLNX→AMD, ATVI→MSFT etc. would hand the acquirer's history to a
    # different business; those stay unmapped (and correctly absent).
    for gone in ('XLNX', 'ATVI', 'PXD', 'SIVB', 'TWTR', 'CERN'):
        assert gone not in MEM.TICKER_RENAMES


def test_historical_membership_uses_current_symbols(fresh_db):
    fresh_db.upsert(fresh_db.index_members, [
        {'index_symbol': 'SPY', 'ticker': t, 'as_of': date(2020, 6, 30),
         'source': 'wiki_revision', 'confidence': 0.9, 'weight': None}
        for t in ('FB', 'ANTM', 'AAPL', 'XLNX')])
    got = MEM.members_asof('SPY', date(2021, 1, 4))
    assert got == ['AAPL', 'ELV', 'META', 'XLNX']


# ── only that date's members are ranked ──────────────────────────

def test_factor_frame_holds_only_the_names_asked_for(monkeypatch):
    # A backtest preloads facts for every ticker of the whole window, and
    # fundamentals come back for all of them. The frame for one date must
    # still be that date's universe.
    idx_all = ['AAPL', 'MSFT', 'FUTURE1', 'FUTURE2']
    monkeypatch.setattr(FA, 'price_factors', lambda tickers, as_of, **k:
                        pd.DataFrame({'MOM_12_1': [0.1, 0.2]}, index=['AAPL', 'MSFT']))
    monkeypatch.setattr(FA, 'fundamental_factors', lambda tickers, as_of, **k:
                        pd.DataFrame({'ROIC': [0.3, 0.2, 0.9, 0.8]}, index=idx_all))
    monkeypatch.setattr(FA, 'alt_factors', lambda tickers, as_of, **k: pd.DataFrame())
    out = FA.build_all(['AAPL', 'MSFT'], date(2020, 1, 31), use_cache=False)
    assert sorted(out.index) == ['AAPL', 'MSFT']


# ── the lean facts reader returns exactly what a plain read does ──

def test_lean_read_matches_a_plain_read_and_shares_strings(fresh_db):
    rows = []
    for t, cik in (('AAPL', '1'), ('MSFT', '2')):
        for q in range(1, 9):
            rows.append({'cik': cik, 'tag': 'Revenues', 'unit': 'USD',
                         'period_start': date(2024, 1, 1), 'period_end': date(2024, q, 28),
                         'filed': date(2024, q + 1, 15), 'form': '10-Q' if q % 4 else '10-K',
                         'ticker': t, 'concept': 'revenue', 'fy': 2024,
                         'fp': None if q == 8 else f'Q{q % 4 or 4}', 'val': float(q)})
    fresh_db.upsert(fresh_db.sec_facts, rows)
    lean = sec._read_lean(
        "SELECT ticker, concept, tag, unit, period_start, period_end, filed, form, fy, fp, val "
        "FROM sec_facts ORDER BY ticker, concept, period_end, filed", {},
        parse_dates=['period_start', 'period_end', 'filed'],
        text_columns=('ticker', 'concept', 'tag', 'unit', 'form', 'fp'), chunk_rows=5)
    plain = fresh_db.read_sql(
        "SELECT ticker, concept, tag, unit, period_start, period_end, filed, form, fy, fp, val "
        "FROM sec_facts ORDER BY ticker, concept, period_end, filed",
        parse_dates=['period_start', 'period_end', 'filed'])
    assert len(lean) == 16
    for col in plain.columns:
        a, b = lean[col], plain[col]
        assert (a.eq(b) | (a.isna() & b.isna())).all(), col
    # Equal values are one object, across chunks — that is the memory saving.
    assert lean['concept'].iloc[0] is lean['concept'].iloc[15]
    assert lean['ticker'].iloc[0] is lean['ticker'].iloc[7]
