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


# ── shortlist: no sector takes more than its share ───────────────

def _ranking(sectors):
    return pd.DataFrame({'composite': [1.0 - i * 0.01 for i in range(len(sectors))],
                         'sector': sectors}, index=[f'T{i:02d}' for i in range(len(sectors))])


def test_shortlist_caps_any_one_sector():
    from quant import engine as EN
    # The top of the ranking is one theme: twelve tech names, then others.
    sc = _ranking(['Information Technology'] * 12 + ['Health Care'] * 6
                  + ['Energy'] * 6 + ['Financials'] * 6)
    top = EN.diversified_head(sc, 20)
    counts = top['sector'].value_counts()
    assert len(top) == 20 and counts.max() == 6          # 30% of 20
    assert list(top.index[:6]) == [f'T{i:02d}' for i in range(6)]   # best names kept, in order
    assert 'T06' not in top.index and 'T12' in top.index           # 7th tech name passed over
    # Passed-over names are still in the ranking itself.
    assert len(sc) == 30


def test_shortlist_cap_never_starves_a_small_list_and_ignores_unknown_sector():
    from quant import engine as EN
    sc = _ranking(['Energy'] * 5 + ['Unknown'] * 5)
    assert len(EN.diversified_head(sc, 4)) == 4          # cap is at least two names…
    assert EN.diversified_head(sc, 4)['sector'].tolist() == ['Energy', 'Energy', 'Unknown', 'Unknown']
    assert len(EN.diversified_head(_ranking(['Unknown'] * 8), 6)) == 6
    assert EN.diversified_head(sc, 10, max_share=1.0).equals(sc.head(10))


def test_backtest_holds_the_capped_shortlist():
    from quant import backtest as BT
    sc = _ranking(['Information Technology'] * 12 + ['Health Care'] * 4
                  + ['Energy'] * 4 + ['Financials'] * 4)
    w = BT._weights(sc, 10, 'equal')
    held = sc.loc[w.index, 'sector'].value_counts()
    assert len(w) == 10 and held['Information Technology'] == 3 and abs(w.sum() - 1) < 1e-9


def test_cap_gives_way_in_a_narrow_universe():
    from quant import engine as EN
    # A single-sector screen must still return a full shortlist.
    one = _ranking(['Information Technology'] * 15)
    assert list(EN.diversified_head(one, 10).index) == list(one.index[:10])
    # Two sectors, ten wanted: as even as the ranking allows, never short.
    two = _ranking(['Information Technology'] * 12 + ['Health Care'] * 10)
    top = EN.diversified_head(two, 10)
    assert len(top) == 10 and top['sector'].value_counts().max() == 5
    assert len(EN.diversified_head(_ranking(['Energy'] * 3), 10)) == 3


# ── the 3-6 month horizon ────────────────────────────────────────

def test_mid_term_means_three_to_six_months():
    from quant import strategies as ST
    from quant import tradeplan as TP
    mid = ST.HORIZONS['mid']
    assert '3-6 months' in mid['label'] and mid['rebalance'] == 'SA'     # hold ~six months
    p = TP.PlanParams()
    assert p.time_stop_days == 126                        # six months of sessions
    assert p.atr_stop_mult >= 3.0                         # not a swing-trade stop


def test_balanced_blends_value_momentum_and_quality():
    from quant import factors as FA
    from quant import strategies as ST
    s = ST.get('balanced')
    assert s.horizon == 'mid' and s.filters.get('min_above_ma200') == 1
    value = {'EARNINGS_YIELD', 'FCF_YIELD', 'BUYBACK_YIELD'}
    momentum = {'MOM_12_1', 'MOM_6_1', 'PCT_FROM_52W_HIGH'}
    w = s.weights
    assert value <= set(w) and momentum <= set(w) and 'ROIC' in w
    assert all(f in FA.FACTOR_DIRECTION for f in w)
    # Neither style dominates the other.
    v, m = sum(w[f] for f in value), sum(w[f] for f in momentum)
    assert 0.8 <= v / m <= 1.25
    assert 'balanced' not in {k for fam in ST.get('consensus').components.values() for k in fam}
