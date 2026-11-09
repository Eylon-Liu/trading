"""AI earnings reviews and the multi-strategy consensus."""

from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from nlp import earnings_review as ERV
from quant import consensus as CONS
from quant import strategies as ST


# ── parsing the model's answer ───────────────────────────────────

def test_review_json_is_parsed_through_fences_and_clamped():
    text = ('Here you go:\n```json\n{"score": 3.5, "verdict": "BUY", '
            '"conviction": "High", "guidance": "raised", "quarter_quality": -9, '
            '"outlook": 1, "valuation_view": "cheap", "positives": ["a","b","c","d"], '
            '"negatives": [], "summary": "Strong quarter."}\n```')
    r = ERV.parse_review(text)
    assert r['score'] == 2.0 and r['quarter_quality'] == -2.0
    assert r['verdict'] == 'buy' and r['conviction'] == 'high'
    assert len(r['positives']) == 3


def test_unknown_labels_fall_back_to_neutral():
    r = ERV.parse_review('{"score": 1, "verdict": "strong buy!!", "guidance": "up"}')
    assert r['verdict'] == 'hold' and r['guidance'] == 'none'
    assert r['conviction'] == 'low'


@pytest.mark.parametrize('bad', [None, '', 'no json here', '{"verdict": "buy"}',
                                 '{"score": "n/a"}', '[1, 2]'])
def test_missing_score_is_rejected_not_stored_as_zero(bad):
    assert ERV.parse_review(bad) is None


# ── fetching the release ─────────────────────────────────────────

INDEX_HTML = """<table class="tableFile"><tr><th>Seq</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/1/000/a8k.htm">a8k.htm</a></td>
    <td>8-K</td><td>1</td></tr>
<tr><td>2</td><td>PRESS RELEASE</td><td><a href="/ix?doc=/Archives/edgar/data/1/000/ex991.htm">ex991.htm</a></td>
    <td>EX-99.1</td><td>1</td></tr></table>"""


def test_press_release_is_exhibit_99(monkeypatch):
    seen = []

    def fake(url, **k):
        seen.append(url)
        if url.endswith('-index.htm'):
            return INDEX_HTML
        return '<html><body><p>Revenue rose  12%.</p><script>x</script></body></html>'
    monkeypatch.setattr(ERV.http, 'fetch_text', fake)
    text = ERV.press_release_text('0000000001', '0000000001-26-000123')
    assert text == 'Revenue rose 12%.'
    assert seen[1].endswith('/000000000126000123/ex991.htm')


# ── storing and reading, point-in-time ───────────────────────────

def _filing(db, ticker='AAA', acc='acc-1', filed=date(2026, 11, 4), form='8-K'):
    db.upsert(db.sec_filings, [{'accession': acc, 'cik': '1', 'ticker': ticker,
                                'form': form, 'filing_date': filed,
                                'primary_doc': 'a8k.htm'}])
    db.upsert(db.corporate_events, [{'ticker': ticker, 'filed': filed,
                                     'accession': acc, 'item_code': '2.02',
                                     'category': 'results'}])


@pytest.fixture
def llm(monkeypatch):
    calls = []
    monkeypatch.setattr(ERV, 'press_release_text', lambda *a, **k: 'Revenue up.')
    monkeypatch.setattr(ERV, 'build_context', lambda t, f, a: {
        'name': t, 'sector': 'X', 'figures': '-', 'reaction': 'n/a',
        'estimate': 'n/a', 'headlines': '-'})
    monkeypatch.setattr(ERV.providers, 'available', lambda: True)
    monkeypatch.setattr(ERV.providers, 'model_name', lambda *a: 'test-model')

    def complete(prompt, system, max_tokens):
        calls.append(prompt)
        return json.dumps({'score': 1.5, 'verdict': 'buy', 'conviction': 'high',
                           'guidance': 'raised', 'summary': 'Good.'})
    monkeypatch.setattr(ERV.providers, 'complete', complete)
    monkeypatch.setattr(config, 'QUANT_LLM_REVIEWS', 'auto')
    return calls


def test_review_is_stored_once_and_dated_when_written(fresh_db, llm):
    _filing(fresh_db)
    today = date(2026, 11, 6)
    done = ERV.review_recent(as_of=today)
    assert len(done) == 1 and done[0]['assessed_on'] == today
    assert ERV.review_recent(as_of=today) == []          # cached by accession
    assert len(llm) == 1


def test_view_is_invisible_before_it_was_written(fresh_db, llm):
    _filing(fresh_db)
    ERV.review_recent(as_of=date(2026, 11, 6))
    assert ERV.view_factor(['AAA'], date(2026, 11, 5)).empty
    assert ERV.view_factor(['AAA'], date(2026, 11, 6))['AAA'] == pytest.approx(1.5)


def test_amended_filings_are_not_reviewed(fresh_db, llm):
    _filing(fresh_db, acc='amend', form='8-K/A')
    assert ERV.pending(as_of=date(2026, 11, 6)).empty


def test_priority_and_cap(fresh_db, llm):
    for i, t in enumerate(['AAA', 'BBB', 'CCC']):
        _filing(fresh_db, ticker=t, acc=f'a{i}', filed=date(2026, 11, 2 + i))
    done = ERV.review_recent(as_of=date(2026, 11, 6), priority=['AAA'], max_reviews=2)
    assert [r['ticker'] for r in done] == ['AAA', 'CCC']


def test_disabled_without_a_provider(fresh_db, monkeypatch):
    monkeypatch.setattr(ERV.providers, 'available', lambda: False)
    _filing(fresh_db)
    assert ERV.review_recent(as_of=date(2026, 11, 6)) == []


def test_ai_strategy_cannot_be_backtested_without_history():
    from quant import availability as AV
    cov = pd.DataFrame([{'source': s, 'first': date(2015, 1, 2),
                         'last': date(2026, 11, 6), 'rows': 1000,
                         'span_days': 4000}
                        for s in ('prices', 'sec_facts', 'corporate_events')]
                       + [{'source': 'llm_reviews', 'first': date(2026, 10, 1),
                           'last': date(2026, 11, 6), 'rows': 40, 'span_days': 36}])
    w = AV.backtest_window('ai_earnings', coverage=cov)
    assert not w.testable
    assert any('live signal' in b for b in w.blockers)


# ── consensus voting ─────────────────────────────────────────────

def _raw(n=50, seed=0):
    rng = np.random.RandomState(seed)
    cols = set()
    for k in ('buffett', 'shareholder_yield', 'quality_momentum', 'momentum_trend',
              'relative_strength', 'insider_conviction', 'insider_cluster',
              'earnings_drift', 'quality_breakdown'):
        cols |= set(ST.get(k).weights)
    raw = pd.DataFrame({c: rng.randn(n) for c in cols},
                       index=[f'T{i}' for i in range(n)])
    raw['sector'] = 'X'
    raw['ABOVE_MA200'] = 1.0
    raw['RSI_14'] = 50.0
    raw['MARKET_CAP'] = 5e10
    return raw


def _strategy(**kw):
    import dataclasses
    return dataclasses.replace(ST.get('consensus'), **kw)


def test_correlated_strategies_vote_once():
    raw = _raw()
    strat = _strategy(components={'Momentum': ('momentum_trend', 'relative_strength',
                                               'quality_momentum'),
                                  'Quality': ('buffett',)},
                      factor_votes={}, veto=())
    sc, _d, _w = CONS.score(raw, strat, '2026-11-06')
    assert sc['votes'].max() <= 2


def test_failing_a_member_filter_counts_as_no():
    raw = _raw()
    raw['ABOVE_MA200'] = 0.0                  # every trend screen excludes all
    strat = _strategy(components={'Momentum': ('momentum_trend', 'relative_strength'),
                                  'Quality': ('buffett',)},
                      factor_votes={}, veto=())
    sc, _d, _w = CONS.score(raw, strat, '2026-11-06')
    assert (sc['FAM::Momentum'] == 0).all()
    assert sc['votes'].max() <= 1


def test_ai_vote_and_veto():
    raw = _raw()
    raw['LLM_EARNINGS_VIEW'] = np.nan
    raw.loc['T1', 'LLM_EARNINGS_VIEW'] = 1.6
    raw.loc['T2', 'LLM_EARNINGS_VIEW'] = -1.4
    base = _strategy(veto=())
    sc, _d, _w = CONS.score(raw, base, '2026-11-06')
    no_ai = _strategy(veto=(), factor_votes={})
    sc0, _d, _w = CONS.score(raw, no_ai, '2026-11-06')
    assert sc.loc['T1', 'votes'] == sc0.loc['T1', 'votes'] + 1
    assert sc.loc['T2', 'votes'] == sc0.loc['T2', 'votes'] - 1

    vetoed, _d, _w = CONS.score(raw, _strategy(factor_votes={}), '2026-11-06')
    hit = vetoed['vetoed']
    assert hit.any()
    assert (vetoed.loc[hit, 'votes'] == sc0.loc[hit[hit].index, 'votes'] - 1).all()


def test_signals_read_the_vote_count():
    sc = pd.DataFrame({'composite': [3.4, 2.4, 0.3], 'votes': [3.0, 2.0, 0.0],
                       'vetoed': [False, False, True], 'families_scored': [4, 4, 4],
                       'FAM::Quality & value': [0.95, 0.9, 0.1]},
                      index=['A', 'B', 'C'])
    out = CONS.signals(sc, ST.get('consensus'))
    assert out.loc['A', 'signal'].startswith('🟢 Strong Buy — 3 of 5')
    assert out.loc['B', 'signal'].startswith('🟢 Buy — 2 of 5')
    assert out.loc['C', 'signal'].startswith('🔴 Avoid')
    assert 'Quality & value ✓ 95%' in out.loc['A', 'reasons']


def test_meta_backtest_drops_families_without_history(monkeypatch):
    from quant import availability as AV

    def fake_window(m, preset='SPY', coverage=None):
        early = date(2018, 1, 2) if m in ('buffett', 'momentum_trend') else date(2025, 1, 2)
        return AV.Window(m, early, date(2026, 11, 6))
    monkeypatch.setattr(AV, 'backtest_window', fake_window)
    strat, notes = AV.meta_members_at('consensus', date(2020, 1, 2),
                                      coverage=pd.DataFrame())
    assert strat.components == {'Quality & value': ('buffett',),
                                'Momentum & trend': ('momentum_trend',)}
    assert strat.factor_votes == {}
    assert any('Insider buying' in n and 'left out' in n for n in notes)


# ── does agreement predict? ──────────────────────────────────────

def test_agreement_study_buckets_and_ic(monkeypatch):
    from quant import engine as EN
    from quant import research as RS

    names = [f'T{i}' for i in range(40)]
    votes = pd.Series([i % 4 for i in range(40)], index=names, dtype=float)

    def fake_run(spec, key, as_of=None, persist=False, **k):
        return SimpleNamespace(scores=pd.DataFrame({'votes': votes}))

    def fake_fwd(tickers, as_of, h):
        r = pd.Series({t: 0.02 * votes.get(t, 0) for t in tickers})
        r[config.BENCHMARK_TICKER] = 0.01
        return r
    monkeypatch.setattr(EN, 'run', fake_run)
    monkeypatch.setattr(RS, 'forward_returns', fake_fwd)
    out = CONS.agreement_study(None, '2024-01-01', '2025-12-31', 3)
    b = out['buckets'].set_index('bucket')
    assert b.loc[3, 'mean_excess'] > b.loc[0, 'mean_excess']
    assert out['agreement_ic'] == pytest.approx(1.0)
    assert out['dates'] >= 6


# ── the report keeps the consensus out of its own count ──────────

def test_consensus_run_is_not_a_voter(fresh_db, monkeypatch, tmp_path):
    from data.universe import UniverseSpec
    from quant import engine as EN
    from reports import builder as RB
    monkeypatch.setattr(config, 'REPORT_DIR', tmp_path)
    monkeypatch.setattr(RB, '_auto_run_prior', lambda *a, **k: (None, None))
    keys = ['buffett', 'momentum_trend', 'consensus']
    monkeypatch.setattr(RB, '_all_strategy_keys', lambda: keys)
    spec = UniverseSpec(preset='SPY')
    for k in keys:
        sc = pd.DataFrame({'composite': [2.4, 1.2], 'rank': [1, 2],
                           'signal': ['🟢 Buy — 2 of 5 families agree'
                                      if k == 'consensus' else '🟢 Buy'] * 2,
                           'signal_score': [2.0, 1.0], 'reasons': 'Quality ✓ 90%',
                           'coverage': 1.0, 'sector': 'X'}, index=['AAA', 'BBB'])
        EN._persist(f'{k}-r', date(2026, 11, 6), spec, ST.get(k), ['AAA', 'BBB'],
                    sc, pd.DataFrame())
    p = RB._gather_cross_strategy(date(2026, 11, 9), spec_json=spec.to_json())
    assert p['n_strategies'] == 2
    assert list(p['consensus_run']['ticker']) == ['AAA', 'BBB']
    html = RB._render_consensus(p['consensus_run'], held={'AAA'})
    assert 'Multi-strategy consensus' in html and 'held' in html
    assert 'BBB' not in html                      # one family is not agreement
    assert RB._current_picks(date(2026, 11, 9), spec.to_json()) == ['AAA']


def test_engine_runs_and_persists_the_consensus(fresh_db, monkeypatch):
    from data.universe import UniverseSpec
    from quant import engine as EN
    raw = _raw(60, seed=3)
    monkeypatch.setattr(EN.FA, 'build_all', lambda *a, **k: raw.copy())
    monkeypatch.setattr(UniverseSpec, 'resolve', lambda self, *a, **k: list(raw.index))
    spec = UniverseSpec(preset='SPY', min_market_cap=None)
    res = EN.run(spec, 'consensus', as_of=date(2026, 11, 6), persist=True)
    assert not res.scores.empty
    assert res.plans.empty                               # long horizon, no plans
    assert res.scores['rank'].tolist() == list(range(1, len(res.scores) + 1))
    assert res.scores['votes'].is_monotonic_decreasing or \
        res.scores['composite'].is_monotonic_decreasing
    stored = fresh_db.read_sql('SELECT params_json FROM runs')
    assert 'components' in stored['params_json'].iloc[0]
    detail = fresh_db.read_sql('SELECT DISTINCT factor FROM factor_scores')
    assert any(f.startswith('family: ') for f in detail['factor'])


def _by_date(wins: int, dates: int = 12) -> pd.DataFrame:
    rows = []
    for d in range(dates):
        hi = 0.03 if d < wins else -0.01
        rows += [{'date': d, 'votes': 2, 'excess': hi},
                 {'date': d, 'votes': 0, 'excess': 0.0}]
    return pd.DataFrame(rows)


def _cli_study(monkeypatch, wins: int):
    from quant import availability as AV
    from quant import backtest as BT
    strat = _strategy(components={'Quality & value': ('buffett',),
                                  'Momentum & trend': ('momentum_trend',)},
                      factor_votes={})
    monkeypatch.setattr(AV, 'meta_members_at', lambda *a, **k: (strat, ['x dropped']))
    buckets = pd.DataFrame({'bucket': [0, 1, 2], 'mean_excess': [-0.01, 0.005, 0.02],
                            'periods': [12, 12, 12], 'names': [300, 120, 60],
                            'hit_rate': [0.45, 0.5, 0.6],
                            'median_excess': [-0.01, 0.0, 0.015],
                            'label': ['0 families', '1 family', '2 families']})
    monkeypatch.setattr(CONS, 'agreement_study', lambda *a, **k: {
        'buckets': buckets, 'agreement_ic': 0.06, 'agreement_ic_hit': 0.75,
        'dates': 12, 'by_date': _by_date(wins)})


def test_inconsistent_agreement_is_only_a_tilt(fresh_db, monkeypatch, capsys):
    import cli
    from quant import backtest as BT
    _cli_study(monkeypatch, wins=5)              # positive mean, 5 of 12 dates
    monkeypatch.setattr(BT, 'run_backtest',
                        lambda *a, **k: SimpleNamespace(metrics={}))
    assert cli.main(['consensus-test', '--start', '2021-01-01']) == 0
    out = capsys.readouterr().out
    assert 'not consistent from date to date' in out
    assert 'ahead on 5 of 12 dates' in out


def test_consensus_test_command_prints_a_verdict(fresh_db, monkeypatch, capsys):
    import cli
    from quant import backtest as BT
    _cli_study(monkeypatch, wins=10)
    m = {'cagr': 0.12, 'excess_cagr': 0.02, 'sharpe': 0.9, 'max_drawdown': -0.2,
         'mean_ic': 0.03}
    monkeypatch.setattr(BT, 'run_backtest',
                        lambda *a, **k: SimpleNamespace(metrics=m))
    monkeypatch.setattr(BT, 'run_trade_backtest',
                        lambda *a, **k: SimpleNamespace(metrics=m))
    assert cli.main(['consensus-test', '--start', '2021-01-01', '--compare']) == 0
    out = capsys.readouterr().out
    assert 'more agreement meant higher returns' in out
    assert 'Multi-Strategy Consensus' in out and '+12.0%' in out


def test_agreement_spread_is_paired_within_dates():
    by_date = pd.DataFrame({
        'date': [1, 1, 1, 2, 2, 2],
        'votes': [0, 2, 3, 0, 2, 1],
        'excess': [-0.02, 0.01, 0.03, 0.01, 0.00, 0.5]})
    s = CONS.agreement_spread(by_date)
    assert s['dates'] == 2 and s['wins'] == 1
    assert s['mean'] == pytest.approx(((0.02 + 0.02) + (-0.01)) / 2)


def test_untestable_veto_is_left_out_of_a_backtest(monkeypatch):
    from quant import availability as AV

    def fake_window(m, preset='SPY', coverage=None):
        if m == 'quality_breakdown':
            return AV.Window(m, None, date(2026, 11, 6), ['needs insider_txns'])
        return AV.Window(m, date(2018, 1, 2), date(2026, 11, 6))
    monkeypatch.setattr(AV, 'backtest_window', fake_window)
    strat, notes = AV.meta_members_at('consensus', date(2020, 1, 2),
                                      coverage=pd.DataFrame())
    assert strat.veto == ()
    assert any('veto quality_breakdown' in n for n in notes)


def test_backtest_preload_includes_the_benchmark(monkeypatch):
    from quant import backtest as BT
    seen = {}
    monkeypatch.setattr(BT.yahoo, 'bulk_ohlc',
                        lambda names, **k: seen.setdefault('names', names) or (pd.DataFrame(),) * 4)
    monkeypatch.setattr(BT.sec, 'facts_asof', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(BT.sec, 'insiders_asof', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(BT.sec, 'events_asof', lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(BT.yahoo, 'split_factors', lambda *a, **k: pd.DataFrame())
    spec = SimpleNamespace(resolve=lambda d: ['AAA', 'BBB'])
    BT._preload(spec, date(2025, 1, 2), date(2025, 6, 30))
    assert config.BENCHMARK_TICKER in seen['names']
