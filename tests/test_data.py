"""Provider-layer tests — parsing, universe reproducibility, NLP."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.members import _normalize_company, normalize_ticker
from data.universe import UniverseSpec, parse_spec
from nlp import extract as EX
from nlp import sentiment as SENT
from nlp import summarize as SUM


# ─────────────────────────────────────────────
# TICKER / NAME NORMALIZATION
# ─────────────────────────────────────────────

def test_normalize_ticker_yahoo_convention():
    assert normalize_ticker('BRK.B') == 'BRK-B'
    assert normalize_ticker(' aapl ') == 'AAPL'
    assert normalize_ticker('MSFT[1]') == 'MSFT'


def test_normalize_ticker_rejects_non_tickers():
    for junk in ('', None, 'this is a company name', '###'):
        assert normalize_ticker(junk) is None


@pytest.mark.parametrize('a,b', [
    ('APPLIED MATERIALS INC /DE', 'Applied Materials Inc'),
    ('AMERICAN TOWER CORP /MA/', 'American Tower Corp'),
    ('SMITH A O CORP', 'A O Smith Corp'),
    ('Air Products & Chemicals, Inc.', 'Air Products and Chemicals Inc'),
    ('BANK OF AMERICA CORP /DE/', 'Bank of America Corp'),
    ('CDW Corp', 'CDW Corp/DE'),
])
def test_company_name_keys_reconcile(a, b):
    """
    SEC and N-PORT disagree in four systematic ways: state-of-incorporation
    suffixes, "&" versus "and", surname-first legal ordering, and corporate
    form words. All four must collapse to the same key.
    """
    assert _normalize_company(a) == _normalize_company(b)


def test_company_key_does_not_collapse_distinct_firms():
    assert _normalize_company('Apple Inc') != _normalize_company('Applied Materials Inc')


# ─────────────────────────────────────────────
# UNIVERSE
# ─────────────────────────────────────────────

def test_spec_roundtrips_through_json():
    spec = UniverseSpec(preset='SPY', sectors=['Health Care'],
                        min_market_cap=5e9, max_names=25)
    assert UniverseSpec.from_json(spec.to_json()) == spec


def test_parse_spec_reads_a_compact_expression():
    spec = parse_spec('SPY,sector=Health Care,mcap>5B,top=40')
    assert spec.preset == 'SPY'
    assert 'Health Care' in spec.sectors
    assert spec.min_market_cap == pytest.approx(5e9)
    assert spec.max_names == 40


def test_parse_spec_number_suffixes():
    assert parse_spec('mcap>2B').min_market_cap == pytest.approx(2e9)
    assert parse_spec('adv>25M').min_dollar_adv == pytest.approx(25e6)
    assert parse_spec('price>10').min_price == pytest.approx(10.0)


def test_universe_resolution_is_reproducible():
    """
    The original shuffled and truncated, so identical requests produced
    different universes and cross-sectional scores were not comparable.
    """
    spec = UniverseSpec(preset='DIA', min_market_cap=None,
                        min_dollar_adv=None, min_price=None)
    first = spec.resolve(date.today())
    if not first:
        pytest.skip('no membership data; run cli.py ingest first')
    for _ in range(3):
        assert spec.resolve(date.today()) == first


# ─────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────

def test_accounting_vocabulary_is_not_negative():
    """
    The whole reason for a finance-specific lexicon: liabilities, costs and
    depreciation are neutral in a filing and negative in everyday English.
    """
    text = ('The company reported a significant increase in liabilities, '
            'costs and depreciation expense.')
    assert abs(SENT.score(text)['polarity']) < 0.25


def test_negation_flips_polarity():
    assert SENT.score('Results were profitable.')['polarity'] > 0
    assert SENT.score('Results were not profitable.')['polarity'] < 0


def test_event_direction_scores_even_without_sentiment_words():
    """
    "Management raised guidance and announced a buyback" contains no lexicon
    sentiment words at all, yet is plainly bullish.
    """
    text = ('Management raised full-year guidance and announced a '
            '$10 billion buyback.')
    assert SENT.score(text)['polarity'] > 0.3


def test_clearly_negative_news_scores_negative():
    text = ('The firm cut its dividend and disclosed an SEC investigation '
            'into accounting irregularities.')
    assert SENT.score(text)['polarity'] < -0.3


def test_sentiment_bounded():
    for t in ['excellent outstanding record growth profit success',
              'fraud bankruptcy lawsuit loss collapse disaster']:
        assert -1.0 <= SENT.score(t)['polarity'] <= 1.0


# ─────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────

def test_guidance_direction_detected():
    assert EX.extract('The company raised its full-year outlook.').guidance == 'raised'
    assert EX.extract('The company cut its guidance.').guidance == 'lowered'
    assert EX.extract('Management reaffirmed guidance.').guidance == 'reaffirmed'


def test_analyst_price_target_is_not_company_guidance():
    """
    A broker trimming its price target is a different event from a company
    cutting guidance, and conflating them misreads the thesis.
    """
    ex = EX.extract('Analysts downgraded the stock, cutting the price '
                    'target to $85.')
    assert ex.guidance is None
    assert ex.analyst_action == 'downgrade'


def test_events_and_direction_notes():
    ex = EX.extract('Board cuts dividend and announces 3,000 layoffs amid '
                    'restructuring.')
    assert 'dividend_cut' in ex.events
    assert 'layoffs' in ex.events
    assert any('Dividend cut' in n for n in ex.direction_notes)
    assert ex.is_material


def test_ma_and_management_change():
    ex = EX.extract('Company to acquire rival for $2.3 billion; CEO steps '
                    'down effective March.')
    assert 'ma' in ex.events and 'management' in ex.events


def test_aggregate_rolls_up_multiple_articles():
    items = [EX.extract('Company raised guidance.'),
             EX.extract('Analysts upgraded the shares.'),
             EX.extract('Board approved a buyback.')]
    agg = EX.aggregate(items)
    assert agg['n_articles'] == 3
    assert agg['material_count'] >= 2


# ─────────────────────────────────────────────
# SUMMARIZATION
# ─────────────────────────────────────────────

def test_textrank_selects_a_subset_of_original_sentences():
    text = (
        'The company reported record quarterly revenue of $5 billion. '
        'Analysts had expected $4.7 billion for the period. '
        'Management raised full-year guidance on the strength of the result. '
        'The stock rose sharply in after-hours trading. '
        'A new chief financial officer will join in the autumn.')
    picked = SUM.textrank(text, top_n=2)
    assert len(picked) == 2
    for sentence in picked:
        assert sentence in text, 'extractive summaries must not invent text'


def test_summarize_respects_length_cap():
    text = ' '.join([f'This is filler sentence number {i} about the company.'
                     for i in range(40)])
    assert len(SUM.summarize(text, max_chars=200)) <= 201


def test_summarize_empty_is_safe():
    assert SUM.summarize('') == ''
    assert SUM.textrank('') == []


# ─────────────────────────────────────────────
# BULK SHARE COUNTS VIA FRAMES
# ─────────────────────────────────────────────

def test_recent_frames_are_quarterly_and_newest_first():
    from datetime import date as _d

    from data import sec
    frames = sec._recent_frames(_d(2026, 8, 11), back=4)
    assert frames[0] == 'CY2026Q3I'
    assert frames == ['CY2026Q3I', 'CY2026Q2I', 'CY2026Q1I', 'CY2025Q4I']
    assert all(f.endswith('I') for f in frames), 'instant frames only'


def test_frames_rows_without_a_known_filing_are_skipped(monkeypatch):
    """A frame row carries an accession but no filing date.

    Point-in-time reads gate on `filed`, so a row we cannot date is dropped
    rather than given an estimate — a guessed filing date is exactly the quiet
    fiction the rest of the pipeline exists to prevent.
    """
    import pandas as pd

    from data import sec

    monkeypatch.setattr(sec, 'share_counts_via_frames', lambda *a, **kw: pd.DataFrame([
        {'cik': '0000320193', 'ticker': 'AAPL', 'shares': 1.46e10,
         'end': pd.Timestamp('2026-06-30'), 'accession': 'known-1'},
        {'cik': '0000789019', 'ticker': 'MSFT', 'shares': 7.4e9,
         'end': pd.Timestamp('2026-06-30'), 'accession': 'unindexed-9'},
    ]))
    monkeypatch.setattr(sec.db, 'read_sql', lambda *a, **kw: pd.DataFrame(
        [{'accession': 'known-1', 'filing_date': pd.Timestamp('2026-07-31')}]))

    written = {}

    def fake_upsert(table, rows):
        written['rows'] = rows
        return len(rows)

    monkeypatch.setattr(sec.db, 'upsert', fake_upsert)
    monkeypatch.setattr(sec.db, 'record_ingest', lambda *a, **kw: None)

    n = sec.update_share_counts()
    assert n == 1
    assert [r['ticker'] for r in written['rows']] == ['AAPL']
    row = written['rows'][0]
    assert row['concept'] == 'shares_outstanding'
    assert str(row['filed']) == '2026-07-31', 'must use the real filing date'


# ─────────────────────────────────────────────
# SIZE FILTERS AND MISSING DATA
# ─────────────────────────────────────────────

def test_unknown_market_cap_does_not_fail_a_size_filter():
    """Absence of a snapshot is not evidence that a company is small.

    The universe filter used `fillna(0)`, so the moment profile_snapshots held
    any rows at all, every name *without* one scored zero market cap and was
    dropped. Nine stored snapshots cut an S&P 500 screen to nine names.
    """
    import pandas as pd

    from data.universe import UniverseSpec
    from quant.engine import _apply_size_filters

    raw = pd.DataFrame({'MARKET_CAP': [5e11, 1e9, float('nan')]},
                       index=['BIG', 'SMALL', 'UNKNOWN'])
    spec = UniverseSpec(preset='SPY', min_market_cap=1e10)

    kept = _apply_size_filters(raw, spec)
    assert 'BIG' in kept.index
    assert 'SMALL' not in kept.index, 'a known-small name must be dropped'
    assert 'UNKNOWN' in kept.index, 'an unknown size must not be assumed small'


def test_size_filters_are_a_noop_without_bounds():
    import pandas as pd

    from data.universe import UniverseSpec
    from quant.engine import _apply_size_filters

    raw = pd.DataFrame({'MARKET_CAP': [1.0, float('nan')]}, index=['A', 'B'])
    spec = UniverseSpec(preset='SPY', min_market_cap=None)
    assert len(_apply_size_filters(raw, spec)) == 2


def test_max_market_cap_also_tolerates_unknowns():
    import pandas as pd

    from data.universe import UniverseSpec
    from quant.engine import _apply_size_filters

    raw = pd.DataFrame({'MARKET_CAP': [5e11, 1e9, float('nan')]},
                       index=['BIG', 'SMALL', 'UNKNOWN'])
    spec = UniverseSpec(preset='SPY', min_market_cap=None, max_market_cap=1e10)
    kept = _apply_size_filters(raw, spec)
    assert set(kept.index) == {'SMALL', 'UNKNOWN'}
