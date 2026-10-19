#!/usr/bin/env python3
"""
Compare the dbt models against the Python implementation they were translated
from, on identical inputs.

The tests in tests/ assert that the SQL produces the values this project
*claims* are correct. That is necessary but not sufficient: both the model and
its expected value were written by the same person at the same time, so a
shared misreading of the source logic would pass. This script introduces the
independent party — the original implementation in the trading repo, which
predates the translation and was debugged against real filings.

Agreement here means the SQL reproduces behaviour that was arrived at
separately. Disagreement means one of them is wrong, and the whole point is to
find out which before an interviewer does.

Requires the trading repo (default: the parent directory) and its
dependencies.

    python scripts/parity_check.py
    QUANT_TRADING_PATH=~/code/trading python scripts/parity_check.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
TRADING = Path(os.environ.get('QUANT_TRADING_PATH', HERE.parent)).expanduser()

# The as_of default in dbt_project.yml. Both sides must evaluate the same date
# or they are answering different questions.
AS_OF = dt.date(2026, 12, 31)

# dbt column -> build_fundamentals column.
COMPARISONS = [
    ('ttm_revenue',    'revenue_ttm'),
    ('ttm_net_income', 'net_income_ttm'),
    ('shares_diluted', 'shares_diluted'),
    ('equity',         'equity'),
]

TOLERANCE = 1.0  # dollars; the two paths differ only in float accumulation order


def python_side() -> pd.DataFrame:
    sys.path.insert(0, str(TRADING))
    from quant import fundamentals as F          # noqa: PLC0415

    facts = pd.read_csv(HERE / 'seeds' / 'raw_sec_facts.csv')
    for column in ('period_start', 'period_end', 'filed'):
        facts[column] = pd.to_datetime(facts[column])

    # build_fundamentals expects facts that are ALREADY point-in-time gated —
    # in the application that is sec.facts_asof's job, and the function has no
    # way to tell gated input from ungated. Handing it the raw fixture made the
    # Python side read a quarter filed a month after as_of and report a TTM
    # 2.5bn higher, which looked like a translation bug in the SQL until the
    # missing gate turned up here. Worth keeping as a comment: the asymmetry is
    # easy to reintroduce.
    facts = facts[facts['filed'] <= pd.Timestamp(AS_OF)]

    splits = pd.read_csv(HERE / 'seeds' / 'raw_splits.csv')
    splits['date'] = pd.to_datetime(splits['date'])

    tickers = sorted(facts['ticker'].unique())
    result = F.build_fundamentals(tickers, AS_OF,
                                  facts_all=facts, splits=splits)
    return result.reset_index().rename(columns={'index': 'ticker'})


def dbt_side() -> pd.DataFrame:
    db = HERE / 'target' / 'quant_dbt.duckdb'
    if not db.exists():
        sys.exit(f'no dbt output at {db} — run `dbt build` first')
    with duckdb.connect(str(db), read_only=True) as con:
        return con.sql(
            'select ticker, ttm_revenue, ttm_net_income, shares_diluted, '
            'equity from main.fct_factor_inputs_pit'
        ).to_df()


def main() -> int:
    py = python_side().set_index('ticker')
    sql = dbt_side().set_index('ticker')

    only_py = sorted(set(py.index) - set(sql.index))
    only_sql = sorted(set(sql.index) - set(py.index))
    if only_py or only_sql:
        print(f'ticker sets differ — python only: {only_py}, dbt only: {only_sql}')

    mismatches = []
    compared = 0

    for ticker in sorted(set(py.index) & set(sql.index)):
        for sql_col, py_col in COMPARISONS:
            if py_col not in py.columns:
                continue
            a, b = sql.loc[ticker, sql_col], py.loc[ticker, py_col]
            a_missing, b_missing = pd.isna(a), pd.isna(b)

            if a_missing and b_missing:
                continue
            compared += 1

            # A value on one side and nothing on the other is the most
            # informative disagreement there is: one path found the figure
            # unknowable and the other did not.
            if a_missing != b_missing:
                mismatches.append(
                    f'{ticker:>5}  {sql_col:<16} dbt='
                    f'{"NULL" if a_missing else f"{a:,.0f}"}  python='
                    f'{"NULL" if b_missing else f"{b:,.0f}"}')
            elif abs(float(a) - float(b)) > TOLERANCE:
                mismatches.append(
                    f'{ticker:>5}  {sql_col:<16} dbt={a:>18,.0f}  '
                    f'python={b:>18,.0f}  diff={float(a) - float(b):>,.0f}')

    if mismatches:
        print(f'{len(mismatches)} of {compared} comparisons disagree:\n')
        for line in mismatches:
            print(f'  {line}')
        return 1

    print(f'{compared} comparisons agree across '
          f'{len(set(py.index) & set(sql.index))} tickers '
          f'(as_of {AS_OF})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
