#!/usr/bin/env python3
"""
Prove the tests can fail.

A green test suite says nothing on its own. These fixtures are hand-built, so
it is entirely possible to write assertions that pass because they never look
anywhere interesting — and a point-in-time test that cannot detect a leak is
worse than no test, because it converts an unknown into a false assurance.

Each case below breaks exactly one guard in one model, runs `dbt build`, and
requires a specific test to go red. If the suite stays green after a guard is
removed, that test is decorative and the run fails loudly.

Every mutation is reverted in a finally block, including on interrupt, so the
working tree is left as it was found.

    python scripts/mutation_check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    find: str
    replace: str
    expect_failing: str
    why: str


MUTATIONS = [
    Mutation(
        name='remove-pit-gate',
        path='models/intermediate/int_facts_pit.sql',
        find='  and {{ pit_gate(\'filed\') }}',
        replace='  and 1 = 1  -- MUTATED: point-in-time gate removed',
        expect_failing='assert_no_lookahead_in_pit_mart',
        why='A quarter filed 2027-01-29 becomes visible at an as_of of '
            '2026-12-31 — the leak the whole pipeline exists to prevent.',
    ),
    Mutation(
        name='admit-cumulative-as-quarter',
        path='models/intermediate/int_quarterly_flows.sql',
        find="    where duration_class = 'Q'",
        replace="    where duration_class in ('Q', '9M')  -- MUTATED",
        expect_failing='assert_no_overlapping_durations_summed',
        why="Apple's nine-month cumulative enters the quarter pool and "
            'overlaps the three quarters it contains.',
    ),
    Mutation(
        name='drop-form-whitelist',
        path='models/intermediate/int_facts_pit.sql',
        # The whole clause, not just its opening: adding a form to the list
        # leaves DEF 14A excluded and mutates nothing. The first version of
        # this case did exactly that and reported the test as broken.
        find="""where form in (
          {%- for f in var('statement_forms') %}
          '{{ f }}'{{ ',' if not loop.last }}
          {%- endfor %}
      )""",
        replace='where 1 = 1  -- MUTATED: form whitelist removed',
        expect_failing='assert_fixture_known_answers',
        why='A DEF 14A reporting in thousands survives to restatement '
            'collapsing and wins on filing date, turning $3.5B into $3,511.',
    ),
    Mutation(
        name='skip-restatement-collapse',
        path='models/intermediate/int_facts_deduped.sql',
        find=') = 1',
        replace=') >= 1  -- MUTATED: keeps every restatement',
        expect_failing='assert_one_row_per_period_after_dedupe',
        why='Both the original and the revised figure survive, so every '
            'downstream sum double-counts the revised periods.',
    ),
    Mutation(
        name='mix-price-bases',
        path='models/intermediate/int_adjusted_prices.sql',
        find='    high * adj_ratio as adj_high,\n    low  * adj_ratio as adj_low,',
        replace='    high as adj_high,  -- MUTATED: unadjusted\n    low  as adj_low,',
        expect_failing='assert_prices_single_adjustment_basis',
        why='Unadjusted extremes against an adjusted close put the close '
            'outside its own bar — the Chevron case.',
    ),
    Mutation(
        name='accept-stale-share-counts',
        path='models/intermediate/int_shares_split_adjusted.sql',
        find='        when filing_age_days <= {{ max_share_age_days }}',
        replace='        when 1 = 1  -- MUTATED: staleness guard removed',
        expect_failing='assert_stale_share_counts_rejected',
        why='A share count frozen since 2013 is treated as current, so every '
            'valuation factor divides by a denominator that predates a decade '
            'of buybacks.',
    ),
    Mutation(
        name='collapse-membership-history',
        path='models/marts/dim_index_membership.sql',
        find='    where is_member\n    group by index_symbol, ticker, run_id',
        replace='    where 1 = 1  -- MUTATED: absence no longer ends an interval\n'
                '    group by index_symbol, ticker, run_id',
        expect_failing='assert_membership_is_point_in_time',
        why='A ticker that left the index keeps an open interval, so every '
            'historical date returns the current constituents — survivorship '
            'bias, reintroduced.',
    ),
    Mutation(
        name='tag-choice-by-magnitude-only',
        path='models/intermediate/int_facts_tagged.sql',
        find='    where tag_stats.newest_filing\n          >= concept_newest.concept_newest_filing - interval 365 day',
        replace='    where 1 = 1  -- MUTATED: retired tags are no longer excluded',
        expect_failing='assert_fixture_known_answers',
        why='A retired tag carrying larger values wins on magnitude, which is '
            'exactly the Microsoft case the recency rule exists for.',
    ),
]


def run_build() -> tuple[bool, str]:
    """Run dbt build. Returns (every_test_passed, combined_output)."""
    # Inherit the environment and override only the profiles directory.
    # Building a minimal env from scratch meant hardcoding PATH, which found
    # dbt locally and not on a CI runner, where setup-python installs it under
    # /opt/hostedtoolcache. There is nothing in the ambient environment this
    # script needs protection from.
    proc = subprocess.run(
        ['dbt', 'build', '--no-partial-parse'],
        cwd=ROOT,
        env={**os.environ, 'DBT_PROFILES_DIR': '.'},
        capture_output=True, text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def failing_tests(output: str) -> set[str]:
    names = set()
    for line in output.splitlines():
        if ' FAIL ' in line or ' ERROR ' in line:
            for token in line.split():
                if token.startswith('assert_') or token.startswith(
                        ('not_null_', 'unique_', 'accepted_values_')):
                    names.add(token)
    return names


def main() -> int:
    ok, output = run_build()
    if not ok:
        print('baseline dbt build is not green — fix that before mutating')
        print(output[-3000:])
        return 1
    print(f'baseline: green ({len(MUTATIONS)} mutations to check)\n')

    failures = []

    for mutation in MUTATIONS:
        target = ROOT / mutation.path
        original = target.read_text()

        if mutation.find not in original:
            failures.append(f'{mutation.name}: anchor text not found in '
                            f'{mutation.path} — the mutation is stale')
            print(f'  {mutation.name}: STALE ANCHOR')
            continue

        try:
            target.write_text(original.replace(mutation.find,
                                               mutation.replace, 1))
            passed, output = run_build()
            red = failing_tests(output)

            if passed:
                failures.append(
                    f'{mutation.name}: suite stayed GREEN with the guard '
                    f'removed — {mutation.expect_failing} is not actually '
                    f'checking anything. {mutation.why}')
                print(f'  {mutation.name}: NOT CAUGHT')
            elif mutation.expect_failing not in red:
                failures.append(
                    f'{mutation.name}: build went red but '
                    f'{mutation.expect_failing} was not among the failures '
                    f'({sorted(red) or "none parsed"})')
                print(f'  {mutation.name}: caught by the wrong test')
            else:
                print(f'  {mutation.name}: caught by '
                      f'{mutation.expect_failing}')
        finally:
            target.write_text(original)

    print()
    ok, _ = run_build()
    if not ok:
        failures.append('tree not restored cleanly — baseline is red again')

    if failures:
        print('FAILED:')
        for f in failures:
            print(f'  - {f}')
        return 1

    print(f'all {len(MUTATIONS)} guards are covered by a test that '
          f'actually fails without them')
    return 0


if __name__ == '__main__':
    sys.exit(main())
