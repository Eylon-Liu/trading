# quant_dbt

Point-in-time financial data modelling in dbt. **Every test here encodes a bug that
actually happened** in a production quant research system, with the number it produced.

The hard part of financial data is not the arithmetic. It is that the same figure arrives
several times, in several shapes, on several dates, and quietly using the wrong one makes
a backtest look better than reality by an amount nobody can measure afterwards. These
models are the defences against that, written as SQL that can be reviewed and tested.

Companion to [eylon-liu/trading](https://github.com/Eylon-Liu/trading), which is where the
bugs were found.

```bash
pip install -r requirements.txt
DBT_PROFILES_DIR=. dbt build          # 15 models, 29 tests, no external state
python scripts/mutation_check.py      # proves the tests can actually fail
```

---

## The bug catalogue

Each row is a fixture in `seeds/`, a guard in `models/`, and an assertion in `tests/`.

| Bug | What it produced | Guard |
|---|---|---|
| **Duration overlap** — Apple's FY26 Q3 files `NetIncomeLoss` twice with the *same* end date and the *same* filing date: 29,789M for the quarter, 101,464M for the nine months. Only `period_start` separates them. | TTM earnings **3.4× too high** | `stg_sec_facts.duration_class`, `int_quarterly_flows` |
| **Restatement** — a period is re-reported with a later filing date | A June backtest using an August revision | `int_facts_deduped` |
| **Look-ahead** — a factor as of *T* referencing a filing after *T* | Silently inflated everything | `pit_gate()` macro |
| **Survivorship** — the S&P 500 in 2021 held ABMD, ATVI, XLNX | Scoring 2021 against today's list = knowing the winners | `dim_index_membership` |
| **Mixed price bases** — feed gives `high`/`low` raw, `adj_close` adjusted. Chevron: raw high $174.63 vs adjusted close $150.55 | Sharpe **1.6 → 3.3**, entirely fictional | `int_adjusted_prices` |
| **Wrong XBRL tag** — Camden files `Revenues` as a fragment beside the real `RealEstateRevenueNet` | **2,486%** net margin | `int_facts_tagged` (magnitude rule) |
| **Retired XBRL tag** — Microsoft kept filing a dead `Revenues` line | **134%** net margin | `int_facts_tagged` (recency rule) |
| **Proxy units** — a DEF 14A reports in thousands and is filed *after* the 10-K | $3.5B became **$3,511** | `int_facts_pit` (form whitelist) |
| **Split restatement** — prices are restated for splits, SEC share counts are not | Lam Research: **$3.1B** market cap vs ~$30B actual, a 66% earnings yield, top-ranked everywhere | `int_shares_split_adjusted` |
| **Frozen share count** — Exxon stopped filing the diluted tag in 2013 and kept buying back stock | A **twelve-year-old** denominator under every valuation factor | `int_shares_split_adjusted` (staleness guard) |

### The three that are really the same bug

Proxy units, retired tags, and split restatement were not wrong logic. The logic was
right and ran in the wrong place:

- Restatement collapsing keeps the newest filing. Correct — unless a DEF 14A is still in
  the set, in which case it wins by being filed later and brings its units with it.
- Picking the tag with the largest typical magnitude is correct — unless a retired tag is
  still in the set, in which case it wins precisely when it is the largest.
- Cross-checking a share count against EPS is correct — unless it runs before split
  adjustment, in which case it reads a 10-for-1 split as an order-of-magnitude error.

**This is the argument for dbt.** Ordering that lived in the order of statements inside a
function becomes a DAG: declared in `ref()`, visible in the lineage graph, and documented
per edge in `models/intermediate/_intermediate.yml`.

---

## The pipeline

```
seeds (fixtures)  or  the app's SQLite store
        │
    staging/        type, rename, classify the reporting span
        │
    intermediate/   1  int_facts_pit            form whitelist, then filed <= as_of
        │           2  int_facts_tagged         one XBRL tag per concept
        │           3  int_facts_deduped        restatements → newest knowable
        │           4  int_quarterly_flows      ← the interesting one
        │           5  int_ttm_fundamentals     trailing four quarters
        │           5b int_latest_stocks        balance-sheet values
        │           6  int_shares_split_adjusted
        │           7  int_adjusted_prices      one basis per bar
        │
    marts/          fct_factor_inputs_pit, dim_security, dim_index_membership
```

Everything is parameterised on one variable:

```bash
dbt build --vars 'as_of: 2026-05-15'
```

Move it and the whole table moves to what was knowable then — restatements disappear,
unfiled quarters vanish, splits stop applying.

---

## The translation worth reading

`int_quarterly_flows` replaces a ~60-line row-wise Python loop that walks each filer's
cumulative filings subtracting as it goes. The set-based version is one pass:

```sql
select *,
       lag(val)        over w as prev_val,
       lag(period_end) over w as prev_period_end,
       lag(filed)      over w as prev_filed
from flows
where duration_class in ('H', '9M', 'A')
window w as (partition by ticker, concept, period_start order by period_end)
```

Partitioning by `period_start` is what keeps it inside one fiscal year: cumulative figures
all share the year's opening date, so consecutive rows are Q1, H1, 9M, FY — and the gaps
between them are the quarters nobody reported.

And the whole of the original's `_latest_filed` helper — which exists to work around
pandas' column-wise aggregation silently assembling a row that was never filed — is one
clause:

```sql
qualify row_number() over (
    partition by ticker, concept, period_start, period_end
    order by filed desc
) = 1
```

One detail that is easy to miss: **a derived quarter carries the later of its two
constituent filing dates.** It is not knowable until both exist, so inheriting the earlier
one would backdate a figure into a window where nobody could have computed it. A
subtraction that is arithmetically correct can still be a look-ahead leak.

---

## Proving the tests can fail

A green suite is not evidence. These fixtures are hand-built, so it is entirely possible
to write assertions that pass because they never look anywhere interesting — and a
point-in-time test that cannot detect a leak is worse than no test, because it turns an
unknown into a false assurance.

`scripts/mutation_check.py` breaks each guard in turn and requires a specific test to go
red:

```
remove-pit-gate                caught by assert_no_lookahead_in_pit_mart
admit-cumulative-as-quarter    caught by assert_no_overlapping_durations_summed
drop-form-whitelist            caught by assert_fixture_known_answers
skip-restatement-collapse      caught by assert_one_row_per_period_after_dedupe
mix-price-bases                caught by assert_prices_single_adjustment_basis
collapse-membership-history    caught by assert_membership_is_point_in_time
tag-choice-by-magnitude-only   caught by assert_fixture_known_answers
```

It runs in CI, so a test that quietly stops checking anything fails the build instead of
passing silently. The first version of the form-whitelist mutation was itself broken — it
added a form to the allowlist rather than removing the filter, so nothing was mutated and
the check correctly reported the test as unproven.

## Checking against the original

The tests assert that the SQL produces the values this project claims are correct. Both
the model and its expected value were written at the same time, so a shared misreading of
the source logic would pass them both. `scripts/parity_check.py` brings in the independent
party: it runs the trading repo's `build_fundamentals` over the same fixtures and compares.

```
18 comparisons agree across 7 tickers (as_of 2026-12-31)
```

It earned its place on the first run, with two disagreements:

- **A missing guard in the SQL.** The Python rejects a share count filed more than 500
  days before the evaluation date; this project had no such rule, so it happily reported a
  number the original would have refused. That is the Exxon row in the table above — found
  by the parity check, not by the tests, because the tests only knew what the models
  already did.
- **A bug in the harness itself.** `build_fundamentals` expects input that is *already*
  point-in-time gated — in the application that is `sec.facts_asof`'s job — and has no way
  to tell gated input from ungated. Handing it the raw fixture made the Python side read a
  quarter filed a month after `as_of` and report a TTM $2.5bn higher, which looked like a
  translation bug in the SQL until the real cause turned up.

The second one is the better story. Two implementations disagreeing tells you something is
wrong; it does not tell you which side, and the obvious suspect was not the culprit.

---

## Why these choices

**dbt-duckdb, attached to SQLite.** DuckDB's `sqlite_scanner` reads the application's
store in place — no export step, nothing to drift out of date. DuckDB also has `QUALIFY`,
real window functions and interval arithmetic; SQLite's dialect is too thin for the
quarter-differencing logic.

**Fixtures, not a database dump.** The `dev` target runs on ~50 rows of committed CSV, so
CI needs no warehouse, no credentials and no network. The fixtures also *are* the
documentation — each row exists to produce one number, and a reviewer can read the input
that causes the failure. `--target local` attaches the real store for checking against
genuine volume and messiness.

**No `snapshots/` directory.** dbt snapshots exist for sources that overwrite in place,
where yesterday's value is gone unless dbt captured it. This source appends — every
observation carries the `as_of` date on which it was knowable — so the history is already
there and the work is reshaping it into intervals. Running a snapshot over an
already-temporal table would record when dbt happened to look, which is a fact about the
pipeline rather than about the index. `dim_index_membership` does it properly, with
gaps-and-islands, because membership is not monotonic: a company can leave an index and
rejoin.

**Incremental on `stg_prices`, with a lookback.** Bars get corrected after the fact and a
split restates every historical `adj_close`, so "newer than my newest row" would lock in
the uncorrected version forever. `delete+insert` on `(ticker, date)` over a trailing
window makes re-running idempotent.

---

## dbt concepts demonstrated

| Concept | Where |
|---|---|
| `ref()` / `source()` and the DAG | throughout; `macros/raw_relation.sql` switches source by target |
| Materializations | `view` staging, `table` marts, `incremental` prices |
| Incremental idempotency + late-arriving data | `stg_prices.sql` |
| Generic tests | `_staging.yml`, `_marts.yml` |
| Singular tests | `tests/` — six of them |
| Macros + Jinja | `classify_duration`, `pit_gate`, `raw` |
| Seeds | `seeds/` — the fixture catalogue |
| Vars and environments | `as_of`, `statement_forms`; `dev` / `local` targets |
| SCD Type 2 | `dim_index_membership.sql` |
| Docs + lineage | `dbt docs generate`, deployed from `target/` |

---

## Running against real data

```bash
# in the trading repo, build a store first
python3 cli.py init && python3 cli.py ingest --index DIA --full

# then, here
DBT_PROFILES_DIR=. dbt build --target local
```

`profiles.yml` attaches `../db/quant.sqlite` read-only, so a dbt run can never write back
into the application's store.

The attach pulls DuckDB's `sqlite_scanner` extension, which is downloaded once and cached
in `~/.duckdb`. On a machine without outbound access it fails with *"Failed to download
extension"* — `duckdb -c 'INSTALL sqlite;'` somewhere with network is the whole fix. The
`dev` target needs none of this, which is the other reason CI uses it.

**Not yet run against real data.** Everything above — 51 tests, 8 mutations, 18 parity
comparisons — is on the fixtures. The fixtures are faithful to the bugs but they are 60
rows; real filings bring volume, missing quarters, tickers spanning two CIKs after a
reorganisation, and concepts no fixture anticipates. Running `--target local` against an
ingested store is the next verification, and nothing here should be presented as
production-proven until it has.

## Extracting to its own repository

This currently lives inside the trading repo so the parity check has the original
implementation on hand and so the work is version-controlled somewhere. It is
self-contained — nothing outside `quant_dbt/` is imported except by
`scripts/parity_check.py`, which takes a path.

```bash
# from the trading repo root, preserving this directory's history
git subtree split --prefix=quant_dbt -b quant_dbt_only
cd .. && git clone -b quant_dbt_only trading quant_dbt && cd quant_dbt
git remote set-url origin git@github.com:<you>/quant_dbt.git && git push -u origin main
```

Then `quant_dbt/.github/workflows/dbt.yml` becomes the active workflow — it checks the
trading repo out into `trading-src/` for the parity step — and
`trading/.github/workflows/quant_dbt.yml` can be deleted. `scripts/parity_check.py`
defaults to the parent directory and takes `QUANT_TRADING_PATH` (absolute or relative)
when the two repos sit elsewhere.

## Docs site

`dbt docs generate` emits a standalone static site in `target/`, including the interactive
lineage graph. `vercel.json` builds and serves it; GitHub Pages works equally well from
the CI artifact.
