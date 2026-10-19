-- Known-answer tests: each fixture ticker exists to produce one number, and
-- this asserts that number exactly.
--
-- The structural tests alongside this one check that the shape is right, which
-- a pipeline can satisfy while still computing the wrong value — pick the
-- wrong XBRL tag and the grain, the intervals and the filing dates all stay
-- perfectly consistent. These assertions are what make the fixtures a
-- regression suite rather than a smoke test.
--
-- Each expected value is derived in the comment beside it, so a failure says
-- which rule broke rather than only that something moved.

with expected as (

    select * from (values

        -- Duration disambiguation. The nine-month cumulative (101,464M) shares
        -- its end date AND its filing date with the 29,789M quarter; only
        -- period_start separates them. Summing both gives 230,394M — 1.79x too
        -- high. The 30,000M quarter filed 2027-01-29 is beyond as_of and must
        -- be invisible.
        ('AAPL', 'ttm_net_income', 42097e6 + 29578e6 + 29789e6 + 27466e6),
        ('AAPL', 'ttm_revenue',    94930e6 + 124300e6 + 95359e6 + 85777e6),

        -- Restatement. Q1 2026 was first reported at 1,000M, then revised to
        -- 900M on 2026-08-01. The default as_of is after the revision, so the
        -- revised figure is the knowable one.
        ('RSTT', 'ttm_revenue',    800e6 + 850e6 + 900e6 + 900e6),

        -- Form whitelist. A DEF 14A filed 2026-04-10 restates Q4 2025 as 3,511
        -- — the same figure in thousands. It is filed later than the 10-K, so
        -- restatement collapsing would prefer it if the form filter had not
        -- already removed it. Leaking gives 10,500,003,511.
        ('PRXY', 'ttm_revenue',    4 * 3500e6),

        -- Tag selection by magnitude. `Revenues` is a 2M fragment filed
        -- alongside the real 500M `RealEstateRevenueNet`, on the same dates.
        -- Recency cannot separate them; median magnitude can.
        ('TAGS', 'ttm_revenue',    4 * 500e6),

        -- Tag selection by currency. The retired `Revenues` tag carries LARGER
        -- values (900M) than the current one (300M), so magnitude alone would
        -- pick exactly the wrong tag. The recency rule has to run first.
        ('RTRD', 'ttm_revenue',    4 * 300e6),

        -- Split restatement. 100M shares filed 2026-06-01, 10-for-1 split on
        -- 2026-09-01. Prices are restated by the feed, share counts are not.
        ('SPLT', 'shares_diluted', 100e6 * 10),

        -- Staleness. STAL's revenue is current, so it must still produce a
        -- TTM — the guard rejects the share count alone, not the whole name.
        ('STAL', 'ttm_revenue',    4 * 600e6)

    ) as t(ticker, measure, expected_value)

),

actual as (

    select ticker, 'ttm_net_income' as measure, ttm_net_income as actual_value
    from {{ ref('fct_factor_inputs_pit') }}
    union all
    select ticker, 'ttm_revenue', ttm_revenue
    from {{ ref('fct_factor_inputs_pit') }}
    union all
    select ticker, 'shares_diluted', shares_diluted
    from {{ ref('fct_factor_inputs_pit') }}

)

select
    expected.ticker,
    expected.measure,
    expected.expected_value,
    actual.actual_value
from expected
left join actual
       on actual.ticker  = expected.ticker
      and actual.measure = expected.measure
-- NULL fails too: a measure that vanished is not a measure that passed.
where actual.actual_value is null
   or abs(actual.actual_value - expected.expected_value) > 1
