-- Step 2: one XBRL tag per (ticker, concept).
--
-- A concept like "revenue" maps to several candidate tags, and a filer may
-- report more than one of them. Mixing them produces nonsense, so a single tag
-- is chosen per ticker and concept and the rest are dropped.
--
-- Two rules, in order, each from a specific failure:
--
--   1. Currency. Keep only tags still in use — those whose newest filing is
--      within a year of the concept's newest filing anywhere. Microsoft kept
--      filing a retired `Revenues` line long after moving to
--      `RevenueFromContractWithCustomerExcludingAssessedTax`; reading the
--      stale one produced a 134% net margin.
--
--   2. Magnitude. Among what survives, take the largest median absolute value.
--      Camden Property reports `Revenues` as a fragment alongside the real
--      `RealEstateRevenueNet`; recency alone picked the fragment and returned
--      a 2,486% net margin.
--
-- Rule 1 must come first. Magnitude alone would keep a retired tag precisely
-- when it is largest, which is the Microsoft case.
--
-- Median, not mean or max: a single misfiled row — an order-of-magnitude typo,
-- a one-off restatement — moves a mean or a max enough to decide the vote, and
-- the question being asked is which tag this filer habitually uses.

with tag_stats as (

    select
        ticker,
        concept,
        tag,
        max(filed)       as newest_filing,
        median(abs(val)) as typical_magnitude
    from {{ ref('int_facts_pit') }}
    group by ticker, concept, tag

),

concept_newest as (

    select
        ticker,
        concept,
        max(newest_filing) as concept_newest_filing
    from tag_stats
    group by ticker, concept

),

still_current as (

    -- The tag holding the concept's newest filing always survives this, so the
    -- set can never empty out.
    select tag_stats.*
    from tag_stats
    join concept_newest using (ticker, concept)
    where tag_stats.newest_filing
          >= concept_newest.concept_newest_filing - interval 365 day

),

chosen as (

    select ticker, concept, tag
    from still_current
    -- `tag` breaks a magnitude tie so the choice is reproducible rather than
    -- dependent on scan order.
    qualify row_number() over (
        partition by ticker, concept
        order by typical_magnitude desc, tag
    ) = 1

)

select facts.*
from {{ ref('int_facts_pit') }} as facts
join chosen using (ticker, concept, tag)
