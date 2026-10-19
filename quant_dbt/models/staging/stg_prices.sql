-- Daily bars, typed, with the adjustment ratio made explicit.
--
-- The feed returns `high` and `low` unadjusted while `adj_close` is
-- back-adjusted for splits and dividends, so the four columns of a single bar
-- are not on the same basis. Carrying the ratio here lets the intermediate
-- layer put the whole bar onto one basis without re-deriving it, and makes the
-- mismatch visible in the staging model rather than buried in a join.
--
-- Mixing the two bases is not a rounding problem. Comparing an adjusted stop
-- against a raw high made every trade appear to hit its target instantly and
-- turned a Sharpe of 1.6 into a fictional 3.3.

{{ config(
    materialized='incremental',
    unique_key=['ticker', 'date'],
    incremental_strategy='delete+insert'
) }}

select
    ticker,
    cast(date as date)      as date,
    cast(open as double)    as open,
    cast(high as double)    as high,
    cast(low as double)     as low,
    cast(close as double)   as close,
    cast(adj_close as double) as adj_close,
    cast(volume as double)  as volume,

    -- NULLIF rather than a CASE: a zero close is bad data, and producing NULL
    -- here makes it propagate visibly instead of dividing by zero or silently
    -- defaulting the bar to an unadjusted basis.
    cast(adj_close as double) / nullif(cast(close as double), 0) as adj_ratio

from {{ raw('prices') }}

{% if is_incremental() %}

    -- Reprocess a trailing window rather than only strictly-new dates. Bars
    -- get corrected after the fact — a bad print is revised, and a split
    -- restates every historical adj_close — so "newer than the newest row I
    -- have" would lock in the uncorrected version forever. delete+insert on
    -- (ticker, date) makes re-running the window idempotent.
    where cast(date as date) >= (
        select coalesce(max(date), date '1900-01-01') - interval 7 day
        from {{ this }}
    )

{% endif %}
