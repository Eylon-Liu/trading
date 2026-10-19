-- Index membership as SCD Type 2 intervals: who was in the index, and when.
--
-- This is the survivorship-bias fix. Scoring a 2021 date against today's
-- constituent list quietly loads the test with winners — the S&P 500 on
-- 2021-06-01 held ABMD, ATVI and XLNX, all since acquired, and a backtest that
-- silently excluded them would be measuring the performance of having known
-- which companies would survive.
--
-- Why this is a model and not a dbt snapshot: `snapshots/` exists for sources
-- that overwrite in place, where yesterday's value is gone unless dbt captured
-- it. This source appends instead — each observation carries the `as_of` date
-- on which it was knowable — so the history is already there and the work is
-- reshaping it into intervals, not preserving it. Running a dbt snapshot over
-- an already-temporal table would record when dbt happened to look, which is a
-- fact about the pipeline rather than about the index.
--
-- Gaps-and-islands, because membership is not monotonic: a company can leave
-- an index and rejoin, and collapsing that to one interval would claim it was
-- a member throughout a period when it was not.

with snapshots as (

    select distinct index_symbol, as_of
    from {{ ref('stg_index_members') }}
    where {{ pit_gate('as_of') }}

),

index_tickers as (

    select distinct index_symbol, ticker
    from {{ ref('stg_index_members') }}
    where {{ pit_gate('as_of') }}

),

grid as (

    -- Every ticker against every observation date, so absence is explicit. A
    -- ticker missing from a snapshot is a removal; without the grid it is
    -- indistinguishable from no observation at all.
    select
        index_tickers.index_symbol,
        index_tickers.ticker,
        snapshots.as_of,
        members.ticker is not null as is_member
    from index_tickers
    join snapshots
      on snapshots.index_symbol = index_tickers.index_symbol
    left join {{ ref('stg_index_members') }} as members
      on  members.index_symbol = index_tickers.index_symbol
      and members.ticker       = index_tickers.ticker
      and members.as_of        = snapshots.as_of

),

marked as (

    select
        *,
        lag(is_member) over (
            partition by index_symbol, ticker order by as_of
        ) as was_member
    from grid

),

islands as (

    select
        *,
        -- Increments once per entry into the index, giving each continuous
        -- run of membership its own id.
        sum(case when is_member and coalesce(was_member, false) = false
                 then 1 else 0 end)
            over (partition by index_symbol, ticker order by as_of) as run_id
    from marked

),

intervals as (

    select
        index_symbol,
        ticker,
        run_id,
        min(as_of) as valid_from,
        max(as_of) as last_observed
    from islands
    where is_member
    group by index_symbol, ticker, run_id

)

select
    intervals.index_symbol,
    intervals.ticker,
    intervals.valid_from,

    -- The first observation that no longer contained the ticker. NULL means
    -- still a member as of the most recent observation — an open interval, not
    -- a missing value.
    (
        select min(snapshots.as_of)
        from snapshots
        where snapshots.index_symbol = intervals.index_symbol
          and snapshots.as_of        > intervals.last_observed
    ) as valid_to,

    intervals.last_observed

from intervals
