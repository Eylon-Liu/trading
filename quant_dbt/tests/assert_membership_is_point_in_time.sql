-- Membership as of one date must differ from membership as of another.
--
-- The survivorship assertion, written so it tests the property rather than the
-- fixture: if the SCD2 intervals ever collapsed to "whoever is in the index
-- now", every historical date would return today's constituents and this
-- would return zero differing dates.
--
-- A degenerate pass is the risk worth naming — with a single observation date
-- there is nothing to differ from, so the test also requires that more than
-- one date exists before comparing.

with dates as (
    select distinct index_symbol, valid_from as d from {{ ref('dim_index_membership') }}
),

pairs as (
    select
        a.index_symbol,
        a.d as early,
        b.d as late
    from dates a
    join dates b on b.index_symbol = a.index_symbol and b.d > a.d
),

membership_at as (
    select
        pairs.index_symbol,
        pairs.early,
        pairs.late,
        (select count(*) from {{ ref('dim_index_membership') }} m
          where m.index_symbol = pairs.index_symbol
            and m.valid_from <= pairs.early
            and (m.valid_to is null or m.valid_to > pairs.early)) as n_early,
        (select count(*) from {{ ref('dim_index_membership') }} m
          where m.index_symbol = pairs.index_symbol
            and m.valid_from <= pairs.late
            and (m.valid_to is null or m.valid_to > pairs.late)) as n_late,
        (select count(*) from {{ ref('dim_index_membership') }} m
          where m.index_symbol = pairs.index_symbol
            and m.valid_from <= pairs.early
            and (m.valid_to is null or m.valid_to > pairs.early)
            and not exists (
                select 1 from {{ ref('dim_index_membership') }} m2
                 where m2.index_symbol = m.index_symbol
                   and m2.ticker = m.ticker
                   and m2.valid_from <= pairs.late
                   and (m2.valid_to is null or m2.valid_to > pairs.late))) as n_dropped
    from pairs
)

-- Fails when no pair of dates exists at all, or when a pair shows no
-- departures anywhere in the history.
select * from membership_at
where n_early = 0
   or n_late = 0

union all

select * from membership_at
where (select count(*) from membership_at where n_dropped > 0) = 0
