-- Point-in-time index membership.
--
-- `as_of` is the date a membership was *knowable*. For N-PORT-sourced rows
-- that is the filing date, which lags the period end by about two months, so
-- treating it as the effective date would be wrong by exactly that lag.

select
    index_symbol,
    ticker,
    cast(as_of as date)        as as_of,
    source,
    cast(confidence as double) as confidence

from {{ raw('index_members') }}
