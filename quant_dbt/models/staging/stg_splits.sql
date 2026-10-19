-- Split events, typed.
--
-- Rare, but they cannot be ignored: the price feed restates history after a
-- split while SEC share counts are whatever was filed at the time. Reconciling
-- the two is int_shares_split_adjusted's job; this model just types the input.

select
    ticker,
    cast(date as date)   as date,
    cast(ratio as double) as ratio    -- 10.0 for a 10-for-1 forward split

from {{ raw('splits') }}
