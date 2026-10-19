-- Step 7: one adjustment basis per bar.
--
-- The feed gives `high` and `low` unadjusted while `adj_close` is back-adjusted
-- for splits and dividends, so the columns of a single bar disagree about what
-- they are denominated in. Scaling the extremes by adj_close/close puts the
-- whole bar on the adjusted basis.
--
-- Chevron in early 2023: a raw high of $174.63 against an adjusted close of
-- $150.55. Every stop and target derived from adjusted closes sat far below
-- the raw bar, so each one looked like it triggered the instant it was placed.
--
-- `raw_close` is kept deliberately and is the correct input for market
-- capitalisation, because share counts are restated for splits separately (see
-- int_shares_split_adjusted). Using the adjusted close there would remove the
-- dividend adjustment too and quietly understate the capitalisation of every
-- dividend payer.

select
    ticker,
    date,
    adj_close,
    high * adj_ratio as adj_high,
    low  * adj_ratio as adj_low,
    close            as raw_close,
    adj_ratio,
    volume
from {{ ref('stg_prices') }}
where {{ pit_gate('date') }}
