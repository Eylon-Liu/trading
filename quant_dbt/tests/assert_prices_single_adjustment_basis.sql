-- Within one bar, the close must lie between the high and the low.
--
-- Trivially true of any single basis and reliably false when two are mixed,
-- which is what makes it the right assertion. The feed returns high and low
-- unadjusted while adj_close is back-adjusted for splits and dividends, so an
-- unscaled comparison puts the close outside its own bar: Chevron in early
-- 2023 had a raw high of $174.63 against an adjusted close of $150.55.
--
-- The consequence was not a wrong number but a wrong conclusion — stops and
-- targets derived from adjusted closes sat below every raw bar, so each one
-- appeared to trigger instantly and a Sharpe of 1.6 came out as 3.3.
--
-- The tolerance absorbs float error in the ratio multiplication, nothing more;
-- a genuine basis mismatch is off by percent, not by 1e-9.

select
    ticker,
    date,
    adj_low,
    adj_close,
    adj_high
from {{ ref('int_adjusted_prices') }}
where adj_close > adj_high + 1e-9
   or adj_close < adj_low  - 1e-9
