-- A share count too old to be current must come through as NULL.
--
-- Exxon stopped filing the diluted share tag in 2013, so the newest value on
-- file stayed frozen for twelve years while the company kept buying back
-- stock. "Most recent" and "current" are different claims, and the gap between
-- them grows silently — nothing in the data says the number stopped being
-- updated.
--
-- Everything built on a share count inherits the error: market cap, earnings
-- yield, every valuation factor. NULL removes the name from those rankings,
-- which is visible. A twelve-year-old denominator is not.
--
-- Asserted at the mart rather than at the intermediate model, because what
-- matters is that the rejection survives the joins rather than being quietly
-- refilled by one of them.

select
    mart.ticker,
    mart.shares_diluted,
    stale.filing_age_days
from {{ ref('fct_factor_inputs_pit') }} as mart
join {{ ref('int_shares_split_adjusted') }} as stale
  on  stale.ticker  = mart.ticker
  and stale.concept = 'shares_diluted'
where stale.filing_age_days > 500
  and mart.shares_diluted is not null
