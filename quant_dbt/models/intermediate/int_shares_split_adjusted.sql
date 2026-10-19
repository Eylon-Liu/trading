-- Step 6: put share counts on the same basis as prices, and refuse the ones
-- too old to mean anything.
--
-- Price history is restated after every split, so a 2018 bar is quoted in
-- today's shares. SEC share counts are not restated — they are whatever was
-- filed at the time. Multiplying one by the other understates the market cap
-- of any company that has since split, by exactly the split ratio.
--
-- Lam Research showed a $3.1B market cap in January 2018 against ~$30B actual:
-- a 66% earnings yield and a P/E of 1.5, which made it the top-ranked name in
-- every valuation strategy at every historical date. A single-column unit
-- mismatch, presenting as a brilliant stock picker.
--
-- The ratios applied are those dated after the count was FILED and at or
-- before as_of. Both halves matter. A split the filer had not yet seen cannot
-- be baked into the number they reported; a split that has not happened yet as
-- of the evaluation date is not knowable at all, and applying it would be
-- look-ahead wearing a unit-conversion costume.
--
-- The staleness guard is separate and equally necessary. Exxon stopped filing
-- the diluted share tag in 2013, so the most recent value on file stayed
-- frozen for twelve years while the company kept buying back stock. "Most
-- recent" and "current" are not the same claim, and a count old enough to
-- predate the buybacks it is being divided into is not a measurement any more.
-- NULL propagates visibly; a stale number does not.

{% set max_share_age_days = 500 %}

with share_facts as (

    select ticker, concept, period_end, filed, val
    from {{ ref('int_facts_deduped') }}
    where concept in ('shares_diluted', 'shares_basic', 'shares_outstanding')

),

with_splits as (

    select
        share_facts.*,
        date_diff('day', share_facts.filed, {{ as_of_date() }}) as filing_age_days,
        coalesce(
            (
                select product(splits.ratio)
                from {{ ref('stg_splits') }} as splits
                where splits.ticker = share_facts.ticker
                  and splits.date   > share_facts.filed
                  and {{ pit_gate('splits.date') }}
            ),
            1.0
        ) as split_factor
    from share_facts

)

select
    *,
    case
        when filing_age_days <= {{ max_share_age_days }}
        then val * split_factor
    end as shares_adjusted
from with_splits
