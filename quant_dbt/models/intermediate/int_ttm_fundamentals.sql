-- Step 5: trailing-twelve-month totals for flow concepts.
--
-- Four quarters, or an annual figure, or nothing. Deliberately not "whatever
-- is available, scaled up": annualising from one quarter invents three, and
-- the invented number looks exactly like a real one downstream. Returning
-- nothing is the honest answer, and a name dropping out of a screen is a
-- visible failure where a fabricated TTM is an invisible one.

with quarters as (

    select
        *,
        row_number() over (
            partition by ticker, concept order by period_end desc
        ) as recency,
        count(*) over (partition by ticker, concept) as quarters_available
    from {{ ref('int_quarterly_flows') }}
    where val is not null

),

from_quarters as (

    select
        ticker,
        concept,
        sum(val)         as ttm_val,
        max(filed)       as last_filed,
        max(period_end)  as last_period_end,
        count(*)         as quarters_used,
        'quarters'       as ttm_source
    from quarters
    where recency <= 4
      and quarters_available >= 4
    group by ticker, concept

),

annual_fallback as (

    -- Sparse coverage — an early filer, or a foreign issuer that files 20-F
    -- annually and never quarterly. The newest annual figure is a real TTM,
    -- just a staler one, so it is labelled rather than silently blended.
    select
        ticker,
        concept,
        val             as ttm_val,
        filed           as last_filed,
        period_end      as last_period_end,
        1               as quarters_used,
        'annual'        as ttm_source
    from {{ ref('int_facts_deduped') }}
    where duration_class = 'A'
      and ticker || '|' || concept not in (
          select ticker || '|' || concept from from_quarters
      )
    qualify row_number() over (
        partition by ticker, concept order by period_end desc, filed desc
    ) = 1

)

select * from from_quarters
union all
select * from annual_fallback
