-- The consumable table: one row per ticker, as of a single date.
--
-- Everything here was knowable on `as_of`. Change the `as_of` var and the
-- whole table moves to what was visible then — restatements disappear,
-- unfiled quarters vanish, splits stop applying. That is the property the
-- pipeline exists to guarantee, and tests/assert_no_lookahead_in_pit_mart.sql
-- asserts it rather than trusting it.
--
-- Ratios are computed here rather than left to the consumer so the guards
-- travel with them. A denominator near zero produces a number that is
-- arithmetically correct and economically meaningless — GoDaddy's sliver of
-- positive equity yields a 13,587% return on it — and the guard belongs with
-- the definition, not in whatever reads it next.

with ttm as (

    select
        ticker,
        max(case when concept = 'revenue'    then ttm_val end) as ttm_revenue,
        max(case when concept = 'net_income' then ttm_val end) as ttm_net_income,
        max(last_filed)      as last_fact_filed,
        max(last_period_end) as last_period_end
    from {{ ref('int_ttm_fundamentals') }}
    group by ticker

),

stocks as (

    select
        ticker,
        max(case when concept = 'equity' then val end) as equity,
        max(case when concept = 'assets' then val end) as assets
    from {{ ref('int_latest_stocks') }}
    group by ticker

),

shares as (

    select ticker, shares_adjusted, split_factor
    from {{ ref('int_shares_split_adjusted') }}
    where concept = 'shares_diluted'

),

latest_price as (

    select ticker, raw_close, adj_close, date as price_date
    from {{ ref('int_adjusted_prices') }}
    qualify row_number() over (partition by ticker order by date desc) = 1

),

joined as (

    select
        securities.ticker,
        {{ as_of_date() }} as as_of,
        securities.name,
        securities.sector,

        ttm.ttm_revenue,
        ttm.ttm_net_income,
        stocks.equity,
        stocks.assets,
        shares.shares_adjusted  as shares_diluted,
        shares.split_factor,
        latest_price.raw_close,
        latest_price.price_date,

        -- Raw close, split-adjusted shares. Both halves are load-bearing; see
        -- int_shares_split_adjusted.
        latest_price.raw_close * shares.shares_adjusted as market_cap,

        ttm.last_fact_filed,
        ttm.last_period_end

    from {{ ref('stg_securities') }} as securities
    left join ttm          on ttm.ticker          = securities.ticker
    left join stocks       on stocks.ticker       = securities.ticker
    left join shares       on shares.ticker       = securities.ticker
    left join latest_price on latest_price.ticker = securities.ticker

)

select
    *,

    -- Earnings yield, not dollar EPS: trailing EPS of $10 says nothing about
    -- cheapness without a price attached, and is not comparable across names.
    ttm_net_income / nullif(market_cap, 0) as earnings_yield,

    ttm_net_income / nullif(ttm_revenue, 0) as net_margin,

    -- Equity below 1% of assets is a rounding artefact, not a capital base.
    case
        when equity > 0.01 * assets then ttm_net_income / nullif(equity, 0)
    end as return_on_equity

from joined
