-- Step 4: recover genuine single-quarter observations for flow concepts.
--
-- This is the model the project exists to show. The original is a row-wise
-- Python loop (quant/fundamentals.py::_quarterly_flows_uncached, ~60 lines)
-- that walks each filer's cumulative filings in order, subtracting as it goes.
-- Expressed as window functions it is one pass and no loop at all.
--
-- Three things happen here:
--
-- 1. Take the facts already classified as a single quarter. This is where the
--    Apple case resolves: FY2026 Q3 reports NetIncomeLoss twice with the same
--    end date (2026-06-27) and the same filing date (2026-07-31) — 29,789M for
--    the quarter and 101,464M for the nine months. The span classification in
--    staging marks the second one '9M', so it never enters the quarter pool.
--    Summing both overstates trailing-twelve-month earnings by ~3.4x.
--
-- 2. Derive the quarters that were only ever reported cumulatively. Q4 is the
--    usual case — many filers publish a full year and never a fourth quarter —
--    and some report year-to-date all the way through. Differencing
--    consecutive cumulative figures within the same fiscal year recovers them.
--
-- 3. Flag implausible derived quarters rather than trusting the arithmetic.
--
-- The filing date of a derived quarter is the LATER of its two constituent
-- filings. It cannot be known until both exist, so inheriting the earlier date
-- would backdate a figure into a window where nobody could have computed it.
-- A subtraction that is arithmetically right can still be a look-ahead leak.

{% set flow_concepts = [
    'revenue', 'net_income', 'operating_income', 'gross_profit',
    'operating_cash_flow', 'capex', 'interest_expense', 'rnd',
    'dividends_paid', 'buybacks', 'eps_diluted'
] %}

with flows as (

    select *
    from {{ ref('int_facts_deduped') }}
    where concept in (
              {%- for c in flow_concepts %}
              '{{ c }}'{{ ',' if not loop.last }}
              {%- endfor %}
          )

),

reported_quarters as (

    select
        ticker, concept, period_start, period_end, filed, fy, fp, val,
        false as is_derived
    from flows
    where duration_class = 'Q'

),

cumulative as (

    -- Partitioning by period_start is what keeps this within one fiscal year:
    -- cumulative figures all share the year's opening date, so consecutive
    -- rows are Q1, H1, 9M, FY and the gaps between them are the quarters.
    select
        ticker, concept, period_start, period_end, filed, fy, fp, val,
        lag(val)        over w as prev_val,
        lag(period_end) over w as prev_period_end,
        lag(filed)      over w as prev_filed
    from flows
    where duration_class in ('H', '9M', 'A')
    window w as (
        partition by ticker, concept, period_start
        order by period_end
    )

),

derived_quarters as (

    select
        ticker,
        concept,
        prev_period_end            as period_start,
        period_end,
        greatest(filed, prev_filed) as filed,
        fy,
        fp,
        val - prev_val             as val,
        true                       as is_derived
    from cumulative
    where prev_period_end is not null
      -- Only accept a gap that is actually a quarter. A missing interim filing
      -- leaves a six-month hole, and calling that a quarter would double the
      -- figure.
      and date_diff('day', prev_period_end, period_end) between 60 and 115

),

combined as (

    select * from reported_quarters
    union all
    select * from derived_quarters

),

one_per_period as (

    select *
    from combined
    -- A reported quarter beats a derived one filed the same day: the filer
    -- stated it, rather than it falling out of a subtraction. (The Python
    -- original resolves this tie by concatenation order, which happens to
    -- prefer the derived row — an accident of implementation rather than a
    -- decision.)
    qualify row_number() over (
        partition by ticker, concept, period_end
        order by filed desc, is_derived asc
    ) = 1

),

checked as (

    select
        *,
        lag(val, 4) over (
            partition by ticker, concept order by period_end
        ) as val_year_ago
    from one_per_period

)

select
    ticker,
    concept,
    period_start,
    period_end,
    filed,
    fy,
    fp,
    is_derived,

    -- A derived quarter that both flips sign and moves by more than 10x
    -- against the same quarter a year earlier is far more likely to be two
    -- misaligned restatements differenced against each other than a real
    -- economic event. NULL says "unknown", which propagates visibly; keeping
    -- the number would let an artefact masquerade as a result.
    case
        when is_derived
             and val_year_ago is not null
             and val_year_ago <> 0
             and abs(val / val_year_ago) > 10
             and sign(val) <> sign(val_year_ago)
        then null
        else val
    end as val

from checked
