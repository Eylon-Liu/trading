-- Balance-sheet values: the most recent observation that was knowable.
--
-- Stocks are measured at an instant, so they need neither duration matching
-- nor summing — just the latest period end, resolved to the newest filing for
-- that period. Restatement collapsing already happened upstream, so this is
-- only the recency pick.

{% set stock_concepts = [
    'assets', 'current_assets', 'liabilities', 'current_liabilities',
    'equity', 'cash', 'long_term_debt', 'short_term_debt', 'total_debt',
    'inventory', 'shares_diluted', 'shares_basic', 'shares_outstanding'
] %}

select
    ticker,
    concept,
    period_end,
    filed,
    val
from {{ ref('int_facts_deduped') }}
where concept in (
          {%- for c in stock_concepts %}
          '{{ c }}'{{ ',' if not loop.last }}
          {%- endfor %}
      )
qualify row_number() over (
    partition by ticker, concept
    order by period_end desc, filed desc
) = 1
