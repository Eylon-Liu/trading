-- The invariant the whole pipeline exists to guarantee: nothing that feeds the
-- as-of view may have been filed after the as-of date.
--
-- This is the failure mode that does not announce itself. A leak does not
-- raise, does not produce an obviously wrong number, and does not change the
-- row count — it just makes every backtest better than reality by an amount
-- nobody can measure after the fact.
--
-- Checked against the models that actually feed the mart rather than against
-- the mart's own provenance columns, because a bug that dropped a row from
-- those columns would make the cheaper check pass while the leak continued.

select 'quarterly_flows' as model, ticker, concept, filed
from {{ ref('int_quarterly_flows') }}
where filed > {{ as_of_date() }}

union all

select 'latest_stocks', ticker, concept, filed
from {{ ref('int_latest_stocks') }}
where filed > {{ as_of_date() }}

union all

select 'adjusted_prices', ticker, 'price', date
from {{ ref('int_adjusted_prices') }}
where date > {{ as_of_date() }}
