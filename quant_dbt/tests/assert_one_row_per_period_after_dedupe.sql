-- After restatement collapsing there is exactly one row per reported period.
--
-- The grain claim the rest of the pipeline is built on. If it fails, every
-- downstream sum silently double-counts the periods that were revised — which
-- is to say the most-scrutinised ones.

select
    ticker,
    concept,
    period_start,
    period_end,
    count(*) as n_rows
from {{ ref('int_facts_deduped') }}
group by ticker, concept, period_start, period_end
having count(*) > 1
