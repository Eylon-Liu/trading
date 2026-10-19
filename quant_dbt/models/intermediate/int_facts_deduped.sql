-- Step 3: collapse restatements to one row per reported period.
--
-- A period is re-reported as figures are revised, so the same (concept,
-- period) carries several rows with different filing dates. The correct row is
-- the newest one *that was already knowable* — which is why the PIT gate had
-- to run in step 1. "Newest" here means newest as of the as_of date, not
-- newest ever; a figure revised in August is invisible to a screen dated June,
-- and that is the whole point of the exercise.
--
-- Note what is NOT happening: no column-wise aggregation. Taking a max or a
-- last-non-null per column would assemble a row from several filings, silently
-- backfilling a field the newest filing left empty with a value from an older
-- one — producing a row that was never actually filed. QUALIFY keeps one
-- coherent filing intact.
--
-- The whole of quant/fundamentals.py::_latest_filed, which exists to work
-- around exactly that pandas behaviour, is this one clause.

select *
from {{ ref('int_facts_tagged') }}
qualify row_number() over (
    partition by ticker, concept, period_start, period_end
    -- form and val only break ties between rows filed the same day, so the
    -- result does not depend on scan order.
    order by filed desc, form, val desc
) = 1
