-- Step 1 of the fact pipeline: restrict to what was knowable, from sources
-- that report in the units everything downstream assumes.
--
-- Both filters belong here, before anything compares rows to each other, and
-- the order of the whole pipeline is the point of this project. Step 3
-- collapses restatements by keeping the newest filing for a period; if a proxy
-- statement is still in the set at that moment, it wins purely by being filed
-- later. DEF 14A reports in thousands, so $3.5B silently became $3,511 for
-- ANET, PCG, ED, SCHW and STZ. The logic in step 3 was never wrong — it just
-- ran against a set that should already have been narrowed.
--
-- The PIT gate is the other half: a figure is knowable on its filing date, not
-- its period end. Apple's quarter ending 2026-06-27 was not public until
-- 2026-07-31.

select *
from {{ ref('stg_sec_facts') }}
where form in (
          {%- for f in var('statement_forms') %}
          '{{ f }}'{{ ',' if not loop.last }}
          {%- endfor %}
      )
  and {{ pit_gate('filed') }}
