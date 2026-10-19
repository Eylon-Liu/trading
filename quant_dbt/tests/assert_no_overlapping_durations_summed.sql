-- No two quarters for the same ticker and concept may cover overlapping time.
--
-- This is what stops the Apple case from being summed. Its FY2026 Q3 filing
-- reports NetIncomeLoss twice with the same end date and the same filing date
-- — 29,789M for three months, 101,464M for nine — and the only thing telling
-- them apart is period_start. If the nine-month row ever reaches the quarter
-- pool it overlaps the three quarters it contains, and trailing-twelve-month
-- earnings come out roughly 3.4x too high.
--
-- Checking for overlap directly, rather than for the specific shape of the
-- known bug, means this also catches a six-month gap mistaken for a quarter or
-- a mis-derived cumulative difference.

select
    a.ticker,
    a.concept,
    a.period_start as a_start, a.period_end as a_end,
    b.period_start as b_start, b.period_end as b_end
from {{ ref('int_quarterly_flows') }} as a
join {{ ref('int_quarterly_flows') }} as b
  on  a.ticker  = b.ticker
  and a.concept = b.concept
  -- Order the pair so each overlap is reported once, not twice.
  and (a.period_start, a.period_end) < (b.period_start, b.period_end)
-- Half-open intervals: one quarter ending exactly where the next begins is
-- adjacency, not overlap, and is how consecutive quarters normally meet.
where a.period_start < b.period_end
  and b.period_start < a.period_end
