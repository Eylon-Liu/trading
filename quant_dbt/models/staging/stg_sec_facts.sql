-- Raw XBRL facts, typed and with the reporting span made explicit.
--
-- Staging does no business logic: rename, cast, and derive only what is purely
-- mechanical. The span classification lands here because it is a function of
-- two columns in the same row and nothing else — every decision that depends
-- on comparing rows happens downstream.
--
-- The casts are not ceremony. The application writes dates through SQLAlchemy
-- into SQLite columns declared DATE, which stores them as 'YYYY-MM-DD' text;
-- read back without a cast they compare lexicographically, which happens to
-- work for ISO dates right up until the first piece of arithmetic.

select
    cik,
    ticker,
    tag,
    concept,
    unit,
    cast(period_start as date)  as period_start,
    cast(period_end   as date)  as period_end,
    cast(filed        as date)  as filed,
    form,
    cast(fy as integer)         as fy,
    fp,
    cast(val as double)         as val,

    date_diff('day', cast(period_start as date), cast(period_end as date))
        as duration_days,

    {{ classify_duration(
        "date_diff('day', cast(period_start as date), cast(period_end as date))"
    ) }}                        as duration_class

from {{ raw('sec_facts') }}
