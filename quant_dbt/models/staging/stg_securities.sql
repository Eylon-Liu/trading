-- Security master — one row per ticker.

select
    ticker,
    cik,
    name,
    sector,
    industry,
    exchange,
    country

from {{ raw('securities') }}
