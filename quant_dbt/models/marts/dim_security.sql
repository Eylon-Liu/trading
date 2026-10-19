-- Security dimension: identity and classification, one row per ticker.
--
-- Sector comes from the SEC's SIC codes rather than a provider's `info`
-- endpoint. SIC is coarser than GICS and imperfect at the margins, but it
-- arrives with the filings and is always there; the provider endpoint is the
-- most heavily rate-limited source in the stack, and when it returned nothing
-- every name classified as Unknown and the sector-relative z-scores collapsed
-- into one meaningless bucket. A mostly-right sector beats a universally
-- absent one.

select
    ticker,
    cik,
    name,
    sector,
    industry,
    exchange,
    country
from {{ ref('stg_securities') }}
