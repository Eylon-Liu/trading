{#
    Resolve a raw table to whichever source the active target provides.

    The project runs against two different inputs and the staging models should
    not care which:

      dev   — committed CSV fixtures in seeds/. No external state, so CI can
              build and test the whole pipeline on a clean checkout.
      local — the trading app's own SQLite store, attached read-only, for
              checking the models against real data volume and messiness.

    Putting the switch here rather than in each staging model means the
    difference is stated once. Everything downstream of staging is identical
    under both targets, which is the point: if a model behaved differently
    depending on where its input came from, the dev run would prove nothing
    about the real one.
#}

{% macro raw(table_name) %}
    {%- if target.name == 'local' -%}
        {{ source('quant', table_name) }}
    {%- else -%}
        {{ ref('raw_' ~ table_name) }}
    {%- endif -%}
{% endmacro %}
