{#
    The point-in-time gate.

    A figure becomes knowable on the date it was *filed*, not the date the
    period ended. Apple's quarter ending 2026-06-27 was not public until
    2026-07-31 — a five-week gap. Gating on period_end instead of filed is the
    single most common way a backtest ends up trading on information that did
    not exist, and it inflates results silently rather than failing.

    Every model that reads a filing-dated table applies this, and
    tests/assert_no_lookahead_in_pit_mart.sql asserts that nothing slipped
    past it.

    Mirrors the `filed <= :ts` gate in data/sec.py::facts_asof.
#}

{% macro pit_gate(column='filed') %}
    {{ column }} <= cast('{{ var("as_of") }}' as date)
{% endmacro %}


{#
    The as_of date itself, as a typed literal — for select lists and joins
    where the gate macro would read oddly.
#}
{% macro as_of_date() %}
    cast('{{ var("as_of") }}' as date)
{% endmacro %}
