{#
    Classify an XBRL fact's reporting span from its day count.

    Flow concepts (revenue, net income, cash flow) are measured *over* a span,
    and filers report the same concept over several overlapping spans in the
    same filing. Apple's FY2026 Q3 10-Q carries NetIncomeLoss twice with an
    identical end date of 2026-06-27 and an identical filing date: 29,789M for
    the three months and 101,464M for the nine months to date. Nothing but the
    span length separates them, so classifying the span is the first thing that
    has to happen — summing without it overstates trailing-twelve-month
    earnings by roughly 3.4x and corrupts every ratio built on top.

    The windows are wide because fiscal quarters are not uniform: a 4-4-5
    retail calendar produces 91-day quarters that drift to 98, and a 52/53-week
    year produces a 371-day "annual". Anything outside the windows is 'other'
    and is discarded rather than guessed at.

    Mirrors quant/fundamentals.py::_classify_duration in the trading repo.
#}

{% macro classify_duration(days_expression) %}
    case
        when {{ days_expression }} between  60 and 115 then 'Q'
        when {{ days_expression }} between 150 and 220 then 'H'
        when {{ days_expression }} between 240 and 310 then '9M'
        when {{ days_expression }} between 330 and 400 then 'A'
        else 'other'
    end
{% endmacro %}
