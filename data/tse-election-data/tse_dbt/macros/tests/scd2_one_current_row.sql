{% test scd2_one_current_row(
    model,
    business_key,
    is_current='is_current'
) %}
select
    {{ business_key }} as business_key,
    count(*) filter (where {{ is_current }}) as current_rows
from {{ model }}
group by 1
having current_rows <> 1
{% endtest %}
