{% test scd2_no_overlaps(
    model,
    business_key,
    valid_from='valid_from',
    valid_to='valid_to'
) %}
with ordered as (
    select
        {{ business_key }} as business_key,
        {{ valid_from }} as valid_from,
        {{ valid_to }} as valid_to,
        lag({{ valid_to }}) over (
            partition by {{ business_key }}
            order by {{ valid_from }}
        ) as previous_valid_to
    from {{ model }}
)
select *
from ordered
where previous_valid_to is not null
  and valid_from < previous_valid_to
{% endtest %}
