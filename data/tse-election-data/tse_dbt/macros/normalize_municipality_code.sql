{% macro normalize_municipality_code(value) %}
case
    when {{ value }} is null then null
    when trim(cast({{ value }} as varchar)) = '' then null
    else lpad(trim(cast({{ value }} as varchar)), 5, '0')
end
{% endmacro %}
