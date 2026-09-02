{% macro normalize_email(column_name) %}

    lower(trim({{ column_name }}))

{% endmacro %}

{% macro amount_for_status(amount_column, status_column, expected_status) %}

    case
        when {{ status_column }} = '{{ expected_status }}'
        then {{ amount_column }}
        else 0
    end

{% endmacro %}


