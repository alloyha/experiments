-- macros/get_date_trunc.sql

{% macro get_date_trunc(column, date_part='day') %}
    DATE_TRUNC('{{ date_part }}', {{ column }})
{% endmacro %}

-- macros/cents_to_dollars.sql

{% macro cents_to_dollars(column) %}
    {{ column }} / 100.0
{% endmacro %}

-- macros/generate_surrogate_key.sql

{% macro generate_surrogate_key(columns) %}
    MD5(CONCAT(
    {%- for column in columns -%}
        CAST({{ column }} AS VARCHAR)
        {%- if not loop.last %}, {% endif -%}
    {%- endfor -%}
    ))
{% endmacro %}

-- macros/generate_alias_hash.sql

{% macro generate_alias_hash(column_name) %}
    SUBSTR(MD5({{ column_name }}), 1, 8)
{% endmacro %}
