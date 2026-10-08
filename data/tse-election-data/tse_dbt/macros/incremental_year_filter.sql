{% macro incremental_election_filter(
    year_column='election_year',
    type_column='election_type'
) %}
  {% if is_incremental() %}
    where {{ incremental_partition_predicate(year_column, type_column) }}
  {% endif %}
{% endmacro %}

{% macro incremental_year_filter(column='election_year') %}
  {{ incremental_election_filter(column, 'election_type') }}
{% endmacro %}
