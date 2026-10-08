{% macro quoted_sql_list(values) %}
  {% set quoted = [] %}
  {% for value in values %}
    {% do quoted.append("'" ~ (value | string | replace("'", "''")) ~ "'") %}
  {% endfor %}
  {{ return(quoted | join(', ')) }}
{% endmacro %}

{% macro incremental_partition_predicate(
    year_column='election_year',
    type_column='election_type'
) %}
  {% set years = var('incremental_years', election_year_list()) %}
  {% set types = var('incremental_election_types', election_type_list()) %}

  {{ return(
    year_column ~ ' in (' ~ (years | join(', ')) ~ ')' ~
    ' and ' ~ type_column ~ ' in (' ~ quoted_sql_list(types) ~ ')'
  ) }}
{% endmacro %}

{% macro partition_replace_pre_hook(
    year_column='election_year',
    type_column='election_type'
) %}
  {#
    TSE analytical resources are authoritative snapshots, not row deltas.

    dbt's ordinary delete+insert keyed by a business key cannot remove a row
    that disappeared from a republished source snapshot.  Therefore every
    incremental run first deletes the selected election partition, then the
    model inserts the complete current snapshot for that same partition.

    Full-refresh runs do not need this hook because dbt replaces the relation.
  #}
  {% if is_incremental() %}
    {{ return(
      'delete from ' ~ this ~
      ' where ' ~ incremental_partition_predicate(year_column, type_column)
    ) }}
  {% else %}
    {{ return('') }}
  {% endif %}
{% endmacro %}

{% macro selected_election_predicate(
    year_column='election_year',
    type_column='election_type'
) %}
  {% set years = election_year_list() %}
  {% set types = election_type_list() %}
  {{ return(
    year_column ~ ' in (' ~ (years | join(', ')) ~ ')' ~
    ' and ' ~ type_column ~ ' in (' ~ quoted_sql_list(types) ~ ')'
  ) }}
{% endmacro %}
