{% macro election_year_list() %}
  {% set years = var(
      'election_years',
      [var('election_year', 2026)]
  ) %}
  {{ return(years) }}
{% endmacro %}


{% macro election_type_list() %}
  {% set types = var(
      'election_types',
      ['general', 'municipal']
  ) %}
  {{ return(types) }}
{% endmacro %}


{% macro current_raw_files(
    domain,
    resource_name_like=none,
    incremental_scope=false
) %}
  {#
    Compile mode must be independent from ingestion state.
  #}
  {% if var('compile_only', false) %}
    {{
      return([
        var(
          'dbt_fixture_root',
          'tse_dbt/fixtures'
        ) ~ '/' ~ domain ~ '.csv'
      ])
    }}
  {% endif %}

  {% if not execute %}
    {{
      return([
        var(
          'dbt_fixture_root',
          'tse_dbt/fixtures'
        ) ~ '/' ~ domain ~ '.csv'
      ])
    }}
  {% endif %}

  {% if incremental_scope %}
    {% set years = var(
        'incremental_years',
        election_year_list()
    ) %}
    {% set types = var(
        'incremental_election_types',
        election_type_list()
    ) %}
  {% else %}
    {% set years = election_year_list() %}
    {% set types = election_type_list() %}
  {% endif %}

  {% set year_sql = years | join(', ') %}
  {% set quoted_types = [] %}

  {% for t in types %}
    {% do quoted_types.append(
      "'" ~ (t | replace("'", "''")) ~ "'"
    ) %}
  {% endfor %}

  {% set index_path =
      var('tse_raw_root')
      ~ '/_metadata/current_objects.jsonl'
  %}

  {% set query %}
    select distinct object

    from read_json_auto(
      '{{ index_path | replace("'", "''") }}'
    )

    where domain =
      '{{ domain | replace("'", "''") }}'

      and year in (
        {{ year_sql }}
      )

      and election_type in (
        {{ quoted_types | join(', ') }}
      )

      {% if resource_name_like is not none %}
      and resource_name ilike
        '{{ resource_name_like | replace("'", "''") }}'
      {% endif %}

    order by
      election_type,
      year,
      resource_id,
      object
  {% endset %}

  {% set result = run_query(query) %}
  {% set paths = [] %}

  {% if result is not none %}
    {% for row in result.rows %}
      {% do paths.append(
        var('tse_raw_root')
        ~ '/'
        ~ row[0]
      ) %}
    {% endfor %}
  {% endif %}

  {% if paths | length == 0 %}
    {{
      exceptions.raise_compiler_error(
        "No active raw objects found for domain='" ~
        domain ~
        "', resource_name_like=" ~
        (resource_name_like | string) ~
        ", years=" ~
        (years | string) ~
        ", election_types=" ~
        (types | string) ~
        ". Run tse_ingest.py first or adjust vars."
      )
    }}
  {% endif %}

  {{ return(paths) }}
{% endmacro %}
