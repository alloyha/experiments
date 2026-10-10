{% macro source_snapshot_relation(
    domain,
    resource_name_like=none,
    incremental_scope=false
) %}
  {% if var('compile_only', false) %}
    select
      cast(null as integer) as election_year,
      cast(null as varchar) as election_type,
      cast(null as varchar) as source_snapshot_id
    where false
  {% else %}

    {% if incremental_scope %}
      {% set years = var('incremental_years', election_year_list()) %}
      {% set types = var('incremental_election_types', election_type_list()) %}
    {% else %}
      {% set years = election_year_list() %}
      {% set types = election_type_list() %}
    {% endif %}

    {% set index_path =
        var('tse_raw_root') ~ '/_metadata/current_objects.jsonl'
    %}

    with selected as (
      select distinct
        cast(year as integer) as election_year,
        election_type,

        coalesce(resource_id, '') || ':' ||
        coalesce(source_sha256, '') || ':' ||
        coalesce(object, '') as snapshot_component

      from read_json_auto(
        '{{ index_path | replace("'", "''") }}'
      )

      where domain = '{{ domain | replace("'", "''") }}'

        and year in (
          {{ years | join(', ') }}
        )

        and election_type in (
          {{ quoted_sql_list(types) }}
        )

        {% if resource_name_like is not none %}
        and resource_name ilike
          '{{ resource_name_like | replace("'", "''") }}'
        {% endif %}
    )

    select
      election_year,
      election_type,

      md5(
        string_agg(
          snapshot_component,
          '|' order by snapshot_component
        )
      ) as source_snapshot_id

    from selected

    group by
      election_year,
      election_type

  {% endif %}
{% endmacro %}


{% macro ensure_source_snapshot_column_pre_hook(
    column_name='source_snapshot_id'
) %}
  {% if is_incremental() %}

    alter table {{ this }}
      add column if not exists
      {{ adapter.quote(column_name) }} varchar

  {% endif %}
{% endmacro %}


{% macro source_aware_partition_replace_pre_hook(
    domain,
    resource_name_like=none,
    year_column='election_year',
    type_column='election_type',
    snapshot_column='source_snapshot_id'
) %}
  {% if is_incremental() %}

    delete from {{ this }} as target

    using (
      {{
        source_snapshot_relation(
          domain,
          resource_name_like,
          incremental_scope=true
        )
      }}
    ) as current_snapshot

    where target.{{ year_column }}
            = current_snapshot.election_year

      and target.{{ type_column }}
            = current_snapshot.election_type

      and coalesce(
            target.{{ snapshot_column }},
            ''
          )
          <> current_snapshot.source_snapshot_id

  {% endif %}
{% endmacro %}


{% macro source_snapshot_changed(
    domain,
    resource_name_like=none,
    snapshot_column='source_snapshot_id'
) %}

  {#
    Parse/compile must remain independent of warehouse state.
    A full refresh also necessarily reads the source.
  #}
  {% if
      var('compile_only', false)
      or not execute
      or not is_incremental()
  %}
    {{ return(true) }}
  {% endif %}

  {#
    Existing warehouses predate source_snapshot_id.

    The first run after this migration must therefore refresh the selected
    partition. The pre-hook adds the column before model execution.
  #}
  {% set column_query %}

    select count(*)

    from information_schema.columns

    where table_schema =
      '{{ this.schema | replace("'", "''") }}'

      and table_name =
      '{{ this.identifier | replace("'", "''") }}'

      and column_name =
      '{{ snapshot_column | replace("'", "''") }}'

  {% endset %}

  {% set column_result = run_query(column_query) %}

  {% if
      column_result is none
      or (column_result.rows[0][0] | int) == 0
  %}
    {{ return(true) }}
  {% endif %}


  {% set change_query %}

    with current_snapshot as (

      {{
        source_snapshot_relation(
          domain,
          resource_name_like,
          incremental_scope=true
        )
      }}

    ),

    existing_snapshot as (

      select
        election_year,
        election_type,

        count(*) as row_count,

        count({{ snapshot_column }})
          as non_null_snapshot_count,

        count(distinct {{ snapshot_column }})
          as distinct_snapshot_count,

        min({{ snapshot_column }})
          as min_source_snapshot_id,

        max({{ snapshot_column }})
          as max_source_snapshot_id

      from {{ this }}

      where {{
        incremental_partition_predicate(
          'election_year',
          'election_type'
        )
      }}

      group by
        election_year,
        election_type

    )

    select
      count(*) as current_count,

      sum(
        case
          when existing_snapshot.row_count is null
          then 1

          when existing_snapshot.non_null_snapshot_count
               <> existing_snapshot.row_count
          then 1

          when existing_snapshot.distinct_snapshot_count <> 1
          then 1

          when existing_snapshot.min_source_snapshot_id is null
          then 1

          when existing_snapshot.min_source_snapshot_id
               <> current_snapshot.source_snapshot_id
          then 1

          when existing_snapshot.max_source_snapshot_id
               <> current_snapshot.source_snapshot_id
          then 1

          else 0
        end
      ) as changed_count

    from current_snapshot

    left join existing_snapshot
      using (
        election_year,
        election_type
      )

  {% endset %}

  {% set result = run_query(change_query) %}

  {% if result is none %}
    {{ return(true) }}
  {% endif %}

  {% set current_count =
      result.rows[0][0] | int
  %}

  {% if current_count == 0 %}

    {{ exceptions.raise_compiler_error(
      "No current source snapshot found for domain='" ~
      domain ~
      "', resource_name_like=" ~
      (resource_name_like | string)
    ) }}

  {% endif %}

  {% set changed_raw = result.rows[0][1] %}

  {% if changed_raw is none %}
    {{ return(true) }}
  {% endif %}

  {{ return((changed_raw | int) > 0) }}

{% endmacro %}
