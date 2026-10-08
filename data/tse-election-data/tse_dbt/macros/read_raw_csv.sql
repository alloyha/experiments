{% macro read_raw_csv(domain, resource_name_like=none) %}
  {% if var('compile_only', false) %}
    {% set fixture = var('dbt_fixture_root', 'tse_dbt/fixtures') ~ '/' ~ domain ~ '.csv' %}
    (
      select
        *,
        cast(null as varchar) as _election_type,
        cast(null as varchar) as _election_scope
      from read_csv(
        '{{ fixture }}',
        delim = ',',
        header = true,
        all_varchar = true,
        union_by_name = true,
        filename = true,
        sample_size = 20480,
        encoding = 'utf-8',
        ignore_errors = false
      )
      where false
    )
  {% else %}
    {% set paths = current_raw_files(domain, resource_name_like) %}
    {% set index_path = var('tse_raw_root') ~ '/_metadata/current_objects.jsonl' %}
    (
      with _index as (
        select distinct
          '{{ var("tse_raw_root") }}' || '/' || object as object_path,
          year as _index_year,
          election_type as _election_type,
          election_scope as _election_scope
        from read_json_auto('{{ index_path }}')
        where domain = '{{ domain }}'
          and year in ({{ election_year_list() | join(', ') }})
          and election_type in ({{ quoted_sql_list(election_type_list()) }})
          {% if resource_name_like is not none %}
          and resource_name ilike '{{ resource_name_like | replace("'", "''") }}'
          {% endif %}
      ),
      _raw as (
        select *
        from read_csv(
          [
            {% for path in paths %}'{{ path | replace("'", "''") }}'{% if not loop.last %},{% endif %}{% endfor %}
          ],
          delim = ';',
          quote = '"',
          escape = '"',
          header = true,
          all_varchar = true,
          union_by_name = true,
          filename = true,
          sample_size = 20480,
          encoding = '{{ var("tse_csv_encoding", "latin-1") }}',
          ignore_errors = false
        )
      )
      select
        _raw.*,
        _index._election_type,
        _index._election_scope
      from _raw
      left join _index
        on replace(_raw.filename, '\\', '/') = replace(_index.object_path, '\\', '/')
    )
  {% endif %}
{% endmacro %}
