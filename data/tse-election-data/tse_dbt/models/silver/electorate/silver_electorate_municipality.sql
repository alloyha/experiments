{{ config(
    pre_hook=[
      ensure_source_snapshot_column_pre_hook(),
      source_aware_partition_replace_pre_hook(
        'electorate',
        'Eleitorado - %'
      )
    ],
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=[
      'election_year',
      'election_type',
      'uf',
      'municipality_code'
    ],
    on_schema_change='sync_all_columns',
    meta={
      'load_semantics': 'source_aware_partition_replace',
      'history_semantics': 'derived_snapshot',
      'partition_key': [
        'election_year',
        'election_type'
      ],
      'source_snapshot': 'current_objects.source_sha256',
      'resource_class': 'heavy_source_aware'
    }
) }}


{% set should_refresh =
    source_snapshot_changed(
      'electorate',
      'Eleitorado - %'
    )
%}


{% if is_incremental() and not should_refresh %}

  {#
    Source snapshot is identical to the persisted partition.

    Crucially this branch does not reference bronze_electorate, so DuckDB
    cannot expand the view into the 11M-row READ_CSV.
  #}

  select *
  from {{ this }}
  where false


{% else %}

with current_snapshot as (

  {{
    source_snapshot_relation(
      'electorate',
      'Eleitorado - %',
      incremental_scope=is_incremental()
    )
  }}

),

source_rows as (

  select
    source.*,
    current_snapshot.source_snapshot_id

  from {{ ref('bronze_electorate') }} as source

  inner join current_snapshot
    using (
      election_year,
      election_type
    )

  {% if is_incremental() %}
  where {{
    incremental_partition_predicate(
      'election_year',
      'election_type'
    )
  }}
  {% endif %}

)

select
    election_year,
    election_type,
    election_scope,

    uf,
    municipality_code,
    municipality,

    sum(electorate) as electorate,

    max(source_snapshot_id)
      as source_snapshot_id

from source_rows

group by
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality

{% endif %}
