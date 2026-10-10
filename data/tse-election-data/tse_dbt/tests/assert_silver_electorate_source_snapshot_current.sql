with current_snapshot as (

  {{
    source_snapshot_relation(
      'electorate',
      'Eleitorado - %',
      incremental_scope=false
    )
  }}

),

stored_snapshot as (

  select
    election_year,
    election_type,

    count(*) as row_count,

    count(source_snapshot_id)
      as non_null_snapshot_count,

    count(distinct source_snapshot_id)
      as distinct_snapshot_count,

    min(source_snapshot_id)
      as min_source_snapshot_id,

    max(source_snapshot_id)
      as max_source_snapshot_id

  from {{ ref('silver_electorate_municipality') }}

  where {{
    selected_election_predicate(
      'election_year',
      'election_type'
    )
  }}

  group by
    election_year,
    election_type

)

select
  current_snapshot.election_year,
  current_snapshot.election_type,

  current_snapshot.source_snapshot_id
    as expected_source_snapshot_id,

  stored_snapshot.min_source_snapshot_id
    as actual_min_source_snapshot_id,

  stored_snapshot.max_source_snapshot_id
    as actual_max_source_snapshot_id,

  stored_snapshot.row_count,
  stored_snapshot.non_null_snapshot_count,
  stored_snapshot.distinct_snapshot_count

from current_snapshot

left join stored_snapshot
  using (
    election_year,
    election_type
  )

where
  stored_snapshot.row_count is null

  or stored_snapshot.non_null_snapshot_count
     <> stored_snapshot.row_count

  or stored_snapshot.distinct_snapshot_count <> 1

  or stored_snapshot.min_source_snapshot_id
     <> current_snapshot.source_snapshot_id

  or stored_snapshot.max_source_snapshot_id
     <> current_snapshot.source_snapshot_id
