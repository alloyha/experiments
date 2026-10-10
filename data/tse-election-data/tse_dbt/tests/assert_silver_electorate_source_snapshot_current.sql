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

  stored_snapshot.source_snapshot_id
    as actual_source_snapshot_id

from current_snapshot

left join stored_snapshot
  using (
    election_year,
    election_type
  )

where
  stored_snapshot.source_snapshot_id is null

  or stored_snapshot.source_snapshot_id
     <> current_snapshot.source_snapshot_id
