{{ config(
    materialized='view',
    meta={
      'load_semantics': 'logical_projection',
      'history_semantics': 'current_source_snapshot',
      'partition_key': ['election_year', 'election_type'],
      'resource_class': 'logical_projection_view'
    }
) }}

select *
from {{ ref('bronze_candidate_votes_raw') }}
