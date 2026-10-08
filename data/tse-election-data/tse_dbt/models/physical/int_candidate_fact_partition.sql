{{ config(
    materialized='table',
    enabled=var('build_candidate_fact_partition', false),
    meta={
        'load_semantics': 'full_snapshot_ctas',
        'history_semantics': 'authoritative_snapshot',
        'partition_key': ['election_year', 'election_type'],
        'physical_publish': 'candidate_fact_partition'
    }
) }}

select *
from {{ ref('silver_candidate_votes') }}
{{ incremental_election_filter('election_year', 'election_type') }}
