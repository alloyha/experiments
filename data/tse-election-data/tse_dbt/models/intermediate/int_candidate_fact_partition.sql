{{ config(
    materialized='table',
    enabled=var('build_candidate_fact_partition', false),
    meta={
        'load_semantics': 'full_snapshot_ctas',
        'history_semantics': 'authoritative_snapshot',
        'partition_key': ['election_year', 'election_type']
    }
) }}

select *
from {{ ref('int_candidate_votes') }}
{{ incremental_election_filter('election_year', 'election_type') }}
