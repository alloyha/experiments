with source_keys as (
    select
        election_year, election_type, election_code, candidate_id
    from {{ ref('bronze_candidates') }}
    where {{ selected_election_predicate() }}
),
target_keys as (
    select
        election_year, election_type, election_code, candidate_id
    from {{ ref('dim_candidate') }}
    where {{ selected_election_predicate() }}
),
diff as (
    (select 'missing_in_target' as issue, * from source_keys
     except
     select 'missing_in_target' as issue, * from target_keys)
    union all
    (select 'stale_in_target' as issue, * from target_keys
     except
     select 'stale_in_target' as issue, * from source_keys)
)
select * from diff
