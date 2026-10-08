{{ config(enabled=var('run_heavy_integrity_tests', false), tags=['integrity_heavy']) }}

with source_keys as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, candidate_id,
        is_transit_vote, nominal_votes
    from {{ ref('int_candidate_votes') }}
    where {{ selected_election_predicate() }}
),
target_keys as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, candidate_id,
        is_transit_vote, nominal_votes
    from {{ ref('fact_candidate_votes') }}
    where {{ selected_election_predicate() }}
),
diff as (
    (select 'missing_or_changed_in_target' as issue, * from source_keys
     except
     select 'missing_or_changed_in_target' as issue, * from target_keys)
    union all
    (select 'stale_or_changed_in_target' as issue, * from target_keys
     except
     select 'stale_or_changed_in_target' as issue, * from source_keys)
)
select * from diff
