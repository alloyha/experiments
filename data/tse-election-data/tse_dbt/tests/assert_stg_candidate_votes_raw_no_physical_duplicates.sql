{{ config(enabled=var('run_heavy_integrity_tests', false), tags=['integrity_heavy']) }}

with grouped as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        candidate_id,
        is_transit_vote,
        nominal_votes,
        nominal_valid_votes,
        totalization_status_code,
        source_file,
        count(*) as copies
    from {{ ref('bronze_candidate_votes_raw') }}
    group by
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        candidate_id,
        is_transit_vote,
        nominal_votes,
        nominal_valid_votes,
        totalization_status_code,
        source_file
)
select *
from grouped
where copies > 1
