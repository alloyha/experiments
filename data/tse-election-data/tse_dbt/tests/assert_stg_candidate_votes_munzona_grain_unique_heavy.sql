{{ config(
    enabled=var('run_heavy_integrity_tests', false),
    tags=['integrity_heavy']
) }}

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
    count(*) as copies
from {{ ref('stg_candidate_votes_munzona') }}
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
    is_transit_vote
having count(*) > 1
