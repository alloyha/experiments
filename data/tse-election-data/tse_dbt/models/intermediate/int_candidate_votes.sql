select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,

    uf,
    municipality_code,
    zone,

    office_code,
    office_scope,

    candidate_id,
    totalization_status_code,
    is_transit_vote,
    nominal_votes,
    nominal_valid_votes,

    generated_at,
    source_file
from {{ ref('stg_candidate_votes_munzona') }}
