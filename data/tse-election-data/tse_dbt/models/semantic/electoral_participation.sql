select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,

    eligible_voters,
    voters_uninstalled_sections,
    uncounted_voters,
    turnout,
    abstentions,
    turnout_rate,
    abstention_rate,

    generated_at
from {{ ref('fact_turnout') }}
