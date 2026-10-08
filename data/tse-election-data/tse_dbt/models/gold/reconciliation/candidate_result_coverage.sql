{{ config(materialized='view') }}

select distinct
    election_year,
    election_type,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    is_transit_vote
from {{ ref('fact_candidate_votes') }}
