select
    f.election_year,
    f.election_type,
    f.election_id,
    f.election_code,
    f.round_number,

    d.office_code,
    d.office,
    d.office_scope,

    f.candidate_id,
    d.candidate_number,
    d.candidate_name,
    d.ballot_name,
    d.party,
    d.party_name,

    sum(f.nominal_valid_votes) as nominal_valid_votes,
    count(distinct f.municipality_code) as municipalities_with_votes,
    count(distinct cast(f.uf as varchar) || ':' || cast(f.zone as varchar)) as zones_with_votes
from {{ ref('fact_candidate_votes') }} f
left join {{ ref('dim_candidate') }} d
  on d.election_year = f.election_year
 and d.election_type = f.election_type
 and d.election_code = f.election_code
 and d.candidate_id = f.candidate_id
group by
    f.election_year,
    f.election_type,
    f.election_id,
    f.election_code,
    f.round_number,
    d.office_code,
    d.office,
    d.office_scope,
    f.candidate_id,
    d.candidate_number,
    d.candidate_name,
    d.ballot_name,
    d.party,
    d.party_name
