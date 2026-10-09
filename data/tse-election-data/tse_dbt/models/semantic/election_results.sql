{{ config(materialized='table') }}

with top_ranked as (
    select *
    from {{ ref('candidate_ranking') }}
    where candidate_rank = 1
),

second_ranked as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        electoral_unit,
        office_code,
        candidate_id as second_candidate_id,
        candidate_number as second_candidate_number,
        candidate_name as second_candidate_name,
        ballot_name as second_ballot_name,
        party_number as second_party_number,
        party as second_party,
        party_name as second_party_name,
        nominal_valid_votes as second_nominal_valid_votes,
        candidate_nominal_vote_share as second_candidate_nominal_vote_share
    from {{ ref('candidate_ranking') }}
    where candidate_rank = 2
)

select
    t.election_year,
    t.election_type,
    t.election_scope,
    t.election_id,
    t.election_code,
    t.round_number,
    t.electoral_unit,
    t.office_code,
    t.office,
    t.office_scope,
    t.contest_candidate_nominal_valid_votes,
    t.candidate_id as top_candidate_id,
    t.candidate_number as top_candidate_number,
    t.candidate_name as top_candidate_name,
    t.ballot_name as top_ballot_name,
    t.party_number as top_party_number,
    t.party as top_party,
    t.party_name as top_party_name,
    t.nominal_valid_votes as top_nominal_valid_votes,
    t.candidate_nominal_vote_share as top_candidate_nominal_vote_share,
    s.second_candidate_id,
    s.second_candidate_number,
    s.second_candidate_name,
    s.second_ballot_name,
    s.second_party_number,
    s.second_party,
    s.second_party_name,
    s.second_nominal_valid_votes,
    s.second_candidate_nominal_vote_share,
    case
        when s.second_nominal_valid_votes is not null
        then t.nominal_valid_votes - s.second_nominal_valid_votes
    end as lead_margin_votes
from top_ranked t
left join second_ranked s
  using (
    election_year,
    election_type,
    election_code,
    round_number,
    electoral_unit,
    office_code
  )
