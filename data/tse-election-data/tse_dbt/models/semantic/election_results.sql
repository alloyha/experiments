with winner as (
    select * from {{ ref('candidate_ranking') }} where candidate_rank=1
),
runner_up as (
    select election_year,election_type,election_code,round_number,electoral_unit,office_code,
           candidate_id as runner_up_candidate_id,
           candidate_name as runner_up_candidate_name,
           party as runner_up_party,
           nominal_valid_votes as runner_up_nominal_valid_votes,
           vote_share as runner_up_vote_share
    from {{ ref('candidate_ranking') }} where candidate_rank=2
)
select
    w.election_year,w.election_type,w.election_scope,w.election_id,w.election_code,w.round_number,
    w.electoral_unit,w.office_code,w.office,w.office_scope,
    w.candidate_id as winner_candidate_id,w.candidate_number as winner_candidate_number,
    w.candidate_name as winner_candidate_name,w.ballot_name as winner_ballot_name,
    w.party_number as winner_party_number,w.party as winner_party,w.party_name as winner_party_name,
    w.nominal_valid_votes as winner_nominal_valid_votes,w.vote_share as winner_vote_share,w.municipalities_won,
    r.runner_up_candidate_id,r.runner_up_candidate_name,r.runner_up_party,
    r.runner_up_nominal_valid_votes,r.runner_up_vote_share,
    w.nominal_valid_votes-coalesce(r.runner_up_nominal_valid_votes,0) as winner_margin_votes
from winner w
left join runner_up r using(election_year,election_type,election_code,round_number,electoral_unit,office_code)
