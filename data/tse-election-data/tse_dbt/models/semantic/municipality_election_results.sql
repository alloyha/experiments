with candidate_votes as (
    select
        f.election_year, f.election_type, f.election_scope, f.election_id,
        f.election_code, f.round_number, f.uf, f.municipality_code, f.office_code,
        d.office, d.office_scope, f.candidate_id, d.candidate_number,
        d.candidate_name, d.ballot_name, d.party_number, d.party, d.party_name,
        sum(f.nominal_valid_votes) as nominal_valid_votes
    from {{ ref('fact_candidate_votes') }} f
    inner join {{ ref('dim_candidate') }} d
      on d.election_year=f.election_year and d.election_type=f.election_type
     and d.election_code=f.election_code and d.candidate_id=f.candidate_id
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
),
tally as (
    select election_year,election_type,election_code,round_number,uf,municipality_code,office_code,
           sum(valid_votes) as valid_votes, sum(turnout) as turnout,
           sum(eligible_voters) as eligible_voters, sum(abstentions) as abstentions
    from {{ ref('fact_tally_munzona') }}
    where not coalesce(is_transit_vote,false)
    group by 1,2,3,4,5,6,7
),
ranked as (
    select c.*, t.valid_votes,t.turnout,t.eligible_voters,t.abstentions,
           case when t.valid_votes>0 then c.nominal_valid_votes::double/t.valid_votes end as vote_share,
           row_number() over (
             partition by c.election_year,c.election_type,c.election_code,c.round_number,c.uf,c.municipality_code,c.office_code
             order by c.nominal_valid_votes desc,c.candidate_id
           ) as candidate_rank,
           lead(c.nominal_valid_votes) over (
             partition by c.election_year,c.election_type,c.election_code,c.round_number,c.uf,c.municipality_code,c.office_code
             order by c.nominal_valid_votes desc,c.candidate_id
           ) as next_candidate_votes
    from candidate_votes c
    left join tally t using(election_year,election_type,election_code,round_number,uf,municipality_code,office_code)
)
select *,
       candidate_rank=1 as winner,
       nominal_valid_votes-coalesce(next_candidate_votes,0) as margin_to_next,
       case when eligible_voters>0 then turnout::double/eligible_voters end as turnout_rate,
       case when eligible_voters>0 then abstentions::double/eligible_voters end as abstention_rate
from ranked
