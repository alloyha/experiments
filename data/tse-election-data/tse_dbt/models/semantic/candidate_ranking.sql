with candidate_votes as (
    select
        f.election_year,f.election_type,f.election_scope,f.election_id,f.election_code,f.round_number,
        coalesce(d.electoral_unit,d.uf,f.uf) as electoral_unit,
        f.office_code,d.office,d.office_scope,f.candidate_id,d.candidate_number,d.candidate_name,
        d.ballot_name,d.party_number,d.party,d.party_name,
        sum(f.nominal_valid_votes) as nominal_valid_votes
    from {{ ref('fact_candidate_votes') }} f
    inner join {{ ref('dim_candidate') }} d
      on d.election_year=f.election_year and d.election_type=f.election_type
     and d.election_code=f.election_code and d.candidate_id=f.candidate_id
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
),
winners as (
    select election_year,election_type,election_code,round_number,office_code,candidate_id,
           count(*) as municipalities_won
    from {{ ref('municipality_election_results') }}
    where winner
    group by 1,2,3,4,5,6
),
ranked as (
    select c.*,
      sum(c.nominal_valid_votes) over (
        partition by c.election_year,c.election_type,c.election_code,c.round_number,c.electoral_unit,c.office_code
      ) as contest_nominal_valid_votes,
      row_number() over (
        partition by c.election_year,c.election_type,c.election_code,c.round_number,c.electoral_unit,c.office_code
        order by c.nominal_valid_votes desc,c.candidate_id
      ) as candidate_rank,
      lag(c.nominal_valid_votes) over (
        partition by c.election_year,c.election_type,c.election_code,c.round_number,c.electoral_unit,c.office_code
        order by c.nominal_valid_votes desc,c.candidate_id
      ) as previous_candidate_votes,
      lead(c.nominal_valid_votes) over (
        partition by c.election_year,c.election_type,c.election_code,c.round_number,c.electoral_unit,c.office_code
        order by c.nominal_valid_votes desc,c.candidate_id
      ) as next_candidate_votes
    from candidate_votes c
)
select r.*,
       case when contest_nominal_valid_votes>0 then nominal_valid_votes::double/contest_nominal_valid_votes end as vote_share,
       candidate_rank=1 as winner,
       nominal_valid_votes-previous_candidate_votes as margin_to_previous,
       nominal_valid_votes-next_candidate_votes as margin_to_next,
       coalesce(w.municipalities_won,0) as municipalities_won
from ranked r
left join winners w using(election_year,election_type,election_code,round_number,office_code,candidate_id)
