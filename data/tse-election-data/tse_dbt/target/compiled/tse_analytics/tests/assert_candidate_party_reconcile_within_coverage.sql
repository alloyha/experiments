with candidate as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,
        sum(nominal_valid_votes) as candidate_nominal_valid_votes
    from "tse_analytics"."main"."fact_candidate_votes"
    group by 1,2,3,4,5,6,7,8,9
),

party as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,
        sum(nominal_valid_votes) as party_nominal_valid_votes
    from "tse_analytics"."main"."fact_party_votes"
    group by 1,2,3,4,5,6,7,8,9
),

coverage as (
    select *
    from "tse_analytics"."main"."candidate_result_coverage"
)

select
    p.*,
    coalesce(c.candidate_nominal_valid_votes, 0) as candidate_nominal_valid_votes,
    coalesce(c.candidate_nominal_valid_votes, 0) - p.party_nominal_valid_votes
        as nominal_valid_delta
from party p
join coverage cv
  on cv.election_year = p.election_year
 and cv.election_type = p.election_type
 and cv.election_code = p.election_code
 and cv.round_number = p.round_number
 and cv.uf = p.uf
 and cv.office_code = p.office_code
left join candidate c
  on c.election_year = p.election_year
 and c.election_type = p.election_type
 and c.election_code = p.election_code
 and c.round_number = p.round_number
 and c.uf = p.uf
 and c.municipality_code = p.municipality_code
 and c.zone = p.zone
 and c.office_code = p.office_code
 and c.is_transit_vote is not distinct from p.is_transit_vote
where coalesce(c.candidate_nominal_valid_votes, 0) <> p.party_nominal_valid_votes