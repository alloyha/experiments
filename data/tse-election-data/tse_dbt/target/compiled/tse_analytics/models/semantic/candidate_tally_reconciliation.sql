

with coverage as (
    select *
    from "tse_analytics"."main"."silver_candidate_result_coverage"
),

candidate as (
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

tally as (
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
        sum(nominal_valid_votes) as tally_nominal_valid_votes
    from "tse_analytics"."main"."fact_tally_munzona"
    group by 1,2,3,4,5,6,7,8,9
)

select
    t.election_year,
    t.election_type,
    t.election_code,
    t.round_number,
    t.uf,
    t.municipality_code,
    t.zone,
    t.office_code,
    t.is_transit_vote,
    coalesce(c.candidate_nominal_valid_votes, 0) as candidate_nominal_valid_votes,
    t.tally_nominal_valid_votes,
    coalesce(c.candidate_nominal_valid_votes, 0) - t.tally_nominal_valid_votes
        as nominal_valid_delta
from tally t
join coverage cv
  on cv.election_year = t.election_year
 and cv.election_type = t.election_type
 and cv.election_code = t.election_code
 and cv.round_number = t.round_number
 and cv.uf = t.uf
 and cv.municipality_code = t.municipality_code
 and cv.zone = t.zone
 and cv.office_code = t.office_code
 and cv.is_transit_vote is not distinct from t.is_transit_vote
left join candidate c
  on c.election_year = t.election_year
 and c.election_type = t.election_type
 and c.election_code = t.election_code
 and c.round_number = t.round_number
 and c.uf = t.uf
 and c.municipality_code = t.municipality_code
 and c.zone = t.zone
 and c.office_code = t.office_code
 and c.is_transit_vote is not distinct from t.is_transit_vote