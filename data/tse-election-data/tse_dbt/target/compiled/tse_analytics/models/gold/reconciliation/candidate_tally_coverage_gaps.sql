

with coverage as (

    select *
    from "tse_analytics"."main"."silver_candidate_result_coverage"

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
        nominal_valid_votes,
        valid_votes,
        total_votes,
        generated_at

    from "tse_analytics"."main"."fact_tally_munzona"

)

select
    t.*,
    'missing_candidate_result_coverage' as gap_reason

from tally t

left join coverage c
  on c.election_year = t.election_year
 and c.election_type = t.election_type
 and c.election_code = t.election_code
 and c.round_number = t.round_number
 and c.uf = t.uf
 and c.municipality_code = t.municipality_code
 and c.zone = t.zone
 and c.office_code = t.office_code
 and c.is_transit_vote = t.is_transit_vote

where c.election_year is null
  and t.nominal_valid_votes > 0