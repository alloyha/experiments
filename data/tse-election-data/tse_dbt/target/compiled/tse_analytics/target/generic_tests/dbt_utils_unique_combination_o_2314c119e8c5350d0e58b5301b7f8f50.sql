





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    from "tse_analytics"."main"."fact_turnout"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors


