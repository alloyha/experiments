





with validation_errors as (

    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."dim_candidate"
    group by election_year, election_type, election_code, candidate_id
    having count(*) > 1

)

select *
from validation_errors


