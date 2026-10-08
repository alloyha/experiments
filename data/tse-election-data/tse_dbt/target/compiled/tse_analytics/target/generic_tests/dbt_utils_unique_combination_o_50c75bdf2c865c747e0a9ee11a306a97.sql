





with validation_errors as (

    select
        election_year, election_type, election_code
    from "tse_analytics"."main"."dim_election"
    group by election_year, election_type, election_code
    having count(*) > 1

)

select *
from validation_errors


