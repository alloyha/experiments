





with validation_errors as (

    select
        election_year, election_type
    from "tse_analytics"."main"."election_calendar"
    group by election_year, election_type
    having count(*) > 1

)

select *
from validation_errors


