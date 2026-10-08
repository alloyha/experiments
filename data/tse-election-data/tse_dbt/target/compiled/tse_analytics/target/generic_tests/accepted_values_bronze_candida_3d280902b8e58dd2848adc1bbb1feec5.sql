
    
    

with all_values as (

    select
        election_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."bronze_candidates"
    group by election_scope

)

select *
from all_values
where value_field not in (
    'federal_state','municipal'
)


