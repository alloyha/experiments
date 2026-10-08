
    
    

with all_values as (

    select
        election_type as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."dim_election"
    group by election_type

)

select *
from all_values
where value_field not in (
    'general','municipal'
)


