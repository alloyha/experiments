
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."election_calendar"
    group by election_scope

)

select *
from all_values
where value_field not in (
    'federal_state','municipal'
)



  
  
      
    ) dbt_internal_test