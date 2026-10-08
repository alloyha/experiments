
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_type as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."election_calendar"
    group by election_type

)

select *
from all_values
where value_field not in (
    'general','municipal'
)



  
  
      
    ) dbt_internal_test