
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        office_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."bronze_candidates"
    group by office_scope

)

select *
from all_values
where value_field not in (
    'federal','state','municipal','other'
)



  
  
      
    ) dbt_internal_test