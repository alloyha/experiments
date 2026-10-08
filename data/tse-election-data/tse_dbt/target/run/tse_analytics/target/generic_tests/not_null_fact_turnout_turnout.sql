
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select turnout
from "tse_analytics"."main"."fact_turnout"
where turnout is null



  
  
      
    ) dbt_internal_test