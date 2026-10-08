
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select abstentions
from "tse_analytics"."main"."fact_turnout"
where abstentions is null



  
  
      
    ) dbt_internal_test