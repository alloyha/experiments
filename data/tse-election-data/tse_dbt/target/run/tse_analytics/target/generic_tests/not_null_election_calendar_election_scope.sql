
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_scope
from "tse_analytics"."main"."election_calendar"
where election_scope is null



  
  
      
    ) dbt_internal_test