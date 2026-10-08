
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select office_scope
from "tse_analytics"."main"."bronze_candidates"
where office_scope is null



  
  
      
    ) dbt_internal_test