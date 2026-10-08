
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party
from "tse_analytics"."main"."dim_party"
where party is null



  
  
      
    ) dbt_internal_test