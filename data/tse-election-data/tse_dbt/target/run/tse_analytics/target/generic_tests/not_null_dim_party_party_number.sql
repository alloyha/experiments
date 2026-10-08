
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_number
from "tse_analytics"."main"."dim_party"
where party_number is null



  
  
      
    ) dbt_internal_test