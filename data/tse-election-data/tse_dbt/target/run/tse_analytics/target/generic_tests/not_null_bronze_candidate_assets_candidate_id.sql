
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_id
from "tse_analytics"."main"."bronze_candidate_assets"
where candidate_id is null



  
  
      
    ) dbt_internal_test