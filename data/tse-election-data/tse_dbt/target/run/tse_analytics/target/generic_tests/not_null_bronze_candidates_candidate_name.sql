
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_name
from "tse_analytics"."main"."bronze_candidates"
where candidate_name is null



  
  
      
    ) dbt_internal_test