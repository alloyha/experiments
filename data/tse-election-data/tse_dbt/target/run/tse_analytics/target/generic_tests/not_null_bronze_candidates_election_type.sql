
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."bronze_candidates"
where election_type is null



  
  
      
    ) dbt_internal_test