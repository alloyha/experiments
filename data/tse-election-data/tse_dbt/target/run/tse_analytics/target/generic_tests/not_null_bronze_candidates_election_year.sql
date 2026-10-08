
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."bronze_candidates"
where election_year is null



  
  
      
    ) dbt_internal_test