
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_votes
from "tse_analytics"."main"."fact_candidate_votes"
where nominal_votes is null



  
  
      
    ) dbt_internal_test