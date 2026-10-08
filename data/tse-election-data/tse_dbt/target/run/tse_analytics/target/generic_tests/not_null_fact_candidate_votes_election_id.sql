
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_id
from "tse_analytics"."main"."fact_candidate_votes"
where election_id is null



  
  
      
    ) dbt_internal_test