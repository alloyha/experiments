
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_valid_votes
from "tse_analytics"."main"."fact_party_votes"
where party_valid_votes is null



  
  
      
    ) dbt_internal_test