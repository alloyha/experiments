
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_legend_valid_votes
from "tse_analytics"."main"."silver_party_votes_munzona"
where total_legend_valid_votes is null



  
  
      
    ) dbt_internal_test