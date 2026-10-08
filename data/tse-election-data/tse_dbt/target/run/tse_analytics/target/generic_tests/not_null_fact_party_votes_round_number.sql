
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select round_number
from "tse_analytics"."main"."fact_party_votes"
where round_number is null



  
  
      
    ) dbt_internal_test