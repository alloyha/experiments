
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select uf
from "tse_analytics"."main"."silver_party_votes_munzona"
where uf is null



  
  
      
    ) dbt_internal_test