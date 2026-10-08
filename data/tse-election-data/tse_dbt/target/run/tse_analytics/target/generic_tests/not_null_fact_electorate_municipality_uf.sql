
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select uf
from "tse_analytics"."main"."fact_electorate_municipality"
where uf is null



  
  
      
    ) dbt_internal_test