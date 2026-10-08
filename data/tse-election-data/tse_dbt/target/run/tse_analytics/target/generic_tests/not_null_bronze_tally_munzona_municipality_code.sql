
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select municipality_code
from "tse_analytics"."main"."bronze_tally_munzona"
where municipality_code is null



  
  
      
    ) dbt_internal_test