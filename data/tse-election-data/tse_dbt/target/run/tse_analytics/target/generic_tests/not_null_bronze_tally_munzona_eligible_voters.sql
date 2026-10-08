
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select eligible_voters
from "tse_analytics"."main"."bronze_tally_munzona"
where eligible_voters is null



  
  
      
    ) dbt_internal_test