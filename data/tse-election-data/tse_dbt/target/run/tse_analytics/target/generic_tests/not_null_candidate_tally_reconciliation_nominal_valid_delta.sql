
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_valid_delta
from "tse_analytics"."main"."candidate_tally_reconciliation"
where nominal_valid_delta is null



  
  
      
    ) dbt_internal_test