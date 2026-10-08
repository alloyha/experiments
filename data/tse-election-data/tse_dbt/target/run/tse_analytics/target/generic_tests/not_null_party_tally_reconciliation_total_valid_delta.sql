
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_valid_delta
from "tse_analytics"."main"."party_tally_reconciliation"
where total_valid_delta is null



  
  
      
    ) dbt_internal_test