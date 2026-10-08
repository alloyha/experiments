
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."candidate_tally_reconciliation"
where nominal_valid_delta <> 0
  
  
      
    ) dbt_internal_test