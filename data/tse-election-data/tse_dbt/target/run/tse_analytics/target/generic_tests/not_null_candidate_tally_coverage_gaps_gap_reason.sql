
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select gap_reason
from "tse_analytics"."main"."candidate_tally_coverage_gaps"
where gap_reason is null



  
  
      
    ) dbt_internal_test