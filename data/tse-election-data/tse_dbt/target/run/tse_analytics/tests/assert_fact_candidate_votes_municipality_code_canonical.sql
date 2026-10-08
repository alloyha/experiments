
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_candidate_votes"
where municipality_code is null
   or length(municipality_code) <> 5
   or not regexp_matches(municipality_code, '^[0-9]{5}$')
limit 1
  
  
      
    ) dbt_internal_test