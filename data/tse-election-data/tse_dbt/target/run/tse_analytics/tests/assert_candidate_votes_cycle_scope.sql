
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_candidate_votes"
where
      (election_type = 'general' and office_scope = 'municipal')
   or (election_type = 'municipal' and office_scope in ('federal', 'state'))
  
  
      
    ) dbt_internal_test