
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."bronze_candidates"
where
    (election_type = 'municipal' and office_scope <> 'municipal')
    or
    (election_type = 'general' and office_scope = 'municipal')
  
  
      
    ) dbt_internal_test