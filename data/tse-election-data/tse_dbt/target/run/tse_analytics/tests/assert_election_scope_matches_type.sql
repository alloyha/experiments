
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."bronze_candidates"
where election_scope <> case
    when election_type = 'general' then 'federal_state'
    when election_type = 'municipal' then 'municipal'
end
  
  
      
    ) dbt_internal_test