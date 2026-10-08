
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_tally_munzona"
where eligible_voters
   <> turnout
    + abstentions
    + coalesce(voters_uninstalled_sections, 0)
  
  
      
    ) dbt_internal_test