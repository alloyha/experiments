
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_turnout"
where turnout_rate < 0 or turnout_rate > 1
   or abstention_rate < 0 or abstention_rate > 1
  
  
      
    ) dbt_internal_test