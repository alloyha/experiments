
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."dim_candidate"
where declared_assets_value < 0
  
  
      
    ) dbt_internal_test