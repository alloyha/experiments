
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, uf, municipality_code
    from "tse_analytics"."main"."dim_geography"
    group by election_year, election_type, uf, municipality_code
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test