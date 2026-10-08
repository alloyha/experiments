
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, party_number
    from "tse_analytics"."main"."dim_party"
    group by election_year, election_type, election_code, party_number
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test