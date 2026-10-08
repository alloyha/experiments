
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."bronze_candidates"
    group by election_year, election_type, election_code, candidate_id
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test