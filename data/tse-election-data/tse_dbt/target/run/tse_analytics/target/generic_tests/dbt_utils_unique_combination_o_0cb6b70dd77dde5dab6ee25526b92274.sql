
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, candidate_id, is_transit_vote
    from "tse_analytics"."main"."fact_candidate_votes"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, candidate_id, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test