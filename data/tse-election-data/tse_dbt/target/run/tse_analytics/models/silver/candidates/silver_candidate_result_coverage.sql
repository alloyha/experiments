
  
  create view "tse_analytics"."main"."silver_candidate_result_coverage__dbt_tmp" as (
    

select distinct
    election_year,
    election_type,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    is_transit_vote
from "tse_analytics"."main"."fact_candidate_votes"
  );
