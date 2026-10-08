
  
  create view "tse_analytics"."main"."silver_candidate_votes_munzona__dbt_tmp" as (
    

select *
from "tse_analytics"."main"."bronze_candidate_votes_raw"
  );
