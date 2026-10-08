
  
  create view "tse_analytics"."main"."candidate_summary__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_scope,
    office_scope,
    office,
    party,
    count(*) as candidates,
    avg(declared_assets_value) as avg_declared_assets_value,
    median(declared_assets_value) as median_declared_assets_value
from "tse_analytics"."main"."dim_candidate"
group by 1,2,3,4,5,6
  );
