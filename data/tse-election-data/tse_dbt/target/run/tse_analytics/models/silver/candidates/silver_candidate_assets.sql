
  
  create view "tse_analytics"."main"."silver_candidate_assets__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_code,
    candidate_id,
    sum(asset_value) as declared_assets_value,
    count(*) as declared_assets_count
from "tse_analytics"."main"."bronze_candidate_assets"
group by 1,2,3,4
  );
