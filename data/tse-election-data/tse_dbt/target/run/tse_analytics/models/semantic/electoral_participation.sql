
  
  create view "tse_analytics"."main"."electoral_participation__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,

    eligible_voters,
    voters_uninstalled_sections,
    uncounted_voters,
    turnout,
    abstentions,
    turnout_rate,
    abstention_rate,

    generated_at
from "tse_analytics"."main"."fact_turnout"
  );
