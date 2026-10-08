{{ config(enabled=var('run_heavy_integrity_tests', false), tags=['integrity_heavy']) }}

select *
from {{ ref('stg_candidate_votes_raw') }}
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
