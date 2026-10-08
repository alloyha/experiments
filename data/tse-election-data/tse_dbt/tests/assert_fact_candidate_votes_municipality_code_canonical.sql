select *
from {{ ref('fact_candidate_votes') }}
where municipality_code is null
   or length(municipality_code) <> 5
   or not regexp_matches(municipality_code, '^[0-9]{5}$')
limit 1
