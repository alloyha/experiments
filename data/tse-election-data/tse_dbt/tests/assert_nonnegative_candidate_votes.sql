select *
from {{ ref('fact_candidate_votes') }}
where nominal_votes < 0
