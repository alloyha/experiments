select *
from {{ ref('fact_candidate_votes') }}
where
      (election_type = 'general' and office_scope = 'municipal')
   or (election_type = 'municipal' and office_scope in ('federal', 'state'))
