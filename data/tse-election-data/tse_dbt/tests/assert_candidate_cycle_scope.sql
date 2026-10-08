select *
from {{ ref('stg_candidates') }}
where
    (election_type = 'municipal' and office_scope <> 'municipal')
    or
    (election_type = 'general' and office_scope = 'municipal')
