select *
from {{ ref('candidate_tally_reconciliation') }}
where nominal_valid_delta <> 0
