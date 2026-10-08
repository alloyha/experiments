select *
from {{ ref('party_tally_reconciliation') }}
where nominal_valid_delta <> 0
   or total_legend_valid_delta <> 0
   or total_valid_delta <> 0
