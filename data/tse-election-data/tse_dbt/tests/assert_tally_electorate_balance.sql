select *
from {{ ref('fact_tally_munzona') }}
where eligible_voters
   <> turnout
    + abstentions
    + coalesce(voters_uninstalled_sections, 0)
