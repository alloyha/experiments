select *
from {{ ref('fact_party_votes') }}
where nominal_valid_votes < 0
   or legend_valid_votes < 0
   or total_legend_valid_votes < 0
   or party_valid_votes < 0
   or nominal_annulled_subjudice_votes < 0
   or legend_annulled_subjudice_votes < 0
