

select distinct
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  
