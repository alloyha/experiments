

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from "tse_analytics"."main"."bronze_electorate"

where election_year in (2026) and election_type in ('general')

group by 1,2,3,4,5,6