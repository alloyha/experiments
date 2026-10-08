

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from "tse_analytics"."main"."stg_electorate"

where election_year in (2018) and election_type in ('general')

group by 1,2,3,4,5,6