select
    election_year,
    election_type,
    election_scope,
    office_scope,
    office,
    party,
    count(*) as candidates,
    avg(declared_assets_value) as avg_declared_assets_value,
    median(declared_assets_value) as median_declared_assets_value
from {{ ref('dim_candidate') }}
group by 1,2,3,4,5,6
