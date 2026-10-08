select
    election_year,
    election_type,
    election_code,
    candidate_id,
    sum(asset_value) as declared_assets_value,
    count(*) as declared_assets_count
from {{ ref('bronze_candidate_assets') }}
group by 1,2,3,4
