select *
from {{ ref('dim_candidate') }}
where declared_assets_value < 0
