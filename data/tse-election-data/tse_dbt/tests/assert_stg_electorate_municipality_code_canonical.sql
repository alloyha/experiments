select *
from {{ ref('stg_electorate') }}
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
