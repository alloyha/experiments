select *
from "tse_analytics"."main"."bronze_tally_munzona"
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )