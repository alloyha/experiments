select
    f.election_year,
    f.election_type,
    f.election_code,
    f.party_number,
    count(*) as missing_rows
from "tse_analytics"."main"."fact_party_votes" f
left join "tse_analytics"."main"."dim_party" d
  on d.election_year = f.election_year
 and d.election_type = f.election_type
 and d.election_code = f.election_code
 and d.party_number = f.party_number
where d.party_number is null
group by 1,2,3,4