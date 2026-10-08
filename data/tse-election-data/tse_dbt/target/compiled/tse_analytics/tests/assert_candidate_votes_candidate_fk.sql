select
    f.election_year,
    f.election_type,
    f.election_code,
    f.candidate_id,
    count(*) as missing_rows
from "tse_analytics"."main"."fact_candidate_votes" f
left join "tse_analytics"."main"."dim_candidate" d
  on d.election_year = f.election_year
 and d.election_type = f.election_type
 and d.election_code = f.election_code
 and d.candidate_id = f.candidate_id
where d.candidate_id is null
group by 1,2,3,4