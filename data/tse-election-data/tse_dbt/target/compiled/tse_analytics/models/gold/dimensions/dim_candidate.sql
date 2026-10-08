

select
    c.election_year,
    c.election_type,
    c.election_scope,
    cast(c.election_year as varchar) || ':' || c.election_type as election_id,
    c.election_code,
    c.election_description,
    c.round_number,
    c.electoral_unit,
    c.office_scope,
    c.candidate_id,
    c.uf,
    c.office_code,
    c.office,
    c.candidate_number,
    c.candidate_name,
    c.ballot_name,
    c.party_number,
    c.party,
    c.party_name,
    c.candidacy_status,
    c.gender,
    c.education,
    c.occupation,
    c.race_color,
    coalesce(a.declared_assets_value, 0) as declared_assets_value,
    coalesce(a.declared_assets_count, 0) as declared_assets_count
from "tse_analytics"."main"."bronze_candidates" c
left join "tse_analytics"."main"."silver_candidate_assets" a
  using (election_year, election_type, election_code, candidate_id)

  
    where c.election_year in (2026) and c.election_type in ('general')
  
