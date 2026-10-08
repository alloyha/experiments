
  
    
    
    create temporary table
      "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479"
  
    as (
      

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
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_candidate" as DBT_INCREMENTAL_TARGET
            using "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479"
            where (
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".candidate_id = DBT_INCREMENTAL_TARGET.candidate_id
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_candidate" ("election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count"
        from "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479"
    )
  