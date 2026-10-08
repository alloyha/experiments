
  
    
    
    create temporary table
      "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,

    uf,
    municipality_code,
    zone,

    office_code,
    office_scope,

    party_number,
    is_transit_vote,

    nominal_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_votes,
    total_legend_valid_votes,

    coalesce(nominal_valid_votes, 0)
      + coalesce(total_legend_valid_votes, 0) as party_valid_votes,

    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    generated_at,
    source_file
from "tse_analytics"."main"."silver_party_votes_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_party_votes" as DBT_INCREMENTAL_TARGET
            using "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3"
            where (
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_party_votes" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file"
        from "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3"
    )
  