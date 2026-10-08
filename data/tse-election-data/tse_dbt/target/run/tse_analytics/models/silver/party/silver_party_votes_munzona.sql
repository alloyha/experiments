
  
    
    
    create temporary table
      "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8"
  
    as (
      

with source_rows as (
    select *
    from "tse_analytics"."main"."bronze_party_votes_raw"
    
    where election_year in (2026) and election_type in ('general')
    
),

collapsed as (
    select
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,

        uf,
        municipality_code,
        zone,

        office_code,
        office_scope,

        party_number,
        is_transit_vote,

        max(party) as party,
        max(party_name) as party_name,

        max(nominal_valid_votes) as nominal_valid_votes,
        max(legend_valid_votes) as legend_valid_votes,
        max(nominal_converted_to_legend_votes) as nominal_converted_to_legend_votes,
        max(total_legend_valid_votes) as total_legend_valid_votes,

        max(nominal_annulled_subjudice_votes) as nominal_annulled_subjudice_votes,
        max(legend_annulled_subjudice_votes) as legend_annulled_subjudice_votes,

        max(generated_at) as generated_at,
        max(source_file) as source_file,

        count(*) as source_row_count,
        count(distinct party_group_type) as source_party_group_types,
        count(distinct coalition_id) as source_coalitions,
        count(distinct federation_number) as source_federations

    from source_rows
    group by
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        office_scope,
        party_number,
        is_transit_vote
)

select *
from collapsed
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."silver_party_votes_munzona" as DBT_INCREMENTAL_TARGET
            using "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8"
            where (
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."silver_party_votes_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations"
        from "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8"
    )
  