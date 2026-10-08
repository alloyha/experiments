
  
    
    
    create temporary table
      "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff"
  
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
    is_transit_vote,

    eligible_voters,
    voters_uninstalled_sections,
    turnout,
    abstentions,

    total_votes,
    competing_votes,

    valid_votes,
    nominal_valid_votes,
    total_legend_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_valid_votes,

    annulled_votes,
    nominal_annulled_votes,
    legend_annulled_votes,

    annulled_subjudice_votes,
    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    blank_votes,
    total_null_votes,
    null_votes,
    technical_null_votes,
    separately_counted_annulled_votes,

    generated_at,
    last_totalization_at,
    source_file
from "tse_analytics"."main"."bronze_tally_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_tally_munzona" as DBT_INCREMENTAL_TARGET
            using "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff"
            where (
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_tally_munzona" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file"
        from "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff"
    )
  