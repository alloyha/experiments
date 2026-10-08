
  
    
    
    create temporary table
      "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    election_id,
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
    eligible_voters - turnout - abstentions as uncounted_voters,
    turnout,
    abstentions,

    case when eligible_voters > 0
         then turnout::double / eligible_voters
    end as turnout_rate,

    case when eligible_voters > 0
         then abstentions::double / eligible_voters
    end as abstention_rate,

    generated_at
from "tse_analytics"."main"."fact_tally_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_turnout" as DBT_INCREMENTAL_TARGET
            using "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47"
            where (
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_turnout" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at"
        from "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47"
    )
  