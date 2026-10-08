
  
    
    
    create temporary table
      "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918"
  
    as (
      

select distinct
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_geography" as DBT_INCREMENTAL_TARGET
            using "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918"
            where (
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_geography" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality"
        from "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918"
    )
  