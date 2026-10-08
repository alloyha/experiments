
  
    
    
    create temporary table
      "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from "tse_analytics"."main"."bronze_electorate"

where election_year in (2026) and election_type in ('general')

group by 1,2,3,4,5,6
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."silver_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578"
            where (
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."silver_electorate_municipality" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate"
        from "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578"
    )
  