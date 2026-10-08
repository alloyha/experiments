
  
    
    
    create temporary table
      "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83"
  
    as (
      

select *
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83"
            where (
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_electorate_municipality" ("election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate")
    (
        select "election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate"
        from "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83"
    )
  