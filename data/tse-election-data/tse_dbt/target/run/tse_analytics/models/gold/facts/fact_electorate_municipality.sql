
  
    
    
    create temporary table
      "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2"
  
    as (
      

select *
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2"
            where (
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_electorate_municipality" ("election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate")
    (
        select "election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate"
        from "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2"
    )
  