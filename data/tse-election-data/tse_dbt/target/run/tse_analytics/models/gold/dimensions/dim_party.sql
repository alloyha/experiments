
  
    
    
    create temporary table
      "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40"
  
    as (
      

with versions as (
    select
        election_year,
        election_type,
        election_scope,
        cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,
        election_code,
        party_number,
        party,
        party_name,
        generated_at,
        row_number() over (
            partition by election_year, election_type, election_code, party_number
            order by generated_at desc nulls last, party desc, party_name desc
        ) as _rank
    from "tse_analytics"."main"."silver_party_votes_munzona"
    
    where election_year in (2026) and election_type in ('general')
    
)

select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    party_number,
    party,
    party_name,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code || ':' ||
      cast(election_code as varchar) || ':' || cast(party_number as varchar) as party_id
from versions
where _rank = 1
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_party" as DBT_INCREMENTAL_TARGET
            using "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40"
            where (
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".party_number = DBT_INCREMENTAL_TARGET.party_number
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_party" ("election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id"
        from "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40"
    )
  