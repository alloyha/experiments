
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with source_rows as (
    select distinct
        election_year,
        election_type,
        election_code,
        party_number
    from "tse_analytics"."main"."silver_party_votes_munzona"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year,
        election_type,
        election_code,
        party_number
    from "tse_analytics"."main"."dim_party"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_in_target' as issue, * from source_rows
     except
     select 'missing_in_target' as issue, * from target_rows)
    union all
    (select 'stale_in_target' as issue, * from target_rows
     except
     select 'stale_in_target' as issue, * from source_rows)
)
select * from diff
  
  
      
    ) dbt_internal_test