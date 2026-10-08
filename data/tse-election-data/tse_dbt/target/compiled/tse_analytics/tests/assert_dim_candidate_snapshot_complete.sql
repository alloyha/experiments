with source_keys as (
    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."bronze_candidates"
    where election_year in (2018) and election_type in ('general')
),
target_keys as (
    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."dim_candidate"
    where election_year in (2018) and election_type in ('general')
),
diff as (
    (select 'missing_in_target' as issue, * from source_keys
     except
     select 'missing_in_target' as issue, * from target_keys)
    union all
    (select 'stale_in_target' as issue, * from target_keys
     except
     select 'stale_in_target' as issue, * from source_keys)
)
select * from diff