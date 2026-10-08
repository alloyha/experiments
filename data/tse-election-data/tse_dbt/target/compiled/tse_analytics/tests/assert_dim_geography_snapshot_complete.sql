with source_rows as (
    select distinct
        election_year, election_type, election_scope,
        uf, municipality_code, municipality
    from "tse_analytics"."main"."silver_electorate_municipality"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year, election_type, election_scope,
        uf, municipality_code, municipality
    from "tse_analytics"."main"."dim_geography"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_or_changed_in_target' as issue, * from source_rows
     except
     select 'missing_or_changed_in_target' as issue, * from target_rows)
    union all
    (select 'stale_or_changed_in_target' as issue, * from target_rows
     except
     select 'stale_or_changed_in_target' as issue, * from source_rows)
)
select * from diff