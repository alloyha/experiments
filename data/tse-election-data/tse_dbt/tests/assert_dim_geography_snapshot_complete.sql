with source_rows as (
    select distinct
        election_year, election_type, election_scope,
        uf, municipality_code, municipality
    from {{ ref('silver_electorate_municipality') }}
    where {{ selected_election_predicate() }}
),
target_rows as (
    select
        election_year, election_type, election_scope,
        uf, municipality_code, municipality
    from {{ ref('dim_geography') }}
    where {{ selected_election_predicate() }}
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
