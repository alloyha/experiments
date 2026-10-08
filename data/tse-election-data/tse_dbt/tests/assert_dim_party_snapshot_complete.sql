with source_rows as (
    select distinct
        election_year,
        election_type,
        election_code,
        party_number
    from {{ ref('silver_party_votes_munzona') }}
    where {{ selected_election_predicate() }}
),
target_rows as (
    select
        election_year,
        election_type,
        election_code,
        party_number
    from {{ ref('dim_party') }}
    where {{ selected_election_predicate() }}
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
