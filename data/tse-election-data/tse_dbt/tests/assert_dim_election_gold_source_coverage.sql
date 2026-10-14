-- Every observed Gold event must be in dim_election with identical scope and
-- evidence provenance. The existing membership test additionally checks the
-- candidate-vote Parquet and turnout against the dimension.
with observed as (
    select distinct election_year, election_type,
        cast(election_code as varchar) as election_code, election_scope,
        'dim_candidate' as evidence_source
    from {{ ref('dim_candidate') }}
    union
    select distinct election_year, election_type,
        cast(election_code as varchar) as election_code, election_scope,
        'dim_party' as evidence_source
    from {{ ref('dim_party') }}
    union
    select distinct election_year, election_type,
        cast(election_code as varchar) as election_code, election_scope,
        'fact_tally_munzona' as evidence_source
    from {{ ref('fact_tally_munzona') }}
    union
    select distinct election_year, election_type,
        cast(election_code as varchar) as election_code, election_scope,
        'fact_party_votes' as evidence_source
    from {{ ref('fact_party_votes') }}
),
expected as (
    select
        cast(election_year as varchar) || ':' || election_type || ':' || election_code
            as election_id,
        min(election_scope) as election_scope,
        string_agg(distinct evidence_source, ',' order by evidence_source)
            as evidence_sources
    from observed
    group by election_year, election_type, election_code
),
actual as (
    select election_id, election_scope, evidence_sources
    from {{ ref('dim_election') }}
),
diffs as (
    (select 'missing_or_mismatched' as violation, * from expected
     except
     select 'missing_or_mismatched', * from actual)
    union all
    (select 'unexpected_or_mismatched', * from actual
     except
     select 'unexpected_or_mismatched', * from expected)
)
select * from diffs
