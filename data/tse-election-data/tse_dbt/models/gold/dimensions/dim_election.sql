{{ config(materialized='table', meta={
    'history_semantics': 'event_registry_derived_from_persisted_gold',
    'source_policy': 'published_gold_event_evidence'
}) }}

-- This is an EVENT dimension (year/type/code), not a cycle dimension.
-- Source it from persisted Gold snapshots to retain events after raw CSV
-- eviction and across partial incremental builds. Do not scan candidate-vote
-- Parquet to discover event codes: the existing membership assertion covers it.
-- The four refs are acyclic: none depends on dim_election or dim_election_cycle.
with observed_events as (
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

conformed_events as (
    select
        election_year,
        election_type,
        election_code,
        case
            when election_year is null or election_type is null
                or election_code is null or trim(election_code) = ''
            then error('dim_election: observed event has a null/empty key')
            when count(*) filter (
                where election_scope is null or trim(election_scope) = ''
            ) > 0
            then error('dim_election: observed event has an unknown scope')
            when count(distinct election_scope) <> 1
            then error('dim_election: conflicting event scopes across Gold sources')
            when min(election_scope) is distinct from
                case election_type
                    when 'general' then 'federal_state'
                    when 'municipal' then 'municipal'
                end
            then error('dim_election: event scope contradicts election type')
            else min(election_scope)
        end as election_scope,
        string_agg(distinct evidence_source, ',' order by evidence_source)
            as evidence_sources
    from observed_events
    group by 1,2,3
)

select
    cast(election_year as varchar) || ':' || election_type || ':' || election_code
        as election_id,
    cast(election_year as varchar) || ':' || election_type as cycle_id,
    election_year,
    election_type,
    election_code,
    election_scope,
    evidence_sources,
    cast(election_year as varchar) || ' ' || election_type || ' ' || election_code
        as cycle_label
from conformed_events
