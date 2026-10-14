-- Election cycle (year:type) is not an election event (year:type:code).
-- Fail any malformed key. Do not join electorate by event_id: it has no event code.
with event_models as (
    select 'dim_election' as model, election_year, election_type,
           cast(election_code as varchar) as election_code, cycle_id, election_id
    from {{ ref('dim_election') }}
    union all
    select 'dim_candidate', election_year, election_type,
           cast(election_code as varchar), cycle_id, election_id
    from {{ ref('dim_candidate') }}
    union all
    select 'dim_party', election_year, election_type,
           cast(election_code as varchar), cycle_id, election_id
    from {{ ref('dim_party') }}
    union all
    select 'fact_tally_munzona', election_year, election_type,
           cast(election_code as varchar), cycle_id, election_id
    from {{ ref('fact_tally_munzona') }}
    union all
    select 'fact_turnout', election_year, election_type,
           cast(election_code as varchar), cycle_id, election_id
    from {{ ref('fact_turnout') }}
    union all
    select 'fact_party_votes', election_year, election_type,
           cast(election_code as varchar), cycle_id, election_id
    from {{ ref('fact_party_votes') }}
    union all
    select 'fact_candidate_votes', election_year, election_type,
           cast(election_code as varchar), cycle_id, election_id
    from {{ ref('fact_candidate_votes') }}
),
cycle_models as (
    select 'dim_election_cycle' as model, election_year, election_type, cycle_id
    from {{ ref('dim_election_cycle') }}
    union all
    select 'dim_geography' as model, election_year, election_type, cycle_id
    from {{ ref('dim_geography') }}
    union all
    select 'fact_electorate_municipality', election_year, election_type, cycle_id
    from {{ ref('fact_electorate_municipality') }}
),
violations as (
    select model, cycle_id, election_id, 'event_identity' as violation
    from event_models
    where election_year is null or election_type is null or election_code is null
       or cycle_id is distinct from (cast(election_year as varchar) || ':' || election_type)
       or election_id is distinct from (cast(election_year as varchar) || ':' || election_type || ':' || election_code)
    union all
    select model, cycle_id, cast(null as varchar) as election_id, 'cycle_identity'
    from cycle_models
    where election_year is null or election_type is null
       or cycle_id is distinct from (cast(election_year as varchar) || ':' || election_type)
)
select * from violations
