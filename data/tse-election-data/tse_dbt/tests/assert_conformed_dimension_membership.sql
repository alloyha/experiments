-- Each referenced event/cycle must have exactly one authoritative dimension member.
-- Deduplicate BEFORE joining: large municipality-zone facts are not dimension rows.
with referenced_events as (
    select distinct election_id, cycle_id from {{ ref('dim_candidate') }}
    union
    select distinct election_id, cycle_id from {{ ref('dim_party') }}
    union
    select distinct election_id, cycle_id from {{ ref('fact_tally_munzona') }}
    union
    select distinct election_id, cycle_id from {{ ref('fact_turnout') }}
    union
    select distinct election_id, cycle_id from {{ ref('fact_party_votes') }}
    union
    select distinct election_id, cycle_id from {{ ref('fact_candidate_votes') }}
),
referenced_cycles as (
    select cycle_id from referenced_events
    union
    select cycle_id from {{ ref('fact_electorate_municipality') }}
    union
    select cycle_id from {{ ref('dim_geography') }}
),
missing_events as (
    select 'missing_election_event' as violation, r.cycle_id, r.election_id
    from referenced_events r
    left join {{ ref('dim_election') }} d on d.election_id = r.election_id
    where d.election_id is null
),
missing_cycles as (
    select 'missing_election_cycle' as violation, r.cycle_id,
           cast(null as varchar) as election_id
    from referenced_cycles r
    left join {{ ref('dim_election_cycle') }} d on d.cycle_id = r.cycle_id
    where d.cycle_id is null
)
select * from missing_events
union all
select * from missing_cycles
