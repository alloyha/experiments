{{ config(materialized='table') }}

with election_sources as (

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from {{ ref('stg_candidates') }}

    union all

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from {{ ref('stg_party_votes_raw') }}

    union all

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from {{ ref('stg_tally_munzona') }}
),

deduped as (
    select
        election_year,
        election_type,
        election_code,
        max(election_scope) as election_scope
    from election_sources
    group by 1,2,3
)

select
    concat(
        cast(election_year as varchar),
        ':',
        election_type,
        ':',
        election_code
    ) as election_id,

    election_year,
    election_type,
    election_code,
    election_scope,

    concat(cast(election_year as varchar), ' ', election_type, ' ', election_code)
        as cycle_label

from deduped
