{{ config(materialized='table') }}

-- One row per election cycle. Never use event-grain dim_election
-- as a cycle bridge: one cycle has multiple election codes.
select distinct
    cast(election_year as varchar) || ':' || election_type as cycle_id,
    election_year,
    election_type
from {{ ref('dim_election') }}
