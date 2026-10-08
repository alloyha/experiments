{{ config(enabled=var('run_heavy_integrity_tests', false), tags=['integrity_heavy']) }}

with raw_counts as (
    select
        election_year,
        election_type,
        count(*) as raw_rows
    from {{ ref('bronze_candidate_votes_raw') }}
    group by 1,2
),
munzona_counts as (
    select
        election_year,
        election_type,
        count(*) as munzona_rows
    from {{ ref('silver_candidate_votes_munzona') }}
    group by 1,2
)
select
    coalesce(r.election_year, m.election_year) as election_year,
    coalesce(r.election_type, m.election_type) as election_type,
    coalesce(r.raw_rows, 0) as raw_rows,
    coalesce(m.munzona_rows, 0) as munzona_rows
from raw_counts r
full outer join munzona_counts m
    using (election_year, election_type)
where coalesce(r.raw_rows, 0) <> coalesce(m.munzona_rows, 0)
