{{ config(
    enabled=var('run_heavy_integrity_tests', false),
    tags=['integrity_heavy']
) }}

with violations as (
    select
        sum(case when election_year is null then 1 else 0 end) as null_election_year,
        sum(case when election_type is null then 1 else 0 end) as null_election_type,
        sum(case when election_code is null then 1 else 0 end) as null_election_code,
        sum(case when round_number is null then 1 else 0 end) as null_round_number,
        sum(case when uf is null then 1 else 0 end) as null_uf,
        sum(case when municipality_code is null then 1 else 0 end) as null_municipality_code,
        sum(case when zone is null then 1 else 0 end) as null_zone,
        sum(case when office_code is null then 1 else 0 end) as null_office_code,
        sum(case when office_scope is null then 1 else 0 end) as null_office_scope,
        sum(case when candidate_id is null then 1 else 0 end) as null_candidate_id,
        sum(case when nominal_votes is null then 1 else 0 end) as null_nominal_votes,
        sum(case when nominal_valid_votes is null then 1 else 0 end) as null_nominal_valid_votes,

        sum(case
            when election_type is not null
             and election_type not in ('general', 'municipal')
            then 1 else 0
        end) as invalid_election_type,

        sum(case
            when office_scope is not null
             and office_scope not in ('federal', 'state', 'municipal', 'other')
            then 1 else 0
        end) as invalid_office_scope

    from {{ ref('stg_candidate_votes_munzona') }}
)

select *
from violations
where null_election_year > 0
   or null_election_type > 0
   or null_election_code > 0
   or null_round_number > 0
   or null_uf > 0
   or null_municipality_code > 0
   or null_zone > 0
   or null_office_code > 0
   or null_office_scope > 0
   or null_candidate_id > 0
   or null_nominal_votes > 0
   or null_nominal_valid_votes > 0
   or invalid_election_type > 0
   or invalid_office_scope > 0
