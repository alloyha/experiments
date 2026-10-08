{{ config(
    pre_hook=partition_replace_pre_hook(),
    meta={'load_semantics': 'partition_replace', 'history_semantics': 'election_snapshot', 'partition_key': ['election_year', 'election_type']},
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=['election_year', 'election_type', 'election_code', 'candidate_id'],
    on_schema_change='sync_all_columns'
) }}

select
    c.election_year,
    c.election_type,
    c.election_scope,
    cast(c.election_year as varchar) || ':' || c.election_type as election_id,
    c.election_code,
    c.election_description,
    c.round_number,
    c.electoral_unit,
    c.office_scope,
    c.candidate_id,
    c.uf,
    c.office_code,
    c.office,
    c.candidate_number,
    c.candidate_name,
    c.ballot_name,
    c.party_number,
    c.party,
    c.party_name,
    c.candidacy_status,
    c.gender,
    c.education,
    c.occupation,
    c.race_color,
    coalesce(a.declared_assets_value, 0) as declared_assets_value,
    coalesce(a.declared_assets_count, 0) as declared_assets_count
from {{ ref('stg_candidates') }} c
left join {{ ref('int_candidate_assets') }} a
  using (election_year, election_type, election_code, candidate_id)
{{ incremental_election_filter('c.election_year', 'c.election_type') }}
