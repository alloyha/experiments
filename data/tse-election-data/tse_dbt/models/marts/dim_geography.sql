{{ config(
    pre_hook=partition_replace_pre_hook(),
    meta={'load_semantics': 'partition_replace', 'history_semantics': 'election_snapshot', 'partition_key': ['election_year', 'election_type']},
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=['election_year', 'election_type', 'uf', 'municipality_code'],
    on_schema_change='sync_all_columns'
) }}

select distinct
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality
from {{ ref('int_electorate_municipality') }}
{{ incremental_election_filter('election_year', 'election_type') }}
