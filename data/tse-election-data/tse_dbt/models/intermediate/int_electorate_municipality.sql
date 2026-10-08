{{ config(
    pre_hook=partition_replace_pre_hook(),
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=['election_year', 'election_type', 'uf', 'municipality_code'],
    on_schema_change='sync_all_columns',
    meta={
      'load_semantics': 'partition_replace',
      'history_semantics': 'derived_snapshot',
      'partition_key': ['election_year', 'election_type'],
      'resource_class': 'heavy'
    }
) }}

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from {{ ref('stg_electorate') }}
{% if is_incremental() %}
where {{ incremental_partition_predicate('election_year', 'election_type') }}
{% endif %}
group by 1,2,3,4,5,6
