{{ config(
    pre_hook=partition_replace_pre_hook(),
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=[
      'election_year', 'election_type', 'election_code', 'round_number',
      'uf', 'municipality_code', 'zone', 'office_code', 'is_transit_vote'
    ],
    on_schema_change='sync_all_columns',
    meta={
      'load_semantics': 'partition_replace',
      'history_semantics': 'current_source_snapshot',
      'partition_key': ['election_year', 'election_type'],
      'resource_class': 'medium'
    }
) }}

with ranked as (
    select
        *,
        row_number() over (
            partition by
                election_year, election_type, election_code, round_number,
                uf, municipality_code, zone, office_code, is_transit_vote
            order by
                generated_at desc nulls last,
                last_totalization_at desc nulls last,
                source_file desc
        ) as _version_rank
    from {{ ref('bronze_tally_raw') }}
    {% if is_incremental() %}
    where {{ incremental_partition_predicate('election_year', 'election_type') }}
    {% endif %}
)

select * exclude (_version_rank)
from ranked
where _version_rank = 1
