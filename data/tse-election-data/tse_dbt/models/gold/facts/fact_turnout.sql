{{ config(
    pre_hook="{{ partition_replace_pre_hook() }}",
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=[
      'election_year', 'election_type', 'election_code', 'round_number',
      'uf', 'municipality_code', 'zone', 'office_code', 'is_transit_vote'
    ],
    on_schema_change='sync_all_columns',
    meta={
      'load_semantics': 'partition_replace',
      'history_semantics': 'derived_snapshot',
      'partition_key': ['election_year', 'election_type']
    }
) }}

select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,
    is_transit_vote,

    eligible_voters,
    voters_uninstalled_sections,
    eligible_voters - turnout - abstentions as uncounted_voters,
    turnout,
    abstentions,

    case when eligible_voters > 0
         then turnout::double / eligible_voters
    end as turnout_rate,

    case when eligible_voters > 0
         then abstentions::double / eligible_voters
    end as abstention_rate,

    generated_at
from {{ ref('fact_tally_munzona') }}
{% if is_incremental() %}
where {{ incremental_partition_predicate('election_year', 'election_type') }}
{% endif %}
