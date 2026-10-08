{{ config(
    pre_hook=partition_replace_pre_hook(),
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=[
      'election_year', 'election_type', 'election_code', 'round_number',
      'uf', 'municipality_code', 'zone', 'office_code', 'party_number',
      'is_transit_vote'
    ],
    on_schema_change='sync_all_columns',
    meta={
      'load_semantics': 'partition_replace',
      'history_semantics': 'authoritative_snapshot',
      'partition_key': ['election_year', 'election_type']
    }
) }}

select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,

    uf,
    municipality_code,
    zone,

    office_code,
    office_scope,

    party_number,
    is_transit_vote,

    nominal_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_votes,
    total_legend_valid_votes,

    coalesce(nominal_valid_votes, 0)
      + coalesce(total_legend_valid_votes, 0) as party_valid_votes,

    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    generated_at,
    source_file
from {{ ref('stg_party_votes_munzona') }}
{% if is_incremental() %}
where {{ incremental_partition_predicate('election_year', 'election_type') }}
{% endif %}
