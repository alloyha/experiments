{{ config(
    pre_hook="{{ partition_replace_pre_hook() }}",
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
      'history_semantics': 'current_source_snapshot',
      'partition_key': ['election_year', 'election_type'],
      'resource_class': 'heavy_group_reduce'
    }
) }}

with source_rows as (
    select *
    from {{ ref('bronze_party_votes_raw') }}
    {% if is_incremental() %}
    where {{ incremental_partition_predicate('election_year', 'election_type') }}
    {% endif %}
),

collapsed as (
    select
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,

        uf,
        municipality_code,
        zone,

        office_code,
        office_scope,

        party_number,
        is_transit_vote,

        max(party) as party,
        max(party_name) as party_name,

        max(nominal_valid_votes) as nominal_valid_votes,
        max(legend_valid_votes) as legend_valid_votes,
        max(nominal_converted_to_legend_votes) as nominal_converted_to_legend_votes,
        max(total_legend_valid_votes) as total_legend_valid_votes,

        max(nominal_annulled_subjudice_votes) as nominal_annulled_subjudice_votes,
        max(legend_annulled_subjudice_votes) as legend_annulled_subjudice_votes,

        max(generated_at) as generated_at,
        max(source_file) as source_file,

        count(*) as source_row_count,
        count(distinct party_group_type) as source_party_group_types,
        count(distinct coalition_id) as source_coalitions,
        count(distinct federation_number) as source_federations

    from source_rows
    group by
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        office_scope,
        party_number,
        is_transit_vote
)

select *
from collapsed
