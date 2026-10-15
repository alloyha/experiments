{{ config(
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key=[
        'election_id', 'round_number', 'uf', 'municipality_code',
        'office_code', 'is_transit_vote', 'candidate_id'
    ],
    pre_hook="{{ partition_replace_pre_hook() }}",
    on_schema_change='sync_all_columns',
    meta={
        'load_semantics': 'partition_replace',
        'history_semantics': 'derived_snapshot',
        'partition_key': ['election_year', 'election_type']
    }
) }}

-- Nominal candidate votes ONLY: zone-to-municipality rollup within one
-- event/round/office/transit slice. Never infer elected status from ranking.
-- Aggregate the Parquet-backed fact BEFORE joining descriptive dimensions.
-- Only ONE explicitly selected cycle is materialized per call. Do not use a
-- global full-refresh: it repeats the huge Parquet/window operation at once.
-- IMPORTANT: when the relation is first created, filter even though dbt
-- is_incremental() is false. The ordinary incremental_election_filter macro
-- is unsuitable because it skips filtering on initial table creation.
with vote_rollup as (
    select
        cycle_id, election_id, election_year, election_type,
        election_scope, election_code, round_number, uf,
        municipality_code, office_code, is_transit_vote, candidate_id,
        sum(nominal_valid_votes) as nominal_valid_votes,
        count(*) as zone_rows,
        count(distinct zone) as zone_count
    from {{ ref('fact_candidate_votes') }}
    where {{ incremental_partition_predicate('election_year', 'election_type') }}
    -- 'year' is the Hive partition key from year=*/data.parquet.
    -- Filter it explicitly so unrelated Parquet files are pruned.
    and {{ incremental_partition_predicate('year', 'election_type') }}
    group by all
),
ranked as (
    select
        *,
        sum(nominal_valid_votes) over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
        ) as observed_candidate_nominal_valid_votes,
        row_number() over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
            order by nominal_valid_votes desc, candidate_id
        ) as municipal_candidate_rank
    from vote_rollup
)
select
    v.*,
    g.municipality,
    (g.municipality_code is not null) as geography_matched,
    (d.candidate_id is not null) as candidate_matched,
    d.candidate_number,
    d.candidate_name,
    d.ballot_name,
    d.party_number,
    d.party,
    d.party_name,
    d.office,
    d.office_scope,
    case when v.observed_candidate_nominal_valid_votes > 0
         then v.nominal_valid_votes::double /
              v.observed_candidate_nominal_valid_votes
    end as candidate_share_of_observed_nominal_votes
from ranked v
left join {{ ref('dim_candidate') }} d
  on d.election_id = v.election_id
 and d.candidate_id = v.candidate_id
left join {{ ref('dim_geography') }} g
  on g.cycle_id = v.cycle_id
 and g.uf = v.uf
 and g.municipality_code = v.municipality_code
