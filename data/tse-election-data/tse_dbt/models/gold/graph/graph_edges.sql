{{ config(materialized='view', tags=['graph', 'expensive'], meta={'graph_role': 'edges', 'history_semantics': 'derived_live_gold'}) }}

-- One direction/type per association. Vote edges preserve the full contest
-- context and each underlying vote measure; never infer individual voters,
-- winner status, same-party equality or temporal continuity from adjacency.
-- Expensive lazy view: use explicit cycle-scoped tests and benchmark before use.
with candidate_vote_rollup as (
    select cycle_id, election_id, round_number, uf, municipality_code,
        office_code, is_transit_vote, candidate_id,
        sum(nominal_valid_votes) as nominal_valid_votes,
        count(*) as zone_rows, count(*) as zone_count
    from {{ ref('fact_candidate_votes') }}
    group by all
),
party_vote_rollup as (
    select cycle_id, election_id, round_number, uf, municipality_code,
        office_code, is_transit_vote, party_number,
        sum(nominal_valid_votes) as nominal_valid_votes,
        sum(legend_valid_votes) as legend_valid_votes,
        sum(total_legend_valid_votes) as total_legend_valid_votes,
        sum(party_valid_votes) as party_valid_votes,
        count(*) as zone_rows, count(distinct zone) as zone_count
    from {{ ref('fact_party_votes') }}
    group by all
),
contest_keys as (
    select distinct cycle_id, election_id, round_number, uf, municipality_code,
        office_code, is_transit_vote from candidate_vote_rollup
    union
    select distinct cycle_id, election_id, round_number, uf, municipality_code,
        office_code, is_transit_vote from party_vote_rollup
)
select
    to_json(list_value('edge:CONTAINS_ELECTION', cast(to_json(list_value('cycle', cast(cycle_id as varchar))) as varchar), cast(to_json(list_value('election', cast(election_id as varchar))) as varchar))) as edge_id,
    'CONTAINS_ELECTION' as edge_type,
    to_json(list_value('cycle', cast(cycle_id as varchar))) as source_node_id,
    to_json(list_value('election', cast(election_id as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    cast(null as bigint) as round_number,
    cast(null as varchar) as office_code,
    cast(null as boolean) as is_transit_vote,
    cast(null as hugeint) as nominal_valid_votes,
    cast(null as hugeint) as legend_valid_votes,
    cast(null as hugeint) as total_legend_valid_votes,
    cast(null as hugeint) as party_valid_votes,
    cast(null as bigint) as zone_rows,
    cast(null as bigint) as zone_count,
    'dim_election' as source_relation
from {{ ref('dim_election') }}
union all
select
    to_json(list_value('edge:HAS_CONTEST', cast(to_json(list_value('election', cast(election_id as varchar))) as varchar), cast(to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as varchar))) as edge_id,
    'HAS_CONTEST' as edge_type,
    to_json(list_value('election', cast(election_id as varchar))) as source_node_id,
    to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    round_number as round_number,
    office_code as office_code,
    is_transit_vote as is_transit_vote,
    cast(null as hugeint) as nominal_valid_votes,
    cast(null as hugeint) as legend_valid_votes,
    cast(null as hugeint) as total_legend_valid_votes,
    cast(null as hugeint) as party_valid_votes,
    cast(null as bigint) as zone_rows,
    cast(null as bigint) as zone_count,
    'fact_votes' as source_relation
from contest_keys
union all
select
    to_json(list_value('edge:LOCATED_IN', cast(to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as varchar), cast(to_json(list_value('municipality', cast(cycle_id as varchar), cast(uf as varchar), cast(municipality_code as varchar))) as varchar))) as edge_id,
    'LOCATED_IN' as edge_type,
    to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as source_node_id,
    to_json(list_value('municipality', cast(cycle_id as varchar), cast(uf as varchar), cast(municipality_code as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    round_number as round_number,
    office_code as office_code,
    is_transit_vote as is_transit_vote,
    cast(null as hugeint) as nominal_valid_votes,
    cast(null as hugeint) as legend_valid_votes,
    cast(null as hugeint) as total_legend_valid_votes,
    cast(null as hugeint) as party_valid_votes,
    cast(null as bigint) as zone_rows,
    cast(null as bigint) as zone_count,
    'fact_votes' as source_relation
from contest_keys
union all
select
    to_json(list_value('edge:HAS_CANDIDATE', cast(to_json(list_value('election', cast(election_id as varchar))) as varchar), cast(to_json(list_value('candidate', cast(election_id as varchar), cast(candidate_id as varchar))) as varchar))) as edge_id,
    'HAS_CANDIDATE' as edge_type,
    to_json(list_value('election', cast(election_id as varchar))) as source_node_id,
    to_json(list_value('candidate', cast(election_id as varchar), cast(candidate_id as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    cast(null as bigint) as round_number,
    cast(null as varchar) as office_code,
    cast(null as boolean) as is_transit_vote,
    cast(null as hugeint) as nominal_valid_votes,
    cast(null as hugeint) as legend_valid_votes,
    cast(null as hugeint) as total_legend_valid_votes,
    cast(null as hugeint) as party_valid_votes,
    cast(null as bigint) as zone_rows,
    cast(null as bigint) as zone_count,
    'dim_candidate' as source_relation
from {{ ref('dim_candidate') }}
union all
select
    to_json(list_value('edge:HAS_PARTY', cast(to_json(list_value('election', cast(election_id as varchar))) as varchar), cast(to_json(list_value('party', cast(election_id as varchar), cast(party_number as varchar))) as varchar))) as edge_id,
    'HAS_PARTY' as edge_type,
    to_json(list_value('election', cast(election_id as varchar))) as source_node_id,
    to_json(list_value('party', cast(election_id as varchar), cast(party_number as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    cast(null as bigint) as round_number,
    cast(null as varchar) as office_code,
    cast(null as boolean) as is_transit_vote,
    cast(null as hugeint) as nominal_valid_votes,
    cast(null as hugeint) as legend_valid_votes,
    cast(null as hugeint) as total_legend_valid_votes,
    cast(null as hugeint) as party_valid_votes,
    cast(null as bigint) as zone_rows,
    cast(null as bigint) as zone_count,
    'dim_party' as source_relation
from {{ ref('dim_party') }}
union all
select
    to_json(list_value('edge:AFFILIATED_WITH', cast(to_json(list_value('candidate', cast(election_id as varchar), cast(candidate_id as varchar))) as varchar), cast(to_json(list_value('party', cast(election_id as varchar), cast(party_number as varchar))) as varchar))) as edge_id,
    'AFFILIATED_WITH' as edge_type,
    to_json(list_value('candidate', cast(election_id as varchar), cast(candidate_id as varchar))) as source_node_id,
    to_json(list_value('party', cast(election_id as varchar), cast(party_number as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    cast(null as bigint) as round_number,
    cast(null as varchar) as office_code,
    cast(null as boolean) as is_transit_vote,
    cast(null as hugeint) as nominal_valid_votes,
    cast(null as hugeint) as legend_valid_votes,
    cast(null as hugeint) as total_legend_valid_votes,
    cast(null as hugeint) as party_valid_votes,
    cast(null as bigint) as zone_rows,
    cast(null as bigint) as zone_count,
    'dim_candidate' as source_relation
from {{ ref('dim_candidate') }} where party_number is not null
union all
select
    to_json(list_value('edge:VOTED_FOR_CANDIDATE', cast(to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as varchar), cast(to_json(list_value('candidate', cast(election_id as varchar), cast(candidate_id as varchar))) as varchar))) as edge_id,
    'VOTED_FOR_CANDIDATE' as edge_type,
    to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as source_node_id,
    to_json(list_value('candidate', cast(election_id as varchar), cast(candidate_id as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    round_number as round_number,
    office_code as office_code,
    is_transit_vote as is_transit_vote,
    nominal_valid_votes as nominal_valid_votes,
    cast(null as hugeint) as legend_valid_votes,
    cast(null as hugeint) as total_legend_valid_votes,
    cast(null as hugeint) as party_valid_votes,
    zone_rows as zone_rows,
    zone_count as zone_count,
    'fact_candidate_votes' as source_relation
from candidate_vote_rollup
union all
select
    to_json(list_value('edge:VOTED_FOR_PARTY', cast(to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as varchar), cast(to_json(list_value('party', cast(election_id as varchar), cast(party_number as varchar))) as varchar))) as edge_id,
    'VOTED_FOR_PARTY' as edge_type,
    to_json(list_value('contest', cast(election_id as varchar), cast(round_number as varchar), cast(uf as varchar), cast(municipality_code as varchar), cast(office_code as varchar), cast(is_transit_vote as varchar))) as source_node_id,
    to_json(list_value('party', cast(election_id as varchar), cast(party_number as varchar))) as target_node_id,
    cycle_id as cycle_id,
    election_id as election_id,
    round_number as round_number,
    office_code as office_code,
    is_transit_vote as is_transit_vote,
    nominal_valid_votes as nominal_valid_votes,
    legend_valid_votes as legend_valid_votes,
    total_legend_valid_votes as total_legend_valid_votes,
    party_valid_votes as party_valid_votes,
    zone_rows as zone_rows,
    zone_count as zone_count,
    'fact_party_votes' as source_relation
from party_vote_rollup
