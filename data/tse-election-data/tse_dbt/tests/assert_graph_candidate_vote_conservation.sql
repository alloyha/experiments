{{ config(tags=['graph', 'expensive']) }}
-- Multiset equality: no missing, duplicate, mutated or cross-contest edge.
with expected_rollup as (
    select cycle_id, election_id, round_number, uf, municipality_code,
        office_code, is_transit_vote, candidate_id,
        sum(nominal_valid_votes) as votes, count(*) as rows,
        count(distinct zone) as zones
    from {{ ref('fact_candidate_votes') }}
    where {{ incremental_partition_predicate('election_year', 'election_type') }}
      and {{ incremental_partition_predicate('year', 'election_type') }}
    group by all
), expected as (
    select
        to_json(list_value('contest', cast(election_id as varchar),
            cast(round_number as varchar), cast(uf as varchar),
            cast(municipality_code as varchar), cast(office_code as varchar),
            cast(is_transit_vote as varchar))) as source_id,
        to_json(list_value('candidate', cast(election_id as varchar),
            cast(candidate_id as varchar))) as target_id,
        cycle_id, election_id, round_number, office_code, is_transit_vote,
        votes as nominal_valid_votes, rows as zone_rows, zones as zone_count
    from expected_rollup
), actual as (
    select source_node_id as source_id, target_node_id as target_id,
        cycle_id, election_id, round_number, office_code, is_transit_vote,
        nominal_valid_votes, zone_rows, zone_count
    from {{ ref('graph_edges') }}
    where edge_type = 'VOTED_FOR_CANDIDATE'
      and {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
)
(select 'missing_or_changed' as issue, * from expected
 except all select 'missing_or_changed', * from actual)
union all
(select 'unexpected_or_changed' as issue, * from actual
 except all select 'unexpected_or_changed', * from expected)
