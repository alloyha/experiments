{{ config(tags=['graph', 'expensive']) }}
-- Multiset equality: no missing, duplicate, mutated or cross-contest edge.
with expected_rollup as (
    select cycle_id, election_id, round_number, uf, municipality_code,
        office_code, is_transit_vote, candidate_id,
        sum(nominal_valid_votes) as votes, count(*) as rows,
        count(*) as zones
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
-- Signed multiset reconciliation. Each row contributes +1/-1; grouping
-- includes all ten typed attributes, preserving NULL and duplicate semantics.
, combined as (
    select source_id, target_id, cycle_id, election_id, round_number,
        office_code, is_transit_vote, nominal_valid_votes, zone_rows, zone_count,
        1::HUGEINT as multiplicity_delta from expected
    union all
    select source_id, target_id, cycle_id, election_id, round_number,
        office_code, is_transit_vote, nominal_valid_votes, zone_rows, zone_count,
        (-1)::HUGEINT as multiplicity_delta from actual
), violations as (
    select source_id, target_id, cycle_id, election_id, round_number,
        office_code, is_transit_vote, nominal_valid_votes, zone_rows, zone_count,
        sum(multiplicity_delta) as signed_multiplicity
    from combined
    group by all
    having sum(multiplicity_delta) <> 0
)
select case when signed_multiplicity > 0 then 'missing_or_changed'
            else 'unexpected_or_changed' end as issue,
       source_id, target_id, cycle_id, election_id, round_number,
       office_code, is_transit_vote, nominal_valid_votes, zone_rows, zone_count,
       abs(signed_multiplicity) as differing_rows
from violations
