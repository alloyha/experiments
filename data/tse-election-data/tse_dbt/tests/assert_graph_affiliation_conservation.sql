{{ config(tags=['graph', 'expensive']) }}
-- Full-dimensional membership equivalence; never infer timeless party membership.
-- EXCEPT ALL (both directions) preserves multiplicity and detects spurious,
-- missing or redirected affiliations even when both endpoint vertices exist.
with expected as (
    select
        to_json(list_value('edge:AFFILIATED_WITH',
            cast(to_json(list_value('candidate', cast(election_id as varchar),
                cast(candidate_id as varchar))) as varchar),
            cast(to_json(list_value('party', cast(election_id as varchar),
                cast(party_number as varchar))) as varchar))) as edge_id,
        to_json(list_value('candidate', cast(election_id as varchar),
            cast(candidate_id as varchar))) as source_node_id,
        to_json(list_value('party', cast(election_id as varchar),
            cast(party_number as varchar))) as target_node_id,
        cycle_id, election_id, 'dim_candidate' as source_relation
    from {{ ref('dim_candidate') }}
    where party_number is not null
      and {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
), actual as (
    select edge_id, source_node_id, target_node_id, cycle_id, election_id,
           source_relation
    from {{ ref('graph_edges') }}
    where edge_type = 'AFFILIATED_WITH'
      and {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
)
(select 'missing_or_changed' as issue, * from expected
 except all select 'missing_or_changed', * from actual)
union all
(select 'unexpected_or_changed' as issue, * from actual
 except all select 'unexpected_or_changed', * from expected)
