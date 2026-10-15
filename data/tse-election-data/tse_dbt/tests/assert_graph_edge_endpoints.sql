{{ config(tags=['graph', 'expensive']) }}
-- Edge endpoints must exist in the same cycle; no implicit cross-cycle bridge.
with nodes as (
    select node_id, cycle_id from {{ ref('graph_vertices') }}
), scoped_edges as (
    select * from {{ ref('graph_edges') }}
    where {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
)
select e.edge_id, e.edge_type, e.cycle_id
from scoped_edges e
left join nodes source_node
  on source_node.node_id = e.source_node_id and source_node.cycle_id = e.cycle_id
left join nodes target_node
  on target_node.node_id = e.target_node_id and target_node.cycle_id = e.cycle_id
where source_node.node_id is null or target_node.node_id is null
