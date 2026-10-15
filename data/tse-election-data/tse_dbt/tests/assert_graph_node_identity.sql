{{ config(tags=['graph', 'expensive']) }}
-- A duplicate typed identity is a failed projection, even if labels differ.
select node_id, count(*) as appearances
from {{ ref('graph_vertices') }}
where {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
group by node_id
having node_id is null or count(*) <> 1
