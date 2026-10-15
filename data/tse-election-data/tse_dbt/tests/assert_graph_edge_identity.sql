{{ config(tags=['graph', 'expensive']) }}
select edge_id, count(*) as appearances
from {{ ref('graph_edges') }}
where {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
group by edge_id
having edge_id is null or count(*) <> 1
