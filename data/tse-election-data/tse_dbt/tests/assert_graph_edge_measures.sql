{{ config(tags=['graph', 'expensive']) }}
-- Structural links carry no voting amounts; vote links have an exact type.
with edges as (
    select * from {{ ref('graph_edges') }}
    where {{ incremental_partition_predicate("cast(split_part(cycle_id, ':', 1) as integer)", "split_part(cycle_id, ':', 2)") }}
)
select edge_id, edge_type
from edges
where
    (edge_type not in ('VOTED_FOR_CANDIDATE', 'VOTED_FOR_PARTY')
      and (nominal_valid_votes is not null or legend_valid_votes is not null
           or total_legend_valid_votes is not null or party_valid_votes is not null
           or zone_rows is not null or zone_count is not null))
    or (edge_type = 'VOTED_FOR_CANDIDATE'
      and (nominal_valid_votes is null or legend_valid_votes is not null
           or total_legend_valid_votes is not null or party_valid_votes is not null
           or zone_rows is null or zone_count is null))
    or (edge_type = 'VOTED_FOR_PARTY'
      and (nominal_valid_votes is null or total_legend_valid_votes is null
           or party_valid_votes is null or party_valid_votes is distinct from
              (coalesce(nominal_valid_votes, 0) + coalesce(total_legend_valid_votes, 0))
           or zone_rows is null or zone_count is null))
