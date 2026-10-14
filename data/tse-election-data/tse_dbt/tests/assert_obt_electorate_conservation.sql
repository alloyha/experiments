-- Separate aggregation on either side catches join fanout or missing rows.
with expected as (
    select cycle_id, uf,
           count(*) as row_count,
           sum(electorate) as total_electorate
    from {{ ref('fact_electorate_municipality') }}
    group by 1, 2
),
actual as (
    select cycle_id, uf,
           count(*) as row_count,
           sum(electorate) as total_electorate
    from {{ ref('obt_electorate_municipality') }}
    group by 1, 2
)
(select 'missing_or_changed' as issue, * from expected
 except all
 select 'missing_or_changed', * from actual)
union all
(select 'unexpected_or_changed' as issue, * from actual
 except all
 select 'unexpected_or_changed', * from expected)
