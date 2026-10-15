-- EXCEPT ALL is essential: EXCEPT DISTINCT conceals duplicate fanout.
-- Scoped to the explicitly selected cycle because the derived cube is loaded
-- one cycle at a time. A PASS here is NOT full-history certification.
with expected as (
    select election_id, round_number, uf, municipality_code,
           office_code, is_transit_vote, candidate_id,
           sum(nominal_valid_votes) as nominal_valid_votes,
           count(*) as zone_rows, count(distinct zone) as zone_count
    from {{ ref('fact_candidate_votes') }}
    where {{ incremental_partition_predicate('election_year', 'election_type') }}
    and {{ incremental_partition_predicate('year', 'election_type') }}
    group by 1,2,3,4,5,6,7
),
actual as (
    select election_id, round_number, uf, municipality_code,
           office_code, is_transit_vote, candidate_id,
           nominal_valid_votes, zone_rows, zone_count
    from {{ ref('cube_candidate_performance_municipality') }}
    where {{ incremental_partition_predicate('election_year', 'election_type') }}
)
(select 'missing_or_changed' as issue, * from expected
 except all
 select 'missing_or_changed', * from actual)
union all
(select 'unexpected_or_changed' as issue, * from actual
 except all
 select 'unexpected_or_changed', * from expected)
