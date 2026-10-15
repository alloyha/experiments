-- Preserve exact party nominal/legend totals AND output multiplicity.
with expected as (
    select election_id, round_number, uf, municipality_code,
           office_code, is_transit_vote, party_number,
           sum(nominal_valid_votes) as nominal_valid_votes,
           sum(legend_valid_votes) as legend_valid_votes,
           sum(total_legend_valid_votes) as total_legend_valid_votes,
           sum(party_valid_votes) as party_valid_votes,
           count(*) as zone_rows, count(distinct zone) as zone_count
    from {{ ref('fact_party_votes') }}
    group by 1,2,3,4,5,6,7
),
actual as (
    select election_id, round_number, uf, municipality_code,
           office_code, is_transit_vote, party_number,
           nominal_valid_votes, legend_valid_votes,
           total_legend_valid_votes, party_valid_votes,
           zone_rows, zone_count
    from {{ ref('cube_party_performance_municipality') }}
)
(select 'missing_or_changed' as issue, * from expected
 except all
 select 'missing_or_changed', * from actual)
union all
(select 'unexpected_or_changed' as issue, * from actual
 except all
 select 'unexpected_or_changed', * from expected)
