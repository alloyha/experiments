-- No loss, duplication or rounding of counts in the zone-to-municipality step.
with expected as (
    select election_id, round_number, uf, municipality_code,
           office_code, is_transit_vote,
           sum(eligible_voters) as eligible_voters,
           sum(turnout) as turnout,
           sum(abstentions) as abstentions,
           sum(uncounted_voters) as uncounted_voters,
           sum(voters_uninstalled_sections) as voters_uninstalled_sections,
           count(*) as zone_rows,
           count(distinct zone) as zone_count
    from {{ ref('fact_turnout') }}
    group by 1, 2, 3, 4, 5, 6
),
actual as (
    select election_id, round_number, uf, municipality_code,
           office_code, is_transit_vote,
           eligible_voters, turnout, abstentions, uncounted_voters,
           voters_uninstalled_sections, zone_rows, zone_count
    from {{ ref('cube_participation_municipality') }}
)
(select 'missing_or_changed' as issue, * from expected
 except all
 select 'missing_or_changed', * from actual)
union all
(select 'unexpected_or_changed' as issue, * from actual
 except all
 select 'unexpected_or_changed', * from expected)
