{{ config(materialized='view') }}

-- The safe additive slice is EVENT x ROUND x OFFICE x MUNICIPALITY x TRANSIT.
-- Turnout/eligible voters are repeated across offices and must never be
-- summed across office_code, event or round without a selection policy.
with municipality_rollup as (
    select
        cycle_id,
        election_id,
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,
        uf,
        municipality_code,
        office_code,
        office_scope,
        is_transit_vote,
        sum(eligible_voters) as eligible_voters,
        sum(turnout) as turnout,
        sum(abstentions) as abstentions,
        sum(uncounted_voters) as uncounted_voters,
        sum(voters_uninstalled_sections) as voters_uninstalled_sections,
        count(*) as zone_rows,
        count(distinct zone) as zone_count,
        max(generated_at) as latest_generated_at
    from {{ ref('fact_turnout') }}
    group by
        cycle_id, election_id, election_year, election_type, election_scope,
        election_code, round_number, uf, municipality_code,
        office_code, office_scope, is_transit_vote
)
select
    p.*,
    g.municipality,
    (g.municipality_code is not null) as geography_matched,
    case when p.eligible_voters > 0
         then p.turnout::double / p.eligible_voters
    end as turnout_rate,
    case when p.eligible_voters > 0
         then p.abstentions::double / p.eligible_voters
    end as abstention_rate
from municipality_rollup p
left join {{ ref('dim_geography') }} g
  on g.cycle_id = p.cycle_id
 and g.uf = p.uf
 and g.municipality_code = p.municipality_code
