{{ config(materialized='view') }}

-- Keep nominal and legend votes separate. total_legend_valid_votes already
-- incorporates any converted-to-legend component; do not add it a second time.
-- Aggregate BEFORE joining dim_party/geography to prevent accidental fanout.
with vote_rollup as (
    select
        cycle_id, election_id, election_year, election_type,
        election_scope, election_code, round_number, uf,
        municipality_code, office_code, is_transit_vote, party_number,
        sum(nominal_valid_votes) as nominal_valid_votes,
        sum(legend_valid_votes) as legend_valid_votes,
        sum(total_legend_valid_votes) as total_legend_valid_votes,
        sum(party_valid_votes) as party_valid_votes,
        count(*) as zone_rows,
        count(distinct zone) as zone_count
    from {{ ref('fact_party_votes') }}
    group by all
),
ranked as (
    select
        *,
        sum(party_valid_votes) over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
        ) as observed_party_valid_votes,
        row_number() over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
            order by party_valid_votes desc, party_number
        ) as municipal_party_rank
    from vote_rollup
)
select
    v.*,
    g.municipality,
    (g.municipality_code is not null) as geography_matched,
    (d.party_number is not null) as party_matched,
    d.party,
    d.party_name,
    d.party_id,
    case when v.observed_party_valid_votes > 0
         then v.party_valid_votes::double / v.observed_party_valid_votes
    end as party_share_of_observed_valid_votes
from ranked v
left join {{ ref('dim_party') }} d
  on d.election_id = v.election_id
 and d.party_number = v.party_number
left join {{ ref('dim_geography') }} g
  on g.cycle_id = v.cycle_id
 and g.uf = v.uf
 and g.municipality_code = v.municipality_code
