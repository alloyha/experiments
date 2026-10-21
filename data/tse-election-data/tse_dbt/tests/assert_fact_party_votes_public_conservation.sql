-- Fail with any violating row; intentionally no vote-total joins or re-aggregation.
-- Used to preserve the published vote-total identity across later snapshots.
select cycle_id, election_id, round_number, uf, municipality_code, zone,
       office_code, party_number, is_transit_vote
from {{ ref('fact_party_votes') }}
where party_valid_votes is distinct from
      coalesce(nominal_valid_votes, 0) + coalesce(total_legend_valid_votes, 0)
   or party_valid_votes < 0
   or cycle_id is distinct from cast(election_year as varchar) || ':' || election_type
   or election_id is distinct from cast(election_year as varchar) || ':' || election_type || ':' || election_code
