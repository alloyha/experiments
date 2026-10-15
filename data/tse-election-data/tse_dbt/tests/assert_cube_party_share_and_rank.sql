-- No summing percentages and no addition of legend votes twice.
with checked as (
    select
        *,
        sum(party_valid_votes) over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
        ) as expected_total,
        row_number() over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
            order by party_valid_votes desc, party_number
        ) as expected_rank
    from {{ ref('cube_party_performance_municipality') }}
)
select election_id, party_number, round_number, uf, municipality_code, office_code
from checked
where observed_party_valid_votes is distinct from expected_total
   or municipal_party_rank is distinct from expected_rank
   or party_share_of_observed_valid_votes is distinct from
      (case when expected_total > 0
            then party_valid_votes::double / expected_total end)
   or party_valid_votes is distinct from
      (coalesce(nominal_valid_votes, 0) + coalesce(total_legend_valid_votes, 0))
