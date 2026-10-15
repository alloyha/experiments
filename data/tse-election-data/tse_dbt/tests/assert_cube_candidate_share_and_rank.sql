-- Validate only the explicitly selected cycle. No full-history certification.
-- Vote share = candidate's observed nominal votes / all observed candidate
-- nominal votes in that exact event/round/office/municipality/transit slice.
with checked as (
    select
        *,
        sum(nominal_valid_votes) over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
        ) as expected_total,
        row_number() over (
            partition by election_id, round_number, uf, municipality_code,
                         office_code, is_transit_vote
            order by nominal_valid_votes desc, candidate_id
        ) as expected_rank
    from {{ ref('cube_candidate_performance_municipality') }}
    where {{ incremental_partition_predicate('election_year', 'election_type') }}
)
select election_id, candidate_id, round_number, uf, municipality_code, office_code
from checked
where observed_candidate_nominal_valid_votes is distinct from expected_total
   or municipal_candidate_rank is distinct from expected_rank
   or candidate_share_of_observed_nominal_votes is distinct from
      (case when expected_total > 0
            then nominal_valid_votes::double / expected_total end)
