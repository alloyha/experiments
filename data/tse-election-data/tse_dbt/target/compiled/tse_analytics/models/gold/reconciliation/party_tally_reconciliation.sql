with party as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,

        sum(nominal_valid_votes) as party_nominal_valid_votes,
        sum(total_legend_valid_votes) as party_total_legend_valid_votes,
        sum(party_valid_votes) as party_valid_votes
    from "tse_analytics"."main"."fact_party_votes"
    group by 1,2,3,4,5,6,7,8,9
)

select
    t.election_year,
    t.election_type,
    t.election_code,
    t.round_number,
    t.uf,
    t.municipality_code,
    t.zone,
    t.office_code,
    t.is_transit_vote,

    t.nominal_valid_votes as tally_nominal_valid_votes,
    coalesce(p.party_nominal_valid_votes, 0) as party_nominal_valid_votes,
    coalesce(p.party_nominal_valid_votes, 0) - t.nominal_valid_votes
      as nominal_valid_delta,

    t.total_legend_valid_votes as tally_total_legend_valid_votes,
    coalesce(p.party_total_legend_valid_votes, 0) as party_total_legend_valid_votes,
    coalesce(p.party_total_legend_valid_votes, 0) - t.total_legend_valid_votes
      as total_legend_valid_delta,

    t.valid_votes as tally_valid_votes,
    coalesce(p.party_valid_votes, 0) as party_valid_votes,
    coalesce(p.party_valid_votes, 0) - t.valid_votes
      as total_valid_delta

from "tse_analytics"."main"."fact_tally_munzona" t
left join party p
  on p.election_year = t.election_year
 and p.election_type = t.election_type
 and p.election_code = t.election_code
 and p.round_number = t.round_number
 and p.uf = t.uf
 and p.municipality_code = t.municipality_code
 and p.zone = t.zone
 and p.office_code = t.office_code
 and p.is_transit_vote is not distinct from t.is_transit_vote

where not exists (
    select 1
    from "tse_analytics"."main"."party_tally_coverage_gaps" g
    where g.election_year = t.election_year
      and g.election_type = t.election_type
      and g.election_code = t.election_code
      and g.round_number = t.round_number
      and g.uf = t.uf
      and g.municipality_code = t.municipality_code
      and g.zone = t.zone
      and g.office_code = t.office_code
      and g.is_transit_vote = t.is_transit_vote
)