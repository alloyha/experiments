with candidate_by_party as (
    select
        f.election_year,
        f.election_type,
        f.election_code,
        f.round_number,
        f.uf,
        f.municipality_code,
        f.zone,
        f.office_code,
        d.party_number,
        f.is_transit_vote,
        sum(f.nominal_valid_votes) as candidate_nominal_valid_votes
    from "tse_analytics"."main"."fact_candidate_votes" f
    inner join "tse_analytics"."main"."dim_candidate" d
      on d.election_year = f.election_year
     and d.election_type = f.election_type
     and d.election_code = f.election_code
     and d.candidate_id = f.candidate_id
    group by 1,2,3,4,5,6,7,8,9,10
)

select
    p.election_year,
    p.election_type,
    p.election_scope,
    p.election_id,
    p.election_code,
    p.round_number,

    p.uf,
    p.municipality_code,
    p.zone,

    p.office_code,
    p.office_scope,

    p.party_number,
    d.party,
    d.party_name,
    d.party_id,

    p.is_transit_vote,

    p.nominal_valid_votes as party_reported_nominal_valid_votes,
    coalesce(c.candidate_nominal_valid_votes, 0) as candidate_nominal_valid_votes,
    p.nominal_valid_votes - coalesce(c.candidate_nominal_valid_votes, 0) as nominal_reconciliation_delta,

    p.legend_valid_votes,
    p.party_valid_votes,

    t.nominal_valid_votes as tally_nominal_valid_votes,
    p.nominal_valid_votes - t.nominal_valid_votes as nominal_tally_delta,

    t.valid_votes,
    t.turnout,
    t.eligible_voters,
    t.abstentions,

    case when t.valid_votes > 0
         then p.party_valid_votes::double / t.valid_votes
    end as vote_share,

    case when t.eligible_voters > 0
         then t.turnout::double / t.eligible_voters
    end as turnout_rate,

    case when t.eligible_voters > 0
         then t.abstentions::double / t.eligible_voters
    end as abstention_rate,

    p.generated_at
from "tse_analytics"."main"."fact_party_votes" p
left join candidate_by_party c
  using (
    election_year,
    election_type,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    party_number,
    is_transit_vote
  )
left join "tse_analytics"."main"."dim_party" d
  on d.election_year = p.election_year
 and d.election_type = p.election_type
 and d.election_code = p.election_code
 and d.party_number = p.party_number
left join "tse_analytics"."main"."fact_tally_munzona" t
  on t.election_year = p.election_year
 and t.election_type = p.election_type
 and t.election_code = p.election_code
 and t.round_number = p.round_number
 and t.uf = p.uf
 and t.municipality_code = p.municipality_code
 and t.zone = p.zone
 and t.office_code = p.office_code
 and t.is_transit_vote is not distinct from p.is_transit_vote