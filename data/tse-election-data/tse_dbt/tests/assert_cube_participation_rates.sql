-- A null rate for a zero denominator is intentional, never 0% by invention.
select election_id, round_number, uf, municipality_code, office_code,
       is_transit_vote, eligible_voters, turnout, abstentions
from {{ ref('cube_participation_municipality') }}
where turnout_rate is distinct from (
    case when eligible_voters > 0 then turnout::double / eligible_voters end
)
or abstention_rate is distinct from (
    case when eligible_voters > 0 then abstentions::double / eligible_voters end
)
or uncounted_voters is distinct from eligible_voters - turnout - abstentions
