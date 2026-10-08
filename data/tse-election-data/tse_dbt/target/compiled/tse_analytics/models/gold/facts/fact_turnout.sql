

select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,
    is_transit_vote,

    eligible_voters,
    voters_uninstalled_sections,
    eligible_voters - turnout - abstentions as uncounted_voters,
    turnout,
    abstentions,

    case when eligible_voters > 0
         then turnout::double / eligible_voters
    end as turnout_rate,

    case when eligible_voters > 0
         then abstentions::double / eligible_voters
    end as abstention_rate,

    generated_at
from "tse_analytics"."main"."fact_tally_munzona"

where election_year in (2018) and election_type in ('general')
