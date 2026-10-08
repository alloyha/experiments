

select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

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
    turnout,
    abstentions,

    total_votes,
    competing_votes,

    valid_votes,
    nominal_valid_votes,
    total_legend_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_valid_votes,

    annulled_votes,
    nominal_annulled_votes,
    legend_annulled_votes,

    annulled_subjudice_votes,
    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    blank_votes,
    total_null_votes,
    null_votes,
    technical_null_votes,
    separately_counted_annulled_votes,

    generated_at,
    last_totalization_at,
    source_file
from "tse_analytics"."main"."bronze_tally_munzona"

where election_year in (2026) and election_type in ('general')
