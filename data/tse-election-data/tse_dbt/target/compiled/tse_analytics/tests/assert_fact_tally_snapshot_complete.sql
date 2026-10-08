with source_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, is_transit_vote,
        eligible_voters, voters_uninstalled_sections, turnout, abstentions,
        total_votes, valid_votes,
        nominal_valid_votes, total_legend_valid_votes,
        blank_votes, total_null_votes,
        annulled_votes, annulled_subjudice_votes
    from "tse_analytics"."main"."bronze_tally_munzona"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, is_transit_vote,
        eligible_voters, voters_uninstalled_sections, turnout, abstentions,
        total_votes, valid_votes,
        nominal_valid_votes, total_legend_valid_votes,
        blank_votes, total_null_votes,
        annulled_votes, annulled_subjudice_votes
    from "tse_analytics"."main"."fact_tally_munzona"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_or_changed_in_target' as issue, * from source_rows
     except
     select 'missing_or_changed_in_target' as issue, * from target_rows)
    union all
    (select 'stale_or_changed_in_target' as issue, * from target_rows
     except
     select 'stale_or_changed_in_target' as issue, * from source_rows)
)
select * from diff