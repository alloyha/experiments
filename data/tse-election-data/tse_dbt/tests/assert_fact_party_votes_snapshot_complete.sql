with source_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, party_number, is_transit_vote,
        nominal_valid_votes, legend_valid_votes,
        nominal_converted_to_legend_votes, total_legend_valid_votes,
        nominal_annulled_subjudice_votes, legend_annulled_subjudice_votes
    from {{ ref('stg_party_votes_munzona') }}
    where {{ selected_election_predicate() }}
),
target_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, party_number, is_transit_vote,
        nominal_valid_votes, legend_valid_votes,
        nominal_converted_to_legend_votes, total_legend_valid_votes,
        nominal_annulled_subjudice_votes, legend_annulled_subjudice_votes
    from {{ ref('fact_party_votes') }}
    where {{ selected_election_predicate() }}
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
