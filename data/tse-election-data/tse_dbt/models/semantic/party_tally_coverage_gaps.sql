{{ config(materialized='view') }}

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

        sum(coalesce(nominal_valid_votes, 0))
            as party_nominal_valid_votes,

        sum(coalesce(legend_valid_votes, 0))
            as party_legend_valid_votes,

        sum(coalesce(total_legend_valid_votes, 0))
            as party_total_legend_valid_votes,

        sum(coalesce(nominal_converted_to_legend_votes, 0))
            as party_nominal_converted_to_legend_votes,

        sum(coalesce(party_valid_votes, 0))
            as party_valid_votes,

        count(*) as party_rows

    from {{ ref('fact_party_votes') }}

    group by
        1,2,3,4,5,6,7,8,9

),

tally as (

    select
        *
    from {{ ref('fact_tally_munzona') }}

),

comparison as (

    select
        t.*,

        p.party_rows,
        p.party_nominal_valid_votes,
        p.party_legend_valid_votes,
        p.party_total_legend_valid_votes,
        p.party_nominal_converted_to_legend_votes,
        p.party_valid_votes,

        case

            -- No party source rows whatsoever at this tally grain.
            when p.party_rows is null
             and coalesce(t.valid_votes, 0) > 0
            then 'missing_party_source_coverage'

            -- Party source exists and its legend component is complete,
            -- but the nominal component is entirely absent.
            when p.party_rows is not null
             and coalesce(p.party_nominal_valid_votes, 0) = 0
             and coalesce(t.nominal_valid_votes, 0) > 0
             and coalesce(p.party_total_legend_valid_votes, 0)
                 = coalesce(t.total_legend_valid_votes, 0)
            then 'missing_party_nominal_coverage'

            -- TSE party totals may include nominal votes converted to
            -- legend totals. This creates a known non-comparable overlap.
            when p.party_rows is not null
             and coalesce(
                    p.party_nominal_converted_to_legend_votes,
                    0
                 ) > 0
             and coalesce(p.party_nominal_valid_votes, 0)
                 = coalesce(t.nominal_valid_votes, 0)
             and (
                    coalesce(p.party_valid_votes, 0)
                    - coalesce(t.valid_votes, 0)
                 )
                 = coalesce(
                     p.party_nominal_converted_to_legend_votes,
                     0
                   )
            then 'converted_vote_overlap'

            else null

        end as gap_reason

    from tally t

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

)

select
    *
from comparison
where gap_reason is not null

