with party_grains as (

    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        party_number,
        is_transit_vote,

        count(distinct nominal_valid_votes)
            filter (where nominal_valid_votes <> 0)
            as distinct_nonzero_nominal_values,

        count(distinct legend_valid_votes)
            filter (where legend_valid_votes <> 0)
            as distinct_nonzero_legend_values,

        count(distinct total_legend_valid_votes)
            filter (where total_legend_valid_votes <> 0)
            as distinct_nonzero_total_legend_values

    from {{ ref('stg_party_votes_raw') }}

    group by
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

select *
from party_grains
where distinct_nonzero_nominal_values > 1
   or distinct_nonzero_legend_values > 1
   or distinct_nonzero_total_legend_values > 1
