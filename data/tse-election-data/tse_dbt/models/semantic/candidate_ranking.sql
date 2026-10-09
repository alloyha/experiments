{{ config(materialized='table') }}

with vote_rollup as (
    -- Collapse municipality/zone cardinality before joining descriptive dimensions.
    select
        election_year,
        election_type,
        election_scope,
        election_id,
        election_code,
        round_number,
        uf,
        office_code,
        candidate_id,
        sum(nominal_valid_votes) as nominal_valid_votes
    from {{ ref('fact_candidate_votes') }}
    group by 1,2,3,4,5,6,7,8,9
),

candidate_enriched as (
    select
        v.election_year,
        v.election_type,
        v.election_scope,
        v.election_id,
        v.election_code,
        v.round_number,
        coalesce(d.electoral_unit, d.uf, v.uf) as electoral_unit,
        v.office_code,
        d.office,
        d.office_scope,
        v.candidate_id,
        d.candidate_number,
        d.candidate_name,
        d.ballot_name,
        d.party_number,
        d.party,
        d.party_name,
        v.nominal_valid_votes
    from vote_rollup v
    inner join {{ ref('dim_candidate') }} d
      on d.election_year = v.election_year
     and d.election_type = v.election_type
     and d.election_code = v.election_code
     and d.candidate_id = v.candidate_id
),

contest_rollup as (
    -- vote_rollup retained UF only so the large fact could be reduced before
    -- electoral_unit became available from dim_candidate.
    select
        election_year,
        election_type,
        election_scope,
        election_id,
        election_code,
        round_number,
        electoral_unit,
        office_code,
        office,
        office_scope,
        candidate_id,
        candidate_number,
        candidate_name,
        ballot_name,
        party_number,
        party,
        party_name,
        sum(nominal_valid_votes) as nominal_valid_votes
    from candidate_enriched
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
),

ranked as (
    select
        *,
        sum(nominal_valid_votes) over (
            partition by
                election_year,
                election_type,
                election_code,
                round_number,
                electoral_unit,
                office_code
        ) as contest_candidate_nominal_valid_votes,
        row_number() over (
            partition by
                election_year,
                election_type,
                election_code,
                round_number,
                electoral_unit,
                office_code
            order by nominal_valid_votes desc, candidate_id
        ) as candidate_rank,
        lag(nominal_valid_votes) over (
            partition by
                election_year,
                election_type,
                election_code,
                round_number,
                electoral_unit,
                office_code
            order by nominal_valid_votes desc, candidate_id
        ) as previous_candidate_votes,
        lead(nominal_valid_votes) over (
            partition by
                election_year,
                election_type,
                election_code,
                round_number,
                electoral_unit,
                office_code
            order by nominal_valid_votes desc, candidate_id
        ) as next_candidate_votes
    from contest_rollup
)

select
    *,
    case
        when contest_candidate_nominal_valid_votes > 0
        then nominal_valid_votes::double / contest_candidate_nominal_valid_votes
    end as candidate_nominal_vote_share,
    candidate_rank = 1 as is_top_ranked,
    case
        when previous_candidate_votes is not null
        then previous_candidate_votes - nominal_valid_votes
    end as votes_behind_previous,
    case
        when next_candidate_votes is not null
        then nominal_valid_votes - next_candidate_votes
    end as lead_over_next_votes
from ranked
