

with expected as (
    select
        '339'::varchar as election_code,
        'PE'::varchar as uf,
        '30015'::varchar as municipality_code,
        4::integer as zone,
        '25'::varchar as office_code,
        1836::bigint as nominal_valid_votes
),

actual as (
    select
        election_code,
        uf,
        municipality_code,
        zone,
        office_code,
        nominal_valid_votes
    from "tse_analytics"."main"."candidate_tally_coverage_gaps"
    where election_year = 2018
      and election_type = 'general'
)

(
    select * from expected
    except
    select * from actual
)
union all
(
    select * from actual
    except
    select * from expected
)