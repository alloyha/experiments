{{ config(materialized='view') }}

-- Cycle-level electorate. Never join a candidate, event, office or round
-- before producing this OBT: those grains multiply the electorate.
with conformed as (
    select
        f.cycle_id,
        f.election_year,
        f.election_type,
        f.election_scope,
        f.uf,
        f.municipality_code,
        coalesce(g.municipality, f.municipality) as municipality,
        f.electorate,
        (g.municipality_code is not null) as geography_matched
    from {{ ref('fact_electorate_municipality') }} f
    left join {{ ref('dim_geography') }} g
      on g.cycle_id = f.cycle_id
     and g.uf = f.uf
     and g.municipality_code = f.municipality_code
),
with_history as (
    select
        *,
        lag(cycle_id) over (
            partition by uf, municipality_code
            order by election_year, election_type
        ) as previous_cycle_id,
        lag(electorate) over (
            partition by uf, municipality_code
            order by election_year, election_type
        ) as previous_electorate
    from conformed
)
select
    cycle_id,
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    geography_matched,
    electorate,
    previous_cycle_id,
    previous_electorate,
    electorate - previous_electorate as electorate_delta,
    case when previous_electorate > 0
         then (electorate - previous_electorate)::double / previous_electorate
    end as electorate_growth_rate
from with_history
