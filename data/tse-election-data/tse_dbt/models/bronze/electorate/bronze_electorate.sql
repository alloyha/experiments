{% set selected_years = var('election_years', []) %}

{% set electorate_year_column =
    'AA_ELEICAO'
    if selected_years and selected_years[0] >= 2022
    else 'ANO_ELEICAO'
%}

{% set electorate_count_column =
    'QT_ELEITORES'
    if selected_years and selected_years[0] >= 2022
    else 'QT_ELEITORES_PERFIL'
%}

with src as (
    -- The ingestion domain also contains temporary-transfer resources.
    -- Only the canonical electorate profile belongs in this staging model.
    select * 
    from {{
       read_raw_csv(
           'electorate',
           'Eleitorado - %',
           strict_mode=false,
           null_padding=true,
           parallel=false
       )
    }}
)
select
    try_cast("{{ electorate_year_column }}" as integer) as election_year,
    _election_type as election_type,
    _election_scope as election_scope,
    "SG_UF" as uf,
    "NM_MUNICIPIO" as municipality,
    {{ normalize_municipality_code('"CD_MUNICIPIO"') }} as municipality_code,
    try_cast("{{ electorate_count_column }}" as bigint) as electorate,
    "DS_GENERO" as gender,
    "DS_ESTADO_CIVIL" as marital_status,
    "DS_FAIXA_ETARIA" as age_band,
    "DS_GRAU_ESCOLARIDADE" as schooling,
    filename as source_file
from src
