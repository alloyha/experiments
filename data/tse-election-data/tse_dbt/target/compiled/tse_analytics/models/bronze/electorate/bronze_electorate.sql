





with src as (
    -- The ingestion domain also contains temporary-transfer resources.
    -- Only the canonical electorate profile belongs in this staging model.
    select * 
    from 
  
    
    (
      select
        *,
        cast(null as varchar) as _election_type,
        cast(null as varchar) as _election_scope
      from read_csv(
        '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/fixtures/electorate.csv',
        delim = ',',
        header = true,
        all_varchar = true,
        union_by_name = true,
        filename = true,
        sample_size = 20480,
        encoding = 'utf-8',
        strict_mode = false,
        null_padding = true,
        ignore_errors = false
      )
      where false
    )
  

)
select
    try_cast("ANO_ELEICAO" as integer) as election_year,
    _election_type as election_type,
    _election_scope as election_scope,
    "SG_UF" as uf,
    "NM_MUNICIPIO" as municipality,
    
case
    when "CD_MUNICIPIO" is null then null
    when trim(cast("CD_MUNICIPIO" as varchar)) = '' then null
    else lpad(trim(cast("CD_MUNICIPIO" as varchar)), 5, '0')
end
 as municipality_code,
    try_cast("QT_ELEITORES_PERFIL" as bigint) as electorate,
    "DS_GENERO" as gender,
    "DS_ESTADO_CIVIL" as marital_status,
    "DS_FAIXA_ETARIA" as age_band,
    "DS_GRAU_ESCOLARIDADE" as schooling,
    filename as source_file
from src