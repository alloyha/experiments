
  
  create view "tse_analytics"."main"."bronze_electorate__dbt_tmp" as (
    





with src as (
    -- The ingestion domain also contains temporary-transfer resources.
    -- Only the canonical electorate profile belongs in this staging model.
    select * 
    from 
  
    
    
    (
      with _index as (
        select distinct
          '/home/pingu/github/experiments/data/tse-election-data/data/tse' || '/' || object as object_path,
          year as _index_year,
          election_type as _election_type,
          election_scope as _election_scope
        from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
        where domain = 'electorate'
          and year in (2026)
          and election_type in ('general')
          
          and resource_name ilike 'Eleitorado - %'
          
      ),
      _raw as (
        select *
        from read_csv(
  [
    '/home/pingu/github/experiments/data/tse-election-data/data/tse/raw/election_type=general/year=2026/domain=electorate/dataset=eleitorado_2026/resource=4b4cf58f_7bff_424c_82ee_0b2768060e2e/sha256=0952e9d9577be19380b444bfcc6a9a99397005e8dc839de9d5ec49914d75648b/extracted/perfil_eleitorado_2026_BRASIL.csv'
  ],
  delim = ';',
  quote = '"',
  escape = '"',
  header = true,
  all_varchar = true,
  union_by_name = true,
  filename = true,
  sample_size = 20480,
  encoding = 'latin-1',
  strict_mode = false,
  null_padding = true,
  ignore_errors = false
)
      )
      select
        _raw.*,
        _index._election_type,
        _index._election_scope
      from _raw
      left join _index
        on replace(_raw.filename, '\\', '/') = replace(_index.object_path, '\\', '/')
    )
  

)
select
    try_cast("AA_ELEICAO" as integer) as election_year,
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
    try_cast("QT_ELEITORES" as bigint) as electorate,
    "DS_GENERO" as gender,
    "DS_ESTADO_CIVIL" as marital_status,
    "DS_FAIXA_ETARIA" as age_band,
    "DS_GRAU_ESCOLARIDADE" as schooling,
    filename as source_file
from src
  );
