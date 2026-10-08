with src as (
    select * from 
  
    
    (
      select
        *,
        cast(null as varchar) as _election_type,
        cast(null as varchar) as _election_scope
      from read_csv(
        '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/fixtures/candidate_assets.csv',
        delim = ',',
        header = true,
        all_varchar = true,
        union_by_name = true,
        filename = true,
        sample_size = 20480,
        encoding = 'utf-8',
        strict_mode = true,
        null_padding = false,
        ignore_errors = false
      )
      where false
    )
  

)
select
    try_cast("ANO_ELEICAO" as integer) as election_year,
    _election_type as election_type,
    _election_scope as election_scope,
    "CD_ELEICAO" as election_code,
    "DS_ELEICAO" as election_description,
    "SG_UF" as uf,
    "SG_UE" as electoral_unit,
    "SQ_CANDIDATO" as candidate_id,
    "DS_TIPO_BEM_CANDIDATO" as asset_type,
    "DS_BEM_CANDIDATO" as asset_description,
    try_cast(replace(replace("VR_BEM_CANDIDATO", '.', ''), ',', '.') as decimal(18,2)) as asset_value,
    filename as source_file
from src