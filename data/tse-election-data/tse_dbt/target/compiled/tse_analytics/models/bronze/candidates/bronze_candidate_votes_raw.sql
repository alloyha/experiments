

with src as (
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
        where domain = 'vote_result'
          and year in (2026)
          and election_type in ('general')
          
          and resource_name ilike 'Votação nominal por município e zona%'
          
      ),
      _raw as (
        select *
        from read_csv(
  [
    '/home/pingu/github/experiments/data/tse-election-data/data/tse/raw/election_type=general/year=2026/domain=vote_result/dataset=resultados_2026/resource=c807b826_21ff_4bcf_97ca_d86482656320/sha256=b5214219ca30bcaf3c31b6bd528d66175538d6a501d520c82c40708bcbba17e7/extracted/votacao_candidato_munzona_2026_BRASIL.csv'
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
  strict_mode = true,
  null_padding = false,
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
  

),

typed as (
    select
        try_cast("ANO_ELEICAO" as integer) as election_year,
        _election_type as election_type,
        _election_scope as election_scope,

        "CD_ELEICAO" as election_code,
        try_cast("NR_TURNO" as integer) as round_number,
        try_strptime(
            trim("DT_GERACAO") || ' ' || trim("HH_GERACAO"),
            '%d/%m/%Y %H:%M:%S'
        ) as generated_at,

        "SG_UF" as uf,
        
case
    when "CD_MUNICIPIO" is null then null
    when trim(cast("CD_MUNICIPIO" as varchar)) = '' then null
    else lpad(trim(cast("CD_MUNICIPIO" as varchar)), 5, '0')
end
 as municipality_code,
        try_cast("NR_ZONA" as integer) as zone,

        "CD_CARGO" as office_code,
        
case
  when upper(trim("DS_CARGO")) in (
    'PRESIDENTE', 'VICE-PRESIDENTE', 'SENADOR', '1º SUPLENTE', '2º SUPLENTE',
    'DEPUTADO FEDERAL'
  ) then 'federal'
  when upper(trim("DS_CARGO")) in (
    'GOVERNADOR', 'VICE-GOVERNADOR', 'DEPUTADO ESTADUAL', 'DEPUTADO DISTRITAL'
  ) then 'state'
  when upper(trim("DS_CARGO")) in (
    'PREFEITO', 'VICE-PREFEITO', 'VEREADOR'
  ) then 'municipal'
  else 'other'
end
 as office_scope,

        "SQ_CANDIDATO" as candidate_id,
        "CD_SIT_TOT_TURNO" as totalization_status_code,

        case
          when upper(trim("ST_VOTO_EM_TRANSITO")) = 'S' then true
          when upper(trim("ST_VOTO_EM_TRANSITO")) = 'N' then false
          else null
        end as is_transit_vote,

        try_cast("QT_VOTOS_NOMINAIS" as bigint) as nominal_votes,
        try_cast("QT_VOTOS_NOMINAIS_VALIDOS" as bigint) as nominal_valid_votes,
        filename as source_file
    from src
)

select *
from typed
