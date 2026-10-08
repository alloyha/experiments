

with src as (
    select *
    from 
  
    
    (
      select
        *,
        cast(null as varchar) as _election_type,
        cast(null as varchar) as _election_scope
      from read_csv(
        '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/fixtures/vote_result.csv',
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
