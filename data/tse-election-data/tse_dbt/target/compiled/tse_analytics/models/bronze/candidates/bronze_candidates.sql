with src as (
    select * from 
  
    
    (
      select
        *,
        cast(null as varchar) as _election_type,
        cast(null as varchar) as _election_scope
      from read_csv(
        '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/fixtures/candidate.csv',
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
  

), renamed as (
    select
        try_cast("ANO_ELEICAO" as integer) as election_year,
        _election_type as election_type,
        _election_scope as election_scope,
        "CD_ELEICAO" as election_code,
        "DS_ELEICAO" as election_description,
        try_cast("NR_TURNO" as integer) as round_number,
        "SG_UE" as electoral_unit,
        
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
        "SG_UF" as uf,
        "CD_CARGO" as office_code,
        "DS_CARGO" as office,
        "SQ_CANDIDATO" as candidate_id,
        "NR_CANDIDATO" as candidate_number,
        "NM_CANDIDATO" as candidate_name,
        "NM_URNA_CANDIDATO" as ballot_name,
        "NR_PARTIDO" as party_number,
        "SG_PARTIDO" as party,
        "NM_PARTIDO" as party_name,
        "DS_SITUACAO_CANDIDATURA" as candidacy_status,
        "DS_GENERO" as gender,
        "DS_GRAU_INSTRUCAO" as education,
        "DS_OCUPACAO" as occupation,
        "DS_COR_RACA" as race_color,
        filename as source_file
    from src
)
select * from renamed