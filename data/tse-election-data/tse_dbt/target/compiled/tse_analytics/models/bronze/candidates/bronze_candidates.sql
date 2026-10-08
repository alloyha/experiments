with src as (
    select * from 
  
    
    
    (
      with _index as (
        select distinct
          '/home/pingu/github/experiments/data/tse-election-data/data/tse' || '/' || object as object_path,
          year as _index_year,
          election_type as _election_type,
          election_scope as _election_scope
        from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
        where domain = 'candidate'
          and year in (2026)
          and election_type in ('general')
          
      ),
      _raw as (
        select *
        from read_csv(
  [
    '/home/pingu/github/experiments/data/tse-election-data/data/tse/raw/election_type=general/year=2026/domain=candidate/dataset=candidatos_2026/resource=7748de82_a23b_47c4_9ec1_35535d945e5b/sha256=fd3589a80235942664bec6b135e9916e241b6010484bb837dfcdf077f83f6eab/extracted/consulta_cand_2026_BRASIL.csv'
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