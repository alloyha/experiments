
  
    
    
    create temporary table
      "bronze_party_votes_raw__dbt_tmp_a944d3a9_a06a_41bf_8bcf_df7dc945f019"
  
    as (
      

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
          
          and resource_name ilike 'Votação em partido por município e zona%'
          
      ),
      _raw as (
        select *
        from read_csv(
  [
    '/home/pingu/github/experiments/data/tse-election-data/data/tse/raw/election_type=general/year=2026/domain=vote_result/dataset=resultados_2026/resource=b255886a_23f5_462f_8385_b2df4c3ddac3/sha256=85dd5241e46e78e1437666e901c1b15fe126e7476686ea8da075e306a6a5e56c/extracted/votacao_partido_munzona_2026_BRASIL.csv'
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

        "TP_AGREMIACAO" as party_group_type,
        "NR_PARTIDO" as party_number,
        "SG_PARTIDO" as party,
        "NM_PARTIDO" as party_name,

        "NR_FEDERACAO" as federation_number,
        "NM_FEDERACAO" as federation_name,
        "SG_FEDERACAO" as federation,
        "DS_COMPOSICAO_FEDERACAO" as federation_composition,

        "SQ_COLIGACAO" as coalition_id,
        "NM_COLIGACAO" as coalition_name,
        "DS_COMPOSICAO_COLIGACAO" as coalition_composition,

        case
          when upper(trim("ST_VOTO_EM_TRANSITO")) = 'S' then true
          when upper(trim("ST_VOTO_EM_TRANSITO")) = 'N' then false
          else null
        end as is_transit_vote,

        try_cast("QT_VOTOS_LEGENDA_VALIDOS" as bigint) as legend_valid_votes,
        
        try_cast("QT_VOTOS_NOM_CONVR_LEG_VALIDOS" as bigint)
         as nominal_converted_to_legend_votes,
        try_cast("QT_TOTAL_VOTOS_LEG_VALIDOS" as bigint) as total_legend_valid_votes,
        try_cast("QT_VOTOS_NOMINAIS_VALIDOS" as bigint) as nominal_valid_votes,

        try_cast("QT_VOTOS_LEGENDA_ANUL_SUBJUD" as bigint) as legend_annulled_subjudice_votes,
        try_cast("QT_VOTOS_NOMINAIS_ANUL_SUBJUD" as bigint) as nominal_annulled_subjudice_votes,

        filename as source_file
    from src
)

select *
from typed

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;
insert into "tse_analytics"."main"."bronze_party_votes_raw" ("election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_group_type", "party_number", "party", "party_name", "federation_number", "federation_name", "federation", "federation_composition", "coalition_id", "coalition_name", "coalition_composition", "is_transit_vote", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_valid_votes", "legend_annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_group_type", "party_number", "party", "party_name", "federation_number", "federation_name", "federation", "federation_composition", "coalition_id", "coalition_name", "coalition_composition", "is_transit_vote", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_valid_votes", "legend_annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "source_file"
        from "bronze_party_votes_raw__dbt_tmp_a944d3a9_a06a_41bf_8bcf_df7dc945f019"
    )


  