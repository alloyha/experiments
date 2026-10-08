
  
    
    
    create temporary table
      "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be"
  
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
          
          and resource_name ilike 'Detalhe da apuração por município e zona%'
          
      ),
      _raw as (
        select *
        from read_csv(
  [
    '/home/pingu/github/experiments/data/tse-election-data/data/tse/raw/election_type=general/year=2026/domain=vote_result/dataset=resultados_2026/resource=db2f6a49_a150_42a5_9aca_9ddd16d297f0/sha256=d0eab689021526657686be44e2db52e5f8553d3907f0f0bda7dd0431e91a4161/extracted/detalhe_votacao_munzona_2026_BRASIL.csv'
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

        case
          when upper(trim("ST_VOTO_EM_TRANSITO")) = 'S' then true
          when upper(trim("ST_VOTO_EM_TRANSITO")) = 'N' then false
          else null
        end as is_transit_vote,

        try_cast("QT_APTOS" as bigint) as eligible_voters,
        try_cast("QT_SECOES_PRINCIPAIS" as bigint) as main_sections,
        try_cast("QT_SECOES_AGREGADAS" as bigint) as aggregated_sections,
        try_cast("QT_SECOES_NAO_INSTALADAS" as bigint) as uninstalled_sections,
        try_cast("QT_TOTAL_SECOES" as bigint) as total_sections,

        try_cast("QT_COMPARECIMENTO" as bigint) as turnout,
        try_cast("QT_ELEITORES_SECOES_NAO_INSTALADAS" as bigint) as voters_uninstalled_sections,
        try_cast("QT_ABSTENCOES" as bigint) as abstentions,

        try_cast("QT_VOTOS" as bigint) as total_votes,
        try_cast("QT_VOTOS_CONCORRENTES" as bigint) as competing_votes,

        try_cast("QT_TOTAL_VOTOS_VALIDOS" as bigint) as valid_votes,
        try_cast("QT_VOTOS_NOMINAIS_VALIDOS" as bigint) as nominal_valid_votes,
        try_cast("QT_TOTAL_VOTOS_LEG_VALIDOS" as bigint) as total_legend_valid_votes,
        try_cast("QT_VOTOS_LEG_VALIDOS" as bigint) as legend_valid_votes,
        try_cast("QT_VOTOS_NOM_CONVR_LEG_VALIDOS" as bigint) as nominal_converted_to_legend_valid_votes,

        try_cast("QT_TOTAL_VOTOS_ANULADOS" as bigint) as annulled_votes,
        try_cast("QT_VOTOS_NOMINAIS_ANULADOS" as bigint) as nominal_annulled_votes,
        try_cast("QT_VOTOS_LEGENDA_ANULADOS" as bigint) as legend_annulled_votes,

        try_cast("QT_TOTAL_VOTOS_ANUL_SUBJUD" as bigint) as annulled_subjudice_votes,
        try_cast("QT_VOTOS_NOMINAIS_ANUL_SUBJUD" as bigint) as nominal_annulled_subjudice_votes,
        try_cast("QT_VOTOS_LEGENDA_ANUL_SUBJUD" as bigint) as legend_annulled_subjudice_votes,

        try_cast("QT_VOTOS_BRANCOS" as bigint) as blank_votes,
        try_cast("QT_TOTAL_VOTOS_NULOS" as bigint) as total_null_votes,
        try_cast("QT_VOTOS_NULOS" as bigint) as null_votes,
        try_cast("QT_VOTOS_NULOS_TECNICOS" as bigint) as technical_null_votes,
        try_cast("QT_VOTOS_ANULADOS_APU_SEP" as bigint) as separately_counted_annulled_votes,

        try_strptime(
            trim("DT_ULTIMA_TOTALIZACAO") || ' ' || trim("HH_ULTIMA_TOTALIZACAO"),
            '%d/%m/%Y %H:%M:%S'
        ) as last_totalization_at,

        filename as source_file
    from src
),

ranked as (
    select
        *,
        row_number() over (
            partition by
                election_year,
                election_type,
                election_code,
                round_number,
                uf,
                municipality_code,
                zone,
                office_code,
                is_transit_vote
            order by
                generated_at desc nulls last,
                last_totalization_at desc nulls last,
                source_file desc
        ) as _version_rank
    from typed
    
    where election_year in (2026) and election_type in ('general')
    
)

select * exclude (_version_rank)
from ranked
where _version_rank = 1
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."bronze_tally_munzona" as DBT_INCREMENTAL_TARGET
            using "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be"
            where (
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."bronze_tally_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file"
        from "bronze_tally_munzona__dbt_tmp_f7afb445_935b_40e3_9558_d1f6762d77be"
    )
  