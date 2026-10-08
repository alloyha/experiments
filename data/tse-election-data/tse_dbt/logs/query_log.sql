-- created_at: 2026-10-08T11:14:23.039992205+00:00
-- finished_at: 2026-10-08T11:14:23.044165779+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: list_relations_in_parallel
SELECT table_catalog, table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'main' AND lower(table_catalog) = lower('tse_analytics');
-- created_at: 2026-10-08T11:14:23.150091901+00:00
-- finished_at: 2026-10-08T11:14:23.151313139+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select schema_name
    from system.information_schema.schemata
    
    where lower(catalog_name) = '"tse_analytics"'
    
  
  ;
-- created_at: 2026-10-08T11:14:23.151749755+00:00
-- finished_at: 2026-10-08T11:14:23.152302821+00:00
-- elapsed: 553us
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
        select type from duckdb_databases()
        where lower(database_name)='tse_analytics'
        and type='sqlite'
    
  ;
-- created_at: 2026-10-08T11:14:23.152535301+00:00
-- finished_at: 2026-10-08T11:14:23.152949280+00:00
-- elapsed: 413us
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    
        create schema if not exists "tse_analytics"."main"
    ;
-- created_at: 2026-10-08T11:14:23.155765855+00:00
-- finished_at: 2026-10-08T11:14:23.162679061+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select distinct object
    from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
    where domain = 'vote_result'
      and year in (2026)
      and election_type in ('general')
      
      and resource_name ilike 'Votação nominal por município e zona%'
      
    order by election_type, year, resource_id, object
  
  ;
-- created_at: 2026-10-08T11:14:23.155766715+00:00
-- finished_at: 2026-10-08T11:14:23.163863298+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select distinct object
    from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
    where domain = 'electorate'
      and year in (2026)
      and election_type in ('general')
      
      and resource_name ilike 'Eleitorado - %'
      
    order by election_type, year, resource_id, object
  
  ;
-- created_at: 2026-10-08T11:14:23.166091866+00:00
-- finished_at: 2026-10-08T11:14:23.174230741+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select distinct object
    from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
    where domain = 'vote_result'
      and year in (2026)
      and election_type in ('general')
      
      and resource_name ilike 'Votação em partido por município e zona%'
      
    order by election_type, year, resource_id, object
  
  ;
-- created_at: 2026-10-08T11:14:23.167491171+00:00
-- finished_at: 2026-10-08T11:14:23.174806900+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select distinct object
    from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
    where domain = 'vote_result'
      and year in (2026)
      and election_type in ('general')
      
      and resource_name ilike 'Detalhe da apuração por município e zona%'
      
    order by election_type, year, resource_id, object
  
  ;
-- created_at: 2026-10-08T11:14:23.179175804+00:00
-- finished_at: 2026-10-08T11:14:23.191942010+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select distinct object
    from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
    where domain = 'candidate'
      and year in (2026)
      and election_type in ('general')
      
    order by election_type, year, resource_id, object
  
  ;
-- created_at: 2026-10-08T11:14:23.178811544+00:00
-- finished_at: 2026-10-08T11:14:23.193177169+00:00
-- elapsed: 14ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select distinct object
    from read_json_auto('/home/pingu/github/experiments/data/tse-election-data/data/tse/_metadata/current_objects.jsonl')
    where domain = 'candidate_assets'
      and year in (2026)
      and election_type in ('general')
      
    order by election_type, year, resource_id, object
  
  ;
-- created_at: 2026-10-08T11:14:23.201282170+00:00
-- finished_at: 2026-10-08T11:14:23.255771107+00:00
-- elapsed: 54ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."fact_candidate_votes__dbt_tmp" as (
    

select *
from read_parquet(
    '/home/pingu/github/experiments/data/tse-election-data/data/warehouse/fact_candidate_votes/election_type=*/year=*/data.parquet',
    hive_partitioning = true,
    union_by_name = true
)
  );
;
-- created_at: 2026-10-08T11:14:23.257429235+00:00
-- finished_at: 2026-10-08T11:14:23.261497516+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes" rename to "fact_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T11:14:23.257634888+00:00
-- finished_at: 2026-10-08T11:14:23.265526218+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "seed.tse_analytics.election_calendar", "profile_name": "tse_analytics", "target_name": "dev"} */
truncate table "tse_analytics"."main"."election_calendar";
-- created_at: 2026-10-08T11:14:23.263547078+00:00
-- finished_at: 2026-10-08T11:14:23.269365306+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes__dbt_tmp" rename to "fact_candidate_votes";
-- created_at: 2026-10-08T11:14:23.271363810+00:00
-- finished_at: 2026-10-08T11:14:23.276583471+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."fact_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:14:23.282283744+00:00
-- finished_at: 2026-10-08T11:14:23.285161381+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_normalize_municipality_code_examples
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_normalize_municipality_code_examples", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with cases(input_value, expected_value) as (
    values
        ('35',    '00035'),
        ('1392',  '01392'),
        ('2550',  '02550'),
        ('4154',  '04154'),
        ('30015', '30015'),
        (' 35 ',  '00035'),
        ('',      null),
        (null,    null)
),
evaluated as (
    select
        input_value,
        expected_value,
        
case
    when input_value is null then null
    when trim(cast(input_value as varchar)) = '' then null
    else lpad(trim(cast(input_value as varchar)), 5, '0')
end
 as actual_value
    from cases
)
select *
from evaluated
where actual_value is distinct from expected_value
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:23.281824096+00:00
-- finished_at: 2026-10-08T11:14:23.299781005+00:00
-- elapsed: 17ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: add_query adapter call

          COPY "tse_analytics"."main"."election_calendar" FROM '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/seeds/election_calendar.csv' (FORMAT CSV, HEADER TRUE, DELIMITER ',')
        ;
-- created_at: 2026-10-08T11:14:23.307168330+00:00
-- finished_at: 2026-10-08T11:14:23.393256369+00:00
-- elapsed: 86ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."stg_candidates__dbt_tmp" as (
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
  );
;
-- created_at: 2026-10-08T11:14:23.395003555+00:00
-- finished_at: 2026-10-08T11:14:23.399629963+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidates" rename to "stg_candidates__dbt_backup";
-- created_at: 2026-10-08T11:14:23.401219454+00:00
-- finished_at: 2026-10-08T11:14:23.405448212+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidates__dbt_tmp" rename to "stg_candidates";
-- created_at: 2026-10-08T11:14:23.407577792+00:00
-- finished_at: 2026-10-08T11:14:23.412576238+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidates__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:14:23.289493842+00:00
-- finished_at: 2026-10-08T11:14:23.602263390+00:00
-- elapsed: 312ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."stg_electorate__dbt_tmp" as (
    





with src as (
    -- The ingestion domain also contains temporary-transfer resources.
    -- Only the canonical electorate profile belongs in this staging model.
    select * from 
  
    
    
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
;
-- created_at: 2026-10-08T11:14:23.603834150+00:00
-- finished_at: 2026-10-08T11:14:23.608731648+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_electorate" rename to "stg_electorate__dbt_backup";
-- created_at: 2026-10-08T11:14:23.609929176+00:00
-- finished_at: 2026-10-08T11:14:23.615548904+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_electorate__dbt_tmp" rename to "stg_electorate";
-- created_at: 2026-10-08T11:14:23.617369615+00:00
-- finished_at: 2026-10-08T11:14:23.621302999+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_electorate__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:14:23.626541581+00:00
-- finished_at: 2026-10-08T11:14:23.760330053+00:00
-- elapsed: 133ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

        delete from "tse_analytics"."main"."stg_party_votes_raw"
        where election_year in (
            
                2026
            
        )
        and election_type in (
            
                'general'
                
            
        )
      ;
-- created_at: 2026-10-08T11:14:23.418277209+00:00
-- finished_at: 2026-10-08T11:14:23.859613102+00:00
-- elapsed: 441ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."stg_candidate_votes_raw__dbt_tmp" as (
    

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

  );
;
-- created_at: 2026-10-08T11:14:23.860991098+00:00
-- finished_at: 2026-10-08T11:14:23.864859081+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_raw" rename to "stg_candidate_votes_raw__dbt_backup";
-- created_at: 2026-10-08T11:14:23.865968695+00:00
-- finished_at: 2026-10-08T11:14:23.870232355+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_raw__dbt_tmp" rename to "stg_candidate_votes_raw";
-- created_at: 2026-10-08T11:14:23.871951156+00:00
-- finished_at: 2026-10-08T11:14:23.877211311+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidate_votes_raw__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:14:23.880953574+00:00
-- finished_at: 2026-10-08T11:14:23.978780197+00:00
-- elapsed: 97ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."stg_candidate_assets__dbt_tmp" as (
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
        where domain = 'candidate_assets'
          and year in (2026)
          and election_type in ('general')
          
      ),
      _raw as (
        select *
        from read_csv(
          [
            '/home/pingu/github/experiments/data/tse-election-data/data/tse/raw/election_type=general/year=2026/domain=candidate_assets/dataset=candidatos_2026/resource=33fbda56_eb41_46f5_a8a0_8b499c285a1d/sha256=f912568aa1275c9ce8a0d029780569aeef93745c76c318081ff7206a951b9277/extracted/bem_candidato_2026_BRASIL.csv'
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
  );
;
-- created_at: 2026-10-08T11:14:23.980497034+00:00
-- finished_at: 2026-10-08T11:14:23.984947534+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_assets" rename to "stg_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T11:14:23.986244145+00:00
-- finished_at: 2026-10-08T11:14:23.991953636+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_assets__dbt_tmp" rename to "stg_candidate_assets";
-- created_at: 2026-10-08T11:14:23.993821640+00:00
-- finished_at: 2026-10-08T11:14:23.997102151+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:14:24.003623220+00:00
-- finished_at: 2026-10-08T11:14:24.122263486+00:00
-- elapsed: 118ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

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
    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:14:23.762131277+00:00
-- finished_at: 2026-10-08T11:14:24.123323670+00:00
-- elapsed: 361ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

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

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:14:24.124768587+00:00
-- finished_at: 2026-10-08T11:14:24.135681321+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'stg_party_votes_raw'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:14:24.124798182+00:00
-- finished_at: 2026-10-08T11:14:24.136111+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'stg_tally_munzona'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:14:24.140338624+00:00
-- finished_at: 2026-10-08T11:14:24.963984571+00:00
-- elapsed: 823ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."stg_tally_munzona" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:14:24.138333312+00:00
-- finished_at: 2026-10-08T11:14:24.973963631+00:00
-- elapsed: 835ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."stg_party_votes_raw" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:14:24.966287293+00:00
-- finished_at: 2026-10-08T11:14:25.107062353+00:00
-- elapsed: 140ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."stg_tally_munzona" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T11:14:24.975419579+00:00
-- finished_at: 2026-10-08T11:14:26.437198422+00:00
-- elapsed: 1.5s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."stg_party_votes_raw" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T11:14:25.109665681+00:00
-- finished_at: 2026-10-08T11:14:26.463945476+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."stg_tally_munzona" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T11:14:26.466707212+00:00
-- finished_at: 2026-10-08T11:14:27.828377453+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."stg_tally_munzona" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "zone" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T11:14:26.438666204+00:00
-- finished_at: 2026-10-08T11:14:27.835565376+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."stg_party_votes_raw" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T11:14:27.830697480+00:00
-- finished_at: 2026-10-08T11:14:27.971002218+00:00
-- elapsed: 140ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."stg_tally_munzona" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T11:14:27.974021050+00:00
-- finished_at: 2026-10-08T11:14:29.241955589+00:00
-- elapsed: 1.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "eligible_voters__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "eligible_voters__dbt_alter" = "eligible_voters";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "eligible_voters" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "eligible_voters__dbt_alter" to "eligible_voters"
  ;
-- created_at: 2026-10-08T11:14:27.836946897+00:00
-- finished_at: 2026-10-08T11:14:29.524697313+00:00
-- elapsed: 1.7s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."stg_party_votes_raw" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "zone" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T11:14:29.244283803+00:00
-- finished_at: 2026-10-08T11:14:29.652303304+00:00
-- elapsed: 408ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "main_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "main_sections__dbt_alter" = "main_sections";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "main_sections" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "main_sections__dbt_alter" to "main_sections"
  ;
-- created_at: 2026-10-08T11:14:29.526345325+00:00
-- finished_at: 2026-10-08T11:14:30.400750054+00:00
-- elapsed: 874ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."stg_party_votes_raw" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T11:14:29.654645650+00:00
-- finished_at: 2026-10-08T11:14:30.408002736+00:00
-- elapsed: 753ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "aggregated_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "aggregated_sections__dbt_alter" = "aggregated_sections";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "aggregated_sections" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "aggregated_sections__dbt_alter" to "aggregated_sections"
  ;
-- created_at: 2026-10-08T11:14:30.411112499+00:00
-- finished_at: 2026-10-08T11:14:31.543511142+00:00
-- elapsed: 1.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "uninstalled_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "uninstalled_sections__dbt_alter" = "uninstalled_sections";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "uninstalled_sections" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "uninstalled_sections__dbt_alter" to "uninstalled_sections"
  ;
-- created_at: 2026-10-08T11:14:30.402388351+00:00
-- finished_at: 2026-10-08T11:14:31.552361874+00:00
-- elapsed: 1.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_raw" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:14:31.545826540+00:00
-- finished_at: 2026-10-08T11:14:31.703272873+00:00
-- elapsed: 157ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "total_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "total_sections__dbt_alter" = "total_sections";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "total_sections" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "total_sections__dbt_alter" to "total_sections"
  ;
-- created_at: 2026-10-08T11:14:31.706001558+00:00
-- finished_at: 2026-10-08T11:14:33.294923532+00:00
-- elapsed: 1.6s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "turnout__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "turnout__dbt_alter" = "turnout";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "turnout" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "turnout__dbt_alter" to "turnout"
  ;
-- created_at: 2026-10-08T11:14:31.554650718+00:00
-- finished_at: 2026-10-08T11:14:33.310387797+00:00
-- elapsed: 1.8s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "nominal_converted_to_legend_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_raw" set "nominal_converted_to_legend_votes__dbt_alter" = "nominal_converted_to_legend_votes";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "nominal_converted_to_legend_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "nominal_converted_to_legend_votes__dbt_alter" to "nominal_converted_to_legend_votes"
  ;
-- created_at: 2026-10-08T11:14:33.297683570+00:00
-- finished_at: 2026-10-08T11:14:33.481448637+00:00
-- elapsed: 183ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "voters_uninstalled_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "voters_uninstalled_sections__dbt_alter" = "voters_uninstalled_sections";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "voters_uninstalled_sections" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "voters_uninstalled_sections__dbt_alter" to "voters_uninstalled_sections"
  ;
-- created_at: 2026-10-08T11:14:33.484819993+00:00
-- finished_at: 2026-10-08T11:14:34.869722464+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "abstentions__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "abstentions__dbt_alter" = "abstentions";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "abstentions" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "abstentions__dbt_alter" to "abstentions"
  ;
-- created_at: 2026-10-08T11:14:33.311938840+00:00
-- finished_at: 2026-10-08T11:14:34.877561953+00:00
-- elapsed: 1.6s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_raw" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:14:34.872656482+00:00
-- finished_at: 2026-10-08T11:14:35.061270+00:00
-- elapsed: 188ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "total_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "total_votes__dbt_alter" = "total_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "total_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "total_votes__dbt_alter" to "total_votes"
  ;
-- created_at: 2026-10-08T11:14:35.063698946+00:00
-- finished_at: 2026-10-08T11:14:36.457645756+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "competing_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "competing_votes__dbt_alter" = "competing_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "competing_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "competing_votes__dbt_alter" to "competing_votes"
  ;
-- created_at: 2026-10-08T11:14:34.878886397+00:00
-- finished_at: 2026-10-08T11:14:36.467496955+00:00
-- elapsed: 1.6s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_raw" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T11:14:36.460368824+00:00
-- finished_at: 2026-10-08T11:14:36.620340219+00:00
-- elapsed: 159ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "valid_votes__dbt_alter" = "valid_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "valid_votes__dbt_alter" to "valid_votes"
  ;
-- created_at: 2026-10-08T11:14:36.622998808+00:00
-- finished_at: 2026-10-08T11:14:37.809474050+00:00
-- elapsed: 1.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T11:14:36.468953022+00:00
-- finished_at: 2026-10-08T11:14:37.818205141+00:00
-- elapsed: 1.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_raw" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:14:37.812415885+00:00
-- finished_at: 2026-10-08T11:14:37.986074055+00:00
-- elapsed: 173ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:14:37.989184488+00:00
-- finished_at: 2026-10-08T11:14:39.182365730+00:00
-- elapsed: 1.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:14:37.820021632+00:00
-- finished_at: 2026-10-08T11:14:39.191852174+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_raw" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_raw" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."stg_party_votes_raw" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_raw" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:14:39.185584761+00:00
-- finished_at: 2026-10-08T11:14:40.701087199+00:00
-- elapsed: 1.5s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "nominal_converted_to_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "nominal_converted_to_legend_valid_votes__dbt_alter" = "nominal_converted_to_legend_valid_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "nominal_converted_to_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "nominal_converted_to_legend_valid_votes__dbt_alter" to "nominal_converted_to_legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:14:40.715573582+00:00
-- finished_at: 2026-10-08T11:14:40.936462312+00:00
-- elapsed: 220ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "annulled_votes__dbt_alter" = "annulled_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "annulled_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "annulled_votes__dbt_alter" to "annulled_votes"
  ;
-- created_at: 2026-10-08T11:14:40.956828096+00:00
-- finished_at: 2026-10-08T11:14:41.163577559+00:00
-- elapsed: 206ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "nominal_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "nominal_annulled_votes__dbt_alter" = "nominal_annulled_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "nominal_annulled_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "nominal_annulled_votes__dbt_alter" to "nominal_annulled_votes"
  ;
-- created_at: 2026-10-08T11:14:41.192873342+00:00
-- finished_at: 2026-10-08T11:14:41.431806128+00:00
-- elapsed: 238ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "legend_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "legend_annulled_votes__dbt_alter" = "legend_annulled_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "legend_annulled_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "legend_annulled_votes__dbt_alter" to "legend_annulled_votes"
  ;
-- created_at: 2026-10-08T11:14:41.466530713+00:00
-- finished_at: 2026-10-08T11:14:41.676914589+00:00
-- elapsed: 210ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "annulled_subjudice_votes__dbt_alter" = "annulled_subjudice_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "annulled_subjudice_votes__dbt_alter" to "annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:14:41.697263224+00:00
-- finished_at: 2026-10-08T11:14:41.919041256+00:00
-- elapsed: 221ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:14:41.943957840+00:00
-- finished_at: 2026-10-08T11:14:42.175372161+00:00
-- elapsed: 231ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:14:42.201397607+00:00
-- finished_at: 2026-10-08T11:14:42.848218984+00:00
-- elapsed: 646ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "blank_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "blank_votes__dbt_alter" = "blank_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "blank_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "blank_votes__dbt_alter" to "blank_votes"
  ;
-- created_at: 2026-10-08T11:14:42.879711480+00:00
-- finished_at: 2026-10-08T11:14:44.285572274+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "total_null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "total_null_votes__dbt_alter" = "total_null_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "total_null_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "total_null_votes__dbt_alter" to "total_null_votes"
  ;
-- created_at: 2026-10-08T11:14:44.293626963+00:00
-- finished_at: 2026-10-08T11:14:44.430306512+00:00
-- elapsed: 136ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "null_votes__dbt_alter" = "null_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "null_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "null_votes__dbt_alter" to "null_votes"
  ;
-- created_at: 2026-10-08T11:14:44.441326338+00:00
-- finished_at: 2026-10-08T11:14:44.607828013+00:00
-- elapsed: 166ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "technical_null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "technical_null_votes__dbt_alter" = "technical_null_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "technical_null_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "technical_null_votes__dbt_alter" to "technical_null_votes"
  ;
-- created_at: 2026-10-08T11:14:44.622175072+00:00
-- finished_at: 2026-10-08T11:14:44.790906436+00:00
-- elapsed: 168ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "separately_counted_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_tally_munzona" set "separately_counted_annulled_votes__dbt_alter" = "separately_counted_annulled_votes";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "separately_counted_annulled_votes" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "separately_counted_annulled_votes__dbt_alter" to "separately_counted_annulled_votes"
  ;
-- created_at: 2026-10-08T11:14:44.812668700+00:00
-- finished_at: 2026-10-08T11:14:45.060962049+00:00
-- elapsed: 248ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_tally_munzona" add column "last_totalization_at__dbt_alter" datetime;
    update "tse_analytics"."main"."stg_tally_munzona" set "last_totalization_at__dbt_alter" = "last_totalization_at";
    alter table "tse_analytics"."main"."stg_tally_munzona" drop column "last_totalization_at" cascade;
    alter table "tse_analytics"."main"."stg_tally_munzona" rename column "last_totalization_at__dbt_alter" to "last_totalization_at"
  ;
-- created_at: 2026-10-08T11:14:45.158601871+00:00
-- finished_at: 2026-10-08T11:14:47.256375321+00:00
-- elapsed: 2.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7"
  
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

        
            delete from "tse_analytics"."main"."stg_tally_munzona" as DBT_INCREMENTAL_TARGET
            using "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7"
            where (
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."stg_tally_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file"
        from "stg_tally_munzona__dbt_tmp_7429becb_81a1_4165_b2ba_56704e865cd7"
    )
  ;
-- created_at: 2026-10-08T11:14:47.273739766+00:00
-- finished_at: 2026-10-08T11:14:47.335755785+00:00
-- elapsed: 62ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_candidate_votes_round_number.16d23c0f0c
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_candidate_votes_round_number.16d23c0f0c", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select round_number
from "tse_analytics"."main"."fact_candidate_votes"
where round_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.347056051+00:00
-- finished_at: 2026-10-08T11:14:47.404116671+00:00
-- elapsed: 57ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_candidate_votes_candidate_id.3747010aa1
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_candidate_votes_candidate_id.3747010aa1", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_id
from "tse_analytics"."main"."fact_candidate_votes"
where candidate_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.418545772+00:00
-- finished_at: 2026-10-08T11:14:47.547923508+00:00
-- elapsed: 129ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_candidate_votes_election_year.327834c39b
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_candidate_votes_election_year.327834c39b", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."fact_candidate_votes"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.558157738+00:00
-- finished_at: 2026-10-08T11:14:47.625440332+00:00
-- elapsed: 67ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_nonnegative_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_nonnegative_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_candidate_votes"
where nominal_votes < 0
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.637156589+00:00
-- finished_at: 2026-10-08T11:14:47.689399300+00:00
-- elapsed: 52ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_candidate_votes_election_type.3e7e08b6c8
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_candidate_votes_election_type.3e7e08b6c8", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."fact_candidate_votes"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.701607933+00:00
-- finished_at: 2026-10-08T11:14:47.759698394+00:00
-- elapsed: 58ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_candidate_votes_nominal_votes.3a44b55974
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_candidate_votes_nominal_votes.3a44b55974", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_votes
from "tse_analytics"."main"."fact_candidate_votes"
where nominal_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.769021613+00:00
-- finished_at: 2026-10-08T11:14:47.828389620+00:00
-- elapsed: 59ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_candidate_votes_election_id.4ca494eb67
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_candidate_votes_election_id.4ca494eb67", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_id
from "tse_analytics"."main"."fact_candidate_votes"
where election_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.839865675+00:00
-- finished_at: 2026-10-08T11:14:47.910312788+00:00
-- elapsed: 70ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_candidate_votes_election_code.b2bf8f6b82
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_candidate_votes_election_code.b2bf8f6b82", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_code
from "tse_analytics"."main"."fact_candidate_votes"
where election_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:47.926837760+00:00
-- finished_at: 2026-10-08T11:14:53.928869430+00:00
-- elapsed: 6.0s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_fact_candidate_votes_municipality_code_canonical
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_fact_candidate_votes_municipality_code_canonical", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_candidate_votes"
where municipality_code is null
   or length(municipality_code) <> 5
   or not regexp_matches(municipality_code, '^[0-9]{5}$')
limit 1
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:53.944212235+00:00
-- finished_at: 2026-10-08T11:14:54.938311457+00:00
-- elapsed: 994ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_candidate_votes_cycle_scope
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_candidate_votes_cycle_scope", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_candidate_votes"
where
      (election_type = 'general' and office_scope = 'municipal')
   or (election_type = 'municipal' and office_scope in ('federal', 'state'))
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:39.197268469+00:00
-- finished_at: 2026-10-08T11:14:57.617741582+00:00
-- elapsed: 18.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "stg_party_votes_raw__dbt_tmp_4556b88d_f7f5_419d_9394_e0c71f39e282"
  
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
insert into "tse_analytics"."main"."stg_party_votes_raw" ("election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_group_type", "party_number", "party", "party_name", "federation_number", "federation_name", "federation", "federation_composition", "coalition_id", "coalition_name", "coalition_composition", "is_transit_vote", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_valid_votes", "legend_annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_group_type", "party_number", "party", "party_name", "federation_number", "federation_name", "federation", "federation_composition", "coalition_id", "coalition_name", "coalition_composition", "is_transit_vote", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_valid_votes", "legend_annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "source_file"
        from "stg_party_votes_raw__dbt_tmp_4556b88d_f7f5_419d_9394_e0c71f39e282"
    )


  ;
-- created_at: 2026-10-08T11:14:57.724682326+00:00
-- finished_at: 2026-10-08T11:14:57.726995088+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_election_calendar_election_type.af3ca8570a
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_election_calendar_election_type.af3ca8570a", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."election_calendar"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:57.737752447+00:00
-- finished_at: 2026-10-08T11:14:57.740012114+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_election_calendar_election_scope.6bc74c69c8
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_election_calendar_election_scope.6bc74c69c8", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_scope
from "tse_analytics"."main"."election_calendar"
where election_scope is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:57.752077989+00:00
-- finished_at: 2026-10-08T11:14:57.754610785+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_election_calendar_election_year.02aac0fd03
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_election_calendar_election_year.02aac0fd03", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."election_calendar"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:57.765683965+00:00
-- finished_at: 2026-10-08T11:14:57.773977263+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_election_calendar_election_year__election_type.cd859d6ff1
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_election_calendar_election_year__election_type.cd859d6ff1", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type
    from "tse_analytics"."main"."election_calendar"
    group by election_year, election_type
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:57.786840618+00:00
-- finished_at: 2026-10-08T11:14:57.791573681+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_election_calendar_election_type__general__municipal.dcfd7d280f
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_election_calendar_election_type__general__municipal.dcfd7d280f", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_type as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."election_calendar"
    group by election_type

)

select *
from all_values
where value_field not in (
    'general','municipal'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:57.806087174+00:00
-- finished_at: 2026-10-08T11:14:57.810466995+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_election_calendar_election_scope__federal_state__municipal.ecdad16a31
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_election_calendar_election_scope__federal_state__municipal.ecdad16a31", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."election_calendar"
    group by election_scope

)

select *
from all_values
where value_field not in (
    'federal_state','municipal'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:57.828103171+00:00
-- finished_at: 2026-10-08T11:14:58.271545150+00:00
-- elapsed: 443ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidates_office_scope.3609814c0e
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidates_office_scope.3609814c0e", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select office_scope
from "tse_analytics"."main"."stg_candidates"
where office_scope is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:58.281508759+00:00
-- finished_at: 2026-10-08T11:14:58.881163958+00:00
-- elapsed: 599ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidates_election_year.1bd340dd75
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidates_election_year.1bd340dd75", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."stg_candidates"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:58.898482665+00:00
-- finished_at: 2026-10-08T11:14:59.505766671+00:00
-- elapsed: 607ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidates_election_type.dab2bf8d97
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidates_election_type.dab2bf8d97", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."stg_candidates"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:59.517170219+00:00
-- finished_at: 2026-10-08T11:15:00.056723876+00:00
-- elapsed: 539ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_stg_candidates_office_scope__federal__state__municipal__other.1c6d651ef2
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_stg_candidates_office_scope__federal__state__municipal__other.1c6d651ef2", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        office_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."stg_candidates"
    group by office_scope

)

select *
from all_values
where value_field not in (
    'federal','state','municipal','other'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:00.064828031+00:00
-- finished_at: 2026-10-08T11:15:00.580196844+00:00
-- elapsed: 515ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidates_candidate_id.15278f3fce
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidates_candidate_id.15278f3fce", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_id
from "tse_analytics"."main"."stg_candidates"
where candidate_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:00.592785185+00:00
-- finished_at: 2026-10-08T11:15:01.168208643+00:00
-- elapsed: 575ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidates_election_scope.93439d22b3
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidates_election_scope.93439d22b3", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_scope
from "tse_analytics"."main"."stg_candidates"
where election_scope is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:01.181127917+00:00
-- finished_at: 2026-10-08T11:15:01.736736926+00:00
-- elapsed: 555ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidates_candidate_name.f8e648de18
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidates_candidate_name.f8e648de18", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_name
from "tse_analytics"."main"."stg_candidates"
where candidate_name is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:01.748225114+00:00
-- finished_at: 2026-10-08T11:15:02.199628338+00:00
-- elapsed: 451ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_stg_candidates_election_type__general__municipal.32d7daaf3c
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_stg_candidates_election_type__general__municipal.32d7daaf3c", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_type as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."stg_candidates"
    group by election_type

)

select *
from all_values
where value_field not in (
    'general','municipal'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:02.208327631+00:00
-- finished_at: 2026-10-08T11:15:02.586432768+00:00
-- elapsed: 378ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_stg_candidates_election_scope__federal_state__municipal.8ecea0502c
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_stg_candidates_election_scope__federal_state__municipal.8ecea0502c", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."stg_candidates"
    group by election_scope

)

select *
from all_values
where value_field not in (
    'federal_state','municipal'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:02.599304468+00:00
-- finished_at: 2026-10-08T11:15:03.200162941+00:00
-- elapsed: 600ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_stg_candidates_election_year__election_type__election_code__candidate_id.b8ac2fff7d
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_stg_candidates_election_year__election_type__election_code__candidate_id.b8ac2fff7d", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."stg_candidates"
    group by election_year, election_type, election_code, candidate_id
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:03.211573517+00:00
-- finished_at: 2026-10-08T11:15:03.944649349+00:00
-- elapsed: 733ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_candidate_cycle_scope
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_candidate_cycle_scope", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."stg_candidates"
where
    (election_type = 'municipal' and office_scope <> 'municipal')
    or
    (election_type = 'general' and office_scope = 'municipal')
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:03.955635446+00:00
-- finished_at: 2026-10-08T11:15:04.321906647+00:00
-- elapsed: 366ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_election_scope_matches_type
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_election_scope_matches_type", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."stg_candidates"
where election_scope <> case
    when election_type = 'general' then 'federal_state'
    when election_type = 'municipal' then 'municipal'
end
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:14:54.951925754+00:00
-- finished_at: 2026-10-08T11:15:58.681189827+00:00
-- elapsed: 1m 4s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_candidate_votes_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__candidate_id__is_transit_vote.14434b774a
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_candidate_votes_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__candidate_id__is_transit_vote.14434b774a", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, candidate_id, is_transit_vote
    from "tse_analytics"."main"."fact_candidate_votes"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, candidate_id, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:00.347704527+00:00
-- finished_at: 2026-10-08T11:16:01.261783902+00:00
-- elapsed: 914ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."stg_candidate_votes_munzona__dbt_tmp" as (
    

select *
from "tse_analytics"."main"."stg_candidate_votes_raw"
  );
;
-- created_at: 2026-10-08T11:16:01.270498523+00:00
-- finished_at: 2026-10-08T11:16:01.281062838+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_munzona" rename to "stg_candidate_votes_munzona__dbt_backup";
-- created_at: 2026-10-08T11:16:01.285176036+00:00
-- finished_at: 2026-10-08T11:16:01.291017050+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_munzona__dbt_tmp" rename to "stg_candidate_votes_munzona";
-- created_at: 2026-10-08T11:16:01.301736550+00:00
-- finished_at: 2026-10-08T11:16:01.307718050+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidate_votes_munzona__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:16:01.322105795+00:00
-- finished_at: 2026-10-08T11:16:01.841555190+00:00
-- elapsed: 519ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidate_assets_candidate_id.be442712e7
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidate_assets_candidate_id.be442712e7", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_id
from "tse_analytics"."main"."stg_candidate_assets"
where candidate_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:01.856671405+00:00
-- finished_at: 2026-10-08T11:16:02.341454457+00:00
-- elapsed: 484ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidate_assets_election_year.d0c05b8c8f
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidate_assets_election_year.d0c05b8c8f", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."stg_candidate_assets"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.350738080+00:00
-- finished_at: 2026-10-08T11:16:02.718187719+00:00
-- elapsed: 367ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_candidate_assets_election_type.8a509cbc40
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_candidate_assets_election_type.8a509cbc40", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."stg_candidate_assets"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.767368169+00:00
-- finished_at: 2026-10-08T11:16:02.893684753+00:00
-- elapsed: 126ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_stg_tally_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.3de2ea84ab
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_stg_tally_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.3de2ea84ab", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    from "tse_analytics"."main"."stg_tally_munzona"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.908231427+00:00
-- finished_at: 2026-10-08T11:16:02.916942642+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_election_year.7f78df217d
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_election_year.7f78df217d", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."stg_tally_munzona"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.926854082+00:00
-- finished_at: 2026-10-08T11:16:02.938323049+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_turnout.fc7cff9f4b
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_turnout.fc7cff9f4b", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select turnout
from "tse_analytics"."main"."stg_tally_munzona"
where turnout is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.948264133+00:00
-- finished_at: 2026-10-08T11:16:02.957616568+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_eligible_voters.b168186c57
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_eligible_voters.b168186c57", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select eligible_voters
from "tse_analytics"."main"."stg_tally_munzona"
where eligible_voters is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.966417173+00:00
-- finished_at: 2026-10-08T11:16:02.968812982+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_zone.4df12c09db
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_zone.4df12c09db", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select zone
from "tse_analytics"."main"."stg_tally_munzona"
where zone is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.976266726+00:00
-- finished_at: 2026-10-08T11:16:02.978337385+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_office_code.b7c6aea509
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_office_code.b7c6aea509", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select office_code
from "tse_analytics"."main"."stg_tally_munzona"
where office_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:02.986896121+00:00
-- finished_at: 2026-10-08T11:16:03.010149934+00:00
-- elapsed: 23ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_stg_tally_munzona_municipality_code_canonical
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_stg_tally_munzona_municipality_code_canonical", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."stg_tally_munzona"
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:03.017849637+00:00
-- finished_at: 2026-10-08T11:16:03.019409954+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_election_code.9bcd8247d5
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_election_code.9bcd8247d5", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_code
from "tse_analytics"."main"."stg_tally_munzona"
where election_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:03.027027338+00:00
-- finished_at: 2026-10-08T11:16:03.029277491+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_round_number.9cff3e4760
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_round_number.9cff3e4760", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select round_number
from "tse_analytics"."main"."stg_tally_munzona"
where round_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:03.036213446+00:00
-- finished_at: 2026-10-08T11:16:03.037564001+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_municipality_code.58a41c012c
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_municipality_code.58a41c012c", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select municipality_code
from "tse_analytics"."main"."stg_tally_munzona"
where municipality_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:03.044298072+00:00
-- finished_at: 2026-10-08T11:16:03.053687447+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_tally_munzona_abstentions.63591ab1e8
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_tally_munzona_abstentions.63591ab1e8", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select abstentions
from "tse_analytics"."main"."stg_tally_munzona"
where abstentions is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:03.076377754+00:00
-- finished_at: 2026-10-08T11:16:03.387149201+00:00
-- elapsed: 310ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_stg_party_votes_raw_municipality_code_canonical
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_stg_party_votes_raw_municipality_code_canonical", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."stg_party_votes_raw"
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:03.398544206+00:00
-- finished_at: 2026-10-08T11:16:08.002977040+00:00
-- elapsed: 4.6s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_party_measurewise_max_semantics
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_party_measurewise_max_semantics", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with party_grains as (

    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        party_number,
        is_transit_vote,

        count(distinct nominal_valid_votes)
            filter (where nominal_valid_votes <> 0)
            as distinct_nonzero_nominal_values,

        count(distinct legend_valid_votes)
            filter (where legend_valid_votes <> 0)
            as distinct_nonzero_legend_values,

        count(distinct total_legend_valid_votes)
            filter (where total_legend_valid_votes <> 0)
            as distinct_nonzero_total_legend_values

    from "tse_analytics"."main"."stg_party_votes_raw"

    group by
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        party_number,
        is_transit_vote
)

select *
from party_grains
where distinct_nonzero_nominal_values > 1
   or distinct_nonzero_legend_values > 1
   or distinct_nonzero_total_legend_values > 1
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:08.015925441+00:00
-- finished_at: 2026-10-08T11:16:08.063083961+00:00
-- elapsed: 47ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."int_candidate_result_coverage__dbt_tmp" as (
    

select distinct
    election_year,
    election_type,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    is_transit_vote
from "tse_analytics"."main"."fact_candidate_votes"
  );
;
-- created_at: 2026-10-08T11:16:08.066740595+00:00
-- finished_at: 2026-10-08T11:16:08.072835132+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_result_coverage" rename to "int_candidate_result_coverage__dbt_backup";
-- created_at: 2026-10-08T11:16:08.077013395+00:00
-- finished_at: 2026-10-08T11:16:08.084253785+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_result_coverage__dbt_tmp" rename to "int_candidate_result_coverage";
-- created_at: 2026-10-08T11:16:08.090508380+00:00
-- finished_at: 2026-10-08T11:16:08.097833229+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."int_candidate_result_coverage__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:16:08.116825392+00:00
-- finished_at: 2026-10-08T11:16:08.351780725+00:00
-- elapsed: 234ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."int_candidate_votes__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,

    uf,
    municipality_code,
    zone,

    office_code,
    office_scope,

    candidate_id,
    totalization_status_code,
    is_transit_vote,
    nominal_votes,
    nominal_valid_votes,

    generated_at,
    source_file
from "tse_analytics"."main"."stg_candidate_votes_munzona"
  );
;
-- created_at: 2026-10-08T11:16:08.355022289+00:00
-- finished_at: 2026-10-08T11:16:08.361369515+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_votes" rename to "int_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T11:16:08.364301839+00:00
-- finished_at: 2026-10-08T11:16:08.369639917+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_votes__dbt_tmp" rename to "int_candidate_votes";
-- created_at: 2026-10-08T11:16:08.374533380+00:00
-- finished_at: 2026-10-08T11:16:08.379861902+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."int_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:16:08.393710624+00:00
-- finished_at: 2026-10-08T11:16:08.531935822+00:00
-- elapsed: 138ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."int_candidate_assets__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_code,
    candidate_id,
    sum(asset_value) as declared_assets_value,
    count(*) as declared_assets_count
from "tse_analytics"."main"."stg_candidate_assets"
group by 1,2,3,4
  );
;
-- created_at: 2026-10-08T11:16:08.535603691+00:00
-- finished_at: 2026-10-08T11:16:08.541623748+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_assets" rename to "int_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T11:16:08.544730038+00:00
-- finished_at: 2026-10-08T11:16:08.550591660+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_assets__dbt_tmp" rename to "int_candidate_assets";
-- created_at: 2026-10-08T11:16:08.555071947+00:00
-- finished_at: 2026-10-08T11:16:08.562295942+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."int_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:16:08.583222432+00:00
-- finished_at: 2026-10-08T11:16:08.587899416+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,
    is_transit_vote,

    eligible_voters,
    voters_uninstalled_sections,
    turnout,
    abstentions,

    total_votes,
    competing_votes,

    valid_votes,
    nominal_valid_votes,
    total_legend_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_valid_votes,

    annulled_votes,
    nominal_annulled_votes,
    legend_annulled_votes,

    annulled_subjudice_votes,
    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    blank_votes,
    total_null_votes,
    null_votes,
    technical_null_votes,
    separately_counted_annulled_votes,

    generated_at,
    last_totalization_at,
    source_file
from "tse_analytics"."main"."stg_tally_munzona"

where election_year in (2026) and election_type in ('general')

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:16:08.598330775+00:00
-- finished_at: 2026-10-08T11:16:08.632777956+00:00
-- elapsed: 34ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'fact_tally_munzona'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:16:08.654497319+00:00
-- finished_at: 2026-10-08T11:16:08.813710297+00:00
-- elapsed: 159ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."fact_tally_munzona" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:16:08.818474444+00:00
-- finished_at: 2026-10-08T11:16:08.940236208+00:00
-- elapsed: 121ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."fact_tally_munzona" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T11:16:08.944087058+00:00
-- finished_at: 2026-10-08T11:16:09.051589480+00:00
-- elapsed: 107ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."fact_tally_munzona" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "zone" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T11:16:09.058098738+00:00
-- finished_at: 2026-10-08T11:16:09.139833190+00:00
-- elapsed: 81ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."fact_tally_munzona" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T11:16:09.148547676+00:00
-- finished_at: 2026-10-08T11:16:09.268287287+00:00
-- elapsed: 119ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "eligible_voters__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "eligible_voters__dbt_alter" = "eligible_voters";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "eligible_voters" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "eligible_voters__dbt_alter" to "eligible_voters"
  ;
-- created_at: 2026-10-08T11:16:09.276595400+00:00
-- finished_at: 2026-10-08T11:16:09.375542957+00:00
-- elapsed: 98ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "voters_uninstalled_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "voters_uninstalled_sections__dbt_alter" = "voters_uninstalled_sections";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "voters_uninstalled_sections" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "voters_uninstalled_sections__dbt_alter" to "voters_uninstalled_sections"
  ;
-- created_at: 2026-10-08T11:16:09.383951464+00:00
-- finished_at: 2026-10-08T11:16:09.522412565+00:00
-- elapsed: 138ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "turnout__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "turnout__dbt_alter" = "turnout";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "turnout" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "turnout__dbt_alter" to "turnout"
  ;
-- created_at: 2026-10-08T11:16:09.527084116+00:00
-- finished_at: 2026-10-08T11:16:09.624764944+00:00
-- elapsed: 97ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "abstentions__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "abstentions__dbt_alter" = "abstentions";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "abstentions" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "abstentions__dbt_alter" to "abstentions"
  ;
-- created_at: 2026-10-08T11:16:09.631924974+00:00
-- finished_at: 2026-10-08T11:16:09.727706693+00:00
-- elapsed: 95ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "total_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "total_votes__dbt_alter" = "total_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "total_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "total_votes__dbt_alter" to "total_votes"
  ;
-- created_at: 2026-10-08T11:16:09.732728159+00:00
-- finished_at: 2026-10-08T11:16:09.846456521+00:00
-- elapsed: 113ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "competing_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "competing_votes__dbt_alter" = "competing_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "competing_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "competing_votes__dbt_alter" to "competing_votes"
  ;
-- created_at: 2026-10-08T11:16:09.852501675+00:00
-- finished_at: 2026-10-08T11:16:09.943085372+00:00
-- elapsed: 90ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "valid_votes__dbt_alter" = "valid_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "valid_votes__dbt_alter" to "valid_votes"
  ;
-- created_at: 2026-10-08T11:16:09.948411802+00:00
-- finished_at: 2026-10-08T11:16:10.029426948+00:00
-- elapsed: 81ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T11:16:10.033716660+00:00
-- finished_at: 2026-10-08T11:16:10.124943339+00:00
-- elapsed: 91ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:16:10.129647122+00:00
-- finished_at: 2026-10-08T11:16:10.248006158+00:00
-- elapsed: 118ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:16:10.254546684+00:00
-- finished_at: 2026-10-08T11:16:10.381584360+00:00
-- elapsed: 127ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "nominal_converted_to_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "nominal_converted_to_legend_valid_votes__dbt_alter" = "nominal_converted_to_legend_valid_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "nominal_converted_to_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "nominal_converted_to_legend_valid_votes__dbt_alter" to "nominal_converted_to_legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:16:10.386249647+00:00
-- finished_at: 2026-10-08T11:16:10.498590715+00:00
-- elapsed: 112ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "annulled_votes__dbt_alter" = "annulled_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "annulled_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "annulled_votes__dbt_alter" to "annulled_votes"
  ;
-- created_at: 2026-10-08T11:16:10.502664331+00:00
-- finished_at: 2026-10-08T11:16:10.635751619+00:00
-- elapsed: 133ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "nominal_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "nominal_annulled_votes__dbt_alter" = "nominal_annulled_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "nominal_annulled_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "nominal_annulled_votes__dbt_alter" to "nominal_annulled_votes"
  ;
-- created_at: 2026-10-08T11:16:10.639765054+00:00
-- finished_at: 2026-10-08T11:16:10.769350990+00:00
-- elapsed: 129ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "legend_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "legend_annulled_votes__dbt_alter" = "legend_annulled_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "legend_annulled_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "legend_annulled_votes__dbt_alter" to "legend_annulled_votes"
  ;
-- created_at: 2026-10-08T11:16:10.773048284+00:00
-- finished_at: 2026-10-08T11:16:10.886524846+00:00
-- elapsed: 113ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "annulled_subjudice_votes__dbt_alter" = "annulled_subjudice_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "annulled_subjudice_votes__dbt_alter" to "annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:16:10.890875447+00:00
-- finished_at: 2026-10-08T11:16:11.002318498+00:00
-- elapsed: 111ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:16:11.005290372+00:00
-- finished_at: 2026-10-08T11:16:11.108862552+00:00
-- elapsed: 103ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:16:11.111691081+00:00
-- finished_at: 2026-10-08T11:16:11.202723653+00:00
-- elapsed: 91ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "blank_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "blank_votes__dbt_alter" = "blank_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "blank_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "blank_votes__dbt_alter" to "blank_votes"
  ;
-- created_at: 2026-10-08T11:16:11.205542406+00:00
-- finished_at: 2026-10-08T11:16:11.313694022+00:00
-- elapsed: 108ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "total_null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "total_null_votes__dbt_alter" = "total_null_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "total_null_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "total_null_votes__dbt_alter" to "total_null_votes"
  ;
-- created_at: 2026-10-08T11:16:11.317671167+00:00
-- finished_at: 2026-10-08T11:16:11.437562817+00:00
-- elapsed: 119ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "null_votes__dbt_alter" = "null_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "null_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "null_votes__dbt_alter" to "null_votes"
  ;
-- created_at: 2026-10-08T11:16:11.440844914+00:00
-- finished_at: 2026-10-08T11:16:11.590570364+00:00
-- elapsed: 149ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "technical_null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "technical_null_votes__dbt_alter" = "technical_null_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "technical_null_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "technical_null_votes__dbt_alter" to "technical_null_votes"
  ;
-- created_at: 2026-10-08T11:16:11.593468696+00:00
-- finished_at: 2026-10-08T11:16:11.715081333+00:00
-- elapsed: 121ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "separately_counted_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_tally_munzona" set "separately_counted_annulled_votes__dbt_alter" = "separately_counted_annulled_votes";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "separately_counted_annulled_votes" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "separately_counted_annulled_votes__dbt_alter" to "separately_counted_annulled_votes"
  ;
-- created_at: 2026-10-08T11:16:11.717948978+00:00
-- finished_at: 2026-10-08T11:16:11.875344637+00:00
-- elapsed: 157ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."fact_tally_munzona" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T11:16:11.879356131+00:00
-- finished_at: 2026-10-08T11:16:11.992184072+00:00
-- elapsed: 112ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_tally_munzona" add column "last_totalization_at__dbt_alter" datetime;
    update "tse_analytics"."main"."fact_tally_munzona" set "last_totalization_at__dbt_alter" = "last_totalization_at";
    alter table "tse_analytics"."main"."fact_tally_munzona" drop column "last_totalization_at" cascade;
    alter table "tse_analytics"."main"."fact_tally_munzona" rename column "last_totalization_at__dbt_alter" to "last_totalization_at"
  ;
-- created_at: 2026-10-08T11:16:12.007533177+00:00
-- finished_at: 2026-10-08T11:16:12.750067266+00:00
-- elapsed: 742ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,
    is_transit_vote,

    eligible_voters,
    voters_uninstalled_sections,
    turnout,
    abstentions,

    total_votes,
    competing_votes,

    valid_votes,
    nominal_valid_votes,
    total_legend_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_valid_votes,

    annulled_votes,
    nominal_annulled_votes,
    legend_annulled_votes,

    annulled_subjudice_votes,
    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    blank_votes,
    total_null_votes,
    null_votes,
    technical_null_votes,
    separately_counted_annulled_votes,

    generated_at,
    last_totalization_at,
    source_file
from "tse_analytics"."main"."stg_tally_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_tally_munzona" as DBT_INCREMENTAL_TARGET
            using "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656"
            where (
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_tally_munzona" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file"
        from "fact_tally_munzona__dbt_tmp_cf64748c_9a37_4a9c_8b58_5c633ee6c656"
    )
  ;
-- created_at: 2026-10-08T11:16:12.763272667+00:00
-- finished_at: 2026-10-08T11:16:13.152259233+00:00
-- elapsed: 388ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create  table
      "tse_analytics"."main"."dim_election__dbt_tmp"
  
    as (
      

with election_sources as (

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from "tse_analytics"."main"."stg_candidates"

    union all

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from "tse_analytics"."main"."stg_party_votes_raw"

    union all

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from "tse_analytics"."main"."stg_tally_munzona"
),

deduped as (
    select
        election_year,
        election_type,
        election_code,
        max(election_scope) as election_scope
    from election_sources
    group by 1,2,3
)

select
    concat(
        cast(election_year as varchar),
        ':',
        election_type,
        ':',
        election_code
    ) as election_id,

    election_year,
    election_type,
    election_code,
    election_scope,

    concat(cast(election_year as varchar), ' ', election_type, ' ', election_code)
        as cycle_label

from deduped
    );
  
    
  ;
-- created_at: 2026-10-08T11:16:13.154642224+00:00
-- finished_at: 2026-10-08T11:16:13.155995579+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */

    SELECT index_name
    FROM duckdb_indexes()
    WHERE schema_name = 'main'
      AND table_name = 'dim_election'
  ;
-- created_at: 2026-10-08T11:16:13.157322603+00:00
-- finished_at: 2026-10-08T11:16:13.158297397+00:00
-- elapsed: 974us
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */

    SELECT COUNT(*) as remaining_indexes
    FROM duckdb_indexes()
    WHERE schema_name = 'main'
      AND table_name = 'dim_election'
  ;
-- created_at: 2026-10-08T11:16:13.160484977+00:00
-- finished_at: 2026-10-08T11:16:13.168176918+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election" rename to "dim_election__dbt_backup";
-- created_at: 2026-10-08T11:16:13.170650757+00:00
-- finished_at: 2026-10-08T11:16:13.176616548+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election__dbt_tmp" rename to "dim_election";
-- created_at: 2026-10-08T11:16:13.181029351+00:00
-- finished_at: 2026-10-08T11:16:13.190902835+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop table if exists "tse_analytics"."main"."dim_election__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:16:13.202545875+00:00
-- finished_at: 2026-10-08T11:16:13.207071549+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

with source_rows as (
    select *
    from "tse_analytics"."main"."stg_party_votes_raw"
    
    where election_year in (2026) and election_type in ('general')
    
),

collapsed as (
    select
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,

        uf,
        municipality_code,
        zone,

        office_code,
        office_scope,

        party_number,
        is_transit_vote,

        max(party) as party,
        max(party_name) as party_name,

        max(nominal_valid_votes) as nominal_valid_votes,
        max(legend_valid_votes) as legend_valid_votes,
        max(nominal_converted_to_legend_votes) as nominal_converted_to_legend_votes,
        max(total_legend_valid_votes) as total_legend_valid_votes,

        max(nominal_annulled_subjudice_votes) as nominal_annulled_subjudice_votes,
        max(legend_annulled_subjudice_votes) as legend_annulled_subjudice_votes,

        max(generated_at) as generated_at,
        max(source_file) as source_file,

        count(*) as source_row_count,
        count(distinct party_group_type) as source_party_group_types,
        count(distinct coalition_id) as source_coalitions,
        count(distinct federation_number) as source_federations

    from source_rows
    group by
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        office_scope,
        party_number,
        is_transit_vote
)

select *
from collapsed
    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:16:13.210899513+00:00
-- finished_at: 2026-10-08T11:16:13.216438057+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'stg_party_votes_munzona'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:16:13.224303184+00:00
-- finished_at: 2026-10-08T11:16:15.217740751+00:00
-- elapsed: 2.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:16:15.222347402+00:00
-- finished_at: 2026-10-08T11:16:16.127954612+00:00
-- elapsed: 905ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T11:16:16.132182385+00:00
-- finished_at: 2026-10-08T11:16:17.056122232+00:00
-- elapsed: 923ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "zone" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T11:16:17.063184722+00:00
-- finished_at: 2026-10-08T11:16:17.877938361+00:00
-- elapsed: 814ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T11:16:17.884935145+00:00
-- finished_at: 2026-10-08T11:16:19.045610476+00:00
-- elapsed: 1.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T11:16:19.049842630+00:00
-- finished_at: 2026-10-08T11:16:21.486715504+00:00
-- elapsed: 2.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:16:21.490151158+00:00
-- finished_at: 2026-10-08T11:16:24.513429603+00:00
-- elapsed: 3.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "nominal_converted_to_legend_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "nominal_converted_to_legend_votes__dbt_alter" = "nominal_converted_to_legend_votes";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "nominal_converted_to_legend_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "nominal_converted_to_legend_votes__dbt_alter" to "nominal_converted_to_legend_votes"
  ;
-- created_at: 2026-10-08T11:16:24.517301081+00:00
-- finished_at: 2026-10-08T11:16:27.835442490+00:00
-- elapsed: 3.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:16:27.838956941+00:00
-- finished_at: 2026-10-08T11:16:30.975535224+00:00
-- elapsed: 3.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:16:30.979802450+00:00
-- finished_at: 2026-10-08T11:16:33.033818436+00:00
-- elapsed: 2.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:16:33.037568092+00:00
-- finished_at: 2026-10-08T11:16:34.444868586+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T11:16:34.448764886+00:00
-- finished_at: 2026-10-08T11:16:36.259679878+00:00
-- elapsed: 1.8s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "source_row_count__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "source_row_count__dbt_alter" = "source_row_count";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "source_row_count" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "source_row_count__dbt_alter" to "source_row_count"
  ;
-- created_at: 2026-10-08T11:16:36.263219748+00:00
-- finished_at: 2026-10-08T11:16:37.782580309+00:00
-- elapsed: 1.5s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "source_party_group_types__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "source_party_group_types__dbt_alter" = "source_party_group_types";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "source_party_group_types" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "source_party_group_types__dbt_alter" to "source_party_group_types"
  ;
-- created_at: 2026-10-08T11:16:37.787209234+00:00
-- finished_at: 2026-10-08T11:16:39.209856086+00:00
-- elapsed: 1.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "source_coalitions__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "source_coalitions__dbt_alter" = "source_coalitions";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "source_coalitions" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "source_coalitions__dbt_alter" to "source_coalitions"
  ;
-- created_at: 2026-10-08T11:16:39.213256330+00:00
-- finished_at: 2026-10-08T11:16:42.136752918+00:00
-- elapsed: 2.9s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."stg_party_votes_munzona" add column "source_federations__dbt_alter" bigint;
    update "tse_analytics"."main"."stg_party_votes_munzona" set "source_federations__dbt_alter" = "source_federations";
    alter table "tse_analytics"."main"."stg_party_votes_munzona" drop column "source_federations" cascade;
    alter table "tse_analytics"."main"."stg_party_votes_munzona" rename column "source_federations__dbt_alter" to "source_federations"
  ;
-- created_at: 2026-10-08T11:16:42.441668938+00:00
-- finished_at: 2026-10-08T11:16:52.803406930+00:00
-- elapsed: 10.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9"
  
    as (
      

with source_rows as (
    select *
    from "tse_analytics"."main"."stg_party_votes_raw"
    
    where election_year in (2026) and election_type in ('general')
    
),

collapsed as (
    select
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,

        uf,
        municipality_code,
        zone,

        office_code,
        office_scope,

        party_number,
        is_transit_vote,

        max(party) as party,
        max(party_name) as party_name,

        max(nominal_valid_votes) as nominal_valid_votes,
        max(legend_valid_votes) as legend_valid_votes,
        max(nominal_converted_to_legend_votes) as nominal_converted_to_legend_votes,
        max(total_legend_valid_votes) as total_legend_valid_votes,

        max(nominal_annulled_subjudice_votes) as nominal_annulled_subjudice_votes,
        max(legend_annulled_subjudice_votes) as legend_annulled_subjudice_votes,

        max(generated_at) as generated_at,
        max(source_file) as source_file,

        count(*) as source_row_count,
        count(distinct party_group_type) as source_party_group_types,
        count(distinct coalition_id) as source_coalitions,
        count(distinct federation_number) as source_federations

    from source_rows
    group by
        election_year,
        election_type,
        election_scope,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        office_scope,
        party_number,
        is_transit_vote
)

select *
from collapsed
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."stg_party_votes_munzona" as DBT_INCREMENTAL_TARGET
            using "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9"
            where (
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."stg_party_votes_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations"
        from "stg_party_votes_munzona__dbt_tmp_c89040b4_9c24_41ab_be96_9b87bd7a6eb9"
    )
  ;
-- created_at: 2026-10-08T11:16:52.878500015+00:00
-- finished_at: 2026-10-08T11:16:52.949460510+00:00
-- elapsed: 70ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_nonnegative_tally
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_nonnegative_tally", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_tally_munzona"
where eligible_voters < 0
   or turnout < 0
   or abstentions < 0
   or total_votes < 0
   or valid_votes < 0
   or nominal_valid_votes < 0
   or total_legend_valid_votes < 0
   or blank_votes < 0
   or total_null_votes < 0
   or annulled_votes < 0
   or annulled_subjudice_votes < 0
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:52.966322885+00:00
-- finished_at: 2026-10-08T11:16:52.995310721+00:00
-- elapsed: 28ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_tally_vote_balance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_tally_vote_balance", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_tally_munzona"
where total_votes <> valid_votes
                   + blank_votes
                   + total_null_votes
                   + annulled_votes
                   + annulled_subjudice_votes
                   + separately_counted_annulled_votes
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.013217346+00:00
-- finished_at: 2026-10-08T11:16:53.046223933+00:00
-- elapsed: 33ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_tally_electorate_balance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_tally_electorate_balance", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_tally_munzona"
where eligible_voters
   <> turnout
    + abstentions
    + coalesce(voters_uninstalled_sections, 0)
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.056892236+00:00
-- finished_at: 2026-10-08T11:16:53.178965625+00:00
-- elapsed: 122ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_tally_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.77bc49bd4e
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_tally_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.77bc49bd4e", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    from "tse_analytics"."main"."fact_tally_munzona"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.189055735+00:00
-- finished_at: 2026-10-08T11:16:53.335935023+00:00
-- elapsed: 146ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_fact_tally_snapshot_complete
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_fact_tally_snapshot_complete", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with source_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, is_transit_vote,
        eligible_voters, voters_uninstalled_sections, turnout, abstentions,
        total_votes, valid_votes,
        nominal_valid_votes, total_legend_valid_votes,
        blank_votes, total_null_votes,
        annulled_votes, annulled_subjudice_votes
    from "tse_analytics"."main"."stg_tally_munzona"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, is_transit_vote,
        eligible_voters, voters_uninstalled_sections, turnout, abstentions,
        total_votes, valid_votes,
        nominal_valid_votes, total_legend_valid_votes,
        blank_votes, total_null_votes,
        annulled_votes, annulled_subjudice_votes
    from "tse_analytics"."main"."fact_tally_munzona"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_or_changed_in_target' as issue, * from source_rows
     except
     select 'missing_or_changed_in_target' as issue, * from target_rows)
    union all
    (select 'stale_or_changed_in_target' as issue, * from target_rows
     except
     select 'stale_or_changed_in_target' as issue, * from source_rows)
)
select * from diff
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.346664449+00:00
-- finished_at: 2026-10-08T11:16:53.351850735+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_election_election_id.b977e96806
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_election_election_id.b977e96806", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_id
from "tse_analytics"."main"."dim_election"
where election_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.360857899+00:00
-- finished_at: 2026-10-08T11:16:53.363870788+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.unique_dim_election_election_id.c93f13d88d
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.unique_dim_election_election_id.c93f13d88d", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    election_id as unique_field,
    count(*) as n_records

from "tse_analytics"."main"."dim_election"
where election_id is not null
group by election_id
having count(*) > 1



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.373381923+00:00
-- finished_at: 2026-10-08T11:16:53.376836668+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_election_election_year__election_type__election_code.acb4acaa61
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_election_election_year__election_type__election_code.acb4acaa61", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code
    from "tse_analytics"."main"."dim_election"
    group by election_year, election_type, election_code
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.384901304+00:00
-- finished_at: 2026-10-08T11:16:53.389649236+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_dim_election_election_type__general__municipal.dbd4ee5830
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_dim_election_election_type__general__municipal.dbd4ee5830", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_type as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."dim_election"
    group by election_type

)

select *
from all_values
where value_field not in (
    'general','municipal'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.420739331+00:00
-- finished_at: 2026-10-08T11:16:53.425113955+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_office_code.b51075d174
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_office_code.b51075d174", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select office_code
from "tse_analytics"."main"."stg_party_votes_munzona"
where office_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.436259616+00:00
-- finished_at: 2026-10-08T11:16:53.461581444+00:00
-- elapsed: 25ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_zone.15b4902401
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_zone.15b4902401", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select zone
from "tse_analytics"."main"."stg_party_votes_munzona"
where zone is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.472747025+00:00
-- finished_at: 2026-10-08T11:16:53.474771413+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_uf.e6fa28c649
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_uf.e6fa28c649", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select uf
from "tse_analytics"."main"."stg_party_votes_munzona"
where uf is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.485946441+00:00
-- finished_at: 2026-10-08T11:16:53.517290953+00:00
-- elapsed: 31ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_nominal_valid_votes.e7f4c96cd2
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_nominal_valid_votes.e7f4c96cd2", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_valid_votes
from "tse_analytics"."main"."stg_party_votes_munzona"
where nominal_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.529658789+00:00
-- finished_at: 2026-10-08T11:16:53.533035387+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_election_code.ef14f49295
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_election_code.ef14f49295", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_code
from "tse_analytics"."main"."stg_party_votes_munzona"
where election_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.544595468+00:00
-- finished_at: 2026-10-08T11:16:53.573737225+00:00
-- elapsed: 29ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_legend_valid_votes.7f0bd09aed
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_legend_valid_votes.7f0bd09aed", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select legend_valid_votes
from "tse_analytics"."main"."stg_party_votes_munzona"
where legend_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.584816140+00:00
-- finished_at: 2026-10-08T11:16:53.619076311+00:00
-- elapsed: 34ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_election_year.0f46fed70a
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_election_year.0f46fed70a", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."stg_party_votes_munzona"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.628647242+00:00
-- finished_at: 2026-10-08T11:16:53.630778962+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_party_number.64a1e2cfb2
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_party_number.64a1e2cfb2", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_number
from "tse_analytics"."main"."stg_party_votes_munzona"
where party_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.639580206+00:00
-- finished_at: 2026-10-08T11:16:53.641882058+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_election_type.4a512e2486
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_election_type.4a512e2486", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."stg_party_votes_munzona"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.650599102+00:00
-- finished_at: 2026-10-08T11:16:53.652753177+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_municipality_code.c759a8d872
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_municipality_code.c759a8d872", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select municipality_code
from "tse_analytics"."main"."stg_party_votes_munzona"
where municipality_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.661498102+00:00
-- finished_at: 2026-10-08T11:16:53.688200069+00:00
-- elapsed: 26ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_round_number.14468de79d
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_round_number.14468de79d", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select round_number
from "tse_analytics"."main"."stg_party_votes_munzona"
where round_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:53.700283678+00:00
-- finished_at: 2026-10-08T11:16:55.727172891+00:00
-- elapsed: 2.0s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_stg_party_votes_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__party_number__is_transit_vote.32c3bc47c3
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_stg_party_votes_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__party_number__is_transit_vote.32c3bc47c3", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, party_number, is_transit_vote
    from "tse_analytics"."main"."stg_party_votes_munzona"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, party_number, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:55.736482853+00:00
-- finished_at: 2026-10-08T11:16:55.770637183+00:00
-- elapsed: 34ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_stg_party_votes_munzona_total_legend_valid_votes.5a6ae8ddea
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_stg_party_votes_munzona_total_legend_valid_votes.5a6ae8ddea", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_legend_valid_votes
from "tse_analytics"."main"."stg_party_votes_munzona"
where total_legend_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:16:55.794292489+00:00
-- finished_at: 2026-10-08T11:16:56.471457178+00:00
-- elapsed: 677ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select
    c.election_year,
    c.election_type,
    c.election_scope,
    cast(c.election_year as varchar) || ':' || c.election_type as election_id,
    c.election_code,
    c.election_description,
    c.round_number,
    c.electoral_unit,
    c.office_scope,
    c.candidate_id,
    c.uf,
    c.office_code,
    c.office,
    c.candidate_number,
    c.candidate_name,
    c.ballot_name,
    c.party_number,
    c.party,
    c.party_name,
    c.candidacy_status,
    c.gender,
    c.education,
    c.occupation,
    c.race_color,
    coalesce(a.declared_assets_value, 0) as declared_assets_value,
    coalesce(a.declared_assets_count, 0) as declared_assets_count
from "tse_analytics"."main"."stg_candidates" c
left join "tse_analytics"."main"."int_candidate_assets" a
  using (election_year, election_type, election_code, candidate_id)

  
    where c.election_year in (2026) and c.election_type in ('general')
  

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:16:56.510967320+00:00
-- finished_at: 2026-10-08T11:16:56.529213001+00:00
-- elapsed: 18ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'dim_candidate'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:16:56.539267467+00:00
-- finished_at: 2026-10-08T11:17:03.175958240+00:00
-- elapsed: 6.6s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."dim_candidate" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."dim_candidate" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."dim_candidate" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."dim_candidate" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:17:03.220357926+00:00
-- finished_at: 2026-10-08T11:17:09.115120081+00:00
-- elapsed: 5.9s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."dim_candidate" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."dim_candidate" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."dim_candidate" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."dim_candidate" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T11:17:09.232205050+00:00
-- finished_at: 2026-10-08T11:17:16.486008234+00:00
-- elapsed: 7.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."dim_candidate" add column "declared_assets_value__dbt_alter" decimal(38, 2);
    update "tse_analytics"."main"."dim_candidate" set "declared_assets_value__dbt_alter" = "declared_assets_value";
    alter table "tse_analytics"."main"."dim_candidate" drop column "declared_assets_value" cascade;
    alter table "tse_analytics"."main"."dim_candidate" rename column "declared_assets_value__dbt_alter" to "declared_assets_value"
  ;
-- created_at: 2026-10-08T11:17:16.524047160+00:00
-- finished_at: 2026-10-08T11:17:21.104579817+00:00
-- elapsed: 4.6s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."dim_candidate" add column "declared_assets_count__dbt_alter" bigint;
    update "tse_analytics"."main"."dim_candidate" set "declared_assets_count__dbt_alter" = "declared_assets_count";
    alter table "tse_analytics"."main"."dim_candidate" drop column "declared_assets_count" cascade;
    alter table "tse_analytics"."main"."dim_candidate" rename column "declared_assets_count__dbt_alter" to "declared_assets_count"
  ;
-- created_at: 2026-10-08T11:17:21.546148337+00:00
-- finished_at: 2026-10-08T11:17:37.682956306+00:00
-- elapsed: 16.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_candidate__dbt_tmp_eff3927f_ba01_41bf_8489_e690b4dfff75"
  
    as (
      

select
    c.election_year,
    c.election_type,
    c.election_scope,
    cast(c.election_year as varchar) || ':' || c.election_type as election_id,
    c.election_code,
    c.election_description,
    c.round_number,
    c.electoral_unit,
    c.office_scope,
    c.candidate_id,
    c.uf,
    c.office_code,
    c.office,
    c.candidate_number,
    c.candidate_name,
    c.ballot_name,
    c.party_number,
    c.party,
    c.party_name,
    c.candidacy_status,
    c.gender,
    c.education,
    c.occupation,
    c.race_color,
    coalesce(a.declared_assets_value, 0) as declared_assets_value,
    coalesce(a.declared_assets_count, 0) as declared_assets_count
from "tse_analytics"."main"."stg_candidates" c
left join "tse_analytics"."main"."int_candidate_assets" a
  using (election_year, election_type, election_code, candidate_id)

  
    where c.election_year in (2026) and c.election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_candidate" as DBT_INCREMENTAL_TARGET
            using "dim_candidate__dbt_tmp_eff3927f_ba01_41bf_8489_e690b4dfff75"
            where (
                
                    "dim_candidate__dbt_tmp_eff3927f_ba01_41bf_8489_e690b4dfff75".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_candidate__dbt_tmp_eff3927f_ba01_41bf_8489_e690b4dfff75".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_candidate__dbt_tmp_eff3927f_ba01_41bf_8489_e690b4dfff75".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_candidate__dbt_tmp_eff3927f_ba01_41bf_8489_e690b4dfff75".candidate_id = DBT_INCREMENTAL_TARGET.candidate_id
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_candidate" ("election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count"
        from "dim_candidate__dbt_tmp_eff3927f_ba01_41bf_8489_e690b4dfff75"
    )
  ;
-- created_at: 2026-10-08T11:17:37.810494954+00:00
-- finished_at: 2026-10-08T11:17:37.927543829+00:00
-- elapsed: 117ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,
    is_transit_vote,

    eligible_voters,
    voters_uninstalled_sections,
    eligible_voters - turnout - abstentions as uncounted_voters,
    turnout,
    abstentions,

    case when eligible_voters > 0
         then turnout::double / eligible_voters
    end as turnout_rate,

    case when eligible_voters > 0
         then abstentions::double / eligible_voters
    end as abstention_rate,

    generated_at
from "tse_analytics"."main"."fact_tally_munzona"

where election_year in (2026) and election_type in ('general')

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:17:37.969194629+00:00
-- finished_at: 2026-10-08T11:17:38.097853934+00:00
-- elapsed: 128ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'fact_turnout'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:17:38.176168424+00:00
-- finished_at: 2026-10-08T11:17:38.932459323+00:00
-- elapsed: 756ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."fact_turnout" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."fact_turnout" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:17:38.996491323+00:00
-- finished_at: 2026-10-08T11:17:39.738213571+00:00
-- elapsed: 741ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."fact_turnout" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."fact_turnout" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T11:17:39.778299946+00:00
-- finished_at: 2026-10-08T11:17:40.331398849+00:00
-- elapsed: 553ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."fact_turnout" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."fact_turnout" drop column "zone" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T11:17:40.361493676+00:00
-- finished_at: 2026-10-08T11:17:40.754314422+00:00
-- elapsed: 392ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."fact_turnout" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."fact_turnout" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T11:17:40.783911022+00:00
-- finished_at: 2026-10-08T11:17:41.275366107+00:00
-- elapsed: 491ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "eligible_voters__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_turnout" set "eligible_voters__dbt_alter" = "eligible_voters";
    alter table "tse_analytics"."main"."fact_turnout" drop column "eligible_voters" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "eligible_voters__dbt_alter" to "eligible_voters"
  ;
-- created_at: 2026-10-08T11:17:41.305546227+00:00
-- finished_at: 2026-10-08T11:17:42.013376582+00:00
-- elapsed: 707ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "voters_uninstalled_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_turnout" set "voters_uninstalled_sections__dbt_alter" = "voters_uninstalled_sections";
    alter table "tse_analytics"."main"."fact_turnout" drop column "voters_uninstalled_sections" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "voters_uninstalled_sections__dbt_alter" to "voters_uninstalled_sections"
  ;
-- created_at: 2026-10-08T11:17:42.075598581+00:00
-- finished_at: 2026-10-08T11:17:42.889705549+00:00
-- elapsed: 814ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "uncounted_voters__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_turnout" set "uncounted_voters__dbt_alter" = "uncounted_voters";
    alter table "tse_analytics"."main"."fact_turnout" drop column "uncounted_voters" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "uncounted_voters__dbt_alter" to "uncounted_voters"
  ;
-- created_at: 2026-10-08T11:17:42.964644721+00:00
-- finished_at: 2026-10-08T11:17:43.850926988+00:00
-- elapsed: 886ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "turnout__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_turnout" set "turnout__dbt_alter" = "turnout";
    alter table "tse_analytics"."main"."fact_turnout" drop column "turnout" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "turnout__dbt_alter" to "turnout"
  ;
-- created_at: 2026-10-08T11:17:43.897486116+00:00
-- finished_at: 2026-10-08T11:17:44.757628680+00:00
-- elapsed: 860ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "abstentions__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_turnout" set "abstentions__dbt_alter" = "abstentions";
    alter table "tse_analytics"."main"."fact_turnout" drop column "abstentions" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "abstentions__dbt_alter" to "abstentions"
  ;
-- created_at: 2026-10-08T11:17:44.821913263+00:00
-- finished_at: 2026-10-08T11:17:46.016630422+00:00
-- elapsed: 1.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "turnout_rate__dbt_alter" double;
    update "tse_analytics"."main"."fact_turnout" set "turnout_rate__dbt_alter" = "turnout_rate";
    alter table "tse_analytics"."main"."fact_turnout" drop column "turnout_rate" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "turnout_rate__dbt_alter" to "turnout_rate"
  ;
-- created_at: 2026-10-08T11:17:46.076094590+00:00
-- finished_at: 2026-10-08T11:17:46.945694710+00:00
-- elapsed: 869ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "abstention_rate__dbt_alter" double;
    update "tse_analytics"."main"."fact_turnout" set "abstention_rate__dbt_alter" = "abstention_rate";
    alter table "tse_analytics"."main"."fact_turnout" drop column "abstention_rate" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "abstention_rate__dbt_alter" to "abstention_rate"
  ;
-- created_at: 2026-10-08T11:17:46.974490678+00:00
-- finished_at: 2026-10-08T11:17:47.553641941+00:00
-- elapsed: 579ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_turnout" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."fact_turnout" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."fact_turnout" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."fact_turnout" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T11:17:47.699342429+00:00
-- finished_at: 2026-10-08T11:17:49.796954936+00:00
-- elapsed: 2.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,
    is_transit_vote,

    eligible_voters,
    voters_uninstalled_sections,
    eligible_voters - turnout - abstentions as uncounted_voters,
    turnout,
    abstentions,

    case when eligible_voters > 0
         then turnout::double / eligible_voters
    end as turnout_rate,

    case when eligible_voters > 0
         then abstentions::double / eligible_voters
    end as abstention_rate,

    generated_at
from "tse_analytics"."main"."fact_tally_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_turnout" as DBT_INCREMENTAL_TARGET
            using "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257"
            where (
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_turnout" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at"
        from "fact_turnout__dbt_tmp_13a0c9e6_2cd3_429d_934f_d28b4bf3a257"
    )
  ;
-- created_at: 2026-10-08T11:17:49.922987100+00:00
-- finished_at: 2026-10-08T11:17:50.530302193+00:00
-- elapsed: 607ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_tally_reconciliation__dbt_tmp" as (
    

with coverage as (
    select *
    from "tse_analytics"."main"."int_candidate_result_coverage"
),

candidate as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,
        sum(nominal_valid_votes) as candidate_nominal_valid_votes
    from "tse_analytics"."main"."fact_candidate_votes"
    group by 1,2,3,4,5,6,7,8,9
),

tally as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,
        sum(nominal_valid_votes) as tally_nominal_valid_votes
    from "tse_analytics"."main"."fact_tally_munzona"
    group by 1,2,3,4,5,6,7,8,9
)

select
    t.election_year,
    t.election_type,
    t.election_code,
    t.round_number,
    t.uf,
    t.municipality_code,
    t.zone,
    t.office_code,
    t.is_transit_vote,
    coalesce(c.candidate_nominal_valid_votes, 0) as candidate_nominal_valid_votes,
    t.tally_nominal_valid_votes,
    coalesce(c.candidate_nominal_valid_votes, 0) - t.tally_nominal_valid_votes
        as nominal_valid_delta
from tally t
join coverage cv
  on cv.election_year = t.election_year
 and cv.election_type = t.election_type
 and cv.election_code = t.election_code
 and cv.round_number = t.round_number
 and cv.uf = t.uf
 and cv.municipality_code = t.municipality_code
 and cv.zone = t.zone
 and cv.office_code = t.office_code
 and cv.is_transit_vote is not distinct from t.is_transit_vote
left join candidate c
  on c.election_year = t.election_year
 and c.election_type = t.election_type
 and c.election_code = t.election_code
 and c.round_number = t.round_number
 and c.uf = t.uf
 and c.municipality_code = t.municipality_code
 and c.zone = t.zone
 and c.office_code = t.office_code
 and c.is_transit_vote is not distinct from t.is_transit_vote
  );
;
-- created_at: 2026-10-08T11:17:50.561119800+00:00
-- finished_at: 2026-10-08T11:17:50.586331354+00:00
-- elapsed: 25ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation" rename to "candidate_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T11:17:50.612915793+00:00
-- finished_at: 2026-10-08T11:17:50.632675637+00:00
-- elapsed: 19ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation__dbt_tmp" rename to "candidate_tally_reconciliation";
-- created_at: 2026-10-08T11:17:50.674111052+00:00
-- finished_at: 2026-10-08T11:17:50.701334211+00:00
-- elapsed: 27ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:17:50.807444535+00:00
-- finished_at: 2026-10-08T11:17:50.974223838+00:00
-- elapsed: 166ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_tmp" as (
    

with coverage as (

    select *
    from "tse_analytics"."main"."int_candidate_result_coverage"

),

tally as (

    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,
        nominal_valid_votes,
        valid_votes,
        total_votes,
        generated_at

    from "tse_analytics"."main"."fact_tally_munzona"

)

select
    t.*,
    'missing_candidate_result_coverage' as gap_reason

from tally t

left join coverage c
  on c.election_year = t.election_year
 and c.election_type = t.election_type
 and c.election_code = t.election_code
 and c.round_number = t.round_number
 and c.uf = t.uf
 and c.municipality_code = t.municipality_code
 and c.zone = t.zone
 and c.office_code = t.office_code
 and c.is_transit_vote = t.is_transit_vote

where c.election_year is null
  and t.nominal_valid_votes > 0
  );
;
-- created_at: 2026-10-08T11:17:51.016844047+00:00
-- finished_at: 2026-10-08T11:17:51.035714362+00:00
-- elapsed: 18ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps" rename to "candidate_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T11:17:51.072960828+00:00
-- finished_at: 2026-10-08T11:17:51.093086722+00:00
-- elapsed: 20ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_tmp" rename to "candidate_tally_coverage_gaps";
-- created_at: 2026-10-08T11:17:51.155840147+00:00
-- finished_at: 2026-10-08T11:17:51.190043007+00:00
-- elapsed: 34ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:17:51.313702735+00:00
-- finished_at: 2026-10-08T11:17:51.341729700+00:00
-- elapsed: 28ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,

    uf,
    municipality_code,
    zone,

    office_code,
    office_scope,

    party_number,
    is_transit_vote,

    nominal_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_votes,
    total_legend_valid_votes,

    coalesce(nominal_valid_votes, 0)
      + coalesce(total_legend_valid_votes, 0) as party_valid_votes,

    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    generated_at,
    source_file
from "tse_analytics"."main"."stg_party_votes_munzona"

where election_year in (2026) and election_type in ('general')

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:17:51.384203044+00:00
-- finished_at: 2026-10-08T11:17:51.452787542+00:00
-- elapsed: 68ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'fact_party_votes'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:17:51.526538649+00:00
-- finished_at: 2026-10-08T11:17:57.037577579+00:00
-- elapsed: 5.5s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."fact_party_votes" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:17:57.118474364+00:00
-- finished_at: 2026-10-08T11:18:02.180110229+00:00
-- elapsed: 5.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."fact_party_votes" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T11:18:02.244006297+00:00
-- finished_at: 2026-10-08T11:18:06.569071891+00:00
-- elapsed: 4.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."fact_party_votes" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "zone" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T11:18:06.576395492+00:00
-- finished_at: 2026-10-08T11:18:08.819691349+00:00
-- elapsed: 2.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."fact_party_votes" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T11:18:08.831180955+00:00
-- finished_at: 2026-10-08T11:18:12.860281074+00:00
-- elapsed: 4.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_party_votes" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T11:18:12.886749947+00:00
-- finished_at: 2026-10-08T11:18:23.623459540+00:00
-- elapsed: 10.7s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_party_votes" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:18:23.632175294+00:00
-- finished_at: 2026-10-08T11:18:28.387750175+00:00
-- elapsed: 4.8s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "nominal_converted_to_legend_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_party_votes" set "nominal_converted_to_legend_votes__dbt_alter" = "nominal_converted_to_legend_votes";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "nominal_converted_to_legend_votes" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "nominal_converted_to_legend_votes__dbt_alter" to "nominal_converted_to_legend_votes"
  ;
-- created_at: 2026-10-08T11:18:28.394748358+00:00
-- finished_at: 2026-10-08T11:18:32.576990150+00:00
-- elapsed: 4.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_party_votes" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T11:18:32.588515589+00:00
-- finished_at: 2026-10-08T11:18:40.601058798+00:00
-- elapsed: 8.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "party_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_party_votes" set "party_valid_votes__dbt_alter" = "party_valid_votes";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "party_valid_votes" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "party_valid_votes__dbt_alter" to "party_valid_votes"
  ;
-- created_at: 2026-10-08T11:18:40.607610664+00:00
-- finished_at: 2026-10-08T11:18:52.675971559+00:00
-- elapsed: 12.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_party_votes" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:18:52.682026252+00:00
-- finished_at: 2026-10-08T11:18:56.529393669+00:00
-- elapsed: 3.8s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."fact_party_votes" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T11:18:56.537881952+00:00
-- finished_at: 2026-10-08T11:19:00.530004955+00:00
-- elapsed: 4.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_party_votes" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."fact_party_votes" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."fact_party_votes" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."fact_party_votes" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T11:19:00.550827879+00:00
-- finished_at: 2026-10-08T11:19:38.543562753+00:00
-- elapsed: 38.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,

    election_code,
    round_number,

    uf,
    municipality_code,
    zone,

    office_code,
    office_scope,

    party_number,
    is_transit_vote,

    nominal_valid_votes,
    legend_valid_votes,
    nominal_converted_to_legend_votes,
    total_legend_valid_votes,

    coalesce(nominal_valid_votes, 0)
      + coalesce(total_legend_valid_votes, 0) as party_valid_votes,

    nominal_annulled_subjudice_votes,
    legend_annulled_subjudice_votes,

    generated_at,
    source_file
from "tse_analytics"."main"."stg_party_votes_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_party_votes" as DBT_INCREMENTAL_TARGET
            using "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd"
            where (
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_party_votes" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file"
        from "fact_party_votes__dbt_tmp_a02d8e05_7597_4e1b_acac_a085a98b2dfd"
    )
  ;
-- created_at: 2026-10-08T11:19:39.808625139+00:00
-- finished_at: 2026-10-08T11:19:40.653568737+00:00
-- elapsed: 844ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_party
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_party", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

with versions as (
    select
        election_year,
        election_type,
        election_scope,
        cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,
        election_code,
        party_number,
        party,
        party_name,
        generated_at,
        row_number() over (
            partition by election_year, election_type, election_code, party_number
            order by generated_at desc nulls last, party desc, party_name desc
        ) as _rank
    from "tse_analytics"."main"."stg_party_votes_munzona"
    
    where election_year in (2026) and election_type in ('general')
    
)

select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    party_number,
    party,
    party_name,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code || ':' ||
      cast(election_code as varchar) || ':' || cast(party_number as varchar) as party_id
from versions
where _rank = 1
    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:19:40.655899223+00:00
-- finished_at: 2026-10-08T11:19:41.158243965+00:00
-- elapsed: 502ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_party
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_party", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'dim_party'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:19:41.198339390+00:00
-- finished_at: 2026-10-08T11:19:41.296254038+00:00
-- elapsed: 97ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_party
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_party", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."dim_party" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."dim_party" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."dim_party" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."dim_party" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:19:41.312350217+00:00
-- finished_at: 2026-10-08T11:19:47.649124582+00:00
-- elapsed: 6.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_party
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_party", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_party__dbt_tmp_97fc7982_31a4_461d_9462_6563b39c759d"
  
    as (
      

with versions as (
    select
        election_year,
        election_type,
        election_scope,
        cast(election_year as varchar) || ':' || election_type || ':' || election_code as election_id,
        election_code,
        party_number,
        party,
        party_name,
        generated_at,
        row_number() over (
            partition by election_year, election_type, election_code, party_number
            order by generated_at desc nulls last, party desc, party_name desc
        ) as _rank
    from "tse_analytics"."main"."stg_party_votes_munzona"
    
    where election_year in (2026) and election_type in ('general')
    
)

select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    party_number,
    party,
    party_name,
    cast(election_year as varchar) || ':' || election_type || ':' || election_code || ':' ||
      cast(election_code as varchar) || ':' || cast(party_number as varchar) as party_id
from versions
where _rank = 1
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_party" as DBT_INCREMENTAL_TARGET
            using "dim_party__dbt_tmp_97fc7982_31a4_461d_9462_6563b39c759d"
            where (
                
                    "dim_party__dbt_tmp_97fc7982_31a4_461d_9462_6563b39c759d".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_party__dbt_tmp_97fc7982_31a4_461d_9462_6563b39c759d".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_party__dbt_tmp_97fc7982_31a4_461d_9462_6563b39c759d".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_party__dbt_tmp_97fc7982_31a4_461d_9462_6563b39c759d".party_number = DBT_INCREMENTAL_TARGET.party_number
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_party" ("election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id"
        from "dim_party__dbt_tmp_97fc7982_31a4_461d_9462_6563b39c759d"
    )
  ;
-- created_at: 2026-10-08T11:19:48.095100299+00:00
-- finished_at: 2026-10-08T11:19:53.810182138+00:00
-- elapsed: 5.7s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_dim_candidate_snapshot_complete
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_dim_candidate_snapshot_complete", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with source_keys as (
    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."stg_candidates"
    where election_year in (2026) and election_type in ('general')
),
target_keys as (
    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."dim_candidate"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_in_target' as issue, * from source_keys
     except
     select 'missing_in_target' as issue, * from target_keys)
    union all
    (select 'stale_in_target' as issue, * from target_keys
     except
     select 'stale_in_target' as issue, * from source_keys)
)
select * from diff
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:19:54.176106996+00:00
-- finished_at: 2026-10-08T11:20:01.793241220+00:00
-- elapsed: 7.6s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_candidate_election_year__election_type__election_code__candidate_id.46c4e45e30
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_candidate_election_year__election_type__election_code__candidate_id.46c4e45e30", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."dim_candidate"
    group by election_year, election_type, election_code, candidate_id
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:01.884045011+00:00
-- finished_at: 2026-10-08T11:20:01.934676739+00:00
-- elapsed: 50ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_nonnegative_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_nonnegative_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."dim_candidate"
where declared_assets_value < 0
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:02.020466406+00:00
-- finished_at: 2026-10-08T11:20:02.039447348+00:00
-- elapsed: 18ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_candidate_election_code.72b3ba2d00
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_candidate_election_code.72b3ba2d00", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_code
from "tse_analytics"."main"."dim_candidate"
where election_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:02.105458092+00:00
-- finished_at: 2026-10-08T11:20:02.133584685+00:00
-- elapsed: 28ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_candidate_election_type.aff628686e
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_candidate_election_type.aff628686e", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."dim_candidate"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:02.261077221+00:00
-- finished_at: 2026-10-08T11:20:02.434001552+00:00
-- elapsed: 172ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_candidate_election_year.fb77fb8393
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_candidate_election_year.fb77fb8393", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."dim_candidate"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:02.510791816+00:00
-- finished_at: 2026-10-08T11:20:02.527676251+00:00
-- elapsed: 16ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_candidate_candidate_id.882821f01b
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_candidate_candidate_id.882821f01b", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_id
from "tse_analytics"."main"."dim_candidate"
where candidate_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:02.593440391+00:00
-- finished_at: 2026-10-08T11:20:02.610641477+00:00
-- elapsed: 17ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_candidate_election_id.e7655e0fda
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_candidate_election_id.e7655e0fda", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_id
from "tse_analytics"."main"."dim_candidate"
where election_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:02.694269184+00:00
-- finished_at: 2026-10-08T11:20:35.069596783+00:00
-- elapsed: 32.4s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_candidate_votes_candidate_fk
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_candidate_votes_candidate_fk", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select
    f.election_year,
    f.election_type,
    f.election_code,
    f.candidate_id,
    count(*) as missing_rows
from "tse_analytics"."main"."fact_candidate_votes" f
left join "tse_analytics"."main"."dim_candidate" d
  on d.election_year = f.election_year
 and d.election_type = f.election_type
 and d.election_code = f.election_code
 and d.candidate_id = f.candidate_id
where d.candidate_id is null
group by 1,2,3,4
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:35.078179958+00:00
-- finished_at: 2026-10-08T11:20:35.097995453+00:00
-- elapsed: 19ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_turnout_turnout.2919d6e02e
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_turnout_turnout.2919d6e02e", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select turnout
from "tse_analytics"."main"."fact_turnout"
where turnout is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:35.102760406+00:00
-- finished_at: 2026-10-08T11:20:35.115065510+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_turnout_abstentions.99322c759a
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_turnout_abstentions.99322c759a", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select abstentions
from "tse_analytics"."main"."fact_turnout"
where abstentions is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:35.119955397+00:00
-- finished_at: 2026-10-08T11:20:35.131565999+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_turnout_eligible_voters.b0ad909ef2
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_turnout_eligible_voters.b0ad909ef2", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select eligible_voters
from "tse_analytics"."main"."fact_turnout"
where eligible_voters is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:35.136924692+00:00
-- finished_at: 2026-10-08T11:20:35.211626547+00:00
-- elapsed: 74ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_turnout_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.6ce3bd8520
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_turnout_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.6ce3bd8520", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    from "tse_analytics"."main"."fact_turnout"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:35.216411998+00:00
-- finished_at: 2026-10-08T11:20:35.249037723+00:00
-- elapsed: 32ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_turnout_rates_bounds
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_turnout_rates_bounds", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_turnout"
where turnout_rate < 0 or turnout_rate > 1
   or abstention_rate < 0 or abstention_rate > 1
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:15:04.339351838+00:00
-- finished_at: 2026-10-08T11:20:48.417759018+00:00
-- elapsed: 5m 44s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_stg_electorate_municipality_code_canonical
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_stg_electorate_municipality_code_canonical", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."stg_electorate"
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:35.254070093+00:00
-- finished_at: 2026-10-08T11:20:48.419224574+00:00
-- elapsed: 13.2s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_candidate_nominal_votes_reconcile_tally
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_candidate_nominal_votes_reconcile_tally", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."candidate_tally_reconciliation"
where nominal_valid_delta <> 0
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.431047639+00:00
-- finished_at: 2026-10-08T11:20:48.475861209+00:00
-- elapsed: 44ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_candidate_tally_coverage_gaps_gap_reason.a9dee62b38
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_candidate_tally_coverage_gaps_gap_reason.a9dee62b38", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select gap_reason
from "tse_analytics"."main"."candidate_tally_coverage_gaps"
where gap_reason is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.483138713+00:00
-- finished_at: 2026-10-08T11:20:48.531401785+00:00
-- elapsed: 48ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_round_number.50bdb5d12f
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_round_number.50bdb5d12f", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select round_number
from "tse_analytics"."main"."fact_party_votes"
where round_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.538199436+00:00
-- finished_at: 2026-10-08T11:20:48.554316974+00:00
-- elapsed: 16ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_election_year.051ecd245d
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_election_year.051ecd245d", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."fact_party_votes"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.560385011+00:00
-- finished_at: 2026-10-08T11:20:48.561553783+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_party_number.708e410565
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_party_number.708e410565", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_number
from "tse_analytics"."main"."fact_party_votes"
where party_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.566690209+00:00
-- finished_at: 2026-10-08T11:20:48.568090931+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_election_type.f488bb9e82
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_election_type.f488bb9e82", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."fact_party_votes"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.573755070+00:00
-- finished_at: 2026-10-08T11:20:48.689666031+00:00
-- elapsed: 115ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_total_legend_valid_votes.ad59122f30
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_total_legend_valid_votes.ad59122f30", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_legend_valid_votes
from "tse_analytics"."main"."fact_party_votes"
where total_legend_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.696817282+00:00
-- finished_at: 2026-10-08T11:20:50.045795739+00:00
-- elapsed: 1.3s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_party_votes_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__party_number__is_transit_vote.5522d87bb6
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_party_votes_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__party_number__is_transit_vote.5522d87bb6", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, party_number, is_transit_vote
    from "tse_analytics"."main"."fact_party_votes"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, party_number, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:50.054604544+00:00
-- finished_at: 2026-10-08T11:20:50.172896340+00:00
-- elapsed: 118ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_legend_valid_votes.1363bff419
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_legend_valid_votes.1363bff419", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select legend_valid_votes
from "tse_analytics"."main"."fact_party_votes"
where legend_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:50.182249392+00:00
-- finished_at: 2026-10-08T11:20:50.287851469+00:00
-- elapsed: 105ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_party_votes_cycle_scope
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_party_votes_cycle_scope", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_party_votes"
where
      (election_type = 'general' and office_scope = 'municipal')
   or (election_type = 'municipal' and office_scope in ('federal', 'state'))
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:50.299377570+00:00
-- finished_at: 2026-10-08T11:20:50.374474161+00:00
-- elapsed: 75ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_party_valid_votes.7983795722
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_party_valid_votes.7983795722", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_valid_votes
from "tse_analytics"."main"."fact_party_votes"
where party_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:50.387287182+00:00
-- finished_at: 2026-10-08T11:20:50.388629139+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_election_code.ca5e6fa4a3
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_election_code.ca5e6fa4a3", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_code
from "tse_analytics"."main"."fact_party_votes"
where election_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:50.394821678+00:00
-- finished_at: 2026-10-08T11:20:50.629107719+00:00
-- elapsed: 234ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_nonnegative_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_nonnegative_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."fact_party_votes"
where nominal_valid_votes < 0
   or legend_valid_votes < 0
   or total_legend_valid_votes < 0
   or party_valid_votes < 0
   or nominal_annulled_subjudice_votes < 0
   or legend_annulled_subjudice_votes < 0
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:50.636923043+00:00
-- finished_at: 2026-10-08T11:20:50.653941433+00:00
-- elapsed: 17ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_party_votes_nominal_valid_votes.496725c01b
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_party_votes_nominal_valid_votes.496725c01b", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_valid_votes
from "tse_analytics"."main"."fact_party_votes"
where nominal_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:50.661726064+00:00
-- finished_at: 2026-10-08T11:20:52.301994637+00:00
-- elapsed: 1.6s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_fact_party_votes_snapshot_complete
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_fact_party_votes_snapshot_complete", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with source_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, party_number, is_transit_vote,
        nominal_valid_votes, legend_valid_votes,
        nominal_converted_to_legend_votes, total_legend_valid_votes,
        nominal_annulled_subjudice_votes, legend_annulled_subjudice_votes
    from "tse_analytics"."main"."stg_party_votes_munzona"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year, election_type, election_code, round_number,
        uf, municipality_code, zone, office_code, party_number, is_transit_vote,
        nominal_valid_votes, legend_valid_votes,
        nominal_converted_to_legend_votes, total_legend_valid_votes,
        nominal_annulled_subjudice_votes, legend_annulled_subjudice_votes
    from "tse_analytics"."main"."fact_party_votes"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_or_changed_in_target' as issue, * from source_rows
     except
     select 'missing_or_changed_in_target' as issue, * from target_rows)
    union all
    (select 'stale_or_changed_in_target' as issue, * from target_rows
     except
     select 'stale_or_changed_in_target' as issue, * from source_rows)
)
select * from diff
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:48.426356528+00:00
-- finished_at: 2026-10-08T11:20:56.806320104+00:00
-- elapsed: 8.4s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_candidate_tally_reconciliation_nominal_valid_delta.6562cea092
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_candidate_tally_reconciliation_nominal_valid_delta.6562cea092", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_valid_delta
from "tse_analytics"."main"."candidate_tally_reconciliation"
where nominal_valid_delta is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:56.908136384+00:00
-- finished_at: 2026-10-08T11:20:56.909083980+00:00
-- elapsed: 947us
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_party_party_number.76c8820921
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_party_party_number.76c8820921", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_number
from "tse_analytics"."main"."dim_party"
where party_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:56.936389234+00:00
-- finished_at: 2026-10-08T11:20:56.937496251+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_party_party.a4ef69c28d
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_party_party.a4ef69c28d", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party
from "tse_analytics"."main"."dim_party"
where party is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:56.950283540+00:00
-- finished_at: 2026-10-08T11:20:56.951436131+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_party_party_name.fadfb15c18
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_party_party_name.fadfb15c18", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_name
from "tse_analytics"."main"."dim_party"
where party_name is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:56.978376499+00:00
-- finished_at: 2026-10-08T11:20:56.981851869+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_party_election_year__election_type__election_code__party_number.cdcd63b462
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_party_election_year__election_type__election_code__party_number.cdcd63b462", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, party_number
    from "tse_analytics"."main"."dim_party"
    group by election_year, election_type, election_code, party_number
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:56.993920291+00:00
-- finished_at: 2026-10-08T11:20:57.027613507+00:00
-- elapsed: 33ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_dim_party_snapshot_complete
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_dim_party_snapshot_complete", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with source_rows as (
    select distinct
        election_year,
        election_type,
        election_code,
        party_number
    from "tse_analytics"."main"."stg_party_votes_munzona"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year,
        election_type,
        election_code,
        party_number
    from "tse_analytics"."main"."dim_party"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_in_target' as issue, * from source_rows
     except
     select 'missing_in_target' as issue, * from target_rows)
    union all
    (select 'stale_in_target' as issue, * from target_rows
     except
     select 'stale_in_target' as issue, * from source_rows)
)
select * from diff
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:57.052634168+00:00
-- finished_at: 2026-10-08T11:20:57.136295627+00:00
-- elapsed: 83ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.unique_dim_party_party_id.9da298a6ce
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.unique_dim_party_party_id.9da298a6ce", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

select
    party_id as unique_field,
    count(*) as n_records

from "tse_analytics"."main"."dim_party"
where party_id is not null
group by party_id
having count(*) > 1



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:57.141693078+00:00
-- finished_at: 2026-10-08T11:20:57.142936527+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_dim_party_party_id.0287c6f11e
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_dim_party_party_id.0287c6f11e", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_id
from "tse_analytics"."main"."dim_party"
where party_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:57.197939987+00:00
-- finished_at: 2026-10-08T11:20:57.420159758+00:00
-- elapsed: 222ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_party_votes_party_fk
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_party_votes_party_fk", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select
    f.election_year,
    f.election_type,
    f.election_code,
    f.party_number,
    count(*) as missing_rows
from "tse_analytics"."main"."fact_party_votes" f
left join "tse_analytics"."main"."dim_party" d
  on d.election_year = f.election_year
 and d.election_type = f.election_type
 and d.election_code = f.election_code
 and d.party_number = f.party_number
where d.party_number is null
group by 1,2,3,4
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:52.309327857+00:00
-- finished_at: 2026-10-08T11:21:06.795464031+00:00
-- elapsed: 14.5s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_candidate_party_reconcile_within_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_candidate_party_reconcile_within_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with candidate as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,
        sum(nominal_valid_votes) as candidate_nominal_valid_votes
    from "tse_analytics"."main"."fact_candidate_votes"
    group by 1,2,3,4,5,6,7,8,9
),

party as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,
        sum(nominal_valid_votes) as party_nominal_valid_votes
    from "tse_analytics"."main"."fact_party_votes"
    group by 1,2,3,4,5,6,7,8,9
),

coverage as (
    select *
    from "tse_analytics"."main"."int_candidate_result_coverage"
)

select
    p.*,
    coalesce(c.candidate_nominal_valid_votes, 0) as candidate_nominal_valid_votes,
    coalesce(c.candidate_nominal_valid_votes, 0) - p.party_nominal_valid_votes
        as nominal_valid_delta
from party p
join coverage cv
  on cv.election_year = p.election_year
 and cv.election_type = p.election_type
 and cv.election_code = p.election_code
 and cv.round_number = p.round_number
 and cv.uf = p.uf
 and cv.office_code = p.office_code
left join candidate c
  on c.election_year = p.election_year
 and c.election_type = p.election_type
 and c.election_code = p.election_code
 and c.round_number = p.round_number
 and c.uf = p.uf
 and c.municipality_code = p.municipality_code
 and c.zone = p.zone
 and c.office_code = p.office_code
 and c.is_transit_vote is not distinct from p.is_transit_vote
where coalesce(c.candidate_nominal_valid_votes, 0) <> p.party_nominal_valid_votes
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:20:57.501588978+00:00
-- finished_at: 2026-10-08T11:21:14.742201291+00:00
-- elapsed: 17.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_summary__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_scope,
    office_scope,
    office,
    party,
    count(*) as candidates,
    avg(declared_assets_value) as avg_declared_assets_value,
    median(declared_assets_value) as median_declared_assets_value
from "tse_analytics"."main"."dim_candidate"
group by 1,2,3,4,5,6
  );
;
-- created_at: 2026-10-08T11:21:06.802421166+00:00
-- finished_at: 2026-10-08T11:21:14.759899182+00:00
-- elapsed: 8.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_vote_summary__dbt_tmp" as (
    select
    f.election_year,
    f.election_type,
    f.election_id,
    f.election_code,
    f.round_number,

    d.office_code,
    d.office,
    d.office_scope,

    f.candidate_id,
    d.candidate_number,
    d.candidate_name,
    d.ballot_name,
    d.party,
    d.party_name,

    sum(f.nominal_valid_votes) as nominal_valid_votes,
    count(distinct f.municipality_code) as municipalities_with_votes,
    count(distinct cast(f.uf as varchar) || ':' || cast(f.zone as varchar)) as zones_with_votes
from "tse_analytics"."main"."fact_candidate_votes" f
left join "tse_analytics"."main"."dim_candidate" d
  on d.election_year = f.election_year
 and d.election_type = f.election_type
 and d.election_code = f.election_code
 and d.candidate_id = f.candidate_id
group by
    f.election_year,
    f.election_type,
    f.election_id,
    f.election_code,
    f.round_number,
    d.office_code,
    d.office,
    d.office_scope,
    f.candidate_id,
    d.candidate_number,
    d.candidate_name,
    d.ballot_name,
    d.party,
    d.party_name
  );
;
-- created_at: 2026-10-08T11:21:14.744989940+00:00
-- finished_at: 2026-10-08T11:21:14.773272458+00:00
-- elapsed: 28ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary" rename to "candidate_summary__dbt_backup";
-- created_at: 2026-10-08T11:21:14.764431529+00:00
-- finished_at: 2026-10-08T11:21:14.777166580+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary" rename to "candidate_vote_summary__dbt_backup";
-- created_at: 2026-10-08T11:21:14.775723042+00:00
-- finished_at: 2026-10-08T11:21:14.780882912+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary__dbt_tmp" rename to "candidate_summary";
-- created_at: 2026-10-08T11:21:14.778385956+00:00
-- finished_at: 2026-10-08T11:21:14.786182276+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary__dbt_tmp" rename to "candidate_vote_summary";
-- created_at: 2026-10-08T11:21:14.782890340+00:00
-- finished_at: 2026-10-08T11:21:14.799704452+00:00
-- elapsed: 16ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:21:14.787645643+00:00
-- finished_at: 2026-10-08T11:21:14.803362044+00:00
-- elapsed: 15ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_vote_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:21:14.804068140+00:00
-- finished_at: 2026-10-08T11:21:14.808091704+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."electoral_participation__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_scope,
    election_id,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    office_scope,

    eligible_voters,
    voters_uninstalled_sections,
    uncounted_voters,
    turnout,
    abstentions,
    turnout_rate,
    abstention_rate,

    generated_at
from "tse_analytics"."main"."fact_turnout"
  );
;
-- created_at: 2026-10-08T11:21:14.809967438+00:00
-- finished_at: 2026-10-08T11:21:14.814565471+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation" rename to "electoral_participation__dbt_backup";
-- created_at: 2026-10-08T11:21:14.815722770+00:00
-- finished_at: 2026-10-08T11:21:14.828996780+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation__dbt_tmp" rename to "electoral_participation";
-- created_at: 2026-10-08T11:21:14.830669930+00:00
-- finished_at: 2026-10-08T11:21:14.834171363+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."electoral_participation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:21:14.838147401+00:00
-- finished_at: 2026-10-08T11:21:14.917317889+00:00
-- elapsed: 79ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."party_tally_coverage_gaps__dbt_tmp" as (
    

with party as (

    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,

        sum(coalesce(nominal_valid_votes, 0))
            as party_nominal_valid_votes,

        sum(coalesce(legend_valid_votes, 0))
            as party_legend_valid_votes,

        sum(coalesce(total_legend_valid_votes, 0))
            as party_total_legend_valid_votes,

        sum(coalesce(nominal_converted_to_legend_votes, 0))
            as party_nominal_converted_to_legend_votes,

        sum(coalesce(party_valid_votes, 0))
            as party_valid_votes,

        count(*) as party_rows

    from "tse_analytics"."main"."fact_party_votes"

    group by
        1,2,3,4,5,6,7,8,9

),

tally as (

    select
        *
    from "tse_analytics"."main"."fact_tally_munzona"

),

comparison as (

    select
        t.*,

        p.party_rows,
        p.party_nominal_valid_votes,
        p.party_legend_valid_votes,
        p.party_total_legend_valid_votes,
        p.party_nominal_converted_to_legend_votes,
        p.party_valid_votes,

        case

            -- No party source rows whatsoever at this tally grain.
            when p.party_rows is null
             and coalesce(t.valid_votes, 0) > 0
            then 'missing_party_source_coverage'

            -- Party source exists and its legend component is complete,
            -- but the nominal component is entirely absent.
            when p.party_rows is not null
             and coalesce(p.party_nominal_valid_votes, 0) = 0
             and coalesce(t.nominal_valid_votes, 0) > 0
             and coalesce(p.party_total_legend_valid_votes, 0)
                 = coalesce(t.total_legend_valid_votes, 0)
            then 'missing_party_nominal_coverage'

            -- TSE party totals may include nominal votes converted to
            -- legend totals. This creates a known non-comparable overlap.
            when p.party_rows is not null
             and coalesce(
                    p.party_nominal_converted_to_legend_votes,
                    0
                 ) > 0
             and coalesce(p.party_nominal_valid_votes, 0)
                 = coalesce(t.nominal_valid_votes, 0)
             and (
                    coalesce(p.party_valid_votes, 0)
                    - coalesce(t.valid_votes, 0)
                 )
                 = coalesce(
                     p.party_nominal_converted_to_legend_votes,
                     0
                   )
            then 'converted_vote_overlap'

            else null

        end as gap_reason

    from tally t

    left join party p
      on p.election_year = t.election_year
     and p.election_type = t.election_type
     and p.election_code = t.election_code
     and p.round_number = t.round_number
     and p.uf = t.uf
     and p.municipality_code = t.municipality_code
     and p.zone = t.zone
     and p.office_code = t.office_code
     and p.is_transit_vote is not distinct from t.is_transit_vote

)

select
    *
from comparison
where gap_reason is not null

  );
;
-- created_at: 2026-10-08T11:21:14.919004964+00:00
-- finished_at: 2026-10-08T11:21:14.924024041+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps" rename to "party_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T11:21:14.925603201+00:00
-- finished_at: 2026-10-08T11:21:14.929904798+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps__dbt_tmp" rename to "party_tally_coverage_gaps";
-- created_at: 2026-10-08T11:21:14.931470078+00:00
-- finished_at: 2026-10-08T11:21:14.944946422+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:21:14.949703085+00:00
-- finished_at: 2026-10-08T11:21:14.965154649+00:00
-- elapsed: 15ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."party_performance__dbt_tmp" as (
    with candidate_by_party as (
    select
        f.election_year,
        f.election_type,
        f.election_code,
        f.round_number,
        f.uf,
        f.municipality_code,
        f.zone,
        f.office_code,
        d.party_number,
        f.is_transit_vote,
        sum(f.nominal_valid_votes) as candidate_nominal_valid_votes
    from "tse_analytics"."main"."fact_candidate_votes" f
    inner join "tse_analytics"."main"."dim_candidate" d
      on d.election_year = f.election_year
     and d.election_type = f.election_type
     and d.election_code = f.election_code
     and d.candidate_id = f.candidate_id
    group by 1,2,3,4,5,6,7,8,9,10
)

select
    p.election_year,
    p.election_type,
    p.election_scope,
    p.election_id,
    p.election_code,
    p.round_number,

    p.uf,
    p.municipality_code,
    p.zone,

    p.office_code,
    p.office_scope,

    p.party_number,
    d.party,
    d.party_name,
    d.party_id,

    p.is_transit_vote,

    p.nominal_valid_votes as party_reported_nominal_valid_votes,
    coalesce(c.candidate_nominal_valid_votes, 0) as candidate_nominal_valid_votes,
    p.nominal_valid_votes - coalesce(c.candidate_nominal_valid_votes, 0) as nominal_reconciliation_delta,

    p.legend_valid_votes,
    p.party_valid_votes,

    t.nominal_valid_votes as tally_nominal_valid_votes,
    p.nominal_valid_votes - t.nominal_valid_votes as nominal_tally_delta,

    t.valid_votes,
    t.turnout,
    t.eligible_voters,
    t.abstentions,

    case when t.valid_votes > 0
         then p.party_valid_votes::double / t.valid_votes
    end as vote_share,

    case when t.eligible_voters > 0
         then t.turnout::double / t.eligible_voters
    end as turnout_rate,

    case when t.eligible_voters > 0
         then t.abstentions::double / t.eligible_voters
    end as abstention_rate,

    p.generated_at
from "tse_analytics"."main"."fact_party_votes" p
left join candidate_by_party c
  using (
    election_year,
    election_type,
    election_code,
    round_number,
    uf,
    municipality_code,
    zone,
    office_code,
    party_number,
    is_transit_vote
  )
left join "tse_analytics"."main"."dim_party" d
  on d.election_year = p.election_year
 and d.election_type = p.election_type
 and d.election_code = p.election_code
 and d.party_number = p.party_number
left join "tse_analytics"."main"."fact_tally_munzona" t
  on t.election_year = p.election_year
 and t.election_type = p.election_type
 and t.election_code = p.election_code
 and t.round_number = p.round_number
 and t.uf = p.uf
 and t.municipality_code = p.municipality_code
 and t.zone = p.zone
 and t.office_code = p.office_code
 and t.is_transit_vote is not distinct from p.is_transit_vote
  );
;
-- created_at: 2026-10-08T11:21:14.967340971+00:00
-- finished_at: 2026-10-08T11:21:14.972625012+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance" rename to "party_performance__dbt_backup";
-- created_at: 2026-10-08T11:21:14.975018934+00:00
-- finished_at: 2026-10-08T11:21:14.982047541+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance__dbt_tmp" rename to "party_performance";
-- created_at: 2026-10-08T11:21:14.984589571+00:00
-- finished_at: 2026-10-08T11:21:14.999410686+00:00
-- elapsed: 14ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_performance__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:21:15.015514793+00:00
-- finished_at: 2026-10-08T11:21:15.034504892+00:00
-- elapsed: 18ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."party_tally_reconciliation__dbt_tmp" as (
    with party as (
    select
        election_year,
        election_type,
        election_code,
        round_number,
        uf,
        municipality_code,
        zone,
        office_code,
        is_transit_vote,

        sum(nominal_valid_votes) as party_nominal_valid_votes,
        sum(total_legend_valid_votes) as party_total_legend_valid_votes,
        sum(party_valid_votes) as party_valid_votes
    from "tse_analytics"."main"."fact_party_votes"
    group by 1,2,3,4,5,6,7,8,9
)

select
    t.election_year,
    t.election_type,
    t.election_code,
    t.round_number,
    t.uf,
    t.municipality_code,
    t.zone,
    t.office_code,
    t.is_transit_vote,

    t.nominal_valid_votes as tally_nominal_valid_votes,
    coalesce(p.party_nominal_valid_votes, 0) as party_nominal_valid_votes,
    coalesce(p.party_nominal_valid_votes, 0) - t.nominal_valid_votes
      as nominal_valid_delta,

    t.total_legend_valid_votes as tally_total_legend_valid_votes,
    coalesce(p.party_total_legend_valid_votes, 0) as party_total_legend_valid_votes,
    coalesce(p.party_total_legend_valid_votes, 0) - t.total_legend_valid_votes
      as total_legend_valid_delta,

    t.valid_votes as tally_valid_votes,
    coalesce(p.party_valid_votes, 0) as party_valid_votes,
    coalesce(p.party_valid_votes, 0) - t.valid_votes
      as total_valid_delta

from "tse_analytics"."main"."fact_tally_munzona" t
left join party p
  on p.election_year = t.election_year
 and p.election_type = t.election_type
 and p.election_code = t.election_code
 and p.round_number = t.round_number
 and p.uf = t.uf
 and p.municipality_code = t.municipality_code
 and p.zone = t.zone
 and p.office_code = t.office_code
 and p.is_transit_vote is not distinct from t.is_transit_vote

where not exists (
    select 1
    from "tse_analytics"."main"."party_tally_coverage_gaps" g
    where g.election_year = t.election_year
      and g.election_type = t.election_type
      and g.election_code = t.election_code
      and g.round_number = t.round_number
      and g.uf = t.uf
      and g.municipality_code = t.municipality_code
      and g.zone = t.zone
      and g.office_code = t.office_code
      and g.is_transit_vote = t.is_transit_vote
)
  );
;
-- created_at: 2026-10-08T11:21:15.036591683+00:00
-- finished_at: 2026-10-08T11:21:15.049962749+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation" rename to "party_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T11:21:15.051810779+00:00
-- finished_at: 2026-10-08T11:21:15.055395258+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation__dbt_tmp" rename to "party_tally_reconciliation";
-- created_at: 2026-10-08T11:21:15.057469165+00:00
-- finished_at: 2026-10-08T11:21:15.061200527+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T11:21:14.810030547+00:00
-- finished_at: 2026-10-08T11:21:15.432305060+00:00
-- elapsed: 622ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_electorate_municipality
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from "tse_analytics"."main"."stg_electorate"

where election_year in (2026) and election_type in ('general')

group by 1,2,3,4,5,6
    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:21:15.435648400+00:00
-- finished_at: 2026-10-08T11:21:15.441082296+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'int_electorate_municipality'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:21:15.445677470+00:00
-- finished_at: 2026-10-08T11:21:15.529542024+00:00
-- elapsed: 83ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."int_electorate_municipality" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."int_electorate_municipality" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."int_electorate_municipality" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."int_electorate_municipality" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:21:15.532348512+00:00
-- finished_at: 2026-10-08T11:21:15.683746177+00:00
-- elapsed: 151ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."int_electorate_municipality" add column "electorate__dbt_alter" decimal(38, 0);
    update "tse_analytics"."main"."int_electorate_municipality" set "electorate__dbt_alter" = "electorate";
    alter table "tse_analytics"."main"."int_electorate_municipality" drop column "electorate" cascade;
    alter table "tse_analytics"."main"."int_electorate_municipality" rename column "electorate__dbt_alter" to "electorate"
  ;
-- created_at: 2026-10-08T11:21:15.071056961+00:00
-- finished_at: 2026-10-08T11:21:16.251634378+00:00
-- elapsed: 1.2s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_party_tally_reconciliation_total_legend_valid_delta.7a778b57fe
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_party_tally_reconciliation_total_legend_valid_delta.7a778b57fe", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_legend_valid_delta
from "tse_analytics"."main"."party_tally_reconciliation"
where total_legend_valid_delta is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:21:16.264783270+00:00
-- finished_at: 2026-10-08T11:21:17.239833279+00:00
-- elapsed: 975ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_party_tally_reconciliation_total_valid_delta.26b6bc9108
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_party_tally_reconciliation_total_valid_delta.26b6bc9108", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_valid_delta
from "tse_analytics"."main"."party_tally_reconciliation"
where total_valid_delta is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:21:17.247106436+00:00
-- finished_at: 2026-10-08T11:21:18.506861828+00:00
-- elapsed: 1.3s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_party_valid_votes_reconcile_tally
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_party_valid_votes_reconcile_tally", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  select *
from "tse_analytics"."main"."party_tally_reconciliation"
where nominal_valid_delta <> 0
   or total_legend_valid_delta <> 0
   or total_valid_delta <> 0
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:21:18.522817908+00:00
-- finished_at: 2026-10-08T11:21:19.658404982+00:00
-- elapsed: 1.1s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_party_tally_reconciliation_nominal_valid_delta.4f6b3ef4ef
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_party_tally_reconciliation_nominal_valid_delta.4f6b3ef4ef", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_valid_delta
from "tse_analytics"."main"."party_tally_reconciliation"
where nominal_valid_delta is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:21:15.688589949+00:00
-- finished_at: 2026-10-08T11:22:05.226577683+00:00
-- elapsed: 49.5s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "int_electorate_municipality__dbt_tmp_f5970944_0c7f_4b86_a96b_f5ae08c34c08"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from "tse_analytics"."main"."stg_electorate"

where election_year in (2026) and election_type in ('general')

group by 1,2,3,4,5,6
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."int_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "int_electorate_municipality__dbt_tmp_f5970944_0c7f_4b86_a96b_f5ae08c34c08"
            where (
                
                    "int_electorate_municipality__dbt_tmp_f5970944_0c7f_4b86_a96b_f5ae08c34c08".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "int_electorate_municipality__dbt_tmp_f5970944_0c7f_4b86_a96b_f5ae08c34c08".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "int_electorate_municipality__dbt_tmp_f5970944_0c7f_4b86_a96b_f5ae08c34c08".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "int_electorate_municipality__dbt_tmp_f5970944_0c7f_4b86_a96b_f5ae08c34c08".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."int_electorate_municipality" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate"
        from "int_electorate_municipality__dbt_tmp_f5970944_0c7f_4b86_a96b_f5ae08c34c08"
    )
  ;
-- created_at: 2026-10-08T11:22:05.248639342+00:00
-- finished_at: 2026-10-08T11:22:05.249935944+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_geography
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_geography", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select distinct
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality
from "tse_analytics"."main"."int_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:22:05.248884496+00:00
-- finished_at: 2026-10-08T11:22:05.250527578+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select *
from "tse_analytics"."main"."int_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T11:22:05.252831704+00:00
-- finished_at: 2026-10-08T11:22:05.260213320+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'fact_electorate_municipality'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:22:05.252088358+00:00
-- finished_at: 2026-10-08T11:22:05.260478366+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_geography
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_geography", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'dim_geography'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T11:22:05.265277365+00:00
-- finished_at: 2026-10-08T11:22:05.494434815+00:00
-- elapsed: 229ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_geography
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_geography", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."dim_geography" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."dim_geography" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."dim_geography" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."dim_geography" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:22:05.265661658+00:00
-- finished_at: 2026-10-08T11:22:05.498180869+00:00
-- elapsed: 232ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_electorate_municipality" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."fact_electorate_municipality" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."fact_electorate_municipality" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."fact_electorate_municipality" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T11:22:05.499656847+00:00
-- finished_at: 2026-10-08T11:22:05.643819235+00:00
-- elapsed: 144ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_geography
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_geography", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_geography__dbt_tmp_411f199b_75b4_486a_b522_9e2ee0239631"
  
    as (
      

select distinct
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality
from "tse_analytics"."main"."int_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_geography" as DBT_INCREMENTAL_TARGET
            using "dim_geography__dbt_tmp_411f199b_75b4_486a_b522_9e2ee0239631"
            where (
                
                    "dim_geography__dbt_tmp_411f199b_75b4_486a_b522_9e2ee0239631".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_geography__dbt_tmp_411f199b_75b4_486a_b522_9e2ee0239631".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_geography__dbt_tmp_411f199b_75b4_486a_b522_9e2ee0239631".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "dim_geography__dbt_tmp_411f199b_75b4_486a_b522_9e2ee0239631".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_geography" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality"
        from "dim_geography__dbt_tmp_411f199b_75b4_486a_b522_9e2ee0239631"
    )
  ;
-- created_at: 2026-10-08T11:22:05.500319139+00:00
-- finished_at: 2026-10-08T11:22:05.657997688+00:00
-- elapsed: 157ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."fact_electorate_municipality" add column "electorate__dbt_alter" decimal(38, 0);
    update "tse_analytics"."main"."fact_electorate_municipality" set "electorate__dbt_alter" = "electorate";
    alter table "tse_analytics"."main"."fact_electorate_municipality" drop column "electorate" cascade;
    alter table "tse_analytics"."main"."fact_electorate_municipality" rename column "electorate__dbt_alter" to "electorate"
  ;
-- created_at: 2026-10-08T11:22:05.664030122+00:00
-- finished_at: 2026-10-08T11:22:05.706346092+00:00
-- elapsed: 42ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_electorate_municipality__dbt_tmp_1b3372d2_9bbb_4866_bfe8_841aece075db"
  
    as (
      

select *
from "tse_analytics"."main"."int_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "fact_electorate_municipality__dbt_tmp_1b3372d2_9bbb_4866_bfe8_841aece075db"
            where (
                
                    "fact_electorate_municipality__dbt_tmp_1b3372d2_9bbb_4866_bfe8_841aece075db".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1b3372d2_9bbb_4866_bfe8_841aece075db".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1b3372d2_9bbb_4866_bfe8_841aece075db".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1b3372d2_9bbb_4866_bfe8_841aece075db".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_electorate_municipality" ("election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate")
    (
        select "election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate"
        from "fact_electorate_municipality__dbt_tmp_1b3372d2_9bbb_4866_bfe8_841aece075db"
    )
  ;
-- created_at: 2026-10-08T11:22:05.665047665+00:00
-- finished_at: 2026-10-08T11:22:05.716742876+00:00
-- elapsed: 51ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_geography_election_year__election_type__uf__municipality_code.132ccfbd61
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_dim_geography_election_year__election_type__uf__municipality_code.132ccfbd61", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, uf, municipality_code
    from "tse_analytics"."main"."dim_geography"
    group by election_year, election_type, uf, municipality_code
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.721042248+00:00
-- finished_at: 2026-10-08T11:22:05.721961908+00:00
-- elapsed: 919us
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_electorate_municipality_municipality_code.e47ac5da0f
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_electorate_municipality_municipality_code.e47ac5da0f", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select municipality_code
from "tse_analytics"."main"."fact_electorate_municipality"
where municipality_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.714798758+00:00
-- finished_at: 2026-10-08T11:22:05.734948322+00:00
-- elapsed: 20ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_dim_geography_snapshot_complete
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_dim_geography_snapshot_complete", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with source_rows as (
    select distinct
        election_year, election_type, election_scope,
        uf, municipality_code, municipality
    from "tse_analytics"."main"."int_electorate_municipality"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year, election_type, election_scope,
        uf, municipality_code, municipality
    from "tse_analytics"."main"."dim_geography"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_or_changed_in_target' as issue, * from source_rows
     except
     select 'missing_or_changed_in_target' as issue, * from target_rows)
    union all
    (select 'stale_or_changed_in_target' as issue, * from target_rows
     except
     select 'stale_or_changed_in_target' as issue, * from source_rows)
)
select * from diff
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.728295321+00:00
-- finished_at: 2026-10-08T11:22:05.738695445+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_electorate_municipality_election_year__election_type__uf__municipality_code.fd8fe21802
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_fact_electorate_municipality_election_year__election_type__uf__municipality_code.fd8fe21802", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, uf, municipality_code
    from "tse_analytics"."main"."fact_electorate_municipality"
    group by election_year, election_type, uf, municipality_code
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.742708525+00:00
-- finished_at: 2026-10-08T11:22:05.743951694+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_electorate_municipality_election_type.95b5da7ded
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_electorate_municipality_election_type.95b5da7ded", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."fact_electorate_municipality"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.745570212+00:00
-- finished_at: 2026-10-08T11:22:05.748056428+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_electorate_municipality_uf.e9271f96e4
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_electorate_municipality_uf.e9271f96e4", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select uf
from "tse_analytics"."main"."fact_electorate_municipality"
where uf is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.751637416+00:00
-- finished_at: 2026-10-08T11:22:05.753877944+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_electorate_municipality_election_year.4966e96aad
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_electorate_municipality_election_year.4966e96aad", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."fact_electorate_municipality"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.757286587+00:00
-- finished_at: 2026-10-08T11:22:05.758189049+00:00
-- elapsed: 902us
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_fact_electorate_municipality_election_scope.5a27d36766
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_fact_electorate_municipality_election_scope.5a27d36766", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_scope
from "tse_analytics"."main"."fact_electorate_municipality"
where election_scope is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T11:22:05.758869811+00:00
-- finished_at: 2026-10-08T11:22:05.777847575+00:00
-- elapsed: 18ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.assert_fact_electorate_snapshot_complete
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.assert_fact_electorate_snapshot_complete", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  with source_rows as (
    select
        election_year,
        election_type,
        election_scope,
        uf,
        municipality_code,
        municipality,
        electorate
    from "tse_analytics"."main"."int_electorate_municipality"
    where election_year in (2026) and election_type in ('general')
),
target_rows as (
    select
        election_year,
        election_type,
        election_scope,
        uf,
        municipality_code,
        municipality,
        electorate
    from "tse_analytics"."main"."fact_electorate_municipality"
    where election_year in (2026) and election_type in ('general')
),
diff as (
    (select 'missing_or_changed_in_target' as issue, * from source_rows
     except
     select 'missing_or_changed_in_target' as issue, * from target_rows)
    union all
    (select 'stale_or_changed_in_target' as issue, * from target_rows
     except
     select 'stale_or_changed_in_target' as issue, * from source_rows)
)
select * from diff
  
  
      
    ) dbt_internal_test;
