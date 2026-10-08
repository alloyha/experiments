-- created_at: 2026-10-08T18:10:35.015341481+00:00
-- finished_at: 2026-10-08T18:10:35.025457899+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: list_relations_in_parallel
SELECT table_catalog, table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'main' AND lower(table_catalog) = lower('tse_analytics');
-- created_at: 2026-10-08T18:10:35.304226590+00:00
-- finished_at: 2026-10-08T18:10:35.306902476+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:10:35.308386848+00:00
-- finished_at: 2026-10-08T18:10:35.311305614+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:10:35.312078205+00:00
-- finished_at: 2026-10-08T18:10:35.312720474+00:00
-- elapsed: 642us
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    
        create schema if not exists "tse_analytics"."main"
    ;
-- created_at: 2026-10-08T18:10:35.322434496+00:00
-- finished_at: 2026-10-08T18:10:35.355737114+00:00
-- elapsed: 33ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
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
-- created_at: 2026-10-08T18:10:35.323254607+00:00
-- finished_at: 2026-10-08T18:10:35.355952920+00:00
-- elapsed: 32ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
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
-- created_at: 2026-10-08T18:10:35.362999424+00:00
-- finished_at: 2026-10-08T18:10:35.374175761+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
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
-- created_at: 2026-10-08T18:10:35.362608332+00:00
-- finished_at: 2026-10-08T18:10:35.375462646+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
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
-- created_at: 2026-10-08T18:10:35.383763910+00:00
-- finished_at: 2026-10-08T18:10:35.406618873+00:00
-- elapsed: 22ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
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
-- created_at: 2026-10-08T18:10:35.383979153+00:00
-- finished_at: 2026-10-08T18:10:35.407146757+00:00
-- elapsed: 23ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
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
-- created_at: 2026-10-08T18:10:35.434360164+00:00
-- finished_at: 2026-10-08T18:10:35.458318802+00:00
-- elapsed: 23ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "seed.tse_analytics.election_calendar", "profile_name": "tse_analytics", "target_name": "dev"} */
truncate table "tse_analytics"."main"."election_calendar";
-- created_at: 2026-10-08T18:10:35.437637112+00:00
-- finished_at: 2026-10-08T18:10:35.499887040+00:00
-- elapsed: 62ms
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
-- created_at: 2026-10-08T18:10:35.505810037+00:00
-- finished_at: 2026-10-08T18:10:35.512085328+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes" rename to "fact_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T18:10:35.476156392+00:00
-- finished_at: 2026-10-08T18:10:35.520896252+00:00
-- elapsed: 44ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: add_query adapter call

          COPY "tse_analytics"."main"."election_calendar" FROM '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/seeds/election_calendar.csv' (FORMAT CSV, HEADER TRUE, DELIMITER ',')
        ;
-- created_at: 2026-10-08T18:10:35.516731430+00:00
-- finished_at: 2026-10-08T18:10:35.527021811+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes__dbt_tmp" rename to "fact_candidate_votes";
-- created_at: 2026-10-08T18:10:35.536664679+00:00
-- finished_at: 2026-10-08T18:10:35.545591295+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."fact_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:10:35.542593558+00:00
-- finished_at: 2026-10-08T18:10:35.559879606+00:00
-- elapsed: 17ms
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
-- created_at: 2026-10-08T18:10:35.573972405+00:00
-- finished_at: 2026-10-08T18:10:35.966178958+00:00
-- elapsed: 392ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."bronze_candidate_assets__dbt_tmp" as (
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
-- created_at: 2026-10-08T18:10:35.974368951+00:00
-- finished_at: 2026-10-08T18:10:35.983638834+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_assets" rename to "bronze_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T18:10:35.991193221+00:00
-- finished_at: 2026-10-08T18:10:36.000776078+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_assets__dbt_tmp" rename to "bronze_candidate_assets";
-- created_at: 2026-10-08T18:10:36.009866176+00:00
-- finished_at: 2026-10-08T18:10:36.020009977+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:10:36.044218250+00:00
-- finished_at: 2026-10-08T18:10:36.304104877+00:00
-- elapsed: 259ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."bronze_candidates__dbt_tmp" as (
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
  );
;
-- created_at: 2026-10-08T18:10:36.310622624+00:00
-- finished_at: 2026-10-08T18:10:36.319115884+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidates" rename to "bronze_candidates__dbt_backup";
-- created_at: 2026-10-08T18:10:36.325118973+00:00
-- finished_at: 2026-10-08T18:10:36.334969272+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidates__dbt_tmp" rename to "bronze_candidates";
-- created_at: 2026-10-08T18:10:36.343354264+00:00
-- finished_at: 2026-10-08T18:10:36.350892205+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_candidates__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:10:36.370801883+00:00
-- finished_at: 2026-10-08T18:10:36.473181819+00:00
-- elapsed: 102ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

        delete from "tse_analytics"."main"."bronze_party_votes_raw"
        where election_year in (
            
                2026
            
        )
        and election_type in (
            
                'general'
                
            
        )
      ;
-- created_at: 2026-10-08T18:10:35.560666059+00:00
-- finished_at: 2026-10-08T18:10:36.546326761+00:00
-- elapsed: 985ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */

  
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
;
-- created_at: 2026-10-08T18:10:36.552750619+00:00
-- finished_at: 2026-10-08T18:10:36.562836828+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_electorate" rename to "bronze_electorate__dbt_backup";
-- created_at: 2026-10-08T18:10:36.569283528+00:00
-- finished_at: 2026-10-08T18:10:36.579752467+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_electorate__dbt_tmp" rename to "bronze_electorate";
-- created_at: 2026-10-08T18:10:36.587303460+00:00
-- finished_at: 2026-10-08T18:10:36.596013189+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_electorate__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:10:36.613069307+00:00
-- finished_at: 2026-10-08T18:10:37.417678589+00:00
-- elapsed: 804ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."bronze_candidate_votes_raw__dbt_tmp" as (
    

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

  );
;
-- created_at: 2026-10-08T18:10:37.423625072+00:00
-- finished_at: 2026-10-08T18:10:37.437902908+00:00
-- elapsed: 14ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_votes_raw" rename to "bronze_candidate_votes_raw__dbt_backup";
-- created_at: 2026-10-08T18:10:37.445222039+00:00
-- finished_at: 2026-10-08T18:10:37.455592994+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_votes_raw__dbt_tmp" rename to "bronze_candidate_votes_raw";
-- created_at: 2026-10-08T18:10:37.463193739+00:00
-- finished_at: 2026-10-08T18:10:37.483599598+00:00
-- elapsed: 20ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_candidate_votes_raw__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:10:36.480003670+00:00
-- finished_at: 2026-10-08T18:10:37.573612231+00:00
-- elapsed: 1.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
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

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:10:37.577995293+00:00
-- finished_at: 2026-10-08T18:10:37.628370795+00:00
-- elapsed: 50ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'bronze_party_votes_raw'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T18:10:37.650212855+00:00
-- finished_at: 2026-10-08T18:10:37.713972549+00:00
-- elapsed: 63ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T18:10:37.721967670+00:00
-- finished_at: 2026-10-08T18:10:37.777813961+00:00
-- elapsed: 55ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T18:10:37.783428285+00:00
-- finished_at: 2026-10-08T18:10:37.830658974+00:00
-- elapsed: 47ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T18:10:37.502281924+00:00
-- finished_at: 2026-10-08T18:10:37.831412740+00:00
-- elapsed: 329ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
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
    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:10:37.841834308+00:00
-- finished_at: 2026-10-08T18:10:37.864706723+00:00
-- elapsed: 22ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'bronze_tally_munzona'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T18:10:37.839636655+00:00
-- finished_at: 2026-10-08T18:10:37.894470586+00:00
-- elapsed: 54ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "zone" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T18:10:37.901315792+00:00
-- finished_at: 2026-10-08T18:10:37.972659179+00:00
-- elapsed: 71ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T18:10:37.879850836+00:00
-- finished_at: 2026-10-08T18:10:38.007059340+00:00
-- elapsed: 127ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."bronze_tally_munzona" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T18:10:37.980469741+00:00
-- finished_at: 2026-10-08T18:10:38.066896870+00:00
-- elapsed: 86ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T18:10:38.018930062+00:00
-- finished_at: 2026-10-08T18:10:38.108142515+00:00
-- elapsed: 89ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."bronze_tally_munzona" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T18:10:38.072526329+00:00
-- finished_at: 2026-10-08T18:10:38.153318529+00:00
-- elapsed: 80ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "nominal_converted_to_legend_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "nominal_converted_to_legend_votes__dbt_alter" = "nominal_converted_to_legend_votes";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "nominal_converted_to_legend_votes" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "nominal_converted_to_legend_votes__dbt_alter" to "nominal_converted_to_legend_votes"
  ;
-- created_at: 2026-10-08T18:10:38.120131921+00:00
-- finished_at: 2026-10-08T18:10:38.440754967+00:00
-- elapsed: 320ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."bronze_tally_munzona" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T18:10:38.161042606+00:00
-- finished_at: 2026-10-08T18:10:38.668825323+00:00
-- elapsed: 507ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T18:10:38.673411480+00:00
-- finished_at: 2026-10-08T18:10:39.875998589+00:00
-- elapsed: 1.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T18:10:38.450211124+00:00
-- finished_at: 2026-10-08T18:10:40.044591055+00:00
-- elapsed: 1.6s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."bronze_tally_munzona" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "zone" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T18:10:39.879931280+00:00
-- finished_at: 2026-10-08T18:10:40.118799375+00:00
-- elapsed: 238ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T18:10:40.051686843+00:00
-- finished_at: 2026-10-08T18:10:40.128417979+00:00
-- elapsed: 76ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."bronze_tally_munzona" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T18:10:40.121485420+00:00
-- finished_at: 2026-10-08T18:10:40.225729254+00:00
-- elapsed: 104ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_party_votes_raw" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_party_votes_raw" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."bronze_party_votes_raw" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."bronze_party_votes_raw" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T18:10:40.132784730+00:00
-- finished_at: 2026-10-08T18:10:40.254668382+00:00
-- elapsed: 121ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "eligible_voters__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "eligible_voters__dbt_alter" = "eligible_voters";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "eligible_voters" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "eligible_voters__dbt_alter" to "eligible_voters"
  ;
-- created_at: 2026-10-08T18:10:40.258911067+00:00
-- finished_at: 2026-10-08T18:10:40.303680149+00:00
-- elapsed: 44ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "main_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "main_sections__dbt_alter" = "main_sections";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "main_sections" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "main_sections__dbt_alter" to "main_sections"
  ;
-- created_at: 2026-10-08T18:10:40.308329804+00:00
-- finished_at: 2026-10-08T18:10:40.359138566+00:00
-- elapsed: 50ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "aggregated_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "aggregated_sections__dbt_alter" = "aggregated_sections";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "aggregated_sections" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "aggregated_sections__dbt_alter" to "aggregated_sections"
  ;
-- created_at: 2026-10-08T18:10:40.363443889+00:00
-- finished_at: 2026-10-08T18:10:40.407821524+00:00
-- elapsed: 44ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "uninstalled_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "uninstalled_sections__dbt_alter" = "uninstalled_sections";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "uninstalled_sections" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "uninstalled_sections__dbt_alter" to "uninstalled_sections"
  ;
-- created_at: 2026-10-08T18:10:40.413127931+00:00
-- finished_at: 2026-10-08T18:10:40.470710592+00:00
-- elapsed: 57ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "total_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "total_sections__dbt_alter" = "total_sections";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "total_sections" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "total_sections__dbt_alter" to "total_sections"
  ;
-- created_at: 2026-10-08T18:10:40.475228812+00:00
-- finished_at: 2026-10-08T18:10:40.523418626+00:00
-- elapsed: 48ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "turnout__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "turnout__dbt_alter" = "turnout";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "turnout" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "turnout__dbt_alter" to "turnout"
  ;
-- created_at: 2026-10-08T18:10:40.528671778+00:00
-- finished_at: 2026-10-08T18:10:40.578176593+00:00
-- elapsed: 49ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "voters_uninstalled_sections__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "voters_uninstalled_sections__dbt_alter" = "voters_uninstalled_sections";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "voters_uninstalled_sections" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "voters_uninstalled_sections__dbt_alter" to "voters_uninstalled_sections"
  ;
-- created_at: 2026-10-08T18:10:40.582702374+00:00
-- finished_at: 2026-10-08T18:10:40.647489943+00:00
-- elapsed: 64ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "abstentions__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "abstentions__dbt_alter" = "abstentions";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "abstentions" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "abstentions__dbt_alter" to "abstentions"
  ;
-- created_at: 2026-10-08T18:10:40.655045362+00:00
-- finished_at: 2026-10-08T18:10:40.730563016+00:00
-- elapsed: 75ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "total_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "total_votes__dbt_alter" = "total_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "total_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "total_votes__dbt_alter" to "total_votes"
  ;
-- created_at: 2026-10-08T18:10:40.736686380+00:00
-- finished_at: 2026-10-08T18:10:40.841817467+00:00
-- elapsed: 105ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "competing_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "competing_votes__dbt_alter" = "competing_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "competing_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "competing_votes__dbt_alter" to "competing_votes"
  ;
-- created_at: 2026-10-08T18:10:40.847324627+00:00
-- finished_at: 2026-10-08T18:10:40.947448018+00:00
-- elapsed: 100ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "valid_votes__dbt_alter" = "valid_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "valid_votes__dbt_alter" to "valid_votes"
  ;
-- created_at: 2026-10-08T18:10:40.953188619+00:00
-- finished_at: 2026-10-08T18:10:41.027967381+00:00
-- elapsed: 74ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T18:10:41.033927433+00:00
-- finished_at: 2026-10-08T18:10:41.080851337+00:00
-- elapsed: 46ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T18:10:41.087472072+00:00
-- finished_at: 2026-10-08T18:10:41.142165983+00:00
-- elapsed: 54ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T18:10:41.149880828+00:00
-- finished_at: 2026-10-08T18:10:41.223798335+00:00
-- elapsed: 73ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "nominal_converted_to_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "nominal_converted_to_legend_valid_votes__dbt_alter" = "nominal_converted_to_legend_valid_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "nominal_converted_to_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "nominal_converted_to_legend_valid_votes__dbt_alter" to "nominal_converted_to_legend_valid_votes"
  ;
-- created_at: 2026-10-08T18:10:41.230278660+00:00
-- finished_at: 2026-10-08T18:10:41.299338816+00:00
-- elapsed: 69ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "annulled_votes__dbt_alter" = "annulled_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "annulled_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "annulled_votes__dbt_alter" to "annulled_votes"
  ;
-- created_at: 2026-10-08T18:10:41.305680896+00:00
-- finished_at: 2026-10-08T18:10:41.368566337+00:00
-- elapsed: 62ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "nominal_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "nominal_annulled_votes__dbt_alter" = "nominal_annulled_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "nominal_annulled_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "nominal_annulled_votes__dbt_alter" to "nominal_annulled_votes"
  ;
-- created_at: 2026-10-08T18:10:41.376114679+00:00
-- finished_at: 2026-10-08T18:10:41.430915905+00:00
-- elapsed: 54ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "legend_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "legend_annulled_votes__dbt_alter" = "legend_annulled_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "legend_annulled_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "legend_annulled_votes__dbt_alter" to "legend_annulled_votes"
  ;
-- created_at: 2026-10-08T18:10:41.439259115+00:00
-- finished_at: 2026-10-08T18:10:41.493695837+00:00
-- elapsed: 54ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "annulled_subjudice_votes__dbt_alter" = "annulled_subjudice_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "annulled_subjudice_votes__dbt_alter" to "annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T18:10:41.500571370+00:00
-- finished_at: 2026-10-08T18:10:41.570919597+00:00
-- elapsed: 70ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T18:10:41.577951009+00:00
-- finished_at: 2026-10-08T18:10:41.647517975+00:00
-- elapsed: 69ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T18:10:41.655621231+00:00
-- finished_at: 2026-10-08T18:10:41.734509512+00:00
-- elapsed: 78ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "blank_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "blank_votes__dbt_alter" = "blank_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "blank_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "blank_votes__dbt_alter" to "blank_votes"
  ;
-- created_at: 2026-10-08T18:10:41.743513802+00:00
-- finished_at: 2026-10-08T18:10:41.802396821+00:00
-- elapsed: 58ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "total_null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "total_null_votes__dbt_alter" = "total_null_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "total_null_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "total_null_votes__dbt_alter" to "total_null_votes"
  ;
-- created_at: 2026-10-08T18:10:41.810230304+00:00
-- finished_at: 2026-10-08T18:10:41.870926570+00:00
-- elapsed: 60ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "null_votes__dbt_alter" = "null_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "null_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "null_votes__dbt_alter" to "null_votes"
  ;
-- created_at: 2026-10-08T18:10:41.882066853+00:00
-- finished_at: 2026-10-08T18:10:41.955939760+00:00
-- elapsed: 73ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "technical_null_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "technical_null_votes__dbt_alter" = "technical_null_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "technical_null_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "technical_null_votes__dbt_alter" to "technical_null_votes"
  ;
-- created_at: 2026-10-08T18:10:41.964636326+00:00
-- finished_at: 2026-10-08T18:10:42.051338908+00:00
-- elapsed: 86ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "separately_counted_annulled_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."bronze_tally_munzona" set "separately_counted_annulled_votes__dbt_alter" = "separately_counted_annulled_votes";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "separately_counted_annulled_votes" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "separately_counted_annulled_votes__dbt_alter" to "separately_counted_annulled_votes"
  ;
-- created_at: 2026-10-08T18:10:42.063939226+00:00
-- finished_at: 2026-10-08T18:10:42.144790813+00:00
-- elapsed: 80ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."bronze_tally_munzona" add column "last_totalization_at__dbt_alter" datetime;
    update "tse_analytics"."main"."bronze_tally_munzona" set "last_totalization_at__dbt_alter" = "last_totalization_at";
    alter table "tse_analytics"."main"."bronze_tally_munzona" drop column "last_totalization_at" cascade;
    alter table "tse_analytics"."main"."bronze_tally_munzona" rename column "last_totalization_at__dbt_alter" to "last_totalization_at"
  ;
-- created_at: 2026-10-08T18:10:42.188933267+00:00
-- finished_at: 2026-10-08T18:10:44.660913069+00:00
-- elapsed: 2.5s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a"
  
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
            using "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a"
            where (
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."bronze_tally_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file"
        from "bronze_tally_munzona__dbt_tmp_f4037138_9ef0_4f4a_8a83_d8eefd2f427a"
    )
  ;
-- created_at: 2026-10-08T18:10:44.699437615+00:00
-- finished_at: 2026-10-08T18:10:44.703527229+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T18:10:44.722151390+00:00
-- finished_at: 2026-10-08T18:10:44.727009002+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T18:10:44.747807182+00:00
-- finished_at: 2026-10-08T18:10:44.752344892+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T18:10:44.782236303+00:00
-- finished_at: 2026-10-08T18:10:44.792591438+00:00
-- elapsed: 10ms
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
-- created_at: 2026-10-08T18:10:44.832869686+00:00
-- finished_at: 2026-10-08T18:10:44.839981374+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T18:10:44.868454001+00:00
-- finished_at: 2026-10-08T18:10:44.881933116+00:00
-- elapsed: 13ms
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
-- created_at: 2026-10-08T18:10:44.926917974+00:00
-- finished_at: 2026-10-08T18:10:45.099379485+00:00
-- elapsed: 172ms
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
-- created_at: 2026-10-08T18:10:45.126739355+00:00
-- finished_at: 2026-10-08T18:10:45.241151418+00:00
-- elapsed: 114ms
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
-- created_at: 2026-10-08T18:10:45.260256811+00:00
-- finished_at: 2026-10-08T18:10:45.369455639+00:00
-- elapsed: 109ms
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
-- created_at: 2026-10-08T18:10:45.390025529+00:00
-- finished_at: 2026-10-08T18:10:45.486945611+00:00
-- elapsed: 96ms
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
-- created_at: 2026-10-08T18:10:45.513546386+00:00
-- finished_at: 2026-10-08T18:10:45.622596543+00:00
-- elapsed: 109ms
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
-- created_at: 2026-10-08T18:10:45.639258289+00:00
-- finished_at: 2026-10-08T18:10:45.761682091+00:00
-- elapsed: 122ms
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
-- created_at: 2026-10-08T18:10:45.774277134+00:00
-- finished_at: 2026-10-08T18:10:45.865769394+00:00
-- elapsed: 91ms
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
-- created_at: 2026-10-08T18:10:45.882544985+00:00
-- finished_at: 2026-10-08T18:10:45.947330037+00:00
-- elapsed: 64ms
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
-- created_at: 2026-10-08T18:10:40.234135908+00:00
-- finished_at: 2026-10-08T18:11:00.926887052+00:00
-- elapsed: 20.7s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "bronze_party_votes_raw__dbt_tmp_7da37efd_6ad4_4071_b5c8_1f3927fa8fb6"
  
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
        from "bronze_party_votes_raw__dbt_tmp_7da37efd_6ad4_4071_b5c8_1f3927fa8fb6"
    )


  ;
-- created_at: 2026-10-08T18:11:01.360508379+00:00
-- finished_at: 2026-10-08T18:11:15.439747078+00:00
-- elapsed: 14.1s
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
-- created_at: 2026-10-08T18:11:15.451663849+00:00
-- finished_at: 2026-10-08T18:11:16.407937685+00:00
-- elapsed: 956ms
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
-- created_at: 2026-10-08T18:11:16.446972735+00:00
-- finished_at: 2026-10-08T18:11:17.469345218+00:00
-- elapsed: 1.0s
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidate_assets_election_year.858bb93deb
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidate_assets_election_year.858bb93deb", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."bronze_candidate_assets"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:17.480789339+00:00
-- finished_at: 2026-10-08T18:11:18.248537810+00:00
-- elapsed: 767ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidate_assets_candidate_id.924275fa98
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidate_assets_candidate_id.924275fa98", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_id
from "tse_analytics"."main"."bronze_candidate_assets"
where candidate_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:18.259719125+00:00
-- finished_at: 2026-10-08T18:11:19.005316287+00:00
-- elapsed: 745ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidate_assets_election_type.b76d8a86d3
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidate_assets_election_type.b76d8a86d3", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."bronze_candidate_assets"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:19.016450712+00:00
-- finished_at: 2026-10-08T18:11:19.700311423+00:00
-- elapsed: 683ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidates_election_type.1eb9b3e5ac
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidates_election_type.1eb9b3e5ac", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."bronze_candidates"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:19.713718796+00:00
-- finished_at: 2026-10-08T18:11:20.217810802+00:00
-- elapsed: 504ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidates_office_scope.197db8c6ce
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidates_office_scope.197db8c6ce", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select office_scope
from "tse_analytics"."main"."bronze_candidates"
where office_scope is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:20.229289161+00:00
-- finished_at: 2026-10-08T18:11:20.842995352+00:00
-- elapsed: 613ms
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
from "tse_analytics"."main"."bronze_candidates"
where election_scope <> case
    when election_type = 'general' then 'federal_state'
    when election_type = 'municipal' then 'municipal'
end
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:20.861117036+00:00
-- finished_at: 2026-10-08T18:11:21.538134541+00:00
-- elapsed: 677ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_bronze_candidates_election_scope__federal_state__municipal.9a5ced23eb
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_bronze_candidates_election_scope__federal_state__municipal.9a5ced23eb", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."bronze_candidates"
    group by election_scope

)

select *
from all_values
where value_field not in (
    'federal_state','municipal'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:21.548733395+00:00
-- finished_at: 2026-10-08T18:11:22.223185863+00:00
-- elapsed: 674ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidates_election_year.8a99824dac
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidates_election_year.8a99824dac", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."bronze_candidates"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:22.231731537+00:00
-- finished_at: 2026-10-08T18:11:22.889110629+00:00
-- elapsed: 657ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidates_election_scope.c20470e33a
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidates_election_scope.c20470e33a", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_scope
from "tse_analytics"."main"."bronze_candidates"
where election_scope is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:22.901789914+00:00
-- finished_at: 2026-10-08T18:11:23.843581060+00:00
-- elapsed: 941ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidates_candidate_name.e2ed96fd17
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidates_candidate_name.e2ed96fd17", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_name
from "tse_analytics"."main"."bronze_candidates"
where candidate_name is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:23.855140615+00:00
-- finished_at: 2026-10-08T18:11:24.810060953+00:00
-- elapsed: 954ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_bronze_candidates_office_scope__federal__state__municipal__other.1593038c12
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_bronze_candidates_office_scope__federal__state__municipal__other.1593038c12", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        office_scope as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."bronze_candidates"
    group by office_scope

)

select *
from all_values
where value_field not in (
    'federal','state','municipal','other'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:24.824705824+00:00
-- finished_at: 2026-10-08T18:11:25.423156807+00:00
-- elapsed: 598ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_candidates_candidate_id.39929d3582
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_candidates_candidate_id.39929d3582", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select candidate_id
from "tse_analytics"."main"."bronze_candidates"
where candidate_id is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:25.435296013+00:00
-- finished_at: 2026-10-08T18:11:26.220057395+00:00
-- elapsed: 784ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_bronze_candidates_election_year__election_type__election_code__candidate_id.72aa2ca911
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_bronze_candidates_election_year__election_type__election_code__candidate_id.72aa2ca911", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, candidate_id
    from "tse_analytics"."main"."bronze_candidates"
    group by election_year, election_type, election_code, candidate_id
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:26.232396033+00:00
-- finished_at: 2026-10-08T18:11:27.313079352+00:00
-- elapsed: 1.1s
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
from "tse_analytics"."main"."bronze_candidates"
where
    (election_type = 'municipal' and office_scope <> 'municipal')
    or
    (election_type = 'general' and office_scope = 'municipal')
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:11:27.323467940+00:00
-- finished_at: 2026-10-08T18:11:27.873353099+00:00
-- elapsed: 549ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.accepted_values_bronze_candidates_election_type__general__municipal.e791b1ecd3
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.accepted_values_bronze_candidates_election_type__general__municipal.e791b1ecd3", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    

with all_values as (

    select
        election_type as value_field,
        count(*) as n_records

    from "tse_analytics"."main"."bronze_candidates"
    group by election_type

)

select *
from all_values
where value_field not in (
    'general','municipal'
)



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:10:45.964255817+00:00
-- finished_at: 2026-10-08T18:12:15.144761825+00:00
-- elapsed: 1m 29s
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
-- created_at: 2026-10-08T18:12:15.167990416+00:00
-- finished_at: 2026-10-08T18:12:15.642875615+00:00
-- elapsed: 474ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."silver_candidate_votes_munzona__dbt_tmp" as (
    

select *
from "tse_analytics"."main"."bronze_candidate_votes_raw"
  );
;
-- created_at: 2026-10-08T18:12:15.649652024+00:00
-- finished_at: 2026-10-08T18:12:15.656205+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes_munzona" rename to "silver_candidate_votes_munzona__dbt_backup";
-- created_at: 2026-10-08T18:12:15.663499564+00:00
-- finished_at: 2026-10-08T18:12:15.670023318+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes_munzona__dbt_tmp" rename to "silver_candidate_votes_munzona";
-- created_at: 2026-10-08T18:12:15.677832889+00:00
-- finished_at: 2026-10-08T18:12:15.685625725+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."silver_candidate_votes_munzona__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:12:15.698424809+00:00
-- finished_at: 2026-10-08T18:12:15.700625521+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_municipality_code.1ace8c4db2
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_municipality_code.1ace8c4db2", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select municipality_code
from "tse_analytics"."main"."bronze_tally_munzona"
where municipality_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.706323074+00:00
-- finished_at: 2026-10-08T18:12:15.708981732+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_turnout.30d5511ea0
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_turnout.30d5511ea0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select turnout
from "tse_analytics"."main"."bronze_tally_munzona"
where turnout is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.715399790+00:00
-- finished_at: 2026-10-08T18:12:15.722299467+00:00
-- elapsed: 6ms
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
from "tse_analytics"."main"."bronze_tally_munzona"
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.728391832+00:00
-- finished_at: 2026-10-08T18:12:15.743368961+00:00
-- elapsed: 14ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_bronze_tally_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.7257dffa85
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_bronze_tally_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__is_transit_vote.7257dffa85", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    from "tse_analytics"."main"."bronze_tally_munzona"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.749442585+00:00
-- finished_at: 2026-10-08T18:12:15.751228321+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_abstentions.042a8ed6c9
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_abstentions.042a8ed6c9", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select abstentions
from "tse_analytics"."main"."bronze_tally_munzona"
where abstentions is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.758577213+00:00
-- finished_at: 2026-10-08T18:12:15.760853538+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_round_number.33899e6566
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_round_number.33899e6566", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select round_number
from "tse_analytics"."main"."bronze_tally_munzona"
where round_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.768239502+00:00
-- finished_at: 2026-10-08T18:12:15.769851726+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_election_code.f5dac92844
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_election_code.f5dac92844", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_code
from "tse_analytics"."main"."bronze_tally_munzona"
where election_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.775957022+00:00
-- finished_at: 2026-10-08T18:12:15.777338870+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_office_code.f00d8c30dc
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_office_code.f00d8c30dc", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select office_code
from "tse_analytics"."main"."bronze_tally_munzona"
where office_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.782652882+00:00
-- finished_at: 2026-10-08T18:12:15.784222911+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_zone.2c13d6d2a0
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_zone.2c13d6d2a0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select zone
from "tse_analytics"."main"."bronze_tally_munzona"
where zone is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.790153593+00:00
-- finished_at: 2026-10-08T18:12:15.792081961+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_eligible_voters.b432787c71
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_eligible_voters.b432787c71", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select eligible_voters
from "tse_analytics"."main"."bronze_tally_munzona"
where eligible_voters is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.798626818+00:00
-- finished_at: 2026-10-08T18:12:15.800887023+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_bronze_tally_munzona_election_year.c6a4ad850f
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_bronze_tally_munzona_election_year.c6a4ad850f", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."bronze_tally_munzona"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.811628874+00:00
-- finished_at: 2026-10-08T18:12:15.884482874+00:00
-- elapsed: 72ms
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
from "tse_analytics"."main"."bronze_party_votes_raw"
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:15.891735707+00:00
-- finished_at: 2026-10-08T18:12:16.699057692+00:00
-- elapsed: 807ms
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

    from "tse_analytics"."main"."bronze_party_votes_raw"

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
-- created_at: 2026-10-08T18:12:16.711169860+00:00
-- finished_at: 2026-10-08T18:12:16.900768089+00:00
-- elapsed: 189ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."silver_candidate_assets__dbt_tmp" as (
    select
    election_year,
    election_type,
    election_code,
    candidate_id,
    sum(asset_value) as declared_assets_value,
    count(*) as declared_assets_count
from "tse_analytics"."main"."bronze_candidate_assets"
group by 1,2,3,4
  );
;
-- created_at: 2026-10-08T18:12:16.904219142+00:00
-- finished_at: 2026-10-08T18:12:16.910152257+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_assets" rename to "silver_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T18:12:16.914157002+00:00
-- finished_at: 2026-10-08T18:12:16.920313495+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_assets__dbt_tmp" rename to "silver_candidate_assets";
-- created_at: 2026-10-08T18:12:16.925729948+00:00
-- finished_at: 2026-10-08T18:12:16.932147808+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."silver_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:12:16.943299196+00:00
-- finished_at: 2026-10-08T18:12:16.962785230+00:00
-- elapsed: 19ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."silver_candidate_result_coverage__dbt_tmp" as (
    

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
-- created_at: 2026-10-08T18:12:16.966589870+00:00
-- finished_at: 2026-10-08T18:12:16.972839951+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_result_coverage" rename to "silver_candidate_result_coverage__dbt_backup";
-- created_at: 2026-10-08T18:12:16.977262439+00:00
-- finished_at: 2026-10-08T18:12:16.985356997+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_result_coverage__dbt_tmp" rename to "silver_candidate_result_coverage";
-- created_at: 2026-10-08T18:12:16.990128678+00:00
-- finished_at: 2026-10-08T18:12:16.997631392+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."silver_candidate_result_coverage__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:12:17.006400691+00:00
-- finished_at: 2026-10-08T18:12:17.253946166+00:00
-- elapsed: 247ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."silver_candidate_votes__dbt_tmp" as (
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
from "tse_analytics"."main"."silver_candidate_votes_munzona"
  );
;
-- created_at: 2026-10-08T18:12:17.259115269+00:00
-- finished_at: 2026-10-08T18:12:17.266148863+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes" rename to "silver_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T18:12:17.271564609+00:00
-- finished_at: 2026-10-08T18:12:17.279022985+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes__dbt_tmp" rename to "silver_candidate_votes";
-- created_at: 2026-10-08T18:12:17.285184840+00:00
-- finished_at: 2026-10-08T18:12:17.291515794+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."silver_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:12:17.306674752+00:00
-- finished_at: 2026-10-08T18:12:17.323102851+00:00
-- elapsed: 16ms
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
from "tse_analytics"."main"."bronze_tally_munzona"

where election_year in (2026) and election_type in ('general')

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:12:17.328362939+00:00
-- finished_at: 2026-10-08T18:12:17.349294995+00:00
-- elapsed: 20ms
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
-- created_at: 2026-10-08T18:12:17.368816988+00:00
-- finished_at: 2026-10-08T18:12:17.473805909+00:00
-- elapsed: 104ms
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
-- created_at: 2026-10-08T18:12:17.479094841+00:00
-- finished_at: 2026-10-08T18:12:17.584450068+00:00
-- elapsed: 105ms
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
-- created_at: 2026-10-08T18:12:17.589593280+00:00
-- finished_at: 2026-10-08T18:12:17.678052024+00:00
-- elapsed: 88ms
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
-- created_at: 2026-10-08T18:12:17.682733272+00:00
-- finished_at: 2026-10-08T18:12:17.759523521+00:00
-- elapsed: 76ms
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
-- created_at: 2026-10-08T18:12:17.765480843+00:00
-- finished_at: 2026-10-08T18:12:17.839268912+00:00
-- elapsed: 73ms
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
-- created_at: 2026-10-08T18:12:17.843605644+00:00
-- finished_at: 2026-10-08T18:12:17.921467369+00:00
-- elapsed: 77ms
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
-- created_at: 2026-10-08T18:12:17.925535358+00:00
-- finished_at: 2026-10-08T18:12:18.005075813+00:00
-- elapsed: 79ms
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
-- created_at: 2026-10-08T18:12:18.008931172+00:00
-- finished_at: 2026-10-08T18:12:18.098352151+00:00
-- elapsed: 89ms
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
-- created_at: 2026-10-08T18:12:18.102489542+00:00
-- finished_at: 2026-10-08T18:12:18.173679342+00:00
-- elapsed: 71ms
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
-- created_at: 2026-10-08T18:12:18.178370724+00:00
-- finished_at: 2026-10-08T18:12:18.264313228+00:00
-- elapsed: 85ms
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
-- created_at: 2026-10-08T18:12:18.268430162+00:00
-- finished_at: 2026-10-08T18:12:18.351319591+00:00
-- elapsed: 82ms
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
-- created_at: 2026-10-08T18:12:18.355660866+00:00
-- finished_at: 2026-10-08T18:12:18.492262992+00:00
-- elapsed: 136ms
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
-- created_at: 2026-10-08T18:12:18.495746584+00:00
-- finished_at: 2026-10-08T18:12:18.618135870+00:00
-- elapsed: 122ms
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
-- created_at: 2026-10-08T18:12:18.623932051+00:00
-- finished_at: 2026-10-08T18:12:18.752653588+00:00
-- elapsed: 128ms
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
-- created_at: 2026-10-08T18:12:18.758869433+00:00
-- finished_at: 2026-10-08T18:12:20.347911891+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T18:12:20.353560109+00:00
-- finished_at: 2026-10-08T18:12:23.561051631+00:00
-- elapsed: 3.2s
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
-- created_at: 2026-10-08T18:12:23.563015399+00:00
-- finished_at: 2026-10-08T18:12:26.491259288+00:00
-- elapsed: 2.9s
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
-- created_at: 2026-10-08T18:12:26.493975007+00:00
-- finished_at: 2026-10-08T18:12:26.651183493+00:00
-- elapsed: 157ms
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
-- created_at: 2026-10-08T18:12:26.653918684+00:00
-- finished_at: 2026-10-08T18:12:26.774506257+00:00
-- elapsed: 120ms
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
-- created_at: 2026-10-08T18:12:26.776450913+00:00
-- finished_at: 2026-10-08T18:12:26.920074363+00:00
-- elapsed: 143ms
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
-- created_at: 2026-10-08T18:12:26.922964955+00:00
-- finished_at: 2026-10-08T18:12:27.055247712+00:00
-- elapsed: 132ms
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
-- created_at: 2026-10-08T18:12:27.057623498+00:00
-- finished_at: 2026-10-08T18:12:27.267464001+00:00
-- elapsed: 209ms
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
-- created_at: 2026-10-08T18:12:27.269368042+00:00
-- finished_at: 2026-10-08T18:12:27.406250922+00:00
-- elapsed: 136ms
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
-- created_at: 2026-10-08T18:12:27.408264264+00:00
-- finished_at: 2026-10-08T18:12:27.523992870+00:00
-- elapsed: 115ms
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
-- created_at: 2026-10-08T18:12:27.526165743+00:00
-- finished_at: 2026-10-08T18:12:27.662668887+00:00
-- elapsed: 136ms
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
-- created_at: 2026-10-08T18:12:27.664699139+00:00
-- finished_at: 2026-10-08T18:12:27.911209996+00:00
-- elapsed: 246ms
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
-- created_at: 2026-10-08T18:12:27.913978698+00:00
-- finished_at: 2026-10-08T18:12:28.030037224+00:00
-- elapsed: 116ms
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
-- created_at: 2026-10-08T18:12:28.032413140+00:00
-- finished_at: 2026-10-08T18:12:28.182136418+00:00
-- elapsed: 149ms
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
-- created_at: 2026-10-08T18:12:28.190775775+00:00
-- finished_at: 2026-10-08T18:12:28.872493057+00:00
-- elapsed: 681ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff"
  
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
from "tse_analytics"."main"."bronze_tally_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_tally_munzona" as DBT_INCREMENTAL_TARGET
            using "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff"
            where (
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_tally_munzona" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file"
        from "fact_tally_munzona__dbt_tmp_abcd273b_66c1_4b3c_bb67_5ea17dcf62ff"
    )
  ;
-- created_at: 2026-10-08T18:12:28.891377031+00:00
-- finished_at: 2026-10-08T18:12:28.894865990+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

with source_rows as (
    select *
    from "tse_analytics"."main"."bronze_party_votes_raw"
    
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
-- created_at: 2026-10-08T18:12:28.900867342+00:00
-- finished_at: 2026-10-08T18:12:28.908631973+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'silver_party_votes_munzona'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T18:12:28.917761527+00:00
-- finished_at: 2026-10-08T18:12:29.632799671+00:00
-- elapsed: 715ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T18:12:29.636438467+00:00
-- finished_at: 2026-10-08T18:12:30.008860515+00:00
-- elapsed: 372ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "round_number__dbt_alter" integer;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "round_number__dbt_alter" = "round_number";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "round_number" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "round_number__dbt_alter" to "round_number"
  ;
-- created_at: 2026-10-08T18:12:30.012310997+00:00
-- finished_at: 2026-10-08T18:12:30.336564994+00:00
-- elapsed: 324ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "zone__dbt_alter" integer;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "zone__dbt_alter" = "zone";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "zone" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "zone__dbt_alter" to "zone"
  ;
-- created_at: 2026-10-08T18:12:30.340923217+00:00
-- finished_at: 2026-10-08T18:12:30.656784755+00:00
-- elapsed: 315ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "is_transit_vote__dbt_alter" boolean;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "is_transit_vote__dbt_alter" = "is_transit_vote";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "is_transit_vote" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "is_transit_vote__dbt_alter" to "is_transit_vote"
  ;
-- created_at: 2026-10-08T18:12:30.662372734+00:00
-- finished_at: 2026-10-08T18:12:31.005823710+00:00
-- elapsed: 343ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "nominal_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "nominal_valid_votes__dbt_alter" = "nominal_valid_votes";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "nominal_valid_votes" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "nominal_valid_votes__dbt_alter" to "nominal_valid_votes"
  ;
-- created_at: 2026-10-08T18:12:31.011198882+00:00
-- finished_at: 2026-10-08T18:12:31.380740419+00:00
-- elapsed: 369ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "legend_valid_votes__dbt_alter" = "legend_valid_votes";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "legend_valid_votes__dbt_alter" to "legend_valid_votes"
  ;
-- created_at: 2026-10-08T18:12:31.385263271+00:00
-- finished_at: 2026-10-08T18:12:31.777102323+00:00
-- elapsed: 391ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "nominal_converted_to_legend_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "nominal_converted_to_legend_votes__dbt_alter" = "nominal_converted_to_legend_votes";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "nominal_converted_to_legend_votes" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "nominal_converted_to_legend_votes__dbt_alter" to "nominal_converted_to_legend_votes"
  ;
-- created_at: 2026-10-08T18:12:31.781959479+00:00
-- finished_at: 2026-10-08T18:12:32.150474835+00:00
-- elapsed: 368ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "total_legend_valid_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "total_legend_valid_votes__dbt_alter" = "total_legend_valid_votes";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "total_legend_valid_votes" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "total_legend_valid_votes__dbt_alter" to "total_legend_valid_votes"
  ;
-- created_at: 2026-10-08T18:12:32.155329390+00:00
-- finished_at: 2026-10-08T18:12:32.516337906+00:00
-- elapsed: 361ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "nominal_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "nominal_annulled_subjudice_votes__dbt_alter" = "nominal_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "nominal_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "nominal_annulled_subjudice_votes__dbt_alter" to "nominal_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T18:12:32.520688670+00:00
-- finished_at: 2026-10-08T18:12:33.005588261+00:00
-- elapsed: 484ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "legend_annulled_subjudice_votes__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "legend_annulled_subjudice_votes__dbt_alter" = "legend_annulled_subjudice_votes";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "legend_annulled_subjudice_votes" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "legend_annulled_subjudice_votes__dbt_alter" to "legend_annulled_subjudice_votes"
  ;
-- created_at: 2026-10-08T18:12:33.010515239+00:00
-- finished_at: 2026-10-08T18:12:33.442693675+00:00
-- elapsed: 432ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "generated_at__dbt_alter" datetime;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "generated_at__dbt_alter" = "generated_at";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "generated_at" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "generated_at__dbt_alter" to "generated_at"
  ;
-- created_at: 2026-10-08T18:12:33.448253747+00:00
-- finished_at: 2026-10-08T18:12:33.869656304+00:00
-- elapsed: 421ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "source_row_count__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "source_row_count__dbt_alter" = "source_row_count";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "source_row_count" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "source_row_count__dbt_alter" to "source_row_count"
  ;
-- created_at: 2026-10-08T18:12:33.876166486+00:00
-- finished_at: 2026-10-08T18:12:34.284631325+00:00
-- elapsed: 408ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "source_party_group_types__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "source_party_group_types__dbt_alter" = "source_party_group_types";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "source_party_group_types" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "source_party_group_types__dbt_alter" to "source_party_group_types"
  ;
-- created_at: 2026-10-08T18:12:34.291417829+00:00
-- finished_at: 2026-10-08T18:12:34.715149816+00:00
-- elapsed: 423ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "source_coalitions__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "source_coalitions__dbt_alter" = "source_coalitions";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "source_coalitions" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "source_coalitions__dbt_alter" to "source_coalitions"
  ;
-- created_at: 2026-10-08T18:12:34.720564792+00:00
-- finished_at: 2026-10-08T18:12:35.114016897+00:00
-- elapsed: 393ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_party_votes_munzona" add column "source_federations__dbt_alter" bigint;
    update "tse_analytics"."main"."silver_party_votes_munzona" set "source_federations__dbt_alter" = "source_federations";
    alter table "tse_analytics"."main"."silver_party_votes_munzona" drop column "source_federations" cascade;
    alter table "tse_analytics"."main"."silver_party_votes_munzona" rename column "source_federations__dbt_alter" to "source_federations"
  ;
-- created_at: 2026-10-08T18:12:35.133150434+00:00
-- finished_at: 2026-10-08T18:12:44.131412523+00:00
-- elapsed: 9.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d"
  
    as (
      

with source_rows as (
    select *
    from "tse_analytics"."main"."bronze_party_votes_raw"
    
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

        
            delete from "tse_analytics"."main"."silver_party_votes_munzona" as DBT_INCREMENTAL_TARGET
            using "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d"
            where (
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."silver_party_votes_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations"
        from "silver_party_votes_munzona__dbt_tmp_32389b65_fe99_46bb_8cb2_31b31ccb4d6d"
    )
  ;
-- created_at: 2026-10-08T18:12:44.144339280+00:00
-- finished_at: 2026-10-08T18:12:44.907376079+00:00
-- elapsed: 763ms
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
    from "tse_analytics"."main"."bronze_candidates"

    union all

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from "tse_analytics"."main"."bronze_party_votes_raw"

    union all

    select
        election_year,
        election_type,
        election_code,
        election_scope
    from "tse_analytics"."main"."bronze_tally_munzona"
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
-- created_at: 2026-10-08T18:12:44.909820580+00:00
-- finished_at: 2026-10-08T18:12:44.911102857+00:00
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
-- created_at: 2026-10-08T18:12:44.911982291+00:00
-- finished_at: 2026-10-08T18:12:44.912581744+00:00
-- elapsed: 599us
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
-- created_at: 2026-10-08T18:12:44.913849928+00:00
-- finished_at: 2026-10-08T18:12:45.274465197+00:00
-- elapsed: 360ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election" rename to "dim_election__dbt_backup";
-- created_at: 2026-10-08T18:12:45.279816953+00:00
-- finished_at: 2026-10-08T18:12:45.289902336+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election__dbt_tmp" rename to "dim_election";
-- created_at: 2026-10-08T18:12:45.293628901+00:00
-- finished_at: 2026-10-08T18:12:45.298378807+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop table if exists "tse_analytics"."main"."dim_election__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:12:45.308747256+00:00
-- finished_at: 2026-10-08T18:12:45.325282517+00:00
-- elapsed: 16ms
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
-- created_at: 2026-10-08T18:12:45.330328915+00:00
-- finished_at: 2026-10-08T18:12:45.341348163+00:00
-- elapsed: 11ms
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
-- created_at: 2026-10-08T18:12:45.346717436+00:00
-- finished_at: 2026-10-08T18:12:45.400766960+00:00
-- elapsed: 54ms
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
-- created_at: 2026-10-08T18:12:45.405998811+00:00
-- finished_at: 2026-10-08T18:12:45.412066850+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T18:12:45.418416581+00:00
-- finished_at: 2026-10-08T18:12:45.503099913+00:00
-- elapsed: 84ms
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
    from "tse_analytics"."main"."bronze_tally_munzona"
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
-- created_at: 2026-10-08T18:12:45.516051902+00:00
-- finished_at: 2026-10-08T18:12:45.520311169+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_total_legend_valid_votes.949adbe8f5
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_total_legend_valid_votes.949adbe8f5", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select total_legend_valid_votes
from "tse_analytics"."main"."silver_party_votes_munzona"
where total_legend_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.526238958+00:00
-- finished_at: 2026-10-08T18:12:45.527603722+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_office_code.ffa812b222
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_office_code.ffa812b222", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select office_code
from "tse_analytics"."main"."silver_party_votes_munzona"
where office_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.532952764+00:00
-- finished_at: 2026-10-08T18:12:45.534653059+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_municipality_code.0a2a5d5c53
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_municipality_code.0a2a5d5c53", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select municipality_code
from "tse_analytics"."main"."silver_party_votes_munzona"
where municipality_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.544690259+00:00
-- finished_at: 2026-10-08T18:12:45.546262783+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_election_type.fb15f66ea5
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_election_type.fb15f66ea5", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_type
from "tse_analytics"."main"."silver_party_votes_munzona"
where election_type is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.553123316+00:00
-- finished_at: 2026-10-08T18:12:45.556947441+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_election_year.e41ec20be5
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_election_year.e41ec20be5", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_year
from "tse_analytics"."main"."silver_party_votes_munzona"
where election_year is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.563276881+00:00
-- finished_at: 2026-10-08T18:12:45.567036234+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_zone.6a7d695ebf
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_zone.6a7d695ebf", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select zone
from "tse_analytics"."main"."silver_party_votes_munzona"
where zone is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.574580492+00:00
-- finished_at: 2026-10-08T18:12:45.578459549+00:00
-- elapsed: 3ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_legend_valid_votes.90f9dbadae
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_legend_valid_votes.90f9dbadae", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select legend_valid_votes
from "tse_analytics"."main"."silver_party_votes_munzona"
where legend_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.583249567+00:00
-- finished_at: 2026-10-08T18:12:45.584508322+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_party_number.97b9b16dd1
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_party_number.97b9b16dd1", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select party_number
from "tse_analytics"."main"."silver_party_votes_munzona"
where party_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.589816474+00:00
-- finished_at: 2026-10-08T18:12:45.591103768+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_election_code.9ade8d9a04
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_election_code.9ade8d9a04", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select election_code
from "tse_analytics"."main"."silver_party_votes_munzona"
where election_code is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.595703572+00:00
-- finished_at: 2026-10-08T18:12:45.598500110+00:00
-- elapsed: 2ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_round_number.930786b7cc
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_round_number.930786b7cc", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select round_number
from "tse_analytics"."main"."silver_party_votes_munzona"
where round_number is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.603312320+00:00
-- finished_at: 2026-10-08T18:12:45.835198026+00:00
-- elapsed: 231ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.dbt_utils_unique_combination_of_columns_silver_party_votes_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__party_number__is_transit_vote.864fb87931
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.dbt_utils_unique_combination_of_columns_silver_party_votes_munzona_election_year__election_type__election_code__round_number__uf__municipality_code__zone__office_code__party_number__is_transit_vote.864fb87931", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  





with validation_errors as (

    select
        election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, party_number, is_transit_vote
    from "tse_analytics"."main"."silver_party_votes_munzona"
    group by election_year, election_type, election_code, round_number, uf, municipality_code, zone, office_code, party_number, is_transit_vote
    having count(*) > 1

)

select *
from validation_errors



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.841303149+00:00
-- finished_at: 2026-10-08T18:12:45.842963537+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_uf.ec64c0f307
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_uf.ec64c0f307", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select uf
from "tse_analytics"."main"."silver_party_votes_munzona"
where uf is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.853692204+00:00
-- finished_at: 2026-10-08T18:12:45.858512873+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: test.tse_analytics.not_null_silver_party_votes_munzona_nominal_valid_votes.7060501af1
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "test.tse_analytics.not_null_silver_party_votes_munzona_nominal_valid_votes.7060501af1", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select nominal_valid_votes
from "tse_analytics"."main"."silver_party_votes_munzona"
where nominal_valid_votes is null



  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:45.867880574+00:00
-- finished_at: 2026-10-08T18:12:45.875456456+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T18:12:45.883385901+00:00
-- finished_at: 2026-10-08T18:12:45.885275372+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:12:45.892721627+00:00
-- finished_at: 2026-10-08T18:12:45.895422761+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:12:45.901088505+00:00
-- finished_at: 2026-10-08T18:12:45.907525620+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T18:12:45.920263371+00:00
-- finished_at: 2026-10-08T18:12:46.457353497+00:00
-- elapsed: 537ms
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
from "tse_analytics"."main"."bronze_candidates" c
left join "tse_analytics"."main"."silver_candidate_assets" a
  using (election_year, election_type, election_code, candidate_id)

  
    where c.election_year in (2026) and c.election_type in ('general')
  

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:12:46.463774991+00:00
-- finished_at: 2026-10-08T18:12:46.472968752+00:00
-- elapsed: 9ms
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
-- created_at: 2026-10-08T18:12:46.482408196+00:00
-- finished_at: 2026-10-08T18:12:47.405319495+00:00
-- elapsed: 922ms
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
-- created_at: 2026-10-08T18:12:47.409686123+00:00
-- finished_at: 2026-10-08T18:12:48.160467584+00:00
-- elapsed: 750ms
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
-- created_at: 2026-10-08T18:12:48.164915878+00:00
-- finished_at: 2026-10-08T18:12:49.272592478+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T18:12:49.276242618+00:00
-- finished_at: 2026-10-08T18:12:50.103645563+00:00
-- elapsed: 827ms
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
-- created_at: 2026-10-08T18:12:50.113715610+00:00
-- finished_at: 2026-10-08T18:12:51.248048575+00:00
-- elapsed: 1.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479"
  
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
from "tse_analytics"."main"."bronze_candidates" c
left join "tse_analytics"."main"."silver_candidate_assets" a
  using (election_year, election_type, election_code, candidate_id)

  
    where c.election_year in (2026) and c.election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_candidate" as DBT_INCREMENTAL_TARGET
            using "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479"
            where (
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479".candidate_id = DBT_INCREMENTAL_TARGET.candidate_id
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_candidate" ("election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count"
        from "dim_candidate__dbt_tmp_8b91cc73_855d_48b3_a4dc_217e1617c479"
    )
  ;
-- created_at: 2026-10-08T18:12:51.267914863+00:00
-- finished_at: 2026-10-08T18:12:51.272776260+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T18:12:51.278046090+00:00
-- finished_at: 2026-10-08T18:12:51.291208431+00:00
-- elapsed: 13ms
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
-- created_at: 2026-10-08T18:12:51.301369700+00:00
-- finished_at: 2026-10-08T18:12:51.491596550+00:00
-- elapsed: 190ms
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
-- created_at: 2026-10-08T18:12:51.502588553+00:00
-- finished_at: 2026-10-08T18:12:51.673166547+00:00
-- elapsed: 170ms
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
-- created_at: 2026-10-08T18:12:51.682061334+00:00
-- finished_at: 2026-10-08T18:12:51.992939471+00:00
-- elapsed: 310ms
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
-- created_at: 2026-10-08T18:12:51.999366634+00:00
-- finished_at: 2026-10-08T18:12:52.119152963+00:00
-- elapsed: 119ms
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
-- created_at: 2026-10-08T18:11:27.911260408+00:00
-- finished_at: 2026-10-08T18:12:52.362236528+00:00
-- elapsed: 1m 24s
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
from "tse_analytics"."main"."bronze_electorate"
where municipality_code is not null
  and (
      length(municipality_code) <> 5
      or not regexp_matches(municipality_code, '^[0-9]{5}$')
  )
  
  
      
    ) dbt_internal_test;
-- created_at: 2026-10-08T18:12:52.378720131+00:00
-- finished_at: 2026-10-08T18:12:54.782385139+00:00
-- elapsed: 2.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_tally_reconciliation__dbt_tmp" as (
    

with coverage as (
    select *
    from "tse_analytics"."main"."silver_candidate_result_coverage"
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
-- created_at: 2026-10-08T18:12:52.127573762+00:00
-- finished_at: 2026-10-08T18:12:54.787576590+00:00
-- elapsed: 2.7s
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
-- created_at: 2026-10-08T18:12:54.786778746+00:00
-- finished_at: 2026-10-08T18:12:54.793784674+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation" rename to "candidate_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T18:12:54.797238826+00:00
-- finished_at: 2026-10-08T18:12:54.814367957+00:00
-- elapsed: 17ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation__dbt_tmp" rename to "candidate_tally_reconciliation";
-- created_at: 2026-10-08T18:12:54.820599520+00:00
-- finished_at: 2026-10-08T18:12:54.827034574+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:12:54.794472207+00:00
-- finished_at: 2026-10-08T18:12:54.985697937+00:00
-- elapsed: 191ms
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
-- created_at: 2026-10-08T18:12:54.990082074+00:00
-- finished_at: 2026-10-08T18:12:55.206920475+00:00
-- elapsed: 216ms
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
-- created_at: 2026-10-08T18:12:54.857023070+00:00
-- finished_at: 2026-10-08T18:12:55.214226590+00:00
-- elapsed: 357ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_tmp" as (
    

with coverage as (

    select *
    from "tse_analytics"."main"."silver_candidate_result_coverage"

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
-- created_at: 2026-10-08T18:12:55.217833860+00:00
-- finished_at: 2026-10-08T18:12:55.227383277+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps" rename to "candidate_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T18:12:55.230210042+00:00
-- finished_at: 2026-10-08T18:12:55.316228982+00:00
-- elapsed: 86ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_tmp" rename to "candidate_tally_coverage_gaps";
-- created_at: 2026-10-08T18:12:55.211012166+00:00
-- finished_at: 2026-10-08T18:12:55.327123870+00:00
-- elapsed: 116ms
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
-- created_at: 2026-10-08T18:12:55.320146922+00:00
-- finished_at: 2026-10-08T18:12:55.333398476+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:12:55.342710009+00:00
-- finished_at: 2026-10-08T18:12:55.351333647+00:00
-- elapsed: 8ms
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
    from "tse_analytics"."main"."silver_party_votes_munzona"
    
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
-- created_at: 2026-10-08T18:12:55.355554518+00:00
-- finished_at: 2026-10-08T18:12:55.433234672+00:00
-- elapsed: 77ms
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
-- created_at: 2026-10-08T18:12:55.331016148+00:00
-- finished_at: 2026-10-08T18:12:55.442874149+00:00
-- elapsed: 111ms
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
-- created_at: 2026-10-08T18:12:55.438977226+00:00
-- finished_at: 2026-10-08T18:12:55.470792726+00:00
-- elapsed: 31ms
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
-- created_at: 2026-10-08T18:12:55.446456302+00:00
-- finished_at: 2026-10-08T18:12:55.591059981+00:00
-- elapsed: 144ms
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
-- created_at: 2026-10-08T18:12:55.594965462+00:00
-- finished_at: 2026-10-08T18:12:55.725472009+00:00
-- elapsed: 130ms
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
-- created_at: 2026-10-08T18:12:55.729391905+00:00
-- finished_at: 2026-10-08T18:12:55.844443596+00:00
-- elapsed: 115ms
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
-- created_at: 2026-10-08T18:12:55.858224508+00:00
-- finished_at: 2026-10-08T18:12:56.140879914+00:00
-- elapsed: 282ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318"
  
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
            using "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318"
            where (
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_turnout" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at"
        from "fact_turnout__dbt_tmp_dd4eecdc_c21b_4148_8dce_62459cb49318"
    )
  ;
-- created_at: 2026-10-08T18:12:56.159463566+00:00
-- finished_at: 2026-10-08T18:12:56.162037461+00:00
-- elapsed: 2ms
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
from "tse_analytics"."main"."silver_party_votes_munzona"

where election_year in (2026) and election_type in ('general')

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:12:56.166016816+00:00
-- finished_at: 2026-10-08T18:12:56.172781741+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T18:12:55.482153062+00:00
-- finished_at: 2026-10-08T18:12:56.590057755+00:00
-- elapsed: 1.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_party
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_party", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_party__dbt_tmp_5ecf6b8c_93d0_482a_81e3_62ea7c225a7c"
  
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
    from "tse_analytics"."main"."silver_party_votes_munzona"
    
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
            using "dim_party__dbt_tmp_5ecf6b8c_93d0_482a_81e3_62ea7c225a7c"
            where (
                
                    "dim_party__dbt_tmp_5ecf6b8c_93d0_482a_81e3_62ea7c225a7c".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_party__dbt_tmp_5ecf6b8c_93d0_482a_81e3_62ea7c225a7c".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_party__dbt_tmp_5ecf6b8c_93d0_482a_81e3_62ea7c225a7c".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_party__dbt_tmp_5ecf6b8c_93d0_482a_81e3_62ea7c225a7c".party_number = DBT_INCREMENTAL_TARGET.party_number
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_party" ("election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id"
        from "dim_party__dbt_tmp_5ecf6b8c_93d0_482a_81e3_62ea7c225a7c"
    )
  ;
-- created_at: 2026-10-08T18:12:56.605145097+00:00
-- finished_at: 2026-10-08T18:12:56.613507226+00:00
-- elapsed: 8ms
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
-- created_at: 2026-10-08T18:12:56.619941998+00:00
-- finished_at: 2026-10-08T18:12:56.621359306+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:12:56.627085472+00:00
-- finished_at: 2026-10-08T18:12:56.628495848+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:12:56.634094090+00:00
-- finished_at: 2026-10-08T18:12:56.635535970+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:12:56.642317153+00:00
-- finished_at: 2026-10-08T18:12:57.549644267+00:00
-- elapsed: 907ms
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
    from "tse_analytics"."main"."bronze_candidates"
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
-- created_at: 2026-10-08T18:12:57.555377034+00:00
-- finished_at: 2026-10-08T18:12:57.573069343+00:00
-- elapsed: 17ms
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
-- created_at: 2026-10-08T18:12:57.578132451+00:00
-- finished_at: 2026-10-08T18:12:57.580820741+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:12:56.180549459+00:00
-- finished_at: 2026-10-08T18:12:57.737382328+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T18:12:57.585119669+00:00
-- finished_at: 2026-10-08T18:12:58.610836021+00:00
-- elapsed: 1.0s
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
-- created_at: 2026-10-08T18:12:57.741799604+00:00
-- finished_at: 2026-10-08T18:12:58.613227380+00:00
-- elapsed: 871ms
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
-- created_at: 2026-10-08T18:12:58.616900676+00:00
-- finished_at: 2026-10-08T18:13:01.317666703+00:00
-- elapsed: 2.7s
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
-- created_at: 2026-10-08T18:13:01.320158870+00:00
-- finished_at: 2026-10-08T18:13:02.013350674+00:00
-- elapsed: 693ms
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
-- created_at: 2026-10-08T18:13:02.017731814+00:00
-- finished_at: 2026-10-08T18:13:03.009062142+00:00
-- elapsed: 991ms
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
-- created_at: 2026-10-08T18:12:58.614606459+00:00
-- finished_at: 2026-10-08T18:13:03.149021985+00:00
-- elapsed: 4.5s
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
-- created_at: 2026-10-08T18:13:03.157935+00:00
-- finished_at: 2026-10-08T18:13:05.415438099+00:00
-- elapsed: 2.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_electorate_municipality
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from "tse_analytics"."main"."bronze_electorate"

where election_year in (2026) and election_type in ('general')

group by 1,2,3,4,5,6
    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:13:05.418644451+00:00
-- finished_at: 2026-10-08T18:13:05.423721932+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    
      select
          column_name,
          data_type,
          character_maximum_length,
          numeric_precision,
          numeric_scale

      from system.information_schema.columns
      where table_name = 'silver_electorate_municipality'
      
      and lower(table_schema) = 'main'
      
      
      and lower(table_catalog) = 'tse_analytics'
      
      order by ordinal_position

    
  ;
-- created_at: 2026-10-08T18:13:03.014105885+00:00
-- finished_at: 2026-10-08T18:13:07.741103455+00:00
-- elapsed: 4.7s
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
-- created_at: 2026-10-08T18:13:05.426197222+00:00
-- finished_at: 2026-10-08T18:13:11.217362194+00:00
-- elapsed: 5.8s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_electorate_municipality" add column "election_year__dbt_alter" integer;
    update "tse_analytics"."main"."silver_electorate_municipality" set "election_year__dbt_alter" = "election_year";
    alter table "tse_analytics"."main"."silver_electorate_municipality" drop column "election_year" cascade;
    alter table "tse_analytics"."main"."silver_electorate_municipality" rename column "election_year__dbt_alter" to "election_year"
  ;
-- created_at: 2026-10-08T18:13:07.744294798+00:00
-- finished_at: 2026-10-08T18:13:11.365424993+00:00
-- elapsed: 3.6s
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
-- created_at: 2026-10-08T18:13:11.222163962+00:00
-- finished_at: 2026-10-08T18:13:11.508998785+00:00
-- elapsed: 286ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

    alter table "tse_analytics"."main"."silver_electorate_municipality" add column "electorate__dbt_alter" decimal(38, 0);
    update "tse_analytics"."main"."silver_electorate_municipality" set "electorate__dbt_alter" = "electorate";
    alter table "tse_analytics"."main"."silver_electorate_municipality" drop column "electorate" cascade;
    alter table "tse_analytics"."main"."silver_electorate_municipality" rename column "electorate__dbt_alter" to "electorate"
  ;
-- created_at: 2026-10-08T18:13:11.368167387+00:00
-- finished_at: 2026-10-08T18:13:12.796052696+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T18:13:12.800912109+00:00
-- finished_at: 2026-10-08T18:13:14.250141369+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T18:13:14.256168299+00:00
-- finished_at: 2026-10-08T18:13:15.886425020+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T18:13:15.894051948+00:00
-- finished_at: 2026-10-08T18:13:17.384308804+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T18:13:17.390065683+00:00
-- finished_at: 2026-10-08T18:13:18.671382655+00:00
-- elapsed: 1.3s
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
-- created_at: 2026-10-08T18:13:18.683371921+00:00
-- finished_at: 2026-10-08T18:13:22.468279056+00:00
-- elapsed: 3.8s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d"
  
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
from "tse_analytics"."main"."silver_party_votes_munzona"

where election_year in (2026) and election_type in ('general')

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_party_votes" as DBT_INCREMENTAL_TARGET
            using "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d"
            where (
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_party_votes" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file"
        from "fact_party_votes__dbt_tmp_f8ab31b5_76dd_4ab8_abdc_c77b45369e0d"
    )
  ;
-- created_at: 2026-10-08T18:13:22.517948450+00:00
-- finished_at: 2026-10-08T18:13:31.265243330+00:00
-- elapsed: 8.7s
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
-- created_at: 2026-10-08T18:13:31.272100373+00:00
-- finished_at: 2026-10-08T18:13:38.616100031+00:00
-- elapsed: 7.3s
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
-- created_at: 2026-10-08T18:13:38.624843991+00:00
-- finished_at: 2026-10-08T18:13:38.650824130+00:00
-- elapsed: 25ms
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
-- created_at: 2026-10-08T18:13:38.656854207+00:00
-- finished_at: 2026-10-08T18:13:38.658877833+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:13:38.664695448+00:00
-- finished_at: 2026-10-08T18:13:38.670260963+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T18:13:38.675760373+00:00
-- finished_at: 2026-10-08T18:13:38.714924859+00:00
-- elapsed: 39ms
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
-- created_at: 2026-10-08T18:13:38.720937359+00:00
-- finished_at: 2026-10-08T18:13:38.722945150+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:13:38.728773734+00:00
-- finished_at: 2026-10-08T18:13:38.732767108+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T18:13:38.742063084+00:00
-- finished_at: 2026-10-08T18:13:38.743446753+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:13:38.747772976+00:00
-- finished_at: 2026-10-08T18:13:38.748966309+00:00
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
-- created_at: 2026-10-08T18:13:38.753617460+00:00
-- finished_at: 2026-10-08T18:13:38.754728743+00:00
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
-- created_at: 2026-10-08T18:13:38.759422402+00:00
-- finished_at: 2026-10-08T18:13:38.760574796+00:00
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
-- created_at: 2026-10-08T18:13:38.765691006+00:00
-- finished_at: 2026-10-08T18:13:38.768292683+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:13:38.774838319+00:00
-- finished_at: 2026-10-08T18:13:38.820101177+00:00
-- elapsed: 45ms
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
    from "tse_analytics"."main"."silver_party_votes_munzona"
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
-- created_at: 2026-10-08T18:13:38.825993581+00:00
-- finished_at: 2026-10-08T18:13:38.828741387+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:13:38.837062757+00:00
-- finished_at: 2026-10-08T18:13:38.856269901+00:00
-- elapsed: 19ms
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
-- created_at: 2026-10-08T18:13:38.858961618+00:00
-- finished_at: 2026-10-08T18:13:38.865381713+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary" rename to "candidate_summary__dbt_backup";
-- created_at: 2026-10-08T18:13:38.867922590+00:00
-- finished_at: 2026-10-08T18:13:38.874770919+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary__dbt_tmp" rename to "candidate_summary";
-- created_at: 2026-10-08T18:13:38.877958909+00:00
-- finished_at: 2026-10-08T18:13:38.883504713+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:13:38.890492137+00:00
-- finished_at: 2026-10-08T18:13:38.907881159+00:00
-- elapsed: 17ms
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
-- created_at: 2026-10-08T18:13:38.910658899+00:00
-- finished_at: 2026-10-08T18:13:38.917137968+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary" rename to "candidate_vote_summary__dbt_backup";
-- created_at: 2026-10-08T18:13:38.919573514+00:00
-- finished_at: 2026-10-08T18:13:38.933089672+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary__dbt_tmp" rename to "candidate_vote_summary";
-- created_at: 2026-10-08T18:13:38.936639874+00:00
-- finished_at: 2026-10-08T18:13:38.942329131+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_vote_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:13:38.947519685+00:00
-- finished_at: 2026-10-08T18:13:38.948941300+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:13:38.954869624+00:00
-- finished_at: 2026-10-08T18:13:38.956311237+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:13:38.960984164+00:00
-- finished_at: 2026-10-08T18:13:38.962177545+00:00
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
-- created_at: 2026-10-08T18:13:38.967782751+00:00
-- finished_at: 2026-10-08T18:13:39.092140752+00:00
-- elapsed: 124ms
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
-- created_at: 2026-10-08T18:13:39.100788349+00:00
-- finished_at: 2026-10-08T18:13:39.102824105+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T18:13:39.109873582+00:00
-- finished_at: 2026-10-08T18:13:40.393807004+00:00
-- elapsed: 1.3s
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
    from "tse_analytics"."main"."silver_party_votes_munzona"
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
-- created_at: 2026-10-08T18:13:40.399559434+00:00
-- finished_at: 2026-10-08T18:13:41.873525020+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T18:13:41.884532073+00:00
-- finished_at: 2026-10-08T18:13:41.920986954+00:00
-- elapsed: 36ms
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
-- created_at: 2026-10-08T18:13:41.929904432+00:00
-- finished_at: 2026-10-08T18:13:41.931676057+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:13:41.941128253+00:00
-- finished_at: 2026-10-08T18:13:41.942891651+00:00
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
-- created_at: 2026-10-08T18:13:41.950014724+00:00
-- finished_at: 2026-10-08T18:13:42.097516070+00:00
-- elapsed: 147ms
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
-- created_at: 2026-10-08T18:13:42.106614265+00:00
-- finished_at: 2026-10-08T18:13:42.243856913+00:00
-- elapsed: 137ms
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
-- created_at: 2026-10-08T18:13:42.250947114+00:00
-- finished_at: 2026-10-08T18:13:42.252687769+00:00
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
-- created_at: 2026-10-08T18:13:42.261907008+00:00
-- finished_at: 2026-10-08T18:13:42.291770066+00:00
-- elapsed: 29ms
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
-- created_at: 2026-10-08T18:13:11.513558011+00:00
-- finished_at: 2026-10-08T18:13:47.165852929+00:00
-- elapsed: 35.7s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "silver_electorate_municipality__dbt_tmp_0d4e4d21_4d3c_4f22_9a2a_266173685830"
  
    as (
      

select
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality,
    sum(electorate) as electorate
from "tse_analytics"."main"."bronze_electorate"

where election_year in (2026) and election_type in ('general')

group by 1,2,3,4,5,6
    );
  
    
  ;

        
            delete from "tse_analytics"."main"."silver_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "silver_electorate_municipality__dbt_tmp_0d4e4d21_4d3c_4f22_9a2a_266173685830"
            where (
                
                    "silver_electorate_municipality__dbt_tmp_0d4e4d21_4d3c_4f22_9a2a_266173685830".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_0d4e4d21_4d3c_4f22_9a2a_266173685830".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_0d4e4d21_4d3c_4f22_9a2a_266173685830".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_0d4e4d21_4d3c_4f22_9a2a_266173685830".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."silver_electorate_municipality" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate"
        from "silver_electorate_municipality__dbt_tmp_0d4e4d21_4d3c_4f22_9a2a_266173685830"
    )
  ;
-- created_at: 2026-10-08T18:13:47.175224258+00:00
-- finished_at: 2026-10-08T18:13:47.181702379+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T18:13:47.184406621+00:00
-- finished_at: 2026-10-08T18:13:47.191567253+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation" rename to "electoral_participation__dbt_backup";
-- created_at: 2026-10-08T18:13:47.194157009+00:00
-- finished_at: 2026-10-08T18:13:47.201330255+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation__dbt_tmp" rename to "electoral_participation";
-- created_at: 2026-10-08T18:13:47.204587501+00:00
-- finished_at: 2026-10-08T18:13:47.209920372+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."electoral_participation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:13:47.216573847+00:00
-- finished_at: 2026-10-08T18:13:47.233876010+00:00
-- elapsed: 17ms
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
-- created_at: 2026-10-08T18:13:47.236689213+00:00
-- finished_at: 2026-10-08T18:13:47.243045658+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance" rename to "party_performance__dbt_backup";
-- created_at: 2026-10-08T18:13:47.245792912+00:00
-- finished_at: 2026-10-08T18:13:47.256736197+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance__dbt_tmp" rename to "party_performance";
-- created_at: 2026-10-08T18:13:47.260298205+00:00
-- finished_at: 2026-10-08T18:13:47.265504283+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_performance__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:13:47.272528233+00:00
-- finished_at: 2026-10-08T18:13:47.281704768+00:00
-- elapsed: 9ms
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
-- created_at: 2026-10-08T18:13:47.284525821+00:00
-- finished_at: 2026-10-08T18:13:47.290679784+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps" rename to "party_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T18:13:47.293200820+00:00
-- finished_at: 2026-10-08T18:13:47.300111281+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps__dbt_tmp" rename to "party_tally_coverage_gaps";
-- created_at: 2026-10-08T18:13:47.303896205+00:00
-- finished_at: 2026-10-08T18:13:47.308786411+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:13:47.326148770+00:00
-- finished_at: 2026-10-08T18:13:47.327784351+00:00
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
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:13:47.331464485+00:00
-- finished_at: 2026-10-08T18:13:47.342192974+00:00
-- elapsed: 10ms
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
-- created_at: 2026-10-08T18:13:47.348799315+00:00
-- finished_at: 2026-10-08T18:13:47.419609967+00:00
-- elapsed: 70ms
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
-- created_at: 2026-10-08T18:13:47.425724331+00:00
-- finished_at: 2026-10-08T18:13:47.467431948+00:00
-- elapsed: 41ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_geography
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_geography", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_geography__dbt_tmp_6e6af3ef_fc67_4fad_b4b2_820197dea627"
  
    as (
      

select distinct
    election_year,
    election_type,
    election_scope,
    uf,
    municipality_code,
    municipality
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."dim_geography" as DBT_INCREMENTAL_TARGET
            using "dim_geography__dbt_tmp_6e6af3ef_fc67_4fad_b4b2_820197dea627"
            where (
                
                    "dim_geography__dbt_tmp_6e6af3ef_fc67_4fad_b4b2_820197dea627".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_geography__dbt_tmp_6e6af3ef_fc67_4fad_b4b2_820197dea627".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_geography__dbt_tmp_6e6af3ef_fc67_4fad_b4b2_820197dea627".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "dim_geography__dbt_tmp_6e6af3ef_fc67_4fad_b4b2_820197dea627".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_geography" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality"
        from "dim_geography__dbt_tmp_6e6af3ef_fc67_4fad_b4b2_820197dea627"
    )
  ;
-- created_at: 2026-10-08T18:13:47.479123651+00:00
-- finished_at: 2026-10-08T18:13:47.480581740+00:00
-- elapsed: 1ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: get_column_schema_from_query adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */
select * from (
        

select *
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T18:13:47.483540452+00:00
-- finished_at: 2026-10-08T18:13:47.544775204+00:00
-- elapsed: 61ms
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
-- created_at: 2026-10-08T18:13:47.550833648+00:00
-- finished_at: 2026-10-08T18:13:47.601391510+00:00
-- elapsed: 50ms
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
-- created_at: 2026-10-08T18:13:47.608996951+00:00
-- finished_at: 2026-10-08T18:13:47.675047732+00:00
-- elapsed: 66ms
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
-- created_at: 2026-10-08T18:13:47.682749943+00:00
-- finished_at: 2026-10-08T18:13:47.727953990+00:00
-- elapsed: 45ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83"
  
    as (
      

select *
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83"
            where (
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_electorate_municipality" ("election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate")
    (
        select "election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate"
        from "fact_electorate_municipality__dbt_tmp_1253276f_6945_49a6_ba34_1c53c7db1b83"
    )
  ;
-- created_at: 2026-10-08T18:13:47.739374324+00:00
-- finished_at: 2026-10-08T18:13:47.759158244+00:00
-- elapsed: 19ms
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
-- created_at: 2026-10-08T18:13:47.762762283+00:00
-- finished_at: 2026-10-08T18:13:47.776222129+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation" rename to "party_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T18:13:47.779193201+00:00
-- finished_at: 2026-10-08T18:13:47.785043017+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation__dbt_tmp" rename to "party_tally_reconciliation";
-- created_at: 2026-10-08T18:13:47.789074854+00:00
-- finished_at: 2026-10-08T18:13:47.794731258+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T18:13:47.801752039+00:00
-- finished_at: 2026-10-08T18:13:47.825307099+00:00
-- elapsed: 23ms
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
    from "tse_analytics"."main"."silver_electorate_municipality"
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
-- created_at: 2026-10-08T18:13:47.831742339+00:00
-- finished_at: 2026-10-08T18:13:47.847596746+00:00
-- elapsed: 15ms
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
-- created_at: 2026-10-08T18:13:47.853743012+00:00
-- finished_at: 2026-10-08T18:13:47.866952713+00:00
-- elapsed: 13ms
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
-- created_at: 2026-10-08T18:13:47.872512131+00:00
-- finished_at: 2026-10-08T18:13:47.873998176+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:13:47.881016474+00:00
-- finished_at: 2026-10-08T18:13:47.883931422+00:00
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
-- created_at: 2026-10-08T18:13:47.889683803+00:00
-- finished_at: 2026-10-08T18:13:47.910139030+00:00
-- elapsed: 20ms
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
    from "tse_analytics"."main"."silver_electorate_municipality"
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
-- created_at: 2026-10-08T18:13:47.917194946+00:00
-- finished_at: 2026-10-08T18:13:47.918691637+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:13:47.923607834+00:00
-- finished_at: 2026-10-08T18:13:47.924869931+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T18:13:47.929847699+00:00
-- finished_at: 2026-10-08T18:13:47.931061818+00:00
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
-- created_at: 2026-10-08T18:13:47.936285459+00:00
-- finished_at: 2026-10-08T18:13:48.803024593+00:00
-- elapsed: 866ms
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
-- created_at: 2026-10-08T18:13:48.807953717+00:00
-- finished_at: 2026-10-08T18:13:49.813216678+00:00
-- elapsed: 1.0s
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
-- created_at: 2026-10-08T18:13:49.819886546+00:00
-- finished_at: 2026-10-08T18:13:50.721447123+00:00
-- elapsed: 901ms
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
-- created_at: 2026-10-08T18:13:50.727291947+00:00
-- finished_at: 2026-10-08T18:13:51.506738204+00:00
-- elapsed: 779ms
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
-- created_at: 2026-10-08T18:13:42.301106699+00:00
-- finished_at: 2026-10-08T18:13:58.816737436+00:00
-- elapsed: 16.5s
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
    from "tse_analytics"."main"."silver_candidate_result_coverage"
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
