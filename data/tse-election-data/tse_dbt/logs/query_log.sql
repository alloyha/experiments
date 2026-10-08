-- created_at: 2026-10-08T19:18:55.709904793+00:00
-- finished_at: 2026-10-08T19:18:55.719615331+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: list_relations_in_parallel
SELECT table_catalog, table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'main' AND lower(table_catalog) = lower('tse_analytics');
-- created_at: 2026-10-08T19:18:55.941418514+00:00
-- finished_at: 2026-10-08T19:18:55.943170515+00:00
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
-- created_at: 2026-10-08T19:18:55.944930798+00:00
-- finished_at: 2026-10-08T19:18:55.947021274+00:00
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
-- created_at: 2026-10-08T19:18:55.947552717+00:00
-- finished_at: 2026-10-08T19:18:55.948099942+00:00
-- elapsed: 547us
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    
        create schema if not exists "tse_analytics"."main"
    ;
-- created_at: 2026-10-08T19:18:55.958851194+00:00
-- finished_at: 2026-10-08T19:18:55.985382387+00:00
-- elapsed: 26ms
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
-- created_at: 2026-10-08T19:18:55.956884633+00:00
-- finished_at: 2026-10-08T19:18:55.985432638+00:00
-- elapsed: 28ms
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
-- created_at: 2026-10-08T19:18:55.992615852+00:00
-- finished_at: 2026-10-08T19:18:56.005062116+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T19:18:55.992810128+00:00
-- finished_at: 2026-10-08T19:18:56.005745741+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T19:18:56.012820895+00:00
-- finished_at: 2026-10-08T19:18:56.026544244+00:00
-- elapsed: 13ms
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
-- created_at: 2026-10-08T19:18:56.012381052+00:00
-- finished_at: 2026-10-08T19:18:56.026574318+00:00
-- elapsed: 14ms
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
-- created_at: 2026-10-08T19:18:56.041141917+00:00
-- finished_at: 2026-10-08T19:18:56.062906665+00:00
-- elapsed: 21ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "seed.tse_analytics.election_calendar", "profile_name": "tse_analytics", "target_name": "dev"} */
truncate table "tse_analytics"."main"."election_calendar";
-- created_at: 2026-10-08T19:18:56.046125197+00:00
-- finished_at: 2026-10-08T19:18:56.101788762+00:00
-- elapsed: 55ms
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
-- created_at: 2026-10-08T19:18:56.077743154+00:00
-- finished_at: 2026-10-08T19:18:56.108910748+00:00
-- elapsed: 31ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: add_query adapter call

          COPY "tse_analytics"."main"."election_calendar" FROM '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/seeds/election_calendar.csv' (FORMAT CSV, HEADER TRUE, DELIMITER ',')
        ;
-- created_at: 2026-10-08T19:18:56.106286736+00:00
-- finished_at: 2026-10-08T19:18:56.113612631+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes" rename to "fact_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T19:18:56.118829840+00:00
-- finished_at: 2026-10-08T19:18:56.127194484+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes__dbt_tmp" rename to "fact_candidate_votes";
-- created_at: 2026-10-08T19:18:56.126328885+00:00
-- finished_at: 2026-10-08T19:18:56.135161456+00:00
-- elapsed: 8ms
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
-- created_at: 2026-10-08T19:18:56.133318289+00:00
-- finished_at: 2026-10-08T19:18:56.143030497+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."fact_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:18:56.157897773+00:00
-- finished_at: 2026-10-08T19:18:56.467733633+00:00
-- elapsed: 309ms
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
-- created_at: 2026-10-08T19:18:56.473571302+00:00
-- finished_at: 2026-10-08T19:18:56.479755221+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_assets" rename to "bronze_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T19:18:56.483652128+00:00
-- finished_at: 2026-10-08T19:18:56.489723953+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_assets__dbt_tmp" rename to "bronze_candidate_assets";
-- created_at: 2026-10-08T19:18:56.495744542+00:00
-- finished_at: 2026-10-08T19:18:56.502126344+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:18:56.516530466+00:00
-- finished_at: 2026-10-08T19:18:56.709483082+00:00
-- elapsed: 192ms
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
-- created_at: 2026-10-08T19:18:56.715998105+00:00
-- finished_at: 2026-10-08T19:18:56.724350981+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidates" rename to "bronze_candidates__dbt_backup";
-- created_at: 2026-10-08T19:18:56.730256894+00:00
-- finished_at: 2026-10-08T19:18:56.737682298+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidates__dbt_tmp" rename to "bronze_candidates";
-- created_at: 2026-10-08T19:18:56.744342695+00:00
-- finished_at: 2026-10-08T19:18:56.750484498+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_candidates__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:18:56.770033655+00:00
-- finished_at: 2026-10-08T19:18:56.851252613+00:00
-- elapsed: 81ms
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
-- created_at: 2026-10-08T19:18:56.144562047+00:00
-- finished_at: 2026-10-08T19:18:56.879153812+00:00
-- elapsed: 734ms
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
-- created_at: 2026-10-08T19:18:56.884583462+00:00
-- finished_at: 2026-10-08T19:18:56.892160942+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_electorate" rename to "bronze_electorate__dbt_backup";
-- created_at: 2026-10-08T19:18:56.898038525+00:00
-- finished_at: 2026-10-08T19:18:56.905884563+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_electorate__dbt_tmp" rename to "bronze_electorate";
-- created_at: 2026-10-08T19:18:56.911514045+00:00
-- finished_at: 2026-10-08T19:18:56.921958613+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_electorate__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:18:56.938405171+00:00
-- finished_at: 2026-10-08T19:18:57.468696980+00:00
-- elapsed: 530ms
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
-- created_at: 2026-10-08T19:18:57.473908193+00:00
-- finished_at: 2026-10-08T19:18:57.497434821+00:00
-- elapsed: 23ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_votes_raw" rename to "bronze_candidate_votes_raw__dbt_backup";
-- created_at: 2026-10-08T19:18:57.501546916+00:00
-- finished_at: 2026-10-08T19:18:57.509384832+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."bronze_candidate_votes_raw__dbt_tmp" rename to "bronze_candidate_votes_raw";
-- created_at: 2026-10-08T19:18:57.517339803+00:00
-- finished_at: 2026-10-08T19:18:57.525689812+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."bronze_candidate_votes_raw__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:18:56.858491062+00:00
-- finished_at: 2026-10-08T19:18:57.732087479+00:00
-- elapsed: 873ms
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
-- created_at: 2026-10-08T19:18:57.736814581+00:00
-- finished_at: 2026-10-08T19:18:57.793653629+00:00
-- elapsed: 56ms
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
-- created_at: 2026-10-08T19:18:57.809060104+00:00
-- finished_at: 2026-10-08T19:18:57.849500603+00:00
-- elapsed: 40ms
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
-- created_at: 2026-10-08T19:18:57.855930368+00:00
-- finished_at: 2026-10-08T19:18:57.900658875+00:00
-- elapsed: 44ms
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
-- created_at: 2026-10-08T19:18:57.548939062+00:00
-- finished_at: 2026-10-08T19:18:57.900856169+00:00
-- elapsed: 351ms
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
-- created_at: 2026-10-08T19:18:57.913244226+00:00
-- finished_at: 2026-10-08T19:18:57.929093967+00:00
-- elapsed: 15ms
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
-- created_at: 2026-10-08T19:18:57.910329928+00:00
-- finished_at: 2026-10-08T19:18:57.959182117+00:00
-- elapsed: 48ms
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
-- created_at: 2026-10-08T19:18:57.947247419+00:00
-- finished_at: 2026-10-08T19:18:58.012991708+00:00
-- elapsed: 65ms
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
-- created_at: 2026-10-08T19:18:57.965313035+00:00
-- finished_at: 2026-10-08T19:18:58.024475308+00:00
-- elapsed: 59ms
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
-- created_at: 2026-10-08T19:18:58.022817732+00:00
-- finished_at: 2026-10-08T19:18:58.070961260+00:00
-- elapsed: 48ms
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
-- created_at: 2026-10-08T19:18:58.030512848+00:00
-- finished_at: 2026-10-08T19:18:58.085333772+00:00
-- elapsed: 54ms
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
-- created_at: 2026-10-08T19:18:58.081626139+00:00
-- finished_at: 2026-10-08T19:18:58.138677498+00:00
-- elapsed: 57ms
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
-- created_at: 2026-10-08T19:18:58.090997858+00:00
-- finished_at: 2026-10-08T19:18:58.151511213+00:00
-- elapsed: 60ms
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
-- created_at: 2026-10-08T19:18:58.151423675+00:00
-- finished_at: 2026-10-08T19:18:58.228794855+00:00
-- elapsed: 77ms
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
-- created_at: 2026-10-08T19:18:58.158158383+00:00
-- finished_at: 2026-10-08T19:18:58.235832257+00:00
-- elapsed: 77ms
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
-- created_at: 2026-10-08T19:18:58.240142492+00:00
-- finished_at: 2026-10-08T19:18:58.288769540+00:00
-- elapsed: 48ms
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
-- created_at: 2026-10-08T19:18:58.241524281+00:00
-- finished_at: 2026-10-08T19:18:58.305656351+00:00
-- elapsed: 64ms
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
-- created_at: 2026-10-08T19:18:58.298166873+00:00
-- finished_at: 2026-10-08T19:18:58.363322731+00:00
-- elapsed: 65ms
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
-- created_at: 2026-10-08T19:18:58.311662407+00:00
-- finished_at: 2026-10-08T19:18:58.377081090+00:00
-- elapsed: 65ms
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
-- created_at: 2026-10-08T19:18:58.374101332+00:00
-- finished_at: 2026-10-08T19:18:58.433148797+00:00
-- elapsed: 59ms
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
-- created_at: 2026-10-08T19:18:58.382354654+00:00
-- finished_at: 2026-10-08T19:18:58.456590679+00:00
-- elapsed: 74ms
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
-- created_at: 2026-10-08T19:18:58.459648561+00:00
-- finished_at: 2026-10-08T19:18:58.570411949+00:00
-- elapsed: 110ms
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
-- created_at: 2026-10-08T19:18:58.473705555+00:00
-- finished_at: 2026-10-08T19:18:58.590605599+00:00
-- elapsed: 116ms
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
-- created_at: 2026-10-08T19:18:58.595043359+00:00
-- finished_at: 2026-10-08T19:18:58.655802053+00:00
-- elapsed: 60ms
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
-- created_at: 2026-10-08T19:18:58.671082215+00:00
-- finished_at: 2026-10-08T19:18:58.770959005+00:00
-- elapsed: 99ms
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
-- created_at: 2026-10-08T19:18:58.797632569+00:00
-- finished_at: 2026-10-08T19:18:58.873105196+00:00
-- elapsed: 75ms
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
-- created_at: 2026-10-08T19:18:58.909495485+00:00
-- finished_at: 2026-10-08T19:18:59.044573683+00:00
-- elapsed: 135ms
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
-- created_at: 2026-10-08T19:18:59.097109651+00:00
-- finished_at: 2026-10-08T19:18:59.231881455+00:00
-- elapsed: 134ms
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
-- created_at: 2026-10-08T19:18:59.270548516+00:00
-- finished_at: 2026-10-08T19:18:59.443498397+00:00
-- elapsed: 172ms
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
-- created_at: 2026-10-08T19:18:59.482926816+00:00
-- finished_at: 2026-10-08T19:18:59.660837126+00:00
-- elapsed: 177ms
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
-- created_at: 2026-10-08T19:18:59.685350360+00:00
-- finished_at: 2026-10-08T19:18:59.773159749+00:00
-- elapsed: 87ms
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
-- created_at: 2026-10-08T19:18:59.791278880+00:00
-- finished_at: 2026-10-08T19:18:59.851696032+00:00
-- elapsed: 60ms
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
-- created_at: 2026-10-08T19:18:59.865131199+00:00
-- finished_at: 2026-10-08T19:18:59.916468606+00:00
-- elapsed: 51ms
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
-- created_at: 2026-10-08T19:18:59.927756821+00:00
-- finished_at: 2026-10-08T19:18:59.998887012+00:00
-- elapsed: 71ms
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
-- created_at: 2026-10-08T19:19:00.011410358+00:00
-- finished_at: 2026-10-08T19:19:00.062925643+00:00
-- elapsed: 51ms
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
-- created_at: 2026-10-08T19:19:00.078680968+00:00
-- finished_at: 2026-10-08T19:19:00.129887509+00:00
-- elapsed: 51ms
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
-- created_at: 2026-10-08T19:19:00.141962693+00:00
-- finished_at: 2026-10-08T19:19:00.215278357+00:00
-- elapsed: 73ms
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
-- created_at: 2026-10-08T19:19:00.237859261+00:00
-- finished_at: 2026-10-08T19:19:00.309348271+00:00
-- elapsed: 71ms
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
-- created_at: 2026-10-08T19:19:00.328624253+00:00
-- finished_at: 2026-10-08T19:19:00.396489832+00:00
-- elapsed: 67ms
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
-- created_at: 2026-10-08T19:19:00.414503266+00:00
-- finished_at: 2026-10-08T19:19:00.484415785+00:00
-- elapsed: 69ms
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
-- created_at: 2026-10-08T19:19:00.498808077+00:00
-- finished_at: 2026-10-08T19:19:00.568517872+00:00
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
-- created_at: 2026-10-08T19:19:00.581422455+00:00
-- finished_at: 2026-10-08T19:19:00.633055735+00:00
-- elapsed: 51ms
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
-- created_at: 2026-10-08T19:19:00.646193404+00:00
-- finished_at: 2026-10-08T19:19:00.707507387+00:00
-- elapsed: 61ms
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
-- created_at: 2026-10-08T19:19:00.718345964+00:00
-- finished_at: 2026-10-08T19:19:00.802169417+00:00
-- elapsed: 83ms
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
-- created_at: 2026-10-08T19:19:00.811776517+00:00
-- finished_at: 2026-10-08T19:19:00.864409808+00:00
-- elapsed: 52ms
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
-- created_at: 2026-10-08T19:19:00.875161105+00:00
-- finished_at: 2026-10-08T19:19:00.923048494+00:00
-- elapsed: 47ms
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
-- created_at: 2026-10-08T19:19:00.935850180+00:00
-- finished_at: 2026-10-08T19:19:00.995216773+00:00
-- elapsed: 59ms
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
-- created_at: 2026-10-08T19:19:01.034423591+00:00
-- finished_at: 2026-10-08T19:19:02.362692958+00:00
-- elapsed: 1.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
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
  ;
-- created_at: 2026-10-08T19:19:02.384503297+00:00
-- finished_at: 2026-10-08T19:19:02.392671023+00:00
-- elapsed: 8ms
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
-- created_at: 2026-10-08T19:19:02.401434041+00:00
-- finished_at: 2026-10-08T19:19:02.403495199+00:00
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
-- created_at: 2026-10-08T19:19:02.413865201+00:00
-- finished_at: 2026-10-08T19:19:02.416844531+00:00
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
-- created_at: 2026-10-08T19:19:02.428288935+00:00
-- finished_at: 2026-10-08T19:19:02.432476793+00:00
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
-- created_at: 2026-10-08T19:19:02.442031541+00:00
-- finished_at: 2026-10-08T19:19:02.445523565+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:19:02.454010193+00:00
-- finished_at: 2026-10-08T19:19:02.461566153+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T19:19:02.470966819+00:00
-- finished_at: 2026-10-08T19:19:02.535376186+00:00
-- elapsed: 64ms
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
-- created_at: 2026-10-08T19:18:58.616653547+00:00
-- finished_at: 2026-10-08T19:19:16.083740826+00:00
-- elapsed: 17.5s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.bronze_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.bronze_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
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


  ;
-- created_at: 2026-10-08T19:19:16.169208946+00:00
-- finished_at: 2026-10-08T19:19:16.236980854+00:00
-- elapsed: 67ms
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
-- created_at: 2026-10-08T19:19:16.248832134+00:00
-- finished_at: 2026-10-08T19:19:16.301334210+00:00
-- elapsed: 52ms
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
-- created_at: 2026-10-08T19:19:16.313031304+00:00
-- finished_at: 2026-10-08T19:19:16.378849685+00:00
-- elapsed: 65ms
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
-- created_at: 2026-10-08T19:19:16.391231332+00:00
-- finished_at: 2026-10-08T19:19:16.451623191+00:00
-- elapsed: 60ms
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
-- created_at: 2026-10-08T19:19:16.462575265+00:00
-- finished_at: 2026-10-08T19:19:16.516912746+00:00
-- elapsed: 54ms
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
-- created_at: 2026-10-08T19:19:16.527351660+00:00
-- finished_at: 2026-10-08T19:19:22.512067310+00:00
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
-- created_at: 2026-10-08T19:19:22.522561726+00:00
-- finished_at: 2026-10-08T19:19:22.613791084+00:00
-- elapsed: 91ms
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
-- created_at: 2026-10-08T19:19:22.626060883+00:00
-- finished_at: 2026-10-08T19:19:22.770033816+00:00
-- elapsed: 143ms
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
-- created_at: 2026-10-08T19:19:22.781886253+00:00
-- finished_at: 2026-10-08T19:19:24.109574243+00:00
-- elapsed: 1.3s
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
-- created_at: 2026-10-08T19:19:24.125673919+00:00
-- finished_at: 2026-10-08T19:19:25.927711572+00:00
-- elapsed: 1.8s
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
-- created_at: 2026-10-08T19:19:25.949244863+00:00
-- finished_at: 2026-10-08T19:19:27.315321316+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T19:19:27.333016838+00:00
-- finished_at: 2026-10-08T19:19:28.872268600+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T19:19:28.896376856+00:00
-- finished_at: 2026-10-08T19:19:30.378899681+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T19:19:30.405647385+00:00
-- finished_at: 2026-10-08T19:19:31.614951986+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T19:19:31.634468609+00:00
-- finished_at: 2026-10-08T19:19:33.622329587+00:00
-- elapsed: 2.0s
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
-- created_at: 2026-10-08T19:19:33.638783633+00:00
-- finished_at: 2026-10-08T19:19:34.260923936+00:00
-- elapsed: 622ms
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
-- created_at: 2026-10-08T19:19:34.276707987+00:00
-- finished_at: 2026-10-08T19:19:35.221100663+00:00
-- elapsed: 944ms
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
-- created_at: 2026-10-08T19:19:35.246279648+00:00
-- finished_at: 2026-10-08T19:19:36.573175514+00:00
-- elapsed: 1.3s
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
-- created_at: 2026-10-08T19:19:36.595397024+00:00
-- finished_at: 2026-10-08T19:19:37.623936141+00:00
-- elapsed: 1.0s
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
-- created_at: 2026-10-08T19:19:37.640693356+00:00
-- finished_at: 2026-10-08T19:19:38.807631494+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T19:19:38.828282306+00:00
-- finished_at: 2026-10-08T19:19:41.086873175+00:00
-- elapsed: 2.3s
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
-- created_at: 2026-10-08T19:19:41.110411918+00:00
-- finished_at: 2026-10-08T19:19:42.532538166+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T19:19:42.556734726+00:00
-- finished_at: 2026-10-08T19:19:43.628280180+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T19:19:43.650597704+00:00
-- finished_at: 2026-10-08T19:19:44.526675970+00:00
-- elapsed: 876ms
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
-- created_at: 2026-10-08T19:19:02.545985047+00:00
-- finished_at: 2026-10-08T19:20:28.800638678+00:00
-- elapsed: 1m 26s
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
-- created_at: 2026-10-08T19:20:28.826126591+00:00
-- finished_at: 2026-10-08T19:20:29.375223599+00:00
-- elapsed: 549ms
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
-- created_at: 2026-10-08T19:20:29.395157839+00:00
-- finished_at: 2026-10-08T19:20:29.406116561+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes_munzona" rename to "silver_candidate_votes_munzona__dbt_backup";
-- created_at: 2026-10-08T19:20:29.422996850+00:00
-- finished_at: 2026-10-08T19:20:29.432549551+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes_munzona__dbt_tmp" rename to "silver_candidate_votes_munzona";
-- created_at: 2026-10-08T19:20:29.451010518+00:00
-- finished_at: 2026-10-08T19:20:29.460681911+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."silver_candidate_votes_munzona__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:20:29.489837157+00:00
-- finished_at: 2026-10-08T19:20:29.511682130+00:00
-- elapsed: 21ms
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
-- created_at: 2026-10-08T19:20:29.531485038+00:00
-- finished_at: 2026-10-08T19:20:29.537359462+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:20:29.556184728+00:00
-- finished_at: 2026-10-08T19:20:29.561923782+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:20:29.594010691+00:00
-- finished_at: 2026-10-08T19:20:29.602000419+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T19:20:29.623627544+00:00
-- finished_at: 2026-10-08T19:20:29.628701239+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:20:29.649116033+00:00
-- finished_at: 2026-10-08T19:20:29.689224380+00:00
-- elapsed: 40ms
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
-- created_at: 2026-10-08T19:20:29.705047273+00:00
-- finished_at: 2026-10-08T19:20:29.709596955+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:20:29.726457439+00:00
-- finished_at: 2026-10-08T19:20:29.729884725+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:20:29.743795879+00:00
-- finished_at: 2026-10-08T19:20:29.748143031+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:20:29.764302121+00:00
-- finished_at: 2026-10-08T19:20:29.768580738+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:20:29.784845688+00:00
-- finished_at: 2026-10-08T19:20:29.787956178+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:20:29.816639056+00:00
-- finished_at: 2026-10-08T19:20:30.062662732+00:00
-- elapsed: 246ms
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
-- created_at: 2026-10-08T19:20:30.080463849+00:00
-- finished_at: 2026-10-08T19:20:33.469236501+00:00
-- elapsed: 3.4s
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
-- created_at: 2026-10-08T19:20:33.485589671+00:00
-- finished_at: 2026-10-08T19:20:33.767304056+00:00
-- elapsed: 281ms
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
-- created_at: 2026-10-08T19:20:33.772296081+00:00
-- finished_at: 2026-10-08T19:20:33.780900987+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_assets" rename to "silver_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T19:20:33.787245149+00:00
-- finished_at: 2026-10-08T19:20:33.795818141+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_assets__dbt_tmp" rename to "silver_candidate_assets";
-- created_at: 2026-10-08T19:20:33.802823770+00:00
-- finished_at: 2026-10-08T19:20:33.810474475+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."silver_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:20:33.825549444+00:00
-- finished_at: 2026-10-08T19:20:33.858978651+00:00
-- elapsed: 33ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_result_coverage__dbt_tmp" as (
    

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
-- created_at: 2026-10-08T19:20:33.867451553+00:00
-- finished_at: 2026-10-08T19:20:33.877019687+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_result_coverage" rename to "candidate_result_coverage__dbt_backup";
-- created_at: 2026-10-08T19:20:33.881666870+00:00
-- finished_at: 2026-10-08T19:20:33.890652180+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_result_coverage__dbt_tmp" rename to "candidate_result_coverage";
-- created_at: 2026-10-08T19:20:33.901069003+00:00
-- finished_at: 2026-10-08T19:20:33.908622075+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_result_coverage__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:20:33.921063699+00:00
-- finished_at: 2026-10-08T19:20:34.480476002+00:00
-- elapsed: 559ms
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
-- created_at: 2026-10-08T19:20:34.493681334+00:00
-- finished_at: 2026-10-08T19:20:34.504723849+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes" rename to "silver_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T19:20:34.523570183+00:00
-- finished_at: 2026-10-08T19:20:34.537643886+00:00
-- elapsed: 14ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."silver_candidate_votes__dbt_tmp" rename to "silver_candidate_votes";
-- created_at: 2026-10-08T19:20:34.558463893+00:00
-- finished_at: 2026-10-08T19:20:34.569832609+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."silver_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:20:34.598439358+00:00
-- finished_at: 2026-10-08T19:20:34.604480791+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T19:20:34.622916576+00:00
-- finished_at: 2026-10-08T19:20:34.683428371+00:00
-- elapsed: 60ms
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
-- created_at: 2026-10-08T19:20:34.707949300+00:00
-- finished_at: 2026-10-08T19:20:34.888504269+00:00
-- elapsed: 180ms
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
-- created_at: 2026-10-08T19:20:34.899680837+00:00
-- finished_at: 2026-10-08T19:20:35.040838979+00:00
-- elapsed: 141ms
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
-- created_at: 2026-10-08T19:20:35.048811252+00:00
-- finished_at: 2026-10-08T19:20:35.198181591+00:00
-- elapsed: 149ms
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
-- created_at: 2026-10-08T19:20:35.207961678+00:00
-- finished_at: 2026-10-08T19:20:35.376233822+00:00
-- elapsed: 168ms
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
-- created_at: 2026-10-08T19:20:35.387139863+00:00
-- finished_at: 2026-10-08T19:20:35.613170879+00:00
-- elapsed: 226ms
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
-- created_at: 2026-10-08T19:20:35.623466739+00:00
-- finished_at: 2026-10-08T19:20:35.801888323+00:00
-- elapsed: 178ms
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
-- created_at: 2026-10-08T19:20:35.812358748+00:00
-- finished_at: 2026-10-08T19:20:36.004939957+00:00
-- elapsed: 192ms
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
-- created_at: 2026-10-08T19:20:36.012794732+00:00
-- finished_at: 2026-10-08T19:20:36.223138247+00:00
-- elapsed: 210ms
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
-- created_at: 2026-10-08T19:20:36.232634118+00:00
-- finished_at: 2026-10-08T19:20:36.425860418+00:00
-- elapsed: 193ms
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
-- created_at: 2026-10-08T19:20:36.437914177+00:00
-- finished_at: 2026-10-08T19:20:36.722455214+00:00
-- elapsed: 284ms
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
-- created_at: 2026-10-08T19:20:36.762053522+00:00
-- finished_at: 2026-10-08T19:20:37.154087870+00:00
-- elapsed: 392ms
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
-- created_at: 2026-10-08T19:20:37.189611216+00:00
-- finished_at: 2026-10-08T19:20:37.412339669+00:00
-- elapsed: 222ms
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
-- created_at: 2026-10-08T19:20:37.426435554+00:00
-- finished_at: 2026-10-08T19:20:37.675809221+00:00
-- elapsed: 249ms
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
-- created_at: 2026-10-08T19:20:37.696726344+00:00
-- finished_at: 2026-10-08T19:20:37.893544041+00:00
-- elapsed: 196ms
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
-- created_at: 2026-10-08T19:20:37.905745419+00:00
-- finished_at: 2026-10-08T19:20:38.113428154+00:00
-- elapsed: 207ms
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
-- created_at: 2026-10-08T19:20:38.132770595+00:00
-- finished_at: 2026-10-08T19:20:38.420707972+00:00
-- elapsed: 287ms
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
-- created_at: 2026-10-08T19:20:38.435242609+00:00
-- finished_at: 2026-10-08T19:20:38.693462276+00:00
-- elapsed: 258ms
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
-- created_at: 2026-10-08T19:20:38.716491159+00:00
-- finished_at: 2026-10-08T19:20:39.048702391+00:00
-- elapsed: 332ms
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
-- created_at: 2026-10-08T19:20:39.090637672+00:00
-- finished_at: 2026-10-08T19:20:39.530668694+00:00
-- elapsed: 440ms
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
-- created_at: 2026-10-08T19:20:39.585149282+00:00
-- finished_at: 2026-10-08T19:20:39.896630744+00:00
-- elapsed: 311ms
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
-- created_at: 2026-10-08T19:20:39.915970880+00:00
-- finished_at: 2026-10-08T19:20:40.114965017+00:00
-- elapsed: 198ms
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
-- created_at: 2026-10-08T19:20:40.135352520+00:00
-- finished_at: 2026-10-08T19:20:40.466492450+00:00
-- elapsed: 331ms
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
-- created_at: 2026-10-08T19:20:40.485857419+00:00
-- finished_at: 2026-10-08T19:20:40.709545386+00:00
-- elapsed: 223ms
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
-- created_at: 2026-10-08T19:20:40.725809194+00:00
-- finished_at: 2026-10-08T19:20:40.905727409+00:00
-- elapsed: 179ms
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
-- created_at: 2026-10-08T19:20:40.918285534+00:00
-- finished_at: 2026-10-08T19:20:41.125241631+00:00
-- elapsed: 206ms
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
-- created_at: 2026-10-08T19:20:41.137359430+00:00
-- finished_at: 2026-10-08T19:20:41.306567368+00:00
-- elapsed: 169ms
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
-- created_at: 2026-10-08T19:20:41.317830612+00:00
-- finished_at: 2026-10-08T19:20:41.487971668+00:00
-- elapsed: 170ms
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
-- created_at: 2026-10-08T19:20:41.500531246+00:00
-- finished_at: 2026-10-08T19:20:41.676320337+00:00
-- elapsed: 175ms
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
-- created_at: 2026-10-08T19:20:41.716607902+00:00
-- finished_at: 2026-10-08T19:20:42.716263434+00:00
-- elapsed: 999ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090"
  
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
            using "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090"
            where (
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_tally_munzona" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file"
        from "fact_tally_munzona__dbt_tmp_02e106fc_b941_4f91_85cd_2b9671753090"
    )
  ;
-- created_at: 2026-10-08T19:20:42.801110779+00:00
-- finished_at: 2026-10-08T19:20:42.821944026+00:00
-- elapsed: 20ms
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
-- created_at: 2026-10-08T19:20:42.855095661+00:00
-- finished_at: 2026-10-08T19:20:42.903619473+00:00
-- elapsed: 48ms
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
-- created_at: 2026-10-08T19:20:42.938302655+00:00
-- finished_at: 2026-10-08T19:20:43.475955842+00:00
-- elapsed: 537ms
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
-- created_at: 2026-10-08T19:20:43.502164652+00:00
-- finished_at: 2026-10-08T19:20:44.117887058+00:00
-- elapsed: 615ms
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
-- created_at: 2026-10-08T19:20:44.136660964+00:00
-- finished_at: 2026-10-08T19:20:44.632521246+00:00
-- elapsed: 495ms
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
-- created_at: 2026-10-08T19:20:44.649053143+00:00
-- finished_at: 2026-10-08T19:20:45.061175829+00:00
-- elapsed: 412ms
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
-- created_at: 2026-10-08T19:20:45.086114224+00:00
-- finished_at: 2026-10-08T19:20:45.587964531+00:00
-- elapsed: 501ms
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
-- created_at: 2026-10-08T19:20:45.602390630+00:00
-- finished_at: 2026-10-08T19:20:46.219466584+00:00
-- elapsed: 617ms
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
-- created_at: 2026-10-08T19:20:46.231202564+00:00
-- finished_at: 2026-10-08T19:20:46.695319298+00:00
-- elapsed: 464ms
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
-- created_at: 2026-10-08T19:20:46.707069411+00:00
-- finished_at: 2026-10-08T19:20:47.270775025+00:00
-- elapsed: 563ms
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
-- created_at: 2026-10-08T19:20:47.279556804+00:00
-- finished_at: 2026-10-08T19:20:47.693375202+00:00
-- elapsed: 413ms
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
-- created_at: 2026-10-08T19:20:47.701607399+00:00
-- finished_at: 2026-10-08T19:20:48.160325018+00:00
-- elapsed: 458ms
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
-- created_at: 2026-10-08T19:20:48.168392681+00:00
-- finished_at: 2026-10-08T19:20:48.675195803+00:00
-- elapsed: 506ms
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
-- created_at: 2026-10-08T19:20:48.684491950+00:00
-- finished_at: 2026-10-08T19:20:49.104100041+00:00
-- elapsed: 419ms
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
-- created_at: 2026-10-08T19:20:49.112684025+00:00
-- finished_at: 2026-10-08T19:20:49.679291517+00:00
-- elapsed: 566ms
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
-- created_at: 2026-10-08T19:20:49.688819587+00:00
-- finished_at: 2026-10-08T19:20:50.177543696+00:00
-- elapsed: 488ms
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
-- created_at: 2026-10-08T19:20:50.191312422+00:00
-- finished_at: 2026-10-08T19:20:50.767766686+00:00
-- elapsed: 576ms
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
-- created_at: 2026-10-08T19:20:50.822246509+00:00
-- finished_at: 2026-10-08T19:21:08.148518708+00:00
-- elapsed: 17.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8"
  
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
            using "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8"
            where (
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."silver_party_votes_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations"
        from "silver_party_votes_munzona__dbt_tmp_67e9c10f_7ad5_4204_8c67_bc755c82f2f8"
    )
  ;
-- created_at: 2026-10-08T19:21:08.272153644+00:00
-- finished_at: 2026-10-08T19:21:09.431276097+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T19:21:09.435421320+00:00
-- finished_at: 2026-10-08T19:21:09.438176045+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T19:21:09.440912429+00:00
-- finished_at: 2026-10-08T19:21:09.443048825+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T19:21:09.449531792+00:00
-- finished_at: 2026-10-08T19:21:09.457060126+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election" rename to "dim_election__dbt_backup";
-- created_at: 2026-10-08T19:21:09.462570405+00:00
-- finished_at: 2026-10-08T19:21:09.470355272+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election__dbt_tmp" rename to "dim_election";
-- created_at: 2026-10-08T19:21:09.485016459+00:00
-- finished_at: 2026-10-08T19:21:09.498808512+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop table if exists "tse_analytics"."main"."dim_election__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:21:09.531913075+00:00
-- finished_at: 2026-10-08T19:21:09.543815410+00:00
-- elapsed: 11ms
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
-- created_at: 2026-10-08T19:21:09.575742204+00:00
-- finished_at: 2026-10-08T19:21:09.767869594+00:00
-- elapsed: 192ms
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
-- created_at: 2026-10-08T19:21:09.787176330+00:00
-- finished_at: 2026-10-08T19:21:09.903702864+00:00
-- elapsed: 116ms
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
-- created_at: 2026-10-08T19:21:09.922519096+00:00
-- finished_at: 2026-10-08T19:21:09.974139491+00:00
-- elapsed: 51ms
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
-- created_at: 2026-10-08T19:21:09.996255618+00:00
-- finished_at: 2026-10-08T19:21:10.024697242+00:00
-- elapsed: 28ms
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
-- created_at: 2026-10-08T19:21:10.057238822+00:00
-- finished_at: 2026-10-08T19:21:10.380162159+00:00
-- elapsed: 322ms
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
-- created_at: 2026-10-08T19:21:10.417274613+00:00
-- finished_at: 2026-10-08T19:21:10.420702995+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:21:10.436648150+00:00
-- finished_at: 2026-10-08T19:21:10.448616981+00:00
-- elapsed: 11ms
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
-- created_at: 2026-10-08T19:21:10.463858965+00:00
-- finished_at: 2026-10-08T19:21:10.474256123+00:00
-- elapsed: 10ms
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
-- created_at: 2026-10-08T19:21:10.487752009+00:00
-- finished_at: 2026-10-08T19:21:10.495238121+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T19:21:10.512210024+00:00
-- finished_at: 2026-10-08T19:21:10.515269635+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:21:10.528361800+00:00
-- finished_at: 2026-10-08T19:21:10.530865636+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T19:21:10.552401032+00:00
-- finished_at: 2026-10-08T19:21:10.564580431+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T19:21:10.581641551+00:00
-- finished_at: 2026-10-08T19:21:10.585779649+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:21:10.624841472+00:00
-- finished_at: 2026-10-08T19:21:10.632381149+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T19:21:10.684837018+00:00
-- finished_at: 2026-10-08T19:21:10.690138040+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:21:10.732569254+00:00
-- finished_at: 2026-10-08T19:21:10.771757155+00:00
-- elapsed: 39ms
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
-- created_at: 2026-10-08T19:21:10.818340877+00:00
-- finished_at: 2026-10-08T19:21:10.856232144+00:00
-- elapsed: 37ms
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
-- created_at: 2026-10-08T19:21:10.936990500+00:00
-- finished_at: 2026-10-08T19:21:12.566044640+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T19:21:12.585754369+00:00
-- finished_at: 2026-10-08T19:21:12.590586521+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:21:12.613760243+00:00
-- finished_at: 2026-10-08T19:21:12.621379449+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T19:21:12.652542049+00:00
-- finished_at: 2026-10-08T19:21:12.673522597+00:00
-- elapsed: 20ms
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
-- created_at: 2026-10-08T19:21:12.715500410+00:00
-- finished_at: 2026-10-08T19:21:14.400321759+00:00
-- elapsed: 1.7s
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
-- created_at: 2026-10-08T19:21:14.414950458+00:00
-- finished_at: 2026-10-08T19:21:14.448363086+00:00
-- elapsed: 33ms
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
-- created_at: 2026-10-08T19:21:14.474942884+00:00
-- finished_at: 2026-10-08T19:21:17.574288048+00:00
-- elapsed: 3.1s
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
-- created_at: 2026-10-08T19:21:17.590295045+00:00
-- finished_at: 2026-10-08T19:21:20.899624099+00:00
-- elapsed: 3.3s
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
-- created_at: 2026-10-08T19:21:20.910316449+00:00
-- finished_at: 2026-10-08T19:21:22.321603820+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T19:21:22.335616970+00:00
-- finished_at: 2026-10-08T19:21:23.187308459+00:00
-- elapsed: 851ms
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
-- created_at: 2026-10-08T19:21:23.217045199+00:00
-- finished_at: 2026-10-08T19:21:25.179870627+00:00
-- elapsed: 2.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_candidate__dbt_tmp_5d0c2a87_a6ce_4cf6_bfdb_ee0652289599"
  
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
            using "dim_candidate__dbt_tmp_5d0c2a87_a6ce_4cf6_bfdb_ee0652289599"
            where (
                
                    "dim_candidate__dbt_tmp_5d0c2a87_a6ce_4cf6_bfdb_ee0652289599".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_candidate__dbt_tmp_5d0c2a87_a6ce_4cf6_bfdb_ee0652289599".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_candidate__dbt_tmp_5d0c2a87_a6ce_4cf6_bfdb_ee0652289599".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_candidate__dbt_tmp_5d0c2a87_a6ce_4cf6_bfdb_ee0652289599".candidate_id = DBT_INCREMENTAL_TARGET.candidate_id
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_candidate" ("election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count"
        from "dim_candidate__dbt_tmp_5d0c2a87_a6ce_4cf6_bfdb_ee0652289599"
    )
  ;
-- created_at: 2026-10-08T19:21:25.199514779+00:00
-- finished_at: 2026-10-08T19:21:25.241743963+00:00
-- elapsed: 42ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_tmp" as (
    

with coverage as (

    select *
    from "tse_analytics"."main"."candidate_result_coverage"

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
-- created_at: 2026-10-08T19:21:25.246455552+00:00
-- finished_at: 2026-10-08T19:21:25.253496446+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps" rename to "candidate_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T19:21:25.257937856+00:00
-- finished_at: 2026-10-08T19:21:25.264494339+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_tmp" rename to "candidate_tally_coverage_gaps";
-- created_at: 2026-10-08T19:21:25.270407504+00:00
-- finished_at: 2026-10-08T19:21:25.277110946+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:21:25.289596643+00:00
-- finished_at: 2026-10-08T19:21:25.408072202+00:00
-- elapsed: 118ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

  
  create view "tse_analytics"."main"."candidate_tally_reconciliation__dbt_tmp" as (
    

with coverage as (
    select *
    from "tse_analytics"."main"."candidate_result_coverage"
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
-- created_at: 2026-10-08T19:21:25.413245298+00:00
-- finished_at: 2026-10-08T19:21:25.420636525+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation" rename to "candidate_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T19:21:25.424994682+00:00
-- finished_at: 2026-10-08T19:21:25.433018638+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation__dbt_tmp" rename to "candidate_tally_reconciliation";
-- created_at: 2026-10-08T19:21:25.439155425+00:00
-- finished_at: 2026-10-08T19:21:25.447845910+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:21:25.463190245+00:00
-- finished_at: 2026-10-08T19:21:25.467376909+00:00
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
-- created_at: 2026-10-08T19:21:25.472351681+00:00
-- finished_at: 2026-10-08T19:21:25.484609287+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T19:21:25.493936701+00:00
-- finished_at: 2026-10-08T19:21:25.632268656+00:00
-- elapsed: 138ms
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
-- created_at: 2026-10-08T19:21:25.639055434+00:00
-- finished_at: 2026-10-08T19:21:25.764926767+00:00
-- elapsed: 125ms
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
-- created_at: 2026-10-08T19:21:25.775074818+00:00
-- finished_at: 2026-10-08T19:21:25.905852090+00:00
-- elapsed: 130ms
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
-- created_at: 2026-10-08T19:21:25.917077298+00:00
-- finished_at: 2026-10-08T19:21:26.024864465+00:00
-- elapsed: 107ms
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
-- created_at: 2026-10-08T19:21:26.033700558+00:00
-- finished_at: 2026-10-08T19:21:26.275247504+00:00
-- elapsed: 241ms
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
-- created_at: 2026-10-08T19:21:26.284416250+00:00
-- finished_at: 2026-10-08T19:21:26.427422808+00:00
-- elapsed: 143ms
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
-- created_at: 2026-10-08T19:21:26.435705908+00:00
-- finished_at: 2026-10-08T19:21:26.571053549+00:00
-- elapsed: 135ms
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
-- created_at: 2026-10-08T19:21:26.579897561+00:00
-- finished_at: 2026-10-08T19:21:26.714241915+00:00
-- elapsed: 134ms
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
-- created_at: 2026-10-08T19:21:26.722630835+00:00
-- finished_at: 2026-10-08T19:21:26.865578483+00:00
-- elapsed: 142ms
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
-- created_at: 2026-10-08T19:21:26.877361454+00:00
-- finished_at: 2026-10-08T19:21:27.036723463+00:00
-- elapsed: 159ms
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
-- created_at: 2026-10-08T19:21:27.046559905+00:00
-- finished_at: 2026-10-08T19:21:27.188729479+00:00
-- elapsed: 142ms
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
-- created_at: 2026-10-08T19:19:44.558668860+00:00
-- finished_at: 2026-10-08T19:21:27.387385854+00:00
-- elapsed: 1m 43s
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
-- created_at: 2026-10-08T19:21:27.405126013+00:00
-- finished_at: 2026-10-08T19:21:27.411074990+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:21:27.418599399+00:00
-- finished_at: 2026-10-08T19:21:27.430757680+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T19:21:27.196706761+00:00
-- finished_at: 2026-10-08T19:21:29.744862385+00:00
-- elapsed: 2.5s
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
-- created_at: 2026-10-08T19:21:27.440765400+00:00
-- finished_at: 2026-10-08T19:21:29.771254027+00:00
-- elapsed: 2.3s
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
-- created_at: 2026-10-08T19:21:29.755683804+00:00
-- finished_at: 2026-10-08T19:21:30.032457272+00:00
-- elapsed: 276ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47"
  
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
            using "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47"
            where (
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_turnout" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at"
        from "fact_turnout__dbt_tmp_cdf1bc9b_8977_4953_8c6b_562789d1ec47"
    )
  ;
-- created_at: 2026-10-08T19:21:30.053694319+00:00
-- finished_at: 2026-10-08T19:21:30.056314697+00:00
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
-- created_at: 2026-10-08T19:21:30.062981467+00:00
-- finished_at: 2026-10-08T19:21:30.073803542+00:00
-- elapsed: 10ms
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
-- created_at: 2026-10-08T19:21:29.779996462+00:00
-- finished_at: 2026-10-08T19:21:31.441088132+00:00
-- elapsed: 1.7s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_party
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_party", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40"
  
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
            using "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40"
            where (
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40".party_number = DBT_INCREMENTAL_TARGET.party_number
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_party" ("election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id"
        from "dim_party__dbt_tmp_7cdb0152_fb3d_4baf_982f_baac63da0c40"
    )
  ;
-- created_at: 2026-10-08T19:21:30.082464885+00:00
-- finished_at: 2026-10-08T19:21:31.664486759+00:00
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
-- created_at: 2026-10-08T19:21:31.450928184+00:00
-- finished_at: 2026-10-08T19:21:32.999390288+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T19:21:33.005036346+00:00
-- finished_at: 2026-10-08T19:21:33.009477375+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:21:33.015336516+00:00
-- finished_at: 2026-10-08T19:21:33.016890203+00:00
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
-- created_at: 2026-10-08T19:21:31.669867668+00:00
-- finished_at: 2026-10-08T19:21:33.191162270+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T19:21:33.023291466+00:00
-- finished_at: 2026-10-08T19:21:33.387077431+00:00
-- elapsed: 363ms
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
-- created_at: 2026-10-08T19:21:33.393562455+00:00
-- finished_at: 2026-10-08T19:21:33.399749740+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T19:21:33.410466834+00:00
-- finished_at: 2026-10-08T19:21:33.412650310+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T19:21:33.419845880+00:00
-- finished_at: 2026-10-08T19:21:33.422899014+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:21:33.431076800+00:00
-- finished_at: 2026-10-08T19:21:33.432759881+00:00
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
-- created_at: 2026-10-08T19:21:33.197414931+00:00
-- finished_at: 2026-10-08T19:21:34.291688755+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T19:21:34.299466006+00:00
-- finished_at: 2026-10-08T19:21:35.089212130+00:00
-- elapsed: 789ms
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
-- created_at: 2026-10-08T19:21:35.097343229+00:00
-- finished_at: 2026-10-08T19:21:36.671667063+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T19:21:36.684359970+00:00
-- finished_at: 2026-10-08T19:21:39.359355755+00:00
-- elapsed: 2.7s
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
-- created_at: 2026-10-08T19:21:33.438899692+00:00
-- finished_at: 2026-10-08T19:21:42.580600489+00:00
-- elapsed: 9.1s
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
-- created_at: 2026-10-08T19:21:42.593304443+00:00
-- finished_at: 2026-10-08T19:21:42.670150386+00:00
-- elapsed: 76ms
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
-- created_at: 2026-10-08T19:21:39.379646637+00:00
-- finished_at: 2026-10-08T19:21:43.809283398+00:00
-- elapsed: 4.4s
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
-- created_at: 2026-10-08T19:21:43.829159289+00:00
-- finished_at: 2026-10-08T19:21:45.659614295+00:00
-- elapsed: 1.8s
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
-- created_at: 2026-10-08T19:21:45.677325874+00:00
-- finished_at: 2026-10-08T19:21:48.532710071+00:00
-- elapsed: 2.9s
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
-- created_at: 2026-10-08T19:21:48.544808929+00:00
-- finished_at: 2026-10-08T19:21:50.111273589+00:00
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
-- created_at: 2026-10-08T19:21:50.163370742+00:00
-- finished_at: 2026-10-08T19:21:52.430138614+00:00
-- elapsed: 2.3s
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
-- created_at: 2026-10-08T19:21:52.491257636+00:00
-- finished_at: 2026-10-08T19:21:53.936921968+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T19:21:53.971853235+00:00
-- finished_at: 2026-10-08T19:22:01.092313057+00:00
-- elapsed: 7.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3"
  
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
            using "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3"
            where (
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_party_votes" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file"
        from "fact_party_votes__dbt_tmp_312bb563_1e9c_41ce_8a62_a1d88e3b8ee3"
    )
  ;
-- created_at: 2026-10-08T19:21:42.683872738+00:00
-- finished_at: 2026-10-08T19:22:03.576791484+00:00
-- elapsed: 20.9s
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
-- created_at: 2026-10-08T19:22:03.606002490+00:00
-- finished_at: 2026-10-08T19:22:04.417869039+00:00
-- elapsed: 811ms
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
-- created_at: 2026-10-08T19:22:04.423314578+00:00
-- finished_at: 2026-10-08T19:22:04.443233006+00:00
-- elapsed: 19ms
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
-- created_at: 2026-10-08T19:22:04.453562008+00:00
-- finished_at: 2026-10-08T19:22:04.497047151+00:00
-- elapsed: 43ms
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
-- created_at: 2026-10-08T19:22:04.506633011+00:00
-- finished_at: 2026-10-08T19:22:04.555895602+00:00
-- elapsed: 49ms
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
-- created_at: 2026-10-08T19:22:01.264118489+00:00
-- finished_at: 2026-10-08T19:22:15.550301281+00:00
-- elapsed: 14.3s
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
-- created_at: 2026-10-08T19:22:15.567060094+00:00
-- finished_at: 2026-10-08T19:22:15.571281563+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:22:15.587032034+00:00
-- finished_at: 2026-10-08T19:22:15.591491544+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:22:15.606841998+00:00
-- finished_at: 2026-10-08T19:22:15.610736665+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:22:15.629211968+00:00
-- finished_at: 2026-10-08T19:22:15.727152153+00:00
-- elapsed: 97ms
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
-- created_at: 2026-10-08T19:22:15.744635685+00:00
-- finished_at: 2026-10-08T19:22:15.750762402+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T19:22:15.766291144+00:00
-- finished_at: 2026-10-08T19:22:15.771406054+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:22:15.789255071+00:00
-- finished_at: 2026-10-08T19:22:15.793322764+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:22:15.814288311+00:00
-- finished_at: 2026-10-08T19:22:15.818333861+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:22:15.835763513+00:00
-- finished_at: 2026-10-08T19:22:15.935229508+00:00
-- elapsed: 99ms
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
-- created_at: 2026-10-08T19:22:15.953169316+00:00
-- finished_at: 2026-10-08T19:22:15.957053108+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:22:15.973902354+00:00
-- finished_at: 2026-10-08T19:22:15.980953840+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T19:22:15.997306181+00:00
-- finished_at: 2026-10-08T19:22:16.002181784+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:22:16.024088973+00:00
-- finished_at: 2026-10-08T19:22:17.222315839+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T19:22:17.229736598+00:00
-- finished_at: 2026-10-08T19:22:17.240472138+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary" rename to "candidate_summary__dbt_backup";
-- created_at: 2026-10-08T19:22:17.247069074+00:00
-- finished_at: 2026-10-08T19:22:17.259285641+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary__dbt_tmp" rename to "candidate_summary";
-- created_at: 2026-10-08T19:22:17.272333758+00:00
-- finished_at: 2026-10-08T19:22:17.284532513+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:22:17.303915852+00:00
-- finished_at: 2026-10-08T19:22:17.356275678+00:00
-- elapsed: 52ms
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
-- created_at: 2026-10-08T19:22:17.365267558+00:00
-- finished_at: 2026-10-08T19:22:17.375282708+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary" rename to "candidate_vote_summary__dbt_backup";
-- created_at: 2026-10-08T19:22:17.382943442+00:00
-- finished_at: 2026-10-08T19:22:17.395569402+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary__dbt_tmp" rename to "candidate_vote_summary";
-- created_at: 2026-10-08T19:22:17.405586520+00:00
-- finished_at: 2026-10-08T19:22:17.414433225+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_vote_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:22:17.430322115+00:00
-- finished_at: 2026-10-08T19:22:17.433905466+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:22:17.448815149+00:00
-- finished_at: 2026-10-08T19:22:17.452117947+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:22:17.466927431+00:00
-- finished_at: 2026-10-08T19:22:19.602541960+00:00
-- elapsed: 2.1s
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
-- created_at: 2026-10-08T19:22:19.614234966+00:00
-- finished_at: 2026-10-08T19:22:22.286283214+00:00
-- elapsed: 2.7s
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
-- created_at: 2026-10-08T19:22:22.326777386+00:00
-- finished_at: 2026-10-08T19:22:22.333848084+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T19:22:22.373803970+00:00
-- finished_at: 2026-10-08T19:22:22.383339185+00:00
-- elapsed: 9ms
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
-- created_at: 2026-10-08T19:22:22.412623639+00:00
-- finished_at: 2026-10-08T19:22:22.419053589+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T19:22:22.446210724+00:00
-- finished_at: 2026-10-08T19:22:22.455864778+00:00
-- elapsed: 9ms
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
-- created_at: 2026-10-08T19:22:22.478889051+00:00
-- finished_at: 2026-10-08T19:22:22.893343410+00:00
-- elapsed: 414ms
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
-- created_at: 2026-10-08T19:22:22.917873497+00:00
-- finished_at: 2026-10-08T19:22:22.922465818+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:22:22.947841929+00:00
-- finished_at: 2026-10-08T19:22:22.953321524+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:22:22.973224088+00:00
-- finished_at: 2026-10-08T19:22:22.978616788+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T19:22:23.001894108+00:00
-- finished_at: 2026-10-08T19:22:23.006512716+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:22:23.027845371+00:00
-- finished_at: 2026-10-08T19:22:23.487343637+00:00
-- elapsed: 459ms
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
-- created_at: 2026-10-08T19:22:23.506282481+00:00
-- finished_at: 2026-10-08T19:23:04.056741047+00:00
-- elapsed: 40.6s
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
    from "tse_analytics"."main"."candidate_result_coverage"
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
-- created_at: 2026-10-08T19:23:04.069445441+00:00
-- finished_at: 2026-10-08T19:23:04.078037529+00:00
-- elapsed: 8ms
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
-- created_at: 2026-10-08T19:23:04.083137552+00:00
-- finished_at: 2026-10-08T19:23:04.091100840+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation" rename to "electoral_participation__dbt_backup";
-- created_at: 2026-10-08T19:23:04.096271999+00:00
-- finished_at: 2026-10-08T19:23:04.102547113+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation__dbt_tmp" rename to "electoral_participation";
-- created_at: 2026-10-08T19:23:04.107473252+00:00
-- finished_at: 2026-10-08T19:23:04.113814936+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."electoral_participation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:23:04.123911337+00:00
-- finished_at: 2026-10-08T19:23:04.137260180+00:00
-- elapsed: 13ms
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
-- created_at: 2026-10-08T19:23:04.141580260+00:00
-- finished_at: 2026-10-08T19:23:04.148283851+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps" rename to "party_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T19:23:04.153589997+00:00
-- finished_at: 2026-10-08T19:23:04.162037572+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps__dbt_tmp" rename to "party_tally_coverage_gaps";
-- created_at: 2026-10-08T19:23:04.167514235+00:00
-- finished_at: 2026-10-08T19:23:04.174314542+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:23:04.188225524+00:00
-- finished_at: 2026-10-08T19:23:04.225978050+00:00
-- elapsed: 37ms
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
-- created_at: 2026-10-08T19:23:04.230403764+00:00
-- finished_at: 2026-10-08T19:23:04.238188176+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance" rename to "party_performance__dbt_backup";
-- created_at: 2026-10-08T19:23:04.242255017+00:00
-- finished_at: 2026-10-08T19:23:04.251743063+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance__dbt_tmp" rename to "party_performance";
-- created_at: 2026-10-08T19:23:04.258530159+00:00
-- finished_at: 2026-10-08T19:23:04.268517134+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_performance__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:23:04.283728227+00:00
-- finished_at: 2026-10-08T19:23:04.307905672+00:00
-- elapsed: 24ms
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
-- created_at: 2026-10-08T19:23:04.312640517+00:00
-- finished_at: 2026-10-08T19:23:04.322442216+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation" rename to "party_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T19:23:04.326855730+00:00
-- finished_at: 2026-10-08T19:23:04.334156239+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation__dbt_tmp" rename to "party_tally_reconciliation";
-- created_at: 2026-10-08T19:23:04.340820247+00:00
-- finished_at: 2026-10-08T19:23:04.347857484+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T19:23:04.366564116+00:00
-- finished_at: 2026-10-08T19:23:05.860864866+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T19:23:05.875245286+00:00
-- finished_at: 2026-10-08T19:23:07.506900921+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T19:22:04.567952350+00:00
-- finished_at: 2026-10-08T19:23:08.044433125+00:00
-- elapsed: 1m 3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.silver_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.silver_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578"
  
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
            using "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578"
            where (
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."silver_electorate_municipality" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate"
        from "silver_electorate_municipality__dbt_tmp_74ba7fa1_61f8_438f_8130_5ffa5bd37578"
    )
  ;
-- created_at: 2026-10-08T19:23:07.518989004+00:00
-- finished_at: 2026-10-08T19:23:09.195410369+00:00
-- elapsed: 1.7s
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
-- created_at: 2026-10-08T19:23:09.237471690+00:00
-- finished_at: 2026-10-08T19:23:09.241709881+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:23:09.250277874+00:00
-- finished_at: 2026-10-08T19:23:09.267956809+00:00
-- elapsed: 17ms
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
-- created_at: 2026-10-08T19:23:09.279370973+00:00
-- finished_at: 2026-10-08T19:23:09.346528403+00:00
-- elapsed: 67ms
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
-- created_at: 2026-10-08T19:23:09.365929441+00:00
-- finished_at: 2026-10-08T19:23:09.473814007+00:00
-- elapsed: 107ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_geography
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_geography", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918"
  
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
            using "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918"
            where (
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_geography" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality"
        from "dim_geography__dbt_tmp_38e7e2ac_9793_45ae_8b1f_322e512c4918"
    )
  ;
-- created_at: 2026-10-08T19:23:09.508918574+00:00
-- finished_at: 2026-10-08T19:23:09.512150524+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:23:09.521751849+00:00
-- finished_at: 2026-10-08T19:23:09.542384736+00:00
-- elapsed: 20ms
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
-- created_at: 2026-10-08T19:23:09.558833508+00:00
-- finished_at: 2026-10-08T19:23:09.663270298+00:00
-- elapsed: 104ms
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
-- created_at: 2026-10-08T19:23:09.678496218+00:00
-- finished_at: 2026-10-08T19:23:09.841263046+00:00
-- elapsed: 162ms
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
-- created_at: 2026-10-08T19:23:09.861676485+00:00
-- finished_at: 2026-10-08T19:23:09.942542472+00:00
-- elapsed: 80ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2"
  
    as (
      

select *
from "tse_analytics"."main"."silver_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2"
            where (
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_electorate_municipality" ("election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate")
    (
        select "election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate"
        from "fact_electorate_municipality__dbt_tmp_3c51f20a_2c0b_4424_807e_f09936fe2bd2"
    )
  ;
-- created_at: 2026-10-08T19:23:09.965572270+00:00
-- finished_at: 2026-10-08T19:23:09.997240466+00:00
-- elapsed: 31ms
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
-- created_at: 2026-10-08T19:23:10.011650264+00:00
-- finished_at: 2026-10-08T19:23:10.051301586+00:00
-- elapsed: 39ms
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
-- created_at: 2026-10-08T19:23:10.067379927+00:00
-- finished_at: 2026-10-08T19:23:10.071034455+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:23:10.084344888+00:00
-- finished_at: 2026-10-08T19:23:10.086927399+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T19:23:10.101663749+00:00
-- finished_at: 2026-10-08T19:23:10.105780084+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T19:23:08.059625215+00:00
-- finished_at: 2026-10-08T19:23:10.106613885+00:00
-- elapsed: 2.0s
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
-- created_at: 2026-10-08T19:23:10.120797734+00:00
-- finished_at: 2026-10-08T19:23:10.144686792+00:00
-- elapsed: 23ms
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
-- created_at: 2026-10-08T19:23:10.121120367+00:00
-- finished_at: 2026-10-08T19:23:10.151758250+00:00
-- elapsed: 30ms
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
-- created_at: 2026-10-08T19:23:10.160161642+00:00
-- finished_at: 2026-10-08T19:23:10.163914134+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T19:23:10.167187112+00:00
-- finished_at: 2026-10-08T19:23:10.171490163+00:00
-- elapsed: 4ms
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
