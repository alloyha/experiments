-- created_at: 2026-10-08T12:23:21.818009920+00:00
-- finished_at: 2026-10-08T12:23:21.843966611+00:00
-- elapsed: 25ms
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: list_relations_in_parallel
SELECT table_catalog, table_schema, table_name, table_type FROM information_schema.tables WHERE table_schema = 'main' AND lower(table_catalog) = lower('tse_analytics');
-- created_at: 2026-10-08T12:23:22.493949807+00:00
-- finished_at: 2026-10-08T12:23:22.496648431+00:00
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
-- created_at: 2026-10-08T12:23:22.498727429+00:00
-- finished_at: 2026-10-08T12:23:22.502373559+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T12:23:22.503309551+00:00
-- finished_at: 2026-10-08T12:23:22.504119925+00:00
-- elapsed: 810us
-- outcome: success
-- dialect: duckdb
-- node_id: not available
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "connection_name": "", "dbt_version": "2.0.0", "profile_name": "tse_analytics", "target_name": "dev"} */

    
    
        create schema if not exists "tse_analytics"."main"
    ;
-- created_at: 2026-10-08T12:23:22.556085863+00:00
-- finished_at: 2026-10-08T12:23:22.627382542+00:00
-- elapsed: 71ms
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
-- created_at: 2026-10-08T12:23:22.557215875+00:00
-- finished_at: 2026-10-08T12:23:22.628122156+00:00
-- elapsed: 70ms
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
-- created_at: 2026-10-08T12:23:22.648334915+00:00
-- finished_at: 2026-10-08T12:23:22.700841751+00:00
-- elapsed: 52ms
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
-- created_at: 2026-10-08T12:23:22.651773728+00:00
-- finished_at: 2026-10-08T12:23:22.707480073+00:00
-- elapsed: 55ms
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
-- created_at: 2026-10-08T12:23:22.724684247+00:00
-- finished_at: 2026-10-08T12:23:22.777118987+00:00
-- elapsed: 52ms
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
-- created_at: 2026-10-08T12:23:22.729789361+00:00
-- finished_at: 2026-10-08T12:23:22.777964783+00:00
-- elapsed: 48ms
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
-- created_at: 2026-10-08T12:23:22.953601558+00:00
-- finished_at: 2026-10-08T12:23:23.072591088+00:00
-- elapsed: 118ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "seed.tse_analytics.election_calendar", "profile_name": "tse_analytics", "target_name": "dev"} */
truncate table "tse_analytics"."main"."election_calendar";
-- created_at: 2026-10-08T12:23:22.877702924+00:00
-- finished_at: 2026-10-08T12:23:23.126864874+00:00
-- elapsed: 249ms
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
-- created_at: 2026-10-08T12:23:23.173287934+00:00
-- finished_at: 2026-10-08T12:23:23.196763361+00:00
-- elapsed: 23ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes" rename to "fact_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T12:23:23.232063287+00:00
-- finished_at: 2026-10-08T12:23:23.254119757+00:00
-- elapsed: 22ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."fact_candidate_votes__dbt_tmp" rename to "fact_candidate_votes";
-- created_at: 2026-10-08T12:23:23.313237690+00:00
-- finished_at: 2026-10-08T12:23:23.339475243+00:00
-- elapsed: 26ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."fact_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:23:23.432774427+00:00
-- finished_at: 2026-10-08T12:23:23.541732981+00:00
-- elapsed: 108ms
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
-- created_at: 2026-10-08T12:23:23.233359755+00:00
-- finished_at: 2026-10-08T12:23:23.622524126+00:00
-- elapsed: 389ms
-- outcome: success
-- dialect: duckdb
-- node_id: seed.tse_analytics.election_calendar
-- query_id: not available
-- desc: add_query adapter call

          COPY "tse_analytics"."main"."election_calendar" FROM '/home/pingu/github/experiments/data/tse-election-data/tse_dbt/seeds/election_calendar.csv' (FORMAT CSV, HEADER TRUE, DELIMITER ',')
        ;
-- created_at: 2026-10-08T12:23:23.908520816+00:00
-- finished_at: 2026-10-08T12:23:25.748422987+00:00
-- elapsed: 1.8s
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
-- created_at: 2026-10-08T12:23:25.771942706+00:00
-- finished_at: 2026-10-08T12:23:25.787926626+00:00
-- elapsed: 15ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_assets" rename to "stg_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T12:23:25.804750504+00:00
-- finished_at: 2026-10-08T12:23:25.821211219+00:00
-- elapsed: 16ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_assets__dbt_tmp" rename to "stg_candidate_assets";
-- created_at: 2026-10-08T12:23:25.850821910+00:00
-- finished_at: 2026-10-08T12:23:25.875492049+00:00
-- elapsed: 24ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:23:25.939850554+00:00
-- finished_at: 2026-10-08T12:23:26.591831927+00:00
-- elapsed: 651ms
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
-- created_at: 2026-10-08T12:23:26.604984577+00:00
-- finished_at: 2026-10-08T12:23:26.623364798+00:00
-- elapsed: 18ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidates" rename to "stg_candidates__dbt_backup";
-- created_at: 2026-10-08T12:23:26.638169747+00:00
-- finished_at: 2026-10-08T12:23:26.649362582+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidates__dbt_tmp" rename to "stg_candidates";
-- created_at: 2026-10-08T12:23:26.664117361+00:00
-- finished_at: 2026-10-08T12:23:26.677946643+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidates
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidates", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidates__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:23:26.721431367+00:00
-- finished_at: 2026-10-08T12:23:26.997706315+00:00
-- elapsed: 276ms
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
-- created_at: 2026-10-08T12:23:23.673063097+00:00
-- finished_at: 2026-10-08T12:23:27.013165785+00:00
-- elapsed: 3.3s
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
-- created_at: 2026-10-08T12:23:27.022290661+00:00
-- finished_at: 2026-10-08T12:23:27.035792521+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_electorate" rename to "stg_electorate__dbt_backup";
-- created_at: 2026-10-08T12:23:27.044404298+00:00
-- finished_at: 2026-10-08T12:23:27.056530411+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_electorate__dbt_tmp" rename to "stg_electorate";
-- created_at: 2026-10-08T12:23:27.069513327+00:00
-- finished_at: 2026-10-08T12:23:27.097056592+00:00
-- elapsed: 27ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_electorate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_electorate", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_electorate__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:23:27.141563450+00:00
-- finished_at: 2026-10-08T12:23:28.266728261+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:23:28.272975821+00:00
-- finished_at: 2026-10-08T12:23:28.282133737+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_raw" rename to "stg_candidate_votes_raw__dbt_backup";
-- created_at: 2026-10-08T12:23:28.287149653+00:00
-- finished_at: 2026-10-08T12:23:28.296745157+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_raw__dbt_tmp" rename to "stg_candidate_votes_raw";
-- created_at: 2026-10-08T12:23:28.305141453+00:00
-- finished_at: 2026-10-08T12:23:28.315442200+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidate_votes_raw__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:23:27.007755986+00:00
-- finished_at: 2026-10-08T12:23:28.658136443+00:00
-- elapsed: 1.7s
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
-- created_at: 2026-10-08T12:23:28.664910200+00:00
-- finished_at: 2026-10-08T12:23:28.709739776+00:00
-- elapsed: 44ms
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
-- created_at: 2026-10-08T12:23:28.343283759+00:00
-- finished_at: 2026-10-08T12:23:28.776914082+00:00
-- elapsed: 433ms
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
-- created_at: 2026-10-08T12:23:28.786984057+00:00
-- finished_at: 2026-10-08T12:23:28.809728265+00:00
-- elapsed: 22ms
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
-- created_at: 2026-10-08T12:23:28.825607402+00:00
-- finished_at: 2026-10-08T12:23:29.016598314+00:00
-- elapsed: 190ms
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
-- created_at: 2026-10-08T12:23:28.723864879+00:00
-- finished_at: 2026-10-08T12:23:30.018860580+00:00
-- elapsed: 1.3s
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
-- created_at: 2026-10-08T12:23:29.026835804+00:00
-- finished_at: 2026-10-08T12:23:30.117642220+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:23:30.127964487+00:00
-- finished_at: 2026-10-08T12:23:31.097935654+00:00
-- elapsed: 969ms
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
-- created_at: 2026-10-08T12:23:30.022681413+00:00
-- finished_at: 2026-10-08T12:23:31.401626560+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T12:23:31.107032162+00:00
-- finished_at: 2026-10-08T12:23:32.498233602+00:00
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
-- created_at: 2026-10-08T12:23:31.405911917+00:00
-- finished_at: 2026-10-08T12:23:32.521172544+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:23:32.506448726+00:00
-- finished_at: 2026-10-08T12:23:32.619635653+00:00
-- elapsed: 113ms
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
-- created_at: 2026-10-08T12:23:32.628156047+00:00
-- finished_at: 2026-10-08T12:23:33.604245137+00:00
-- elapsed: 976ms
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
-- created_at: 2026-10-08T12:23:32.527228285+00:00
-- finished_at: 2026-10-08T12:23:33.877197985+00:00
-- elapsed: 1.3s
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
-- created_at: 2026-10-08T12:23:33.609540353+00:00
-- finished_at: 2026-10-08T12:23:33.999866577+00:00
-- elapsed: 390ms
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
-- created_at: 2026-10-08T12:23:34.005799917+00:00
-- finished_at: 2026-10-08T12:23:34.827843957+00:00
-- elapsed: 822ms
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
-- created_at: 2026-10-08T12:23:33.880094899+00:00
-- finished_at: 2026-10-08T12:23:34.960062369+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:23:34.832628188+00:00
-- finished_at: 2026-10-08T12:23:35.093323705+00:00
-- elapsed: 260ms
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
-- created_at: 2026-10-08T12:23:35.097683111+00:00
-- finished_at: 2026-10-08T12:23:36.689064178+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T12:23:34.962685291+00:00
-- finished_at: 2026-10-08T12:23:36.696090487+00:00
-- elapsed: 1.7s
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
-- created_at: 2026-10-08T12:23:36.692307878+00:00
-- finished_at: 2026-10-08T12:23:36.833985511+00:00
-- elapsed: 141ms
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
-- created_at: 2026-10-08T12:23:36.837338595+00:00
-- finished_at: 2026-10-08T12:23:37.785356154+00:00
-- elapsed: 948ms
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
-- created_at: 2026-10-08T12:23:36.698665726+00:00
-- finished_at: 2026-10-08T12:23:37.796783829+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:23:37.788966128+00:00
-- finished_at: 2026-10-08T12:23:37.920135394+00:00
-- elapsed: 131ms
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
-- created_at: 2026-10-08T12:23:37.924757132+00:00
-- finished_at: 2026-10-08T12:23:38.849031101+00:00
-- elapsed: 924ms
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
-- created_at: 2026-10-08T12:23:37.798749728+00:00
-- finished_at: 2026-10-08T12:23:39.212102979+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T12:23:38.852691961+00:00
-- finished_at: 2026-10-08T12:23:39.319540437+00:00
-- elapsed: 466ms
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
-- created_at: 2026-10-08T12:23:39.323410571+00:00
-- finished_at: 2026-10-08T12:23:40.205647658+00:00
-- elapsed: 882ms
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
-- created_at: 2026-10-08T12:23:39.214432151+00:00
-- finished_at: 2026-10-08T12:23:40.394064839+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T12:23:40.208808803+00:00
-- finished_at: 2026-10-08T12:23:40.491836021+00:00
-- elapsed: 283ms
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
-- created_at: 2026-10-08T12:23:40.495187829+00:00
-- finished_at: 2026-10-08T12:23:41.421004829+00:00
-- elapsed: 925ms
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
-- created_at: 2026-10-08T12:23:40.396156094+00:00
-- finished_at: 2026-10-08T12:23:41.554278987+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T12:23:41.424290317+00:00
-- finished_at: 2026-10-08T12:23:41.660257982+00:00
-- elapsed: 235ms
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
-- created_at: 2026-10-08T12:23:41.663504759+00:00
-- finished_at: 2026-10-08T12:23:42.502908303+00:00
-- elapsed: 839ms
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
-- created_at: 2026-10-08T12:23:41.556235870+00:00
-- finished_at: 2026-10-08T12:23:42.614266424+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:23:42.506705239+00:00
-- finished_at: 2026-10-08T12:23:42.725594530+00:00
-- elapsed: 218ms
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
-- created_at: 2026-10-08T12:23:42.729750785+00:00
-- finished_at: 2026-10-08T12:23:42.837809633+00:00
-- elapsed: 108ms
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
-- created_at: 2026-10-08T12:23:42.841927474+00:00
-- finished_at: 2026-10-08T12:23:42.951718735+00:00
-- elapsed: 109ms
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
-- created_at: 2026-10-08T12:23:42.955909442+00:00
-- finished_at: 2026-10-08T12:23:43.071342401+00:00
-- elapsed: 115ms
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
-- created_at: 2026-10-08T12:23:43.075803837+00:00
-- finished_at: 2026-10-08T12:23:43.205208859+00:00
-- elapsed: 129ms
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
-- created_at: 2026-10-08T12:23:43.210235949+00:00
-- finished_at: 2026-10-08T12:23:43.363451087+00:00
-- elapsed: 153ms
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
-- created_at: 2026-10-08T12:23:43.369071626+00:00
-- finished_at: 2026-10-08T12:23:43.500639562+00:00
-- elapsed: 131ms
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
-- created_at: 2026-10-08T12:23:43.506892742+00:00
-- finished_at: 2026-10-08T12:23:43.680680962+00:00
-- elapsed: 173ms
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
-- created_at: 2026-10-08T12:23:43.687528840+00:00
-- finished_at: 2026-10-08T12:23:43.852246391+00:00
-- elapsed: 164ms
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
-- created_at: 2026-10-08T12:23:43.858432877+00:00
-- finished_at: 2026-10-08T12:23:44.035733162+00:00
-- elapsed: 177ms
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
-- created_at: 2026-10-08T12:23:44.042110853+00:00
-- finished_at: 2026-10-08T12:23:44.213086696+00:00
-- elapsed: 170ms
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
-- created_at: 2026-10-08T12:23:44.219464308+00:00
-- finished_at: 2026-10-08T12:23:44.374585965+00:00
-- elapsed: 155ms
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
-- created_at: 2026-10-08T12:23:44.404791526+00:00
-- finished_at: 2026-10-08T12:23:47.540077029+00:00
-- elapsed: 3.1s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035"
  
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
            using "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035"
            where (
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."stg_tally_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "generated_at", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "main_sections", "aggregated_sections", "uninstalled_sections", "total_sections", "turnout", "voters_uninstalled_sections", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "last_totalization_at", "source_file"
        from "stg_tally_munzona__dbt_tmp_ec43700a_c39e_4cbe_b3f1_34c8df2b8035"
    )
  ;
-- created_at: 2026-10-08T12:23:47.647900352+00:00
-- finished_at: 2026-10-08T12:23:47.819981138+00:00
-- elapsed: 172ms
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
-- created_at: 2026-10-08T12:23:47.848490482+00:00
-- finished_at: 2026-10-08T12:23:48.006740569+00:00
-- elapsed: 158ms
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
-- created_at: 2026-10-08T12:23:48.024064274+00:00
-- finished_at: 2026-10-08T12:23:48.142112170+00:00
-- elapsed: 118ms
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
-- created_at: 2026-10-08T12:23:48.166649018+00:00
-- finished_at: 2026-10-08T12:23:48.274010299+00:00
-- elapsed: 107ms
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
-- created_at: 2026-10-08T12:23:48.282622194+00:00
-- finished_at: 2026-10-08T12:23:48.583035462+00:00
-- elapsed: 300ms
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
-- created_at: 2026-10-08T12:23:48.634874183+00:00
-- finished_at: 2026-10-08T12:23:54.415810025+00:00
-- elapsed: 5.8s
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
-- created_at: 2026-10-08T12:23:54.426452518+00:00
-- finished_at: 2026-10-08T12:23:55.131998011+00:00
-- elapsed: 705ms
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
-- created_at: 2026-10-08T12:23:55.143649171+00:00
-- finished_at: 2026-10-08T12:23:55.238248475+00:00
-- elapsed: 94ms
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
-- created_at: 2026-10-08T12:23:55.253457536+00:00
-- finished_at: 2026-10-08T12:23:55.362326856+00:00
-- elapsed: 108ms
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
-- created_at: 2026-10-08T12:23:55.389071835+00:00
-- finished_at: 2026-10-08T12:23:55.515647661+00:00
-- elapsed: 126ms
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
-- created_at: 2026-10-08T12:23:42.621677426+00:00
-- finished_at: 2026-10-08T12:24:03.978918727+00:00
-- elapsed: 21.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_raw
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_raw", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "stg_party_votes_raw__dbt_tmp_c75e9c16_c76f_41f5_9459_5e21094d2b40"
  
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
        from "stg_party_votes_raw__dbt_tmp_c75e9c16_c76f_41f5_9459_5e21094d2b40"
    )


  ;
-- created_at: 2026-10-08T12:24:04.041294315+00:00
-- finished_at: 2026-10-08T12:24:04.082981990+00:00
-- elapsed: 41ms
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
-- created_at: 2026-10-08T12:24:04.098389740+00:00
-- finished_at: 2026-10-08T12:24:04.102390597+00:00
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
-- created_at: 2026-10-08T12:24:04.119947794+00:00
-- finished_at: 2026-10-08T12:24:04.123369101+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T12:24:04.140765236+00:00
-- finished_at: 2026-10-08T12:24:04.144833001+00:00
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
-- created_at: 2026-10-08T12:24:04.158269508+00:00
-- finished_at: 2026-10-08T12:24:04.161287575+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T12:24:04.172986918+00:00
-- finished_at: 2026-10-08T12:24:04.175669151+00:00
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
-- created_at: 2026-10-08T12:24:04.190419437+00:00
-- finished_at: 2026-10-08T12:24:05.366927240+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T12:24:05.382691542+00:00
-- finished_at: 2026-10-08T12:24:06.478828387+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:24:06.545620794+00:00
-- finished_at: 2026-10-08T12:24:08.208329342+00:00
-- elapsed: 1.7s
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
-- created_at: 2026-10-08T12:24:08.227669201+00:00
-- finished_at: 2026-10-08T12:24:09.349493964+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:24:09.438939907+00:00
-- finished_at: 2026-10-08T12:24:10.589821531+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T12:24:10.625075419+00:00
-- finished_at: 2026-10-08T12:24:11.834024590+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T12:24:11.856096770+00:00
-- finished_at: 2026-10-08T12:24:12.602447803+00:00
-- elapsed: 746ms
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
-- created_at: 2026-10-08T12:24:12.630251058+00:00
-- finished_at: 2026-10-08T12:24:13.248866946+00:00
-- elapsed: 618ms
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
-- created_at: 2026-10-08T12:24:13.284686177+00:00
-- finished_at: 2026-10-08T12:24:13.760330095+00:00
-- elapsed: 475ms
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
-- created_at: 2026-10-08T12:24:13.773329914+00:00
-- finished_at: 2026-10-08T12:24:14.342197541+00:00
-- elapsed: 568ms
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
-- created_at: 2026-10-08T12:24:14.417402621+00:00
-- finished_at: 2026-10-08T12:24:14.988456676+00:00
-- elapsed: 571ms
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
-- created_at: 2026-10-08T12:24:15.033247228+00:00
-- finished_at: 2026-10-08T12:24:15.731674009+00:00
-- elapsed: 698ms
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
-- created_at: 2026-10-08T12:24:15.786234091+00:00
-- finished_at: 2026-10-08T12:24:16.480282425+00:00
-- elapsed: 694ms
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
-- created_at: 2026-10-08T12:24:16.519790928+00:00
-- finished_at: 2026-10-08T12:24:17.377385785+00:00
-- elapsed: 857ms
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
-- created_at: 2026-10-08T12:24:17.423657914+00:00
-- finished_at: 2026-10-08T12:24:18.817031233+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T12:23:55.556270373+00:00
-- finished_at: 2026-10-08T12:25:46.326327305+00:00
-- elapsed: 1m 51s
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
-- created_at: 2026-10-08T12:25:46.421819644+00:00
-- finished_at: 2026-10-08T12:25:47.473631855+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:25:47.495618151+00:00
-- finished_at: 2026-10-08T12:25:47.508428465+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_munzona" rename to "stg_candidate_votes_munzona__dbt_backup";
-- created_at: 2026-10-08T12:25:47.517216867+00:00
-- finished_at: 2026-10-08T12:25:47.526162102+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."stg_candidate_votes_munzona__dbt_tmp" rename to "stg_candidate_votes_munzona";
-- created_at: 2026-10-08T12:25:47.548258140+00:00
-- finished_at: 2026-10-08T12:25:47.566222076+00:00
-- elapsed: 17ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_candidate_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_candidate_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."stg_candidate_votes_munzona__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:25:47.606693280+00:00
-- finished_at: 2026-10-08T12:25:47.613910976+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T12:25:47.626542780+00:00
-- finished_at: 2026-10-08T12:25:47.628775099+00:00
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
-- created_at: 2026-10-08T12:25:47.644435219+00:00
-- finished_at: 2026-10-08T12:25:47.646636726+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:25:47.659491598+00:00
-- finished_at: 2026-10-08T12:25:47.662107059+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:25:47.676480886+00:00
-- finished_at: 2026-10-08T12:25:47.679057775+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:25:47.695352088+00:00
-- finished_at: 2026-10-08T12:25:47.837000999+00:00
-- elapsed: 141ms
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
-- created_at: 2026-10-08T12:25:47.852556012+00:00
-- finished_at: 2026-10-08T12:25:47.854622246+00:00
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
-- created_at: 2026-10-08T12:25:47.866596125+00:00
-- finished_at: 2026-10-08T12:25:47.870737161+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T12:25:47.885450122+00:00
-- finished_at: 2026-10-08T12:25:47.924230703+00:00
-- elapsed: 38ms
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
-- created_at: 2026-10-08T12:25:47.938502043+00:00
-- finished_at: 2026-10-08T12:25:47.940968686+00:00
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
-- created_at: 2026-10-08T12:25:47.953796137+00:00
-- finished_at: 2026-10-08T12:25:47.956308198+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:25:47.993882841+00:00
-- finished_at: 2026-10-08T12:25:48.438128307+00:00
-- elapsed: 444ms
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
-- created_at: 2026-10-08T12:25:48.451263118+00:00
-- finished_at: 2026-10-08T12:25:57.369158894+00:00
-- elapsed: 8.9s
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
-- created_at: 2026-10-08T12:25:57.699701598+00:00
-- finished_at: 2026-10-08T12:25:59.103688321+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T12:25:59.106330313+00:00
-- finished_at: 2026-10-08T12:25:59.889403446+00:00
-- elapsed: 783ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_assets" rename to "int_candidate_assets__dbt_backup";
-- created_at: 2026-10-08T12:25:59.891665402+00:00
-- finished_at: 2026-10-08T12:25:59.902196631+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_assets__dbt_tmp" rename to "int_candidate_assets";
-- created_at: 2026-10-08T12:25:59.905408294+00:00
-- finished_at: 2026-10-08T12:25:59.923038546+00:00
-- elapsed: 17ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_assets
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_assets", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."int_candidate_assets__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:25:59.932542308+00:00
-- finished_at: 2026-10-08T12:26:00.114849784+00:00
-- elapsed: 182ms
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
-- created_at: 2026-10-08T12:26:00.117948892+00:00
-- finished_at: 2026-10-08T12:26:00.123405816+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_result_coverage" rename to "int_candidate_result_coverage__dbt_backup";
-- created_at: 2026-10-08T12:26:00.125917310+00:00
-- finished_at: 2026-10-08T12:26:00.131458725+00:00
-- elapsed: 5ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_result_coverage__dbt_tmp" rename to "int_candidate_result_coverage";
-- created_at: 2026-10-08T12:26:00.134435908+00:00
-- finished_at: 2026-10-08T12:26:00.147537114+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_result_coverage
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_result_coverage", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."int_candidate_result_coverage__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:26:00.157221742+00:00
-- finished_at: 2026-10-08T12:26:00.368947800+00:00
-- elapsed: 211ms
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
-- created_at: 2026-10-08T12:26:00.372530571+00:00
-- finished_at: 2026-10-08T12:26:00.377369167+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_votes" rename to "int_candidate_votes__dbt_backup";
-- created_at: 2026-10-08T12:26:00.379439901+00:00
-- finished_at: 2026-10-08T12:26:00.384220961+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."int_candidate_votes__dbt_tmp" rename to "int_candidate_votes";
-- created_at: 2026-10-08T12:26:00.387746050+00:00
-- finished_at: 2026-10-08T12:26:00.392382728+00:00
-- elapsed: 4ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_candidate_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_candidate_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."int_candidate_votes__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:26:00.477717163+00:00
-- finished_at: 2026-10-08T12:26:00.504076414+00:00
-- elapsed: 26ms
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
-- created_at: 2026-10-08T12:26:00.567092816+00:00
-- finished_at: 2026-10-08T12:26:00.718528921+00:00
-- elapsed: 151ms
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
-- created_at: 2026-10-08T12:26:00.739709028+00:00
-- finished_at: 2026-10-08T12:26:01.095548800+00:00
-- elapsed: 355ms
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
-- created_at: 2026-10-08T12:26:01.099653737+00:00
-- finished_at: 2026-10-08T12:26:01.438662072+00:00
-- elapsed: 339ms
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
-- created_at: 2026-10-08T12:26:01.442601547+00:00
-- finished_at: 2026-10-08T12:26:01.687779809+00:00
-- elapsed: 245ms
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
-- created_at: 2026-10-08T12:26:01.693399572+00:00
-- finished_at: 2026-10-08T12:26:01.915019485+00:00
-- elapsed: 221ms
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
-- created_at: 2026-10-08T12:26:01.921090153+00:00
-- finished_at: 2026-10-08T12:26:02.239623966+00:00
-- elapsed: 318ms
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
-- created_at: 2026-10-08T12:26:02.243220894+00:00
-- finished_at: 2026-10-08T12:26:02.680982253+00:00
-- elapsed: 437ms
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
-- created_at: 2026-10-08T12:26:02.685534017+00:00
-- finished_at: 2026-10-08T12:26:02.944169394+00:00
-- elapsed: 258ms
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
-- created_at: 2026-10-08T12:26:02.948164824+00:00
-- finished_at: 2026-10-08T12:26:03.307351202+00:00
-- elapsed: 359ms
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
-- created_at: 2026-10-08T12:26:03.311272392+00:00
-- finished_at: 2026-10-08T12:26:03.550128831+00:00
-- elapsed: 238ms
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
-- created_at: 2026-10-08T12:26:03.555193337+00:00
-- finished_at: 2026-10-08T12:26:03.733337874+00:00
-- elapsed: 178ms
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
-- created_at: 2026-10-08T12:26:03.737889160+00:00
-- finished_at: 2026-10-08T12:26:04.064982647+00:00
-- elapsed: 327ms
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
-- created_at: 2026-10-08T12:26:04.069900095+00:00
-- finished_at: 2026-10-08T12:26:04.357778581+00:00
-- elapsed: 287ms
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
-- created_at: 2026-10-08T12:26:04.364318195+00:00
-- finished_at: 2026-10-08T12:26:04.567073326+00:00
-- elapsed: 202ms
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
-- created_at: 2026-10-08T12:26:04.571539227+00:00
-- finished_at: 2026-10-08T12:26:04.788904684+00:00
-- elapsed: 217ms
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
-- created_at: 2026-10-08T12:26:04.794010689+00:00
-- finished_at: 2026-10-08T12:26:04.992146207+00:00
-- elapsed: 198ms
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
-- created_at: 2026-10-08T12:26:04.996723616+00:00
-- finished_at: 2026-10-08T12:26:05.231871934+00:00
-- elapsed: 235ms
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
-- created_at: 2026-10-08T12:26:05.235711696+00:00
-- finished_at: 2026-10-08T12:26:05.460180334+00:00
-- elapsed: 224ms
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
-- created_at: 2026-10-08T12:26:05.464993730+00:00
-- finished_at: 2026-10-08T12:26:05.810377517+00:00
-- elapsed: 345ms
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
-- created_at: 2026-10-08T12:26:05.816541182+00:00
-- finished_at: 2026-10-08T12:26:06.116053041+00:00
-- elapsed: 299ms
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
-- created_at: 2026-10-08T12:26:06.121372930+00:00
-- finished_at: 2026-10-08T12:26:06.378658655+00:00
-- elapsed: 257ms
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
-- created_at: 2026-10-08T12:26:06.383247124+00:00
-- finished_at: 2026-10-08T12:26:06.788426044+00:00
-- elapsed: 405ms
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
-- created_at: 2026-10-08T12:26:06.797882610+00:00
-- finished_at: 2026-10-08T12:26:07.039448352+00:00
-- elapsed: 241ms
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
-- created_at: 2026-10-08T12:26:07.043994420+00:00
-- finished_at: 2026-10-08T12:26:07.307402513+00:00
-- elapsed: 263ms
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
-- created_at: 2026-10-08T12:26:07.313824260+00:00
-- finished_at: 2026-10-08T12:26:07.645212901+00:00
-- elapsed: 331ms
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
-- created_at: 2026-10-08T12:24:18.870029261+00:00
-- finished_at: 2026-10-08T12:26:07.669273875+00:00
-- elapsed: 1m 49s
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
-- created_at: 2026-10-08T12:26:07.649499422+00:00
-- finished_at: 2026-10-08T12:26:08.748805152+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:26:07.764251968+00:00
-- finished_at: 2026-10-08T12:26:08.944731098+00:00
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
-- created_at: 2026-10-08T12:26:08.946665941+00:00
-- finished_at: 2026-10-08T12:26:08.964661255+00:00
-- elapsed: 17ms
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
-- created_at: 2026-10-08T12:26:08.966655664+00:00
-- finished_at: 2026-10-08T12:26:08.967760853+00:00
-- elapsed: 1ms
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
-- created_at: 2026-10-08T12:26:08.970154005+00:00
-- finished_at: 2026-10-08T12:26:09.104869470+00:00
-- elapsed: 134ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election" rename to "dim_election__dbt_backup";
-- created_at: 2026-10-08T12:26:09.107211404+00:00
-- finished_at: 2026-10-08T12:26:09.284175800+00:00
-- elapsed: 176ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */
alter table "tse_analytics"."main"."dim_election__dbt_tmp" rename to "dim_election";
-- created_at: 2026-10-08T12:26:08.752539170+00:00
-- finished_at: 2026-10-08T12:26:09.330322569+00:00
-- elapsed: 577ms
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
-- created_at: 2026-10-08T12:26:09.298087416+00:00
-- finished_at: 2026-10-08T12:26:09.403474179+00:00
-- elapsed: 105ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_election
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_election", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop table if exists "tse_analytics"."main"."dim_election__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:26:09.415607229+00:00
-- finished_at: 2026-10-08T12:26:09.434595909+00:00
-- elapsed: 18ms
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
-- created_at: 2026-10-08T12:26:09.438772452+00:00
-- finished_at: 2026-10-08T12:26:09.443950624+00:00
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
-- created_at: 2026-10-08T12:26:09.334463372+00:00
-- finished_at: 2026-10-08T12:26:09.636359397+00:00
-- elapsed: 301ms
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
-- created_at: 2026-10-08T12:26:09.639192315+00:00
-- finished_at: 2026-10-08T12:26:10.859840125+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T12:26:09.449912558+00:00
-- finished_at: 2026-10-08T12:26:11.247193770+00:00
-- elapsed: 1.8s
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
-- created_at: 2026-10-08T12:26:10.904819421+00:00
-- finished_at: 2026-10-08T12:26:11.939184964+00:00
-- elapsed: 1.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_tally_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_tally_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f"
  
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
            using "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f"
            where (
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_tally_munzona" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "turnout", "abstentions", "total_votes", "competing_votes", "valid_votes", "nominal_valid_votes", "total_legend_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_valid_votes", "annulled_votes", "nominal_annulled_votes", "legend_annulled_votes", "annulled_subjudice_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "blank_votes", "total_null_votes", "null_votes", "technical_null_votes", "separately_counted_annulled_votes", "generated_at", "last_totalization_at", "source_file"
        from "fact_tally_munzona__dbt_tmp_bd16f5f4_b476_4998_8d6b_57e652d3742f"
    )
  ;
-- created_at: 2026-10-08T12:26:12.036855362+00:00
-- finished_at: 2026-10-08T12:26:12.577404670+00:00
-- elapsed: 540ms
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
-- created_at: 2026-10-08T12:26:12.599244114+00:00
-- finished_at: 2026-10-08T12:26:12.610064396+00:00
-- elapsed: 10ms
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
-- created_at: 2026-10-08T12:26:11.252715121+00:00
-- finished_at: 2026-10-08T12:26:13.973534580+00:00
-- elapsed: 2.7s
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
-- created_at: 2026-10-08T12:26:12.617735114+00:00
-- finished_at: 2026-10-08T12:26:14.173955751+00:00
-- elapsed: 1.6s
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
-- created_at: 2026-10-08T12:26:14.180599200+00:00
-- finished_at: 2026-10-08T12:26:18.263079837+00:00
-- elapsed: 4.1s
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
-- created_at: 2026-10-08T12:26:13.986382464+00:00
-- finished_at: 2026-10-08T12:26:18.842076492+00:00
-- elapsed: 4.9s
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
-- created_at: 2026-10-08T12:26:18.850172923+00:00
-- finished_at: 2026-10-08T12:26:21.084499018+00:00
-- elapsed: 2.2s
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
-- created_at: 2026-10-08T12:26:21.100869172+00:00
-- finished_at: 2026-10-08T12:26:26.444269529+00:00
-- elapsed: 5.3s
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
-- created_at: 2026-10-08T12:26:26.500324249+00:00
-- finished_at: 2026-10-08T12:26:32.732638758+00:00
-- elapsed: 6.2s
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
-- created_at: 2026-10-08T12:26:32.774197266+00:00
-- finished_at: 2026-10-08T12:26:37.276254826+00:00
-- elapsed: 4.5s
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
-- created_at: 2026-10-08T12:26:37.328453250+00:00
-- finished_at: 2026-10-08T12:26:41.418076398+00:00
-- elapsed: 4.1s
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
-- created_at: 2026-10-08T12:26:41.474492159+00:00
-- finished_at: 2026-10-08T12:26:45.813355693+00:00
-- elapsed: 4.3s
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
-- created_at: 2026-10-08T12:26:45.876465885+00:00
-- finished_at: 2026-10-08T12:26:50.935797702+00:00
-- elapsed: 5.1s
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
-- created_at: 2026-10-08T12:26:50.988969175+00:00
-- finished_at: 2026-10-08T12:26:55.364407454+00:00
-- elapsed: 4.4s
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
-- created_at: 2026-10-08T12:26:55.404387184+00:00
-- finished_at: 2026-10-08T12:26:59.158396935+00:00
-- elapsed: 3.8s
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
-- created_at: 2026-10-08T12:26:59.193985888+00:00
-- finished_at: 2026-10-08T12:27:02.912667910+00:00
-- elapsed: 3.7s
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
-- created_at: 2026-10-08T12:27:02.922297086+00:00
-- finished_at: 2026-10-08T12:27:05.651086724+00:00
-- elapsed: 2.7s
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
-- created_at: 2026-10-08T12:27:05.660236225+00:00
-- finished_at: 2026-10-08T12:27:07.979974559+00:00
-- elapsed: 2.3s
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
-- created_at: 2026-10-08T12:27:08.012215977+00:00
-- finished_at: 2026-10-08T12:27:19.187512504+00:00
-- elapsed: 11.2s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.stg_party_votes_munzona
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.stg_party_votes_munzona", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86"
  
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
            using "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86"
            where (
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."stg_party_votes_munzona" ("election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations")
    (
        select "election_year", "election_type", "election_scope", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "party", "party_name", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file", "source_row_count", "source_party_group_types", "source_coalitions", "source_federations"
        from "stg_party_votes_munzona__dbt_tmp_6257b271_87b5_4aad_b078_0df72e8eaa86"
    )
  ;
-- created_at: 2026-10-08T12:27:19.211989824+00:00
-- finished_at: 2026-10-08T12:27:19.216919994+00:00
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
-- created_at: 2026-10-08T12:27:19.242004045+00:00
-- finished_at: 2026-10-08T12:27:19.250081126+00:00
-- elapsed: 8ms
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
-- created_at: 2026-10-08T12:27:19.262739084+00:00
-- finished_at: 2026-10-08T12:27:19.273621412+00:00
-- elapsed: 10ms
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
-- created_at: 2026-10-08T12:27:19.301388225+00:00
-- finished_at: 2026-10-08T12:27:19.307920941+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T12:27:19.336845869+00:00
-- finished_at: 2026-10-08T12:27:19.345381707+00:00
-- elapsed: 8ms
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
-- created_at: 2026-10-08T12:27:19.375417781+00:00
-- finished_at: 2026-10-08T12:27:20.110543394+00:00
-- elapsed: 735ms
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
-- created_at: 2026-10-08T12:27:20.115637160+00:00
-- finished_at: 2026-10-08T12:27:20.127775467+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T12:27:20.136816499+00:00
-- finished_at: 2026-10-08T12:27:21.038294209+00:00
-- elapsed: 901ms
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
-- created_at: 2026-10-08T12:27:21.047335907+00:00
-- finished_at: 2026-10-08T12:27:22.497152403+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T12:27:22.513597774+00:00
-- finished_at: 2026-10-08T12:27:24.710667269+00:00
-- elapsed: 2.2s
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
-- created_at: 2026-10-08T12:27:24.723061854+00:00
-- finished_at: 2026-10-08T12:27:26.562194709+00:00
-- elapsed: 1.8s
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
-- created_at: 2026-10-08T12:27:26.682788586+00:00
-- finished_at: 2026-10-08T12:27:29.665270050+00:00
-- elapsed: 3.0s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_candidate
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_candidate", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_candidate__dbt_tmp_61e9e922_a0f9_49ac_b63d_8b111ca52aaa"
  
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
            using "dim_candidate__dbt_tmp_61e9e922_a0f9_49ac_b63d_8b111ca52aaa"
            where (
                
                    "dim_candidate__dbt_tmp_61e9e922_a0f9_49ac_b63d_8b111ca52aaa".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_candidate__dbt_tmp_61e9e922_a0f9_49ac_b63d_8b111ca52aaa".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_candidate__dbt_tmp_61e9e922_a0f9_49ac_b63d_8b111ca52aaa".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_candidate__dbt_tmp_61e9e922_a0f9_49ac_b63d_8b111ca52aaa".candidate_id = DBT_INCREMENTAL_TARGET.candidate_id
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_candidate" ("election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "election_description", "round_number", "electoral_unit", "office_scope", "candidate_id", "uf", "office_code", "office", "candidate_number", "candidate_name", "ballot_name", "party_number", "party", "party_name", "candidacy_status", "gender", "education", "occupation", "race_color", "declared_assets_value", "declared_assets_count"
        from "dim_candidate__dbt_tmp_61e9e922_a0f9_49ac_b63d_8b111ca52aaa"
    )
  ;
-- created_at: 2026-10-08T12:27:29.701915980+00:00
-- finished_at: 2026-10-08T12:27:29.795468734+00:00
-- elapsed: 93ms
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
-- created_at: 2026-10-08T12:27:29.823498989+00:00
-- finished_at: 2026-10-08T12:27:29.854936168+00:00
-- elapsed: 31ms
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
-- created_at: 2026-10-08T12:27:29.904579362+00:00
-- finished_at: 2026-10-08T12:27:31.444071919+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T12:27:31.489611388+00:00
-- finished_at: 2026-10-08T12:27:32.438379512+00:00
-- elapsed: 948ms
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
-- created_at: 2026-10-08T12:27:32.732828528+00:00
-- finished_at: 2026-10-08T12:27:33.483659349+00:00
-- elapsed: 750ms
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
-- created_at: 2026-10-08T12:27:33.551535529+00:00
-- finished_at: 2026-10-08T12:27:33.568798473+00:00
-- elapsed: 17ms
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
-- created_at: 2026-10-08T12:27:33.653999766+00:00
-- finished_at: 2026-10-08T12:27:33.669801656+00:00
-- elapsed: 15ms
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
-- created_at: 2026-10-08T12:27:33.712319783+00:00
-- finished_at: 2026-10-08T12:27:34.111293069+00:00
-- elapsed: 398ms
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
-- created_at: 2026-10-08T12:27:34.141536890+00:00
-- finished_at: 2026-10-08T12:27:34.158364164+00:00
-- elapsed: 16ms
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
-- created_at: 2026-10-08T12:27:34.200025955+00:00
-- finished_at: 2026-10-08T12:27:34.205200169+00:00
-- elapsed: 5ms
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
-- created_at: 2026-10-08T12:27:34.248964777+00:00
-- finished_at: 2026-10-08T12:27:34.261945564+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T12:27:34.325728454+00:00
-- finished_at: 2026-10-08T12:27:34.333335837+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T12:27:34.376835967+00:00
-- finished_at: 2026-10-08T12:27:34.390033014+00:00
-- elapsed: 13ms
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
-- created_at: 2026-10-08T12:27:34.418903718+00:00
-- finished_at: 2026-10-08T12:27:34.425722673+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T12:27:34.459508143+00:00
-- finished_at: 2026-10-08T12:27:38.130145074+00:00
-- elapsed: 3.7s
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
-- created_at: 2026-10-08T12:27:38.148428721+00:00
-- finished_at: 2026-10-08T12:27:38.299360062+00:00
-- elapsed: 150ms
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
-- created_at: 2026-10-08T12:27:38.320127556+00:00
-- finished_at: 2026-10-08T12:27:38.323396549+00:00
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
-- created_at: 2026-10-08T12:27:38.340139724+00:00
-- finished_at: 2026-10-08T12:27:38.368745555+00:00
-- elapsed: 28ms
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
-- created_at: 2026-10-08T12:27:38.384761107+00:00
-- finished_at: 2026-10-08T12:27:38.389014526+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T12:27:38.408011007+00:00
-- finished_at: 2026-10-08T12:27:39.905845156+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T12:27:39.929283023+00:00
-- finished_at: 2026-10-08T12:27:39.933727858+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T12:27:40.003266183+00:00
-- finished_at: 2026-10-08T12:27:40.006981862+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T12:27:40.206774521+00:00
-- finished_at: 2026-10-08T12:27:50.670303974+00:00
-- elapsed: 10.5s
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
-- created_at: 2026-10-08T12:27:50.674784794+00:00
-- finished_at: 2026-10-08T12:27:50.719680336+00:00
-- elapsed: 44ms
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
-- created_at: 2026-10-08T12:26:18.275653893+00:00
-- finished_at: 2026-10-08T12:27:56.246890933+00:00
-- elapsed: 1m 38s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.int_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.int_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "int_electorate_municipality__dbt_tmp_eff815d8_97fa_4b96_a1dd_3f7c172edebd"
  
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
            using "int_electorate_municipality__dbt_tmp_eff815d8_97fa_4b96_a1dd_3f7c172edebd"
            where (
                
                    "int_electorate_municipality__dbt_tmp_eff815d8_97fa_4b96_a1dd_3f7c172edebd".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "int_electorate_municipality__dbt_tmp_eff815d8_97fa_4b96_a1dd_3f7c172edebd".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "int_electorate_municipality__dbt_tmp_eff815d8_97fa_4b96_a1dd_3f7c172edebd".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "int_electorate_municipality__dbt_tmp_eff815d8_97fa_4b96_a1dd_3f7c172edebd".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."int_electorate_municipality" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality", "electorate"
        from "int_electorate_municipality__dbt_tmp_eff815d8_97fa_4b96_a1dd_3f7c172edebd"
    )
  ;
-- created_at: 2026-10-08T12:27:56.274733172+00:00
-- finished_at: 2026-10-08T12:27:56.278391781+00:00
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
-- created_at: 2026-10-08T12:27:56.312213704+00:00
-- finished_at: 2026-10-08T12:27:56.330848361+00:00
-- elapsed: 18ms
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
-- created_at: 2026-10-08T12:27:56.346703508+00:00
-- finished_at: 2026-10-08T12:27:56.387500618+00:00
-- elapsed: 40ms
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
-- created_at: 2026-10-08T12:27:56.408943899+00:00
-- finished_at: 2026-10-08T12:27:56.874047597+00:00
-- elapsed: 465ms
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
-- created_at: 2026-10-08T12:27:56.886995390+00:00
-- finished_at: 2026-10-08T12:27:57.345737566+00:00
-- elapsed: 458ms
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
-- created_at: 2026-10-08T12:27:57.365129045+00:00
-- finished_at: 2026-10-08T12:27:57.705326042+00:00
-- elapsed: 340ms
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
-- created_at: 2026-10-08T12:27:57.718916033+00:00
-- finished_at: 2026-10-08T12:27:57.924484793+00:00
-- elapsed: 205ms
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
-- created_at: 2026-10-08T12:27:57.937790063+00:00
-- finished_at: 2026-10-08T12:27:58.226809495+00:00
-- elapsed: 289ms
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
-- created_at: 2026-10-08T12:27:58.237975238+00:00
-- finished_at: 2026-10-08T12:27:58.551535845+00:00
-- elapsed: 313ms
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
-- created_at: 2026-10-08T12:27:58.566792997+00:00
-- finished_at: 2026-10-08T12:27:58.894612760+00:00
-- elapsed: 327ms
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
-- created_at: 2026-10-08T12:27:58.908044879+00:00
-- finished_at: 2026-10-08T12:27:59.218827217+00:00
-- elapsed: 310ms
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
-- created_at: 2026-10-08T12:27:59.236513436+00:00
-- finished_at: 2026-10-08T12:27:59.616966916+00:00
-- elapsed: 380ms
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
-- created_at: 2026-10-08T12:27:59.633042785+00:00
-- finished_at: 2026-10-08T12:28:00.020026607+00:00
-- elapsed: 386ms
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
-- created_at: 2026-10-08T12:28:00.044479114+00:00
-- finished_at: 2026-10-08T12:28:00.474171812+00:00
-- elapsed: 429ms
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
-- created_at: 2026-10-08T12:28:00.496240740+00:00
-- finished_at: 2026-10-08T12:28:00.923442674+00:00
-- elapsed: 427ms
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
-- created_at: 2026-10-08T12:28:01.008536231+00:00
-- finished_at: 2026-10-08T12:28:02.759086734+00:00
-- elapsed: 1.8s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_turnout
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_turnout", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1"
  
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
            using "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1"
            where (
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_turnout" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "is_transit_vote", "eligible_voters", "voters_uninstalled_sections", "uncounted_voters", "turnout", "abstentions", "turnout_rate", "abstention_rate", "generated_at"
        from "fact_turnout__dbt_tmp_597d52f6_8007_47f1_be0c_53f883b899c1"
    )
  ;
-- created_at: 2026-10-08T12:28:02.818121367+00:00
-- finished_at: 2026-10-08T12:28:02.929314006+00:00
-- elapsed: 111ms
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
-- created_at: 2026-10-08T12:28:02.942431599+00:00
-- finished_at: 2026-10-08T12:28:02.955019214+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation" rename to "candidate_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T12:28:02.964679160+00:00
-- finished_at: 2026-10-08T12:28:02.975977355+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_reconciliation__dbt_tmp" rename to "candidate_tally_reconciliation";
-- created_at: 2026-10-08T12:28:02.988627213+00:00
-- finished_at: 2026-10-08T12:28:03.007632569+00:00
-- elapsed: 19ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:03.038107132+00:00
-- finished_at: 2026-10-08T12:28:03.221398369+00:00
-- elapsed: 183ms
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
-- created_at: 2026-10-08T12:28:03.233106186+00:00
-- finished_at: 2026-10-08T12:28:03.253297842+00:00
-- elapsed: 20ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps" rename to "candidate_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T12:28:03.261242206+00:00
-- finished_at: 2026-10-08T12:28:03.272189557+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_tmp" rename to "candidate_tally_coverage_gaps";
-- created_at: 2026-10-08T12:28:03.286395980+00:00
-- finished_at: 2026-10-08T12:28:03.298308772+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:03.346331754+00:00
-- finished_at: 2026-10-08T12:28:03.354505259+00:00
-- elapsed: 8ms
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
-- created_at: 2026-10-08T12:28:03.375605465+00:00
-- finished_at: 2026-10-08T12:28:03.408277662+00:00
-- elapsed: 32ms
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
-- created_at: 2026-10-08T12:28:03.431606480+00:00
-- finished_at: 2026-10-08T12:28:06.345636696+00:00
-- elapsed: 2.9s
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
-- created_at: 2026-10-08T12:27:50.796410341+00:00
-- finished_at: 2026-10-08T12:28:07.238697881+00:00
-- elapsed: 16.4s
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
-- created_at: 2026-10-08T12:28:07.279954057+00:00
-- finished_at: 2026-10-08T12:28:07.308768729+00:00
-- elapsed: 28ms
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
-- created_at: 2026-10-08T12:28:07.333854226+00:00
-- finished_at: 2026-10-08T12:28:09.682914308+00:00
-- elapsed: 2.3s
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
-- created_at: 2026-10-08T12:28:06.357447029+00:00
-- finished_at: 2026-10-08T12:28:10.863116414+00:00
-- elapsed: 4.5s
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
-- created_at: 2026-10-08T12:28:09.693173654+00:00
-- finished_at: 2026-10-08T12:28:10.919018341+00:00
-- elapsed: 1.2s
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
-- created_at: 2026-10-08T12:28:10.947433794+00:00
-- finished_at: 2026-10-08T12:28:14.390861588+00:00
-- elapsed: 3.4s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_party
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_party", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_party__dbt_tmp_9995d8e8_c0fd_4b95_8197_ba7ed889a482"
  
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
            using "dim_party__dbt_tmp_9995d8e8_c0fd_4b95_8197_ba7ed889a482"
            where (
                
                    "dim_party__dbt_tmp_9995d8e8_c0fd_4b95_8197_ba7ed889a482".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_party__dbt_tmp_9995d8e8_c0fd_4b95_8197_ba7ed889a482".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_party__dbt_tmp_9995d8e8_c0fd_4b95_8197_ba7ed889a482".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "dim_party__dbt_tmp_9995d8e8_c0fd_4b95_8197_ba7ed889a482".party_number = DBT_INCREMENTAL_TARGET.party_number
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_party" ("election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "party_number", "party", "party_name", "party_id"
        from "dim_party__dbt_tmp_9995d8e8_c0fd_4b95_8197_ba7ed889a482"
    )
  ;
-- created_at: 2026-10-08T12:28:14.443791856+00:00
-- finished_at: 2026-10-08T12:28:14.745965930+00:00
-- elapsed: 302ms
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
-- created_at: 2026-10-08T12:28:10.884012639+00:00
-- finished_at: 2026-10-08T12:28:14.752148995+00:00
-- elapsed: 3.9s
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
-- created_at: 2026-10-08T12:28:14.750838224+00:00
-- finished_at: 2026-10-08T12:28:14.787821238+00:00
-- elapsed: 36ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary" rename to "candidate_summary__dbt_backup";
-- created_at: 2026-10-08T12:28:14.791203311+00:00
-- finished_at: 2026-10-08T12:28:14.797593803+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_summary__dbt_tmp" rename to "candidate_summary";
-- created_at: 2026-10-08T12:28:14.802364724+00:00
-- finished_at: 2026-10-08T12:28:14.808503163+00:00
-- elapsed: 6ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:14.820626890+00:00
-- finished_at: 2026-10-08T12:28:14.824031642+00:00
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
-- created_at: 2026-10-08T12:28:14.833048414+00:00
-- finished_at: 2026-10-08T12:28:14.835964968+00:00
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
-- created_at: 2026-10-08T12:28:14.855873190+00:00
-- finished_at: 2026-10-08T12:28:14.865182297+00:00
-- elapsed: 9ms
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
-- created_at: 2026-10-08T12:28:14.880020354+00:00
-- finished_at: 2026-10-08T12:28:14.883560549+00:00
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
-- created_at: 2026-10-08T12:28:14.892641835+00:00
-- finished_at: 2026-10-08T12:28:14.973084773+00:00
-- elapsed: 80ms
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
-- created_at: 2026-10-08T12:28:14.758384254+00:00
-- finished_at: 2026-10-08T12:28:16.300427295+00:00
-- elapsed: 1.5s
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
-- created_at: 2026-10-08T12:28:16.307862808+00:00
-- finished_at: 2026-10-08T12:28:19.728003350+00:00
-- elapsed: 3.4s
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
-- created_at: 2026-10-08T12:28:19.733489208+00:00
-- finished_at: 2026-10-08T12:28:23.153844875+00:00
-- elapsed: 3.4s
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
-- created_at: 2026-10-08T12:28:23.162358372+00:00
-- finished_at: 2026-10-08T12:28:25.554380860+00:00
-- elapsed: 2.4s
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
-- created_at: 2026-10-08T12:28:25.560408474+00:00
-- finished_at: 2026-10-08T12:28:27.769375553+00:00
-- elapsed: 2.2s
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
-- created_at: 2026-10-08T12:28:27.773935446+00:00
-- finished_at: 2026-10-08T12:28:29.711729363+00:00
-- elapsed: 1.9s
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
-- created_at: 2026-10-08T12:28:29.717808188+00:00
-- finished_at: 2026-10-08T12:28:32.093067010+00:00
-- elapsed: 2.4s
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
-- created_at: 2026-10-08T12:28:32.097794942+00:00
-- finished_at: 2026-10-08T12:28:34.593836149+00:00
-- elapsed: 2.5s
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
-- created_at: 2026-10-08T12:28:14.985495633+00:00
-- finished_at: 2026-10-08T12:28:34.606235728+00:00
-- elapsed: 19.6s
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
-- created_at: 2026-10-08T12:28:34.598151977+00:00
-- finished_at: 2026-10-08T12:28:40.119627531+00:00
-- elapsed: 5.5s
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
-- created_at: 2026-10-08T12:28:40.159317038+00:00
-- finished_at: 2026-10-08T12:28:45.490069808+00:00
-- elapsed: 5.3s
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_party_votes
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_party_votes", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69"
  
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
            using "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69"
            where (
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".election_code = DBT_INCREMENTAL_TARGET.election_code
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".round_number = DBT_INCREMENTAL_TARGET.round_number
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".zone = DBT_INCREMENTAL_TARGET.zone
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".office_code = DBT_INCREMENTAL_TARGET.office_code
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".party_number = DBT_INCREMENTAL_TARGET.party_number
                    and 
                
                    "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69".is_transit_vote = DBT_INCREMENTAL_TARGET.is_transit_vote
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_party_votes" ("election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file")
    (
        select "election_year", "election_type", "election_scope", "election_id", "election_code", "round_number", "uf", "municipality_code", "zone", "office_code", "office_scope", "party_number", "is_transit_vote", "nominal_valid_votes", "legend_valid_votes", "nominal_converted_to_legend_votes", "total_legend_valid_votes", "party_valid_votes", "nominal_annulled_subjudice_votes", "legend_annulled_subjudice_votes", "generated_at", "source_file"
        from "fact_party_votes__dbt_tmp_bc0acd37_ce6e_4ace_889f_4ffe2e254c69"
    )
  ;
-- created_at: 2026-10-08T12:28:45.553704836+00:00
-- finished_at: 2026-10-08T12:28:45.635765796+00:00
-- elapsed: 82ms
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
-- created_at: 2026-10-08T12:28:45.659733537+00:00
-- finished_at: 2026-10-08T12:28:45.780302363+00:00
-- elapsed: 120ms
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
-- created_at: 2026-10-08T12:28:45.787363357+00:00
-- finished_at: 2026-10-08T12:28:45.795802485+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary" rename to "candidate_vote_summary__dbt_backup";
-- created_at: 2026-10-08T12:28:45.801286166+00:00
-- finished_at: 2026-10-08T12:28:45.811840271+00:00
-- elapsed: 10ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."candidate_vote_summary__dbt_tmp" rename to "candidate_vote_summary";
-- created_at: 2026-10-08T12:28:45.821498492+00:00
-- finished_at: 2026-10-08T12:28:45.835292895+00:00
-- elapsed: 13ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.candidate_vote_summary
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.candidate_vote_summary", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."candidate_vote_summary__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:45.869330852+00:00
-- finished_at: 2026-10-08T12:28:45.873611601+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T12:28:45.884747359+00:00
-- finished_at: 2026-10-08T12:28:45.906985552+00:00
-- elapsed: 22ms
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
-- created_at: 2026-10-08T12:28:45.931779099+00:00
-- finished_at: 2026-10-08T12:28:46.035019801+00:00
-- elapsed: 103ms
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
-- created_at: 2026-10-08T12:28:46.048515949+00:00
-- finished_at: 2026-10-08T12:28:46.197071384+00:00
-- elapsed: 148ms
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
-- created_at: 2026-10-08T12:28:46.222163077+00:00
-- finished_at: 2026-10-08T12:28:46.320450942+00:00
-- elapsed: 98ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.fact_electorate_municipality
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.fact_electorate_municipality", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "fact_electorate_municipality__dbt_tmp_75359b35_ac1d_446b_9d07_7920cff87ced"
  
    as (
      

select *
from "tse_analytics"."main"."int_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    );
  
    
  ;

        
            delete from "tse_analytics"."main"."fact_electorate_municipality" as DBT_INCREMENTAL_TARGET
            using "fact_electorate_municipality__dbt_tmp_75359b35_ac1d_446b_9d07_7920cff87ced"
            where (
                
                    "fact_electorate_municipality__dbt_tmp_75359b35_ac1d_446b_9d07_7920cff87ced".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_75359b35_ac1d_446b_9d07_7920cff87ced".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_75359b35_ac1d_446b_9d07_7920cff87ced".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "fact_electorate_municipality__dbt_tmp_75359b35_ac1d_446b_9d07_7920cff87ced".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."fact_electorate_municipality" ("election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate")
    (
        select "election_type", "election_scope", "uf", "municipality_code", "municipality", "election_year", "electorate"
        from "fact_electorate_municipality__dbt_tmp_75359b35_ac1d_446b_9d07_7920cff87ced"
    )
  ;
-- created_at: 2026-10-08T12:28:46.343869684+00:00
-- finished_at: 2026-10-08T12:28:46.348023108+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T12:28:46.360276275+00:00
-- finished_at: 2026-10-08T12:28:46.370376250+00:00
-- elapsed: 10ms
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
-- created_at: 2026-10-08T12:28:46.385852964+00:00
-- finished_at: 2026-10-08T12:28:46.389164502+00:00
-- elapsed: 3ms
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
-- created_at: 2026-10-08T12:28:46.402583376+00:00
-- finished_at: 2026-10-08T12:28:46.515452754+00:00
-- elapsed: 112ms
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
-- created_at: 2026-10-08T12:28:46.529668921+00:00
-- finished_at: 2026-10-08T12:28:46.542131201+00:00
-- elapsed: 12ms
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
-- created_at: 2026-10-08T12:28:46.568664866+00:00
-- finished_at: 2026-10-08T12:28:46.573598387+00:00
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
from "tse_analytics"."main"."int_electorate_municipality"

  
    where election_year in (2026) and election_type in ('general')
  

    ) as __dbt_sbq
    where false
    limit 0
;
-- created_at: 2026-10-08T12:28:46.583885762+00:00
-- finished_at: 2026-10-08T12:28:46.604663640+00:00
-- elapsed: 20ms
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
-- created_at: 2026-10-08T12:28:46.616181960+00:00
-- finished_at: 2026-10-08T12:28:46.740461057+00:00
-- elapsed: 124ms
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
-- created_at: 2026-10-08T12:28:46.770735358+00:00
-- finished_at: 2026-10-08T12:28:46.889873523+00:00
-- elapsed: 119ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.dim_geography
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.dim_geography", "profile_name": "tse_analytics", "target_name": "dev"} */

  
    
    
    create temporary table
      "dim_geography__dbt_tmp_94f75bc9_8742_411b_8b12_b610daac06fc"
  
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
            using "dim_geography__dbt_tmp_94f75bc9_8742_411b_8b12_b610daac06fc"
            where (
                
                    "dim_geography__dbt_tmp_94f75bc9_8742_411b_8b12_b610daac06fc".election_year = DBT_INCREMENTAL_TARGET.election_year
                    and 
                
                    "dim_geography__dbt_tmp_94f75bc9_8742_411b_8b12_b610daac06fc".election_type = DBT_INCREMENTAL_TARGET.election_type
                    and 
                
                    "dim_geography__dbt_tmp_94f75bc9_8742_411b_8b12_b610daac06fc".uf = DBT_INCREMENTAL_TARGET.uf
                    and 
                
                    "dim_geography__dbt_tmp_94f75bc9_8742_411b_8b12_b610daac06fc".municipality_code = DBT_INCREMENTAL_TARGET.municipality_code
                    
                
                
            );
        
    

    insert into "tse_analytics"."main"."dim_geography" ("election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality")
    (
        select "election_year", "election_type", "election_scope", "uf", "municipality_code", "municipality"
        from "dim_geography__dbt_tmp_94f75bc9_8742_411b_8b12_b610daac06fc"
    )
  ;
-- created_at: 2026-10-08T12:28:46.914239375+00:00
-- finished_at: 2026-10-08T12:28:46.918901465+00:00
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
-- created_at: 2026-10-08T12:28:46.937586846+00:00
-- finished_at: 2026-10-08T12:28:46.943702731+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T12:28:46.965596908+00:00
-- finished_at: 2026-10-08T12:28:46.991523283+00:00
-- elapsed: 25ms
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
-- created_at: 2026-10-08T12:28:47.000465994+00:00
-- finished_at: 2026-10-08T12:28:47.019786635+00:00
-- elapsed: 19ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation" rename to "electoral_participation__dbt_backup";
-- created_at: 2026-10-08T12:28:47.026662744+00:00
-- finished_at: 2026-10-08T12:28:47.036342904+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."electoral_participation__dbt_tmp" rename to "electoral_participation";
-- created_at: 2026-10-08T12:28:47.047854788+00:00
-- finished_at: 2026-10-08T12:28:47.056686509+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.electoral_participation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.electoral_participation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."electoral_participation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:34.617646013+00:00
-- finished_at: 2026-10-08T12:28:47.510162799+00:00
-- elapsed: 12.9s
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
-- created_at: 2026-10-08T12:28:47.534240682+00:00
-- finished_at: 2026-10-08T12:28:47.540299203+00:00
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
-- created_at: 2026-10-08T12:28:47.563414140+00:00
-- finished_at: 2026-10-08T12:28:47.570442417+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T12:28:47.589190608+00:00
-- finished_at: 2026-10-08T12:28:47.595282382+00:00
-- elapsed: 6ms
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
-- created_at: 2026-10-08T12:28:47.614139902+00:00
-- finished_at: 2026-10-08T12:28:47.621454102+00:00
-- elapsed: 7ms
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
-- created_at: 2026-10-08T12:28:47.647308781+00:00
-- finished_at: 2026-10-08T12:28:47.651587399+00:00
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
-- created_at: 2026-10-08T12:28:47.669614831+00:00
-- finished_at: 2026-10-08T12:28:47.674178308+00:00
-- elapsed: 4ms
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
-- created_at: 2026-10-08T12:28:47.696195526+00:00
-- finished_at: 2026-10-08T12:28:47.701913469+00:00
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
-- created_at: 2026-10-08T12:28:47.722237802+00:00
-- finished_at: 2026-10-08T12:28:47.727819376+00:00
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
-- created_at: 2026-10-08T12:28:47.750615572+00:00
-- finished_at: 2026-10-08T12:28:48.204426418+00:00
-- elapsed: 453ms
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
-- created_at: 2026-10-08T12:28:48.229010092+00:00
-- finished_at: 2026-10-08T12:28:48.559735434+00:00
-- elapsed: 330ms
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
-- created_at: 2026-10-08T12:28:47.073461836+00:00
-- finished_at: 2026-10-08T12:28:49.888160631+00:00
-- elapsed: 2.8s
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
-- created_at: 2026-10-08T12:28:49.899856379+00:00
-- finished_at: 2026-10-08T12:28:49.902470047+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:28:49.914154166+00:00
-- finished_at: 2026-10-08T12:28:49.917077869+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:28:48.578635433+00:00
-- finished_at: 2026-10-08T12:28:51.761744598+00:00
-- elapsed: 3.2s
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
-- created_at: 2026-10-08T12:28:51.776897787+00:00
-- finished_at: 2026-10-08T12:28:51.781355036+00:00
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
-- created_at: 2026-10-08T12:28:51.793167920+00:00
-- finished_at: 2026-10-08T12:28:51.795572104+00:00
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
-- created_at: 2026-10-08T12:28:51.808615486+00:00
-- finished_at: 2026-10-08T12:28:51.810849455+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:28:51.823983057+00:00
-- finished_at: 2026-10-08T12:28:51.826116653+00:00
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
-- created_at: 2026-10-08T12:28:51.836902094+00:00
-- finished_at: 2026-10-08T12:28:51.898022148+00:00
-- elapsed: 61ms
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
-- created_at: 2026-10-08T12:28:51.907554354+00:00
-- finished_at: 2026-10-08T12:28:51.909686032+00:00
-- elapsed: 2ms
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
-- created_at: 2026-10-08T12:28:51.919504248+00:00
-- finished_at: 2026-10-08T12:28:51.947638786+00:00
-- elapsed: 28ms
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
-- created_at: 2026-10-08T12:28:51.961131732+00:00
-- finished_at: 2026-10-08T12:28:52.001164298+00:00
-- elapsed: 40ms
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
-- created_at: 2026-10-08T12:28:52.011216170+00:00
-- finished_at: 2026-10-08T12:28:52.055542757+00:00
-- elapsed: 44ms
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
-- created_at: 2026-10-08T12:28:52.071256340+00:00
-- finished_at: 2026-10-08T12:28:52.748227533+00:00
-- elapsed: 676ms
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
-- created_at: 2026-10-08T12:28:52.758059846+00:00
-- finished_at: 2026-10-08T12:28:52.769229100+00:00
-- elapsed: 11ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance" rename to "party_performance__dbt_backup";
-- created_at: 2026-10-08T12:28:52.777210708+00:00
-- finished_at: 2026-10-08T12:28:52.791929634+00:00
-- elapsed: 14ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_performance__dbt_tmp" rename to "party_performance";
-- created_at: 2026-10-08T12:28:52.805835220+00:00
-- finished_at: 2026-10-08T12:28:52.814095320+00:00
-- elapsed: 8ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_performance
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_performance", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_performance__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:52.843077934+00:00
-- finished_at: 2026-10-08T12:28:52.862783503+00:00
-- elapsed: 19ms
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
-- created_at: 2026-10-08T12:28:52.872530545+00:00
-- finished_at: 2026-10-08T12:28:52.881620487+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps" rename to "party_tally_coverage_gaps__dbt_backup";
-- created_at: 2026-10-08T12:28:52.889702444+00:00
-- finished_at: 2026-10-08T12:28:52.902504542+00:00
-- elapsed: 12ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_coverage_gaps__dbt_tmp" rename to "party_tally_coverage_gaps";
-- created_at: 2026-10-08T12:28:52.924439260+00:00
-- finished_at: 2026-10-08T12:28:52.977095322+00:00
-- elapsed: 52ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_coverage_gaps
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_coverage_gaps", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_coverage_gaps__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:53.061060342+00:00
-- finished_at: 2026-10-08T12:28:53.104533250+00:00
-- elapsed: 43ms
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
-- created_at: 2026-10-08T12:28:53.112768795+00:00
-- finished_at: 2026-10-08T12:28:53.120258334+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation" rename to "party_tally_reconciliation__dbt_backup";
-- created_at: 2026-10-08T12:28:53.125240236+00:00
-- finished_at: 2026-10-08T12:28:53.132611274+00:00
-- elapsed: 7ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */
alter view "tse_analytics"."main"."party_tally_reconciliation__dbt_tmp" rename to "party_tally_reconciliation";
-- created_at: 2026-10-08T12:28:53.138996623+00:00
-- finished_at: 2026-10-08T12:28:53.148386664+00:00
-- elapsed: 9ms
-- outcome: success
-- dialect: duckdb
-- node_id: model.tse_analytics.party_tally_reconciliation
-- query_id: not available
-- desc: execute adapter call
/* {"app": "dbt", "dbt_version": "2.0.0", "node_id": "model.tse_analytics.party_tally_reconciliation", "profile_name": "tse_analytics", "target_name": "dev"} */

      drop view if exists "tse_analytics"."main"."party_tally_reconciliation__dbt_backup" cascade
    ;
-- created_at: 2026-10-08T12:28:53.176399577+00:00
-- finished_at: 2026-10-08T12:28:54.239305747+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:28:54.247306069+00:00
-- finished_at: 2026-10-08T12:28:55.303433295+00:00
-- elapsed: 1.1s
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
-- created_at: 2026-10-08T12:28:55.312875219+00:00
-- finished_at: 2026-10-08T12:28:56.675417814+00:00
-- elapsed: 1.4s
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
-- created_at: 2026-10-08T12:28:56.685407739+00:00
-- finished_at: 2026-10-08T12:29:00.184930657+00:00
-- elapsed: 3.5s
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
-- created_at: 2026-10-08T12:28:49.928643385+00:00
-- finished_at: 2026-10-08T12:29:51.801647058+00:00
-- elapsed: 1m 2s
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
