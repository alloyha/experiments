-- ==============================================================================
-- BÔNUS: THE POSTGRES DAG ENGINE (Airflow + dbt inside Postgres)
-- ==============================================================================
-- Cansado de pagar caro no Airflow ou de manter infraestrutura Python injetando queries?
-- Vamos construir uma topologia de DAG completa com dependências, catch-up de erros
-- e telemetria nativa puramente em PostgreSQL, usando pg_cron como nosso Scheduler!
-- ==============================================================================

-- 1. Habilitamos extensões (pg_cron é opcional — requer shared_preload_libraries='pg_cron' no postgresql.conf)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_cron;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '⚠️  pg_cron não disponível (%). Agendamento automático desabilitado — chame fn_check_alerts() manualmente.', SQLERRM;
END $$;
CREATE EXTENSION IF NOT EXISTS dblink;

CREATE SCHEMA IF NOT EXISTS dag_engine;

-- 2. TABELA DE TAREFAS (A definição do nosso DAG Topológico)
CREATE TABLE IF NOT EXISTS dag_engine.tasks (
    task_id SERIAL PRIMARY KEY,
    task_name VARCHAR(100) UNIQUE NOT NULL,
    dag_name  VARCHAR(100),           -- nome do DAG/manifest ao qual esta task pertence
    procedure_call TEXT NOT NULL, 
    dependencies VARCHAR(100)[] DEFAULT '{}', -- Array de dependências topológicas (Quais tarefas precisam rodar antes?)
    max_retries INT DEFAULT 0,
    retry_delay_seconds INT DEFAULT 5,
    medallion_layer VARCHAR(25),      -- NOVO: Camada Medallion (BRONZE, SILVER, GOLD, etc.)
    sla_ms_override BIGINT DEFAULT NULL       -- NOVO: Âncora externa de SLA (manual)
);

-- Migração: garante coluna dag_name em tabelas que possam ter sido criadas antes deste campo
ALTER TABLE dag_engine.tasks ADD COLUMN IF NOT EXISTS dag_name VARCHAR(100);
ALTER TABLE dag_engine.tasks ADD COLUMN IF NOT EXISTS medallion_layer VARCHAR(25);
ALTER TABLE dag_engine.tasks ADD COLUMN IF NOT EXISTS call_fn   TEXT;
ALTER TABLE dag_engine.tasks ADD COLUMN IF NOT EXISTS call_args JSONB NOT NULL DEFAULT '["$1"]';
ALTER TABLE dag_engine.tasks ADD COLUMN IF NOT EXISTS layer     TEXT;
CREATE INDEX IF NOT EXISTS idx_tasks_dag_name ON dag_engine.tasks(dag_name);

-- Migração 3: UNLOGGED para tabelas de estado transitório
-- Reduz WAL em 60-80% para dados efêmeros por run (descartáveis em crash/catchup)
-- As tabelas analíticas finais mantêm logging: dag_runs, dag_versions, fato_task_exec
-- Cada ALTER é guardado: executa somente se a tabela ainda for LOGGED (p=permanent),
-- evitando o AccessExclusiveLock desnecessário quando já é UNLOGGED (u).
DO $$
DECLARE r RECORD;
BEGIN
    FOR r IN
        SELECT schemaname, tablename FROM (VALUES
            ('dag_engine',    'task_instances'),
            ('dag_engine',    'state_transitions'),
            ('dag_engine',    'async_workers'),
            ('dag_medallion', 'brnz_task_instances_snap'),
            ('dag_medallion', 'brnz_state_transitions_snap')
        ) AS t(schemaname, tablename)
        JOIN pg_class c ON c.relname = t.tablename
        JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.schemaname
        WHERE c.relpersistence = 'p'   -- somente se ainda LOGGED
    LOOP
        EXECUTE format('ALTER TABLE %I.%I SET UNLOGGED', r.schemaname, r.tablename);
    END LOOP;
END $$;

-- Trigger de sincronização: mantém procedure_call derivada de call_fn/call_args
-- para retrocompatibilidade com views e procedures que ainda a referenciam.
CREATE OR REPLACE FUNCTION dag_engine.trg_sync_procedure_call()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.call_fn IS NOT NULL THEN
        NEW.procedure_call := 'CALL ' || NEW.call_fn || '(' ||
            COALESCE(
                (SELECT string_agg(arg #>> '{}', ', ' ORDER BY ordinality)
                 FROM jsonb_array_elements(COALESCE(NEW.call_args, '["$1"]'::JSONB))
                      WITH ORDINALITY AS arr(arg, ordinality)),
                ''
            ) || ')';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_sync_procedure_call ON dag_engine.tasks;
CREATE TRIGGER trg_sync_procedure_call
BEFORE INSERT OR UPDATE OF call_fn, call_args ON dag_engine.tasks
FOR EACH ROW EXECUTE PROCEDURE dag_engine.trg_sync_procedure_call();

-- fn_build_call_sql: constrói o SQL de execução a partir de call_fn, call_args e p_date
-- Substitui o REPLACE(procedure_call, '$1', ...) usado anteriormente em proc_run_dag.
-- Detecta prokind via pg_proc: 'p' (procedure) → CALL, demais (função) → SELECT.
-- Também aceita p_chunk_index + p_chunk_config para substituir o token $chunk_filter
-- com o predicado de hash modular concreto quando method=hash.
CREATE OR REPLACE FUNCTION dag_engine.fn_build_call_sql(
    p_call_fn      TEXT,
    p_call_args    JSONB,
    p_date         DATE,
    p_chunk_index  INT  DEFAULT NULL,
    p_chunk_config JSONB DEFAULT NULL
) RETURNS TEXT LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_kind        CHAR;
    v_args        TEXT;
    v_chunk_filter TEXT := '';
    v_column       TEXT;
    v_buckets      INT;
BEGIN
    SELECT prokind INTO v_kind
    FROM pg_catalog.pg_proc
    WHERE proname = split_part(p_call_fn, '.', 2)
      AND pronamespace = (SELECT oid FROM pg_namespace
                          WHERE nspname = split_part(p_call_fn, '.', 1))
      AND pronargs >= jsonb_array_length(COALESCE(p_call_args, '["$1"]'::JSONB))
      AND (pronargs - pronargdefaults) <= jsonb_array_length(COALESCE(p_call_args, '["$1"]'::JSONB))
    LIMIT 1;

    -- Fallback: tenta sem qualificador de schema
    IF v_kind IS NULL THEN
        SELECT prokind INTO v_kind FROM pg_catalog.pg_proc
        WHERE oid = p_call_fn::regproc;
    END IF;

    -- Constrói predicado de hash modular se method=hash
    IF p_chunk_index IS NOT NULL
       AND p_chunk_config IS NOT NULL
       AND (p_chunk_config->>'method') = 'hash'
    THEN
        v_column  := p_chunk_config->>'column';
        v_buckets := (p_chunk_config->>'buckets')::INT;
        v_chunk_filter := format('hashtext(%I::text) %% %s = %s', v_column, v_buckets, p_chunk_index);
    END IF;

    v_args := COALESCE(
        (SELECT string_agg(
            REPLACE(
                REPLACE(arg #>> '{}', '$1', quote_literal(p_date)),
                '$chunk_filter', quote_literal(v_chunk_filter)
            ),
            ', '
            ORDER BY ordinality
        )
        FROM jsonb_array_elements(COALESCE(p_call_args, '["$1"]'::JSONB))
             WITH ORDINALITY AS arr(arg, ordinality)),
        ''
    );

    -- 'p' = procedure → CALL; qualquer outro prokind (f/w/a) → SELECT
    RETURN CASE v_kind
        WHEN 'p' THEN 'CALL ' || p_call_fn || '(' || v_args || ')'
        ELSE           'SELECT ' || p_call_fn || '(' || v_args || ')'
    END;
END;
$$;

-- proc_noop: stub para tasks de orquestração pura sem operação real (ex: 9_envio_relatorio).
-- Aceita p_date opcionalmente para compatibilidade com call_args: ["$1"].
CREATE OR REPLACE PROCEDURE dag_engine.proc_noop(p_date DATE DEFAULT NULL)
LANGUAGE plpgsql AS $$
BEGIN
    -- Tarefa de orquestração pura: sem operação.
    NULL;
END;
$$;

-- proc_snapshot_clientes com suporte a chunk_filter (hash chunking).
-- p_filter vazio (default) = comportamento original sem WHERE.
-- p_filter preenchido = WHERE hashtext(col::text) % N = i (gerado por fn_build_call_sql).
-- O overload (DATE) original é removido para evitar ambiguidade de resolução de overload.
DROP PROCEDURE IF EXISTS varejo.proc_snapshot_clientes(DATE);
CREATE OR REPLACE PROCEDURE varejo.proc_snapshot_clientes(
    p_data   DATE,
    p_filter TEXT DEFAULT ''
)
LANGUAGE plpgsql AS $$
BEGIN
    EXECUTE format(
        'INSERT INTO varejo.cliente_snapshot_diario
             (cliente_id, data_snapshot, nome, estado, segmento)
         SELECT cliente_id, %L, nome, estado, segmento
         FROM varejo.origem_cliente
         %s
         ON CONFLICT (cliente_id, data_snapshot) DO UPDATE SET
             nome     = EXCLUDED.nome,
             estado   = EXCLUDED.estado,
             segmento = EXCLUDED.segmento',
        p_data,
        CASE WHEN p_filter <> '' THEN 'WHERE ' || p_filter ELSE '' END
    );
END;
$$;

-- proc_acumular_atividade com suporte a chunk_filter (hash chunking por cliente_id).
-- p_filter vazio (default) = comportamento original: processa todos os clientes.
-- p_filter preenchido = WHERE hashtext(cliente_id::text) % N = i — gerado por fn_build_call_sql.
-- Ambas as CTEs (yesterday e today) filtram pelo mesmo predicado para manter consistência.
DROP PROCEDURE IF EXISTS varejo.proc_acumular_atividade(DATE, DATE);
CREATE OR REPLACE PROCEDURE varejo.proc_acumular_atividade(
    p_data_ontem DATE,
    p_data_hoje  DATE,
    p_filter     TEXT DEFAULT ''
)
LANGUAGE plpgsql AS $$
DECLARE
    v_where TEXT := CASE WHEN p_filter <> '' THEN ' AND (' || p_filter || ')' ELSE '' END;
BEGIN
    EXECUTE format($sql$
        WITH yesterday AS (
            SELECT cliente_id, datas_atividade, activity_bitmap_32d, activity_bitmap_32m
            FROM varejo.fct_atividade_acum
            WHERE data_snapshot = %L::DATE %s
        ),
        today AS (
            SELECT DISTINCT c.cliente_id, f.data_venda AS data_evento
            FROM varejo.fato_vendas f
            JOIN varejo.dim_cliente_type2 c ON c.cliente_sk = f.cliente_sk
            WHERE f.data_venda = %L::DATE %s
        ),
        merged AS (
            SELECT
                COALESCE(y.cliente_id, t.cliente_id) AS cliente_id,
                %L::DATE                             AS data_snapshot,
                CASE
                    WHEN t.cliente_id IS NOT NULL
                    THEN array_append(COALESCE(y.datas_atividade, ARRAY[]::DATE[]), t.data_evento)
                    ELSE COALESCE(y.datas_atividade, ARRAY[]::DATE[])
                END AS datas_atividade,
                varejo.shift_activity_bitmap(y.activity_bitmap_32d, t.cliente_id IS NOT NULL) AS activity_bitmap_32d,
                (CASE WHEN EXTRACT(month FROM %L::DATE) != EXTRACT(month FROM %L::DATE)
                      THEN COALESCE(y.activity_bitmap_32m, 0::BIGINT) >> 1
                      ELSE COALESCE(y.activity_bitmap_32m, 0::BIGINT)
                 END) | CASE WHEN t.cliente_id IS NOT NULL THEN (1::BIGINT << 31) ELSE 0::BIGINT END AS activity_bitmap_32m
            FROM yesterday y
            FULL OUTER JOIN today t ON y.cliente_id = t.cliente_id
        )
        INSERT INTO varejo.fct_atividade_acum
            (cliente_id, data_snapshot, datas_atividade, activity_bitmap_32d, activity_bitmap_32m)
        SELECT cliente_id, data_snapshot, datas_atividade, activity_bitmap_32d, activity_bitmap_32m
        FROM merged
        ON CONFLICT (cliente_id, data_snapshot) DO UPDATE SET
            datas_atividade     = EXCLUDED.datas_atividade,
            activity_bitmap_32d = EXCLUDED.activity_bitmap_32d,
            activity_bitmap_32m = EXCLUDED.activity_bitmap_32m
    $sql$,
        p_data_ontem, v_where,
        p_data_hoje,  v_where,
        p_data_hoje,
        p_data_hoje, p_data_ontem
    );
END;
$$;

-- Protege contra inserção de deadlocks e dependências circulares na arvore
CREATE OR REPLACE FUNCTION dag_engine.trg_prevent_cycles() RETURNS TRIGGER AS $$
DECLARE
    v_has_cycle BOOLEAN;
BEGIN
    IF NEW.task_name = ANY(NEW.dependencies) THEN
        RAISE EXCEPTION 'Acyclic DAG Error: % cannot depend on itself', NEW.task_name;
    END IF;

    WITH RECURSIVE dep_tree AS (
        SELECT dep as ancestor, 1 AS depth FROM unnest(NEW.dependencies) as dep
        UNION ALL
        SELECT parent_dep as ancestor, dt.depth + 1
        FROM dag_engine.tasks t
        JOIN dep_tree dt ON dt.ancestor = t.task_name,
        unnest(t.dependencies) as parent_dep
        WHERE dt.depth < 100 -- Segurança adicional caso a view seja adulterada
    )
    SELECT EXISTS (SELECT 1 FROM dep_tree WHERE ancestor = NEW.task_name)
    INTO v_has_cycle;

    IF v_has_cycle THEN
        RAISE EXCEPTION 'Acyclic DAG Error: Cyclic dependency detected for %!', NEW.task_name;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_check_cycles ON dag_engine.tasks;
CREATE TRIGGER trg_check_cycles
BEFORE INSERT OR UPDATE ON dag_engine.tasks
FOR EACH ROW EXECUTE PROCEDURE dag_engine.trg_prevent_cycles();

-- 2.2 TOPOLOGICAL SORTING (Ordens de Execução e Waves Paralelas)
-- Mapeia a árvore gerando em quais "Layers" (Níveis) de paralelismo as tarefas podem correr independentes
-- DROP CASCADE pois CREATE OR REPLACE VIEW não pode renomear colunas (ex: procedure_call → dag_name)
-- table_layer_registry: mapa de tabelas → camada medallion
CREATE TABLE IF NOT EXISTS dag_engine.table_layer_registry (
    table_schema TEXT NOT NULL,
    table_name   TEXT NOT NULL,
    layer        TEXT NOT NULL
        CHECK (layer IN ('Bronze','Silver','Gold','Semantic')),
    PRIMARY KEY (table_schema, table_name)
);

-- vw_topological_sort: migração inicial (sem fn_resolve_task_layer ainda)
-- será substituída pela versão final após fn_resolve_task_layer ser definida
DROP VIEW IF EXISTS dag_engine.vw_topological_sort CASCADE;
CREATE OR REPLACE VIEW dag_engine.vw_topological_sort AS
WITH RECURSIVE topo_sort AS (
    SELECT task_name, dag_name, call_fn, call_args, procedure_call,
           dependencies, layer, 0 AS execution_level
    FROM dag_engine.tasks
    WHERE array_length(dependencies, 1) IS NULL OR array_length(dependencies, 1) = 0
    UNION ALL
    SELECT t.task_name, t.dag_name, t.call_fn, t.call_args, t.procedure_call,
           t.dependencies, t.layer, ts.execution_level + 1
    FROM dag_engine.tasks t
    JOIN topo_sort ts ON ts.task_name = ANY(t.dependencies)
    WHERE ts.execution_level < 100
)
SELECT task_name, dag_name, call_fn, call_args, procedure_call,
       dependencies, layer,
       MAX(execution_level) AS topological_layer
FROM topo_sort
GROUP BY task_name, dag_name, call_fn, call_args, procedure_call, dependencies, layer
ORDER BY dag_name, topological_layer, task_name;

-- 3. TABELAS DE METADADOS E TELEMETRIA DE EXECUÇÃO
CREATE TABLE IF NOT EXISTS dag_engine.dag_runs (
    run_id   SERIAL PRIMARY KEY,
    dag_name VARCHAR(100),           -- nome do DAG que gerou esta run
    run_date DATE NOT NULL,
    status   VARCHAR(20) DEFAULT 'RUNNING',
    run_type VARCHAR(20) DEFAULT 'INCREMENTAL', -- NOVO: Diferencia BACKFILL de rotina diária
    start_ts TIMESTAMP DEFAULT clock_timestamp(),
    end_ts   TIMESTAMP,
    UNIQUE(dag_name, run_date)        -- permite múltiplos DAGs rodarem na mesma data
);
-- Migração: garante dag_name e atualiza constraint UNIQUE em tabelas pré-existentes
ALTER TABLE dag_engine.dag_runs ADD COLUMN IF NOT EXISTS dag_name VARCHAR(100);
DO $$ BEGIN
    ALTER TABLE dag_engine.dag_runs DROP CONSTRAINT IF EXISTS dag_runs_run_date_key;
EXCEPTION WHEN OTHERS THEN NULL; END $$;
DO $$ BEGIN
    ALTER TABLE dag_engine.dag_runs
        ADD CONSTRAINT dag_runs_dag_name_run_date_key UNIQUE(dag_name, run_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS dag_engine.task_instances (
    instance_id SERIAL PRIMARY KEY,
    run_id INT REFERENCES dag_engine.dag_runs(run_id),
    task_name VARCHAR(100) REFERENCES dag_engine.tasks(task_name),
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED', 'UPSTREAM_FAILED', 'SKIPPED')),
    attempt INT DEFAULT 1,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    retry_after_ts TIMESTAMP,
    duration_ms NUMERIC,
    rows_processed BIGINT DEFAULT NULL,       -- NOVO: Telemetria de volume (row-level)
    error_text TEXT,
    UNIQUE(run_id, task_name)
);

CREATE INDEX IF NOT EXISTS idx_task_instances_run_status 
ON dag_engine.task_instances (run_id, status);

-- ==============================================================================
-- NOVO: STATE MACHINE TRACKING (AUDIT TRAIL LOGGING)
-- ==============================================================================
-- Uma máquina de estados real precisa de histórico de transições para MLOps e Observabilidade.
CREATE TABLE IF NOT EXISTS dag_engine.state_transitions (
    transition_id SERIAL PRIMARY KEY,
    run_id INT REFERENCES dag_engine.dag_runs(run_id),
    task_name VARCHAR(100), -- NULL if indicating a DAG-level transition
    old_state VARCHAR(20),
    new_state VARCHAR(20),
    transition_ts TIMESTAMP DEFAULT clock_timestamp()
);

-- Trigger Function para registrar qualquer mudança de STATE das tarefas
CREATE OR REPLACE FUNCTION dag_engine.log_task_state_transition()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') OR (NEW.status IS DISTINCT FROM OLD.status) THEN
        INSERT INTO dag_engine.state_transitions (run_id, task_name, old_state, new_state)
        VALUES (
            NEW.run_id,
            NEW.task_name,
            CASE WHEN TG_OP = 'INSERT' THEN 'NONE' ELSE OLD.status END,
            NEW.status
        );
        
        -- Canal existente: eventos MLOps assíncronos em tempo real
        PERFORM pg_notify(
            'dag_events',
            json_build_object(
                'run_id',    NEW.run_id,
                'task',      NEW.task_name,
                'old_state', CASE WHEN TG_OP = 'INSERT' THEN 'NONE' ELSE OLD.status END,
                'new_state', NEW.status,
                'ts',        clock_timestamp()
            )::text
        );

        -- NOVO: canal de wake-up para o loop do motor (Melhoria 2: LISTEN/NOTIFY)
        IF NEW.status IN ('SUCCESS', 'FAILED', 'UPSTREAM_FAILED') THEN
            PERFORM pg_notify(
                'dag_task_done',
                json_build_object('run_id', NEW.run_id, 'task', NEW.task_name)::text
            );
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger Function para registrar mudanças do STATE da DAG
CREATE OR REPLACE FUNCTION dag_engine.log_dag_state_transition()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') OR (NEW.status IS DISTINCT FROM OLD.status) THEN
        INSERT INTO dag_engine.state_transitions (run_id, task_name, old_state, new_state)
        VALUES (
            NEW.run_id,
            NULL, -- DAG level tem task_name NULL
            CASE WHEN TG_OP = 'INSERT' THEN 'NONE' ELSE OLD.status END,
            NEW.status
        );
        
        -- BROADCAST DE EVENTOS DA DAG EM REAL-TIME
        PERFORM pg_notify(
            'dag_events',
            json_build_object(
                'run_id',    NEW.run_id,
                'task',      NULL,
                'old_state', CASE WHEN TG_OP = 'INSERT' THEN 'NONE' ELSE OLD.status END,
                'new_state', NEW.status,
                'ts',        clock_timestamp()
            )::text
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Anexa as Triggers nas Tabelas Centrais!
DROP TRIGGER IF EXISTS trg_task_status_change ON dag_engine.task_instances;
CREATE TRIGGER trg_task_status_change
AFTER INSERT OR UPDATE OF status ON dag_engine.task_instances
FOR EACH ROW EXECUTE PROCEDURE dag_engine.log_task_state_transition();

DROP TRIGGER IF EXISTS trg_dag_status_change ON dag_engine.dag_runs;
CREATE TRIGGER trg_dag_status_change
AFTER INSERT OR UPDATE OF status ON dag_engine.dag_runs
FOR EACH ROW EXECUTE PROCEDURE dag_engine.log_dag_state_transition();

-- 4. O MOTOR RESOLVEDOR DE DEPENDÊNCIAS (DAG RUNNER)
-- Utilitário de Cadência: Retorna a data de execução agendada imediatamente anterior
CREATE OR REPLACE FUNCTION dag_engine.fn_prev_scheduled_date(
    p_schedule TEXT,
    p_date     DATE
) RETURNS DATE LANGUAGE sql AS $$
    SELECT CASE
        WHEN p_schedule LIKE '% * * 1-5' THEN
            CASE EXTRACT(DOW FROM p_date)
                WHEN 1 THEN p_date - 3
                ELSE p_date - 1
            END
        ELSE p_date - 1
    END;
$$;

DROP PROCEDURE IF EXISTS dag_engine.proc_run_dag(DATE);
CREATE OR REPLACE PROCEDURE dag_engine.proc_run_dag(
    p_dag_name TEXT, 
    p_data DATE, 
    p_verbose BOOLEAN DEFAULT TRUE,
    p_run_type VARCHAR(20) DEFAULT 'INCREMENTAL'
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_run_id INT;
    v_task RECORD;
    v_pending_count INT;
    v_running_count INT;
    v_sql TEXT;
BEGIN
    IF p_verbose THEN
        RAISE NOTICE '=================================================';
        RAISE NOTICE '🚀 Iniciando DAG Topológica [%] para a data: %', p_dag_name, p_data;
    END IF;
    
    -- Cria nova execução (ou avisa se já existe pra data, necessitando intervenção de re-run)
    BEGIN
        INSERT INTO dag_engine.dag_runs (dag_name, run_date) VALUES (p_dag_name, p_data) RETURNING run_id INTO v_run_id;
    EXCEPTION WHEN unique_violation THEN
        IF p_verbose THEN RAISE WARNING 'Já existe execução do DAG "%" para %! Faça clear manual se quiser rodar de novo.', p_dag_name, p_data; END IF;
        RETURN;
    END;
    
    -- Instancia todas as tarefas do DAG base como PENDING
    INSERT INTO dag_engine.task_instances (run_id, task_name)
    SELECT v_run_id, task_name FROM dag_engine.tasks WHERE dag_name = p_dag_name;

    LOOP
        -- 1. Topo-Sort Select: Busca a próxima tarefa PENDING com propriedades resolvidas
        -- O SEGREDO DO PARALELISMO NATIVO PG: FOR UPDATE SKIP LOCKED
        v_task := NULL;
        SELECT ti.task_name, t.procedure_call, t.call_fn, t.call_args
        INTO v_task
        FROM dag_engine.task_instances ti
        JOIN dag_engine.tasks t ON ti.task_name = t.task_name
        WHERE ti.run_id = v_run_id AND ti.status = 'PENDING'
          AND (ti.retry_after_ts IS NULL OR ti.retry_after_ts <= clock_timestamp())
          AND NOT EXISTS (
              -- O pai tem que ter sucesso obrigatoriamente
              SELECT 1 FROM unnest(t.dependencies) as dep
              JOIN dag_engine.task_instances dep_ti ON dep_ti.run_id = v_run_id AND dep_ti.task_name = dep
              WHERE dep_ti.status != 'SUCCESS'
          )
        ORDER BY t.task_id
        FOR UPDATE OF ti SKIP LOCKED
        LIMIT 1;

        -- Se achamos algo que tá livre para rodar na topologia:
        IF v_task IS NOT NULL THEN
            -- Inicia a Tarefa e salva imediatamente o state (WAL / Commit) (Libera bloqueios para outros workers paralelos trabalharem)
            UPDATE dag_engine.task_instances SET status = 'RUNNING', start_ts = clock_timestamp() 
            WHERE run_id = v_run_id AND task_name = v_task.task_name;
            COMMIT;

            BEGIN
                -- Constrói SQL de execução a partir de call_fn/call_args (substituindo $1 pela data)
                v_sql := dag_engine.fn_build_call_sql(v_task.call_fn, v_task.call_args, p_data);
                IF p_verbose THEN RAISE NOTICE '  --> 🔄 Executando: [ % ] %', v_task.task_name, v_sql; END IF;
                
                -- Executa de Fato a Proc do Pipeline Original
                EXECUTE v_sql;
                
                -- Marca Sucesso com Duração
                UPDATE dag_engine.task_instances 
                SET status = 'SUCCESS', end_ts = clock_timestamp(), duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - start_ts)) * 1000
                WHERE run_id = v_run_id AND task_name = v_task.task_name;
                -- NÃO PODE HAVER COMMIT AQUI DENTRO DO BLOCO COM EXCEPTION.
            EXCEPTION WHEN OTHERS THEN
                DECLARE
                    v_current_attempt INT;
                    v_retry_delay     INT;
                    v_max_retries     INT;
                BEGIN
                    -- Captura explicitamente os valores atuais da linha antes de qualquer UPDATE
                    SELECT ti.attempt, t.retry_delay_seconds, t.max_retries
                    INTO v_current_attempt, v_retry_delay, v_max_retries
                    FROM dag_engine.task_instances ti
                    JOIN dag_engine.tasks t ON t.task_name = ti.task_name
                    WHERE ti.run_id = v_run_id AND ti.task_name = v_task.task_name;

                    -- RETRY COM BACKOFF: Temos mais tentativas pra gastar nessa task? (Ex: Deadlock ou Timeout Transitório)
                    IF v_current_attempt < v_max_retries + 1 THEN
                        UPDATE dag_engine.task_instances 
                        SET status         = 'PENDING', 
                            attempt        = attempt + 1, 
                            retry_after_ts = clock_timestamp() + (v_retry_delay * (v_current_attempt + 1)) * INTERVAL '1 second',
                            error_text     = 'Retry acionado | Ultimo erro: ' || SQLERRM
                        WHERE run_id = v_run_id AND task_name = v_task.task_name;
                        IF p_verbose THEN RAISE WARNING '🔄 [ % ] Falha temporária! Iniciando retry...', v_task.task_name; END IF;
                    ELSE
                        -- FATAL ERROR: Estourou o limite de tentativas
                        -- Captura Erro e Evita "Crash" Geral do Banco - Simulando um Stack Trace
                        UPDATE dag_engine.task_instances 
                        SET status = 'FAILED', end_ts = clock_timestamp(), duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - start_ts)) * 1000, error_text = SQLERRM
                        WHERE run_id = v_run_id AND task_name = v_task.task_name;
                        
                        -- PROPAGAÇÃO AIRFLOW/DBT (UPSTREAM_FAILED CASCADE)
                        WITH RECURSIVE fail_cascade AS (
                            SELECT t.task_name, 1 AS depth FROM dag_engine.tasks t WHERE v_task.task_name = ANY(t.dependencies)
                            UNION ALL
                            SELECT t.task_name, fc.depth + 1 FROM dag_engine.tasks t JOIN fail_cascade fc ON fc.task_name = ANY(t.dependencies)
                            WHERE fc.depth < 100 -- Safety Valve: quebra infinite loop em DAGs adulteradas manuais
                        )
                        UPDATE dag_engine.task_instances 
                        SET status = 'UPSTREAM_FAILED', end_ts = clock_timestamp(), error_text = 'Falha propagada do upstream: ' || v_task.task_name
                        WHERE run_id = v_run_id AND task_name IN (SELECT task_name FROM fail_cascade) AND status = 'PENDING';
                    END IF;
                END;
            END;

            -- O Commit deve ocorrer AQUI FORA para fechar apropriadamente o sub-estado
            COMMIT;
        ELSE
            -- Nenhuma tarefa solta no momento. Verifica o quadro geral da DAG.
            SELECT COUNT(*) INTO v_pending_count FROM dag_engine.task_instances WHERE run_id = v_run_id AND status = 'PENDING';
            SELECT COUNT(*) INTO v_running_count FROM dag_engine.task_instances WHERE run_id = v_run_id AND status = 'RUNNING';
            
            IF v_running_count > 0 THEN
                -- Outros threads podem estar calculando dependências (Simulação de Queue Listening Loop de Worker)
                PERFORM pg_sleep(1);
            ELSIF v_pending_count > 0 THEN
                -- Ninguém Running e ainda há PENDINGS. Pode ser o Backoff do Retry esperando!
                IF EXISTS (SELECT 1 FROM dag_engine.task_instances WHERE run_id = v_run_id AND status = 'PENDING' AND retry_after_ts > clock_timestamp()) THEN
                    PERFORM pg_sleep(1);
                ELSE
                    -- Ninguém resolvendo nem em Backoff (DeadLock Acyclic bypass real)
                    UPDATE dag_engine.dag_runs SET status = 'DEADLOCK', end_ts = clock_timestamp() WHERE run_id = v_run_id;
                    COMMIT;
                    IF p_verbose THEN RAISE WARNING '💀 Deadlock Topológico Encontrado: Tarefas pendentes irresolvíveis!'; END IF;
                    CALL dag_medallion.proc_run_medallion(v_run_id);  -- DEADLOCK também alimenta o medallion
                    EXIT;
                END IF;
            ELSE
                -- TUDO TERMINADO! Analisar o status geral da Tabela.
                IF EXISTS (SELECT 1 FROM dag_engine.task_instances WHERE run_id = v_run_id AND status IN ('FAILED', 'UPSTREAM_FAILED')) THEN
                    UPDATE dag_engine.dag_runs SET status = 'FAILED', end_ts = clock_timestamp() WHERE run_id = v_run_id;
                    IF p_verbose THEN RAISE WARNING '❌ DAG % Finalizada com Falhas Parciais/Totais!', p_data; END IF;
                ELSE
                    UPDATE dag_engine.dag_runs SET status = 'SUCCESS', end_ts = clock_timestamp() WHERE run_id = v_run_id;
                    IF p_verbose THEN RAISE NOTICE '✅ DAG % Finalizada com Sucesso Total!', p_data; END IF;
                END IF;
                COMMIT;
                
                EXIT;
            END IF;
        END IF;
    END LOOP;

    -- O DAG do DAG: Após o Pipeline finalizar (independente se SUCCESS, FAILED ou DEADLOCK), geramos o Analytical Medallion com todo o histórico rico!
    CALL dag_medallion.proc_run_medallion(v_run_id);
END;
$$;

-- ==============================================================================
-- 4.1. PROCEDURES DE UTILIDADE (CATCH-UP)
-- ==============================================================================
-- Se o servidor cair por 3 dias, o Cron só chamaria uma vez. Isso preenche o Gap.
-- NOTA: a definição completa e correta (com run_type='BACKFILL') está na seção 8.6
-- que substitui esta ao longo do script. A pré-definição abaixo existe apenas para
-- garantir que DROP PROCEDURE na 8.6 encontre a assinatura certa mesmo num banco limpo.
DROP PROCEDURE IF EXISTS dag_engine.proc_catchup(TEXT, DATE, DATE, BOOLEAN, INT);

-- ==============================================================================
-- NOVO: 4.2. VIEW DE ANOMALIAS E CRITICAL PATH (Health & Performance Z-Score)
-- ==============================================================================
CREATE OR REPLACE VIEW dag_engine.v_task_health AS
WITH stats AS (
    SELECT
        task_name, run_id, duration_ms,
        ROUND(AVG(duration_ms)    OVER (PARTITION BY task_name), 2) AS media_ms,
        ROUND(STDDEV(duration_ms) OVER (PARTITION BY task_name), 2) AS stddev_ms
    FROM dag_engine.task_instances
    WHERE status = 'SUCCESS'
),
scored AS (
    SELECT *,
        ROUND((duration_ms - media_ms) / NULLIF(stddev_ms, 0), 2) AS z_score
    FROM stats
)
SELECT *,
    CASE
        WHEN z_score > 2.0 THEN '🔴 ANOMALIA ESTATISTICA (Lento)'
        WHEN z_score > 1.0 THEN '🟡 DEGRADACAO LENTA'
        ELSE                    '🟢 OK / DENTRO DO NORMAL'
    END AS health_flag
FROM scored;

-- ==============================================================================
-- 4.3 VIEW DE PERCENTIS DE PERFORMANCE (Distribuição e Rolagem Total)
-- ==============================================================================
-- DROP CASCADE pois CREATE OR REPLACE VIEW não pode alterar tipo de coluna (ex: text → varchar)
DROP VIEW IF EXISTS dag_engine.v_task_percentiles CASCADE;
CREATE OR REPLACE VIEW dag_engine.v_task_percentiles AS
SELECT 
    COALESCE(dr.dag_name, 'unknown') AS pipeline_name,
    COALESCE(ti.task_name, '--- TOTAL DAG (Soma) ---') AS step_name,
    COUNT(*) as num_execucoes,
    ROUND(SUM(ti.duration_ms), 2) as sum_ms,
    ROUND((SUM(ti.duration_ms) / (NULLIF(SUM(SUM(ti.duration_ms)) OVER(PARTITION BY COALESCE(dr.dag_name, 'unknown')), 0) / 2)) * 100, 2) as pct_total,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ti.duration_ms)::NUMERIC, 2) AS p25_ms,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY ti.duration_ms)::NUMERIC, 2) AS p50_mediana_ms,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ti.duration_ms)::NUMERIC, 2) AS p75_ms,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY ti.duration_ms)::NUMERIC, 2) AS p90_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ti.duration_ms)::NUMERIC, 2) AS p95_ms,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ti.duration_ms)::NUMERIC, 2) AS p99_ms
FROM dag_engine.task_instances ti
JOIN dag_engine.dag_runs dr ON dr.run_id = ti.run_id
WHERE ti.status = 'SUCCESS'
GROUP BY dr.dag_name, ROLLUP(ti.task_name)
ORDER BY pipeline_name, step_name;

-- ==============================================================================
-- 4.2 PROCEDURES DE MANUTENÇÃO (CLEAR RUN)
-- ==============================================================================
-- Limpa completamente o rastro de uma RUN passada, permitindo reexecução do marco zero.
DROP PROCEDURE IF EXISTS dag_engine.proc_clear_run(DATE);
CREATE OR REPLACE PROCEDURE dag_engine.proc_clear_run(p_dag_name TEXT, p_date DATE, p_verbose BOOLEAN DEFAULT TRUE)
LANGUAGE plpgsql AS $$
DECLARE v_run_id INT;
BEGIN
    SELECT run_id INTO v_run_id FROM dag_engine.dag_runs
    WHERE dag_name = p_dag_name AND run_date = p_date;
    IF NOT FOUND THEN 
        RAISE EXCEPTION 'Nenhuma execução encontrada para o DAG "%" na data %', p_dag_name, p_date; 
    END IF;

    IF EXISTS (SELECT 1 FROM dag_engine.dag_runs WHERE run_id = v_run_id AND status = 'RUNNING') THEN
        RAISE EXCEPTION '🚫 Run de % está RUNNING ativamente. Interrompa o worker antes de limpar.', p_date;
    END IF;

    -- Limpa metadados derivados do Medallion para não existirem FK/PKs Órfãos temporários
    DELETE FROM dag_medallion.brnz_state_transitions_snap WHERE run_id = v_run_id;
    DELETE FROM dag_medallion.brnz_task_instances_snap    WHERE run_id = v_run_id;
    DELETE FROM dag_medallion.fato_task_exec              WHERE run_id = v_run_id;

    -- Limpeza bruta do Engine Base
    DELETE FROM dag_engine.state_transitions WHERE run_id = v_run_id;
    DELETE FROM dag_engine.task_instances     WHERE run_id = v_run_id;
    DELETE FROM dag_engine.dag_runs           WHERE run_id = v_run_id;

    IF p_verbose THEN RAISE NOTICE '🗑️ DAG Run referenciando a data % limpada com sucesso! Pronto para re-execução.', p_date; END IF;
END;
$$;

-- ==============================================================================
-- 4.3 PROCEDURES DE DEPLOYMENT (DAG SPEC AS JSON)
-- ==============================================================================
-- Permite carregar a topologia do DAG via um payload JSON (estilo dbt/Airflow configs).

CREATE OR REPLACE FUNCTION dag_engine.fn_validate_single_graph(
    p_spec JSONB
) RETURNS VOID
LANGUAGE plpgsql AS $$
DECLARE
    v_tasks     JSONB := COALESCE(p_spec->'tasks', p_spec);
    v_total     INT;
    v_reachable INT;
    v_start     TEXT;
    v_isolated  TEXT;
BEGIN
    v_total := jsonb_array_length(v_tasks);

    -- Grafo com 0 ou 1 nó é trivialmente conexo
    IF v_total <= 1 THEN RETURN; END IF;

    -- Ponto de partida: qualquer task (BFS não-dirigido é agnóstico à origem)
    SELECT elem->>'task_name'
    INTO v_start
    FROM jsonb_array_elements(v_tasks) AS elem
    LIMIT 1;

    -- BFS sobre grafo não-dirigido: cada dependência A→B gera arcos A↔B
    WITH RECURSIVE
    edges(a, b) AS (
        SELECT elem->>'task_name' AS a, dep AS b
        FROM jsonb_array_elements(v_tasks) AS elem,
             jsonb_array_elements_text(elem->'dependencies') AS dep
        UNION ALL
        SELECT dep AS a, elem->>'task_name' AS b
        FROM jsonb_array_elements(v_tasks) AS elem,
             jsonb_array_elements_text(elem->'dependencies') AS dep
    ),
    bfs(task_name) AS (
        SELECT v_start
        UNION                   -- UNION (não ALL) garante visita única por nó
        SELECT e.b
        FROM bfs
        JOIN edges e ON e.a = bfs.task_name
    )
    SELECT COUNT(*) INTO v_reachable FROM bfs;

    IF v_reachable < v_total THEN
        WITH RECURSIVE
        edges(a, b) AS (
            SELECT elem->>'task_name' AS a, dep AS b
            FROM jsonb_array_elements(v_tasks) AS elem,
                 jsonb_array_elements_text(elem->'dependencies') AS dep
            UNION ALL
            SELECT dep AS a, elem->>'task_name' AS b
            FROM jsonb_array_elements(v_tasks) AS elem,
                 jsonb_array_elements_text(elem->'dependencies') AS dep
        ),
        bfs(task_name) AS (
            SELECT v_start
            UNION
            SELECT e.b FROM bfs JOIN edges e ON e.a = bfs.task_name
        )
        SELECT string_agg(elem->>'task_name', ', ' ORDER BY elem->>'task_name')
        INTO v_isolated
        FROM jsonb_array_elements(v_tasks) AS elem
        WHERE (elem->>'task_name') NOT IN (SELECT task_name FROM bfs);

        RAISE EXCEPTION
            'DAG Connectivity Error: spec contém componentes desconexos. Tasks isoladas: %',
            v_isolated;
    END IF;
END;
$$;

CREATE OR REPLACE PROCEDURE dag_engine.proc_load_dag_spec(p_spec JSONB)
LANGUAGE plpgsql AS $$
DECLARE
    v_task JSONB;
    v_deps VARCHAR(100)[];
    v_missing_dep TEXT;
    v_tasks JSONB  := COALESCE(p_spec->'tasks', p_spec);  -- suporta manifest envelope E array legado
    v_dag_name TEXT := COALESCE(p_spec->>'name', 'default');
BEGIN
    -- PASSO 0: Remove tarefas que não estão mais no spec (deprecação limpa)
    IF jsonb_array_length(v_tasks) = 0 THEN
        RAISE EXCEPTION 'DAG Spec Error: spec vazio recebido. Operação abortada para proteger o engine.';
    END IF;

    -- Tasks que tem histórico acoplado falharão com Foreign Key Violation, exigindo a limpeza consciente do DBA
    DELETE FROM dag_engine.tasks
    WHERE dag_name = v_dag_name
      AND task_name NOT IN (
        SELECT t->>'task_name' FROM jsonb_array_elements(v_tasks) AS t
    );

    -- PASSO 1: Insere todas as tarefas sem validar deps (usando array vazio temporário para permitir Forward References sem quebras JSON de ordem)
    FOR v_task IN SELECT * FROM jsonb_array_elements(v_tasks)
    LOOP
        IF v_task->>'task_name' IS NULL OR v_task->>'procedure_call' IS NULL THEN
            RAISE EXCEPTION 'DAG Spec Error: campos "task_name" e "procedure_call" são obrigatórios. Payload recebido: %', v_task;
        END IF;

        INSERT INTO dag_engine.tasks (
            task_name,
            dag_name,
            procedure_call, 
            dependencies, 
            max_retries, 
            retry_delay_seconds
        ) VALUES (
            v_task->>'task_name',
            v_dag_name,
            v_task->>'procedure_call',
            '{}',
            COALESCE((v_task->>'max_retries')::INT, 0),
            COALESCE((v_task->>'retry_delay_seconds')::INT, 5)
        )
        ON CONFLICT (task_name) DO UPDATE SET 
            dag_name        = EXCLUDED.dag_name,
            procedure_call  = EXCLUDED.procedure_call,
            dependencies    = '{}',   -- garante que o passo 2 parte do zero em caso de re-deploy
            max_retries     = EXCLUDED.max_retries,
            retry_delay_seconds = EXCLUDED.retry_delay_seconds;
    END LOOP;

    -- PASSO 2: Agora que todos existem no motor, aplica as dependências (trigger de ciclo e validação protegem integridade da malha)
    FOR v_task IN SELECT * FROM jsonb_array_elements(v_tasks)
    LOOP
        -- Converte o array JSON abstrato de dependências para Array Nativo Postgres
        SELECT array_agg(d::VARCHAR) INTO v_deps 
        FROM jsonb_array_elements_text(v_task->'dependencies') d;
        
        -- Garante fallback preventivo caso as chaves não venham preenchidas integralmente
        v_deps := COALESCE(v_deps, '{}'::VARCHAR(100)[]);

        -- Valida que toda dependência declarada e processada no passo 1 existe (protege contra Erros de Typos no JSON)
        SELECT d INTO v_missing_dep
        FROM unnest(v_deps) AS d
        WHERE NOT EXISTS (SELECT 1 FROM dag_engine.tasks WHERE task_name = d)
        LIMIT 1;

        IF FOUND THEN
            RAISE EXCEPTION 'DAG Spec Error: dependência "%" declarada em "%" não existe no engine.',
                v_missing_dep, v_task->>'task_name';
        END IF;
        
        UPDATE dag_engine.tasks 
        SET dependencies = v_deps
        WHERE task_name = v_task->>'task_name';
    END LOOP;
    
    RAISE NOTICE '✅ DAG Specification declarativa carregada em Engine! % tarefas interpretadas.', jsonb_array_length(v_tasks);
END;
$$;

-- ==============================================================================
-- 5. MEDALLION DAG ENGINE OBSERVABILITY (O "DAG DO DAG")
-- ==============================================================================
-- O Motor agora se observa! Ele processa eventos crus do próprio fluxo de execução
-- para criar um DW próprio de observabilidade topológica, utilizando dimensões e score!

CREATE SCHEMA IF NOT EXISTS dag_medallion;

-- Funções Bitmap Escalares para monitoramento de saúde histórica compacta
CREATE OR REPLACE FUNCTION dag_medallion.shift_health_bitmap(
    p_bitmap  BIGINT,
    p_healthy BOOLEAN
) RETURNS BIGINT LANGUAGE sql IMMUTABLE AS $$
    SELECT (COALESCE(p_bitmap, 0::BIGINT) >> 1)
         | CASE WHEN p_healthy THEN (1::BIGINT << 31) ELSE 0::BIGINT END;
$$;

CREATE OR REPLACE FUNCTION dag_medallion.was_healthy_on_day(
    p_bitmap   BIGINT,
    p_days_ago INT
) RETURNS BOOLEAN LANGUAGE sql IMMUTABLE AS $$
    SELECT (p_bitmap & (1::BIGINT << (31 - p_days_ago))) > 0;
$$;

-- As tabelas de saúde cumulativa são criadas após dag_versions (dependência de FK)
-- Veja: seção 8 BACKLOG: DAG VERSIONING


-- ============================================================
-- BRONZE: Snapshots point-in-time raw preservados para auditoria
-- ============================================================

CREATE TABLE IF NOT EXISTS dag_medallion.brnz_task_instances_snap (
    snap_id      SERIAL PRIMARY KEY,
    snapped_at   TIMESTAMP DEFAULT clock_timestamp(),
    run_id       INT,
    task_name    VARCHAR(100),
    status       VARCHAR(20),
    attempt      INT,
    duration_ms  NUMERIC,
    error_text   TEXT,
    retry_after_ts TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dag_medallion.brnz_state_transitions_snap (
    snap_id       SERIAL PRIMARY KEY,
    snapped_at    TIMESTAMP DEFAULT clock_timestamp(),
    transition_id INT,
    run_id        INT,
    task_name     VARCHAR(100),
    old_state     VARCHAR(20),
    new_state     VARCHAR(20),
    transition_ts TIMESTAMP
);

-- Procedure de ingestão Bronze (idempotente)
CREATE OR REPLACE PROCEDURE dag_medallion.proc_ingest_bronze(p_run_id INT)
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM dag_medallion.brnz_task_instances_snap   WHERE run_id = p_run_id;
    DELETE FROM dag_medallion.brnz_state_transitions_snap WHERE run_id = p_run_id;

    INSERT INTO dag_medallion.brnz_task_instances_snap
        (run_id, task_name, status, attempt, duration_ms, error_text, retry_after_ts)
    SELECT run_id, task_name, status, attempt, duration_ms, error_text, retry_after_ts
    FROM dag_engine.task_instances WHERE run_id = p_run_id;

    INSERT INTO dag_medallion.brnz_state_transitions_snap
        (transition_id, run_id, task_name, old_state, new_state, transition_ts)
    SELECT transition_id, run_id, task_name, old_state, new_state, transition_ts
    FROM dag_engine.state_transitions WHERE run_id = p_run_id;
END;
$$;

-- ============================================================
-- SILVER: dim_task (SCD1 — O Grafo Materializado como Dimensão)
-- ============================================================
CREATE TABLE IF NOT EXISTS dag_medallion.dim_task (
    task_sk              SERIAL PRIMARY KEY,
    task_name            VARCHAR(100) UNIQUE NOT NULL,
    procedure_call       TEXT,
    dependencies         VARCHAR(100)[],
    dependency_count     INT,
    topological_layer    INT,        
    max_retries          INT,
    retry_delay_seconds  INT,
    is_root              BOOLEAN,    
    is_leaf              BOOLEAN,    
    updated_at           TIMESTAMP DEFAULT clock_timestamp()
);

CREATE OR REPLACE PROCEDURE dag_medallion.proc_upsert_dim_task()
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO dag_medallion.dim_task (
        task_name, procedure_call, dependencies, dependency_count,
        topological_layer, max_retries, retry_delay_seconds, is_root, is_leaf
    )
    SELECT
        t.task_name,
        t.procedure_call,
        t.dependencies,
        COALESCE(array_length(t.dependencies, 1), 0),
        ts.topological_layer,
        t.max_retries,
        t.retry_delay_seconds,
        COALESCE(array_length(t.dependencies, 1), 0) = 0  AS is_root,
        NOT EXISTS (
            SELECT 1 FROM dag_engine.tasks t2
            WHERE t.task_name = ANY(t2.dependencies)
        ) AS is_leaf
    FROM dag_engine.tasks t
    LEFT JOIN dag_engine.vw_topological_sort ts ON ts.task_name = t.task_name
    ON CONFLICT (task_name) DO UPDATE SET
        procedure_call      = EXCLUDED.procedure_call,
        dependencies        = EXCLUDED.dependencies,
        dependency_count    = EXCLUDED.dependency_count,
        topological_layer   = EXCLUDED.topological_layer,
        max_retries         = EXCLUDED.max_retries,
        retry_delay_seconds = EXCLUDED.retry_delay_seconds,
        is_root             = EXCLUDED.is_root,
        is_leaf             = EXCLUDED.is_leaf,
        updated_at          = clock_timestamp();
END;
$$;

-- ============================================================
-- SILVER: dim_error_class (Regex Taxonomy Dimension)
-- ============================================================
CREATE TABLE IF NOT EXISTS dag_medallion.dim_error_class (
    error_class_sk   SERIAL PRIMARY KEY,
    error_class_name VARCHAR(50) UNIQUE NOT NULL,
    error_pattern    TEXT,   
    description      TEXT
);

-- Seed Extensível
INSERT INTO dag_medallion.dim_error_class (error_class_name, error_pattern, description) VALUES
    ('DEADLOCK',         'deadlock detected',               'Conflito de lock entre transações'),
    ('TIMEOUT',          'timeout|canceling statement',     'Execução excedeu tempo limite'),
    ('FK_VIOLATION',     'foreign key|violates.*constraint','Violação de integridade referencial'),
    ('RELATION_MISSING', 'relation.*does not exist',        'Tabela/view não encontrada'),
    ('NULL_VIOLATION',   'null value.*column',              'Violação de NOT NULL'),
    ('SYNTAX_ERROR',     'syntax error',                    'Erro de sintaxe na procedure'),
    ('UNKNOWN',          '.*',                              'Erro não classificado (fallback)')
ON CONFLICT (error_class_name) DO NOTHING;

-- ============================================================
-- SILVER: fato_task_exec (Grain Universal: run × task × attempt)
-- ============================================================
CREATE TABLE IF NOT EXISTS dag_medallion.fato_task_exec (
    exec_sk             SERIAL PRIMARY KEY,
    run_id              INT NOT NULL,
    run_date            DATE NOT NULL,
    task_name           VARCHAR(100) NOT NULL,
    task_sk             INT REFERENCES dag_medallion.dim_task(task_sk),
    error_class_sk      INT REFERENCES dag_medallion.dim_error_class(error_class_sk),
    attempt             INT,
    final_status        VARCHAR(20),
    duration_ms         NUMERIC,
    queue_wait_ms       NUMERIC,     
    had_retry           BOOLEAN,
    is_upstream_victim  BOOLEAN,     
    run_type            VARCHAR(20), -- NOVO: BACKFILL / INCREMENTAL
    rows_processed      BIGINT,      -- NOVO: Telemetria de volume
    error_text          TEXT,
    start_ts            TIMESTAMP,
    end_ts              TIMESTAMP,
    UNIQUE (run_id, task_name)
);

CREATE OR REPLACE PROCEDURE dag_medallion.proc_upsert_fato_task_exec(p_run_id INT)
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO dag_medallion.fato_task_exec (
        run_id, run_date, task_name, task_sk, error_class_sk,
        attempt, final_status, duration_ms, queue_wait_ms,
        had_retry, is_upstream_victim, run_type, rows_processed,
        error_text, start_ts, end_ts
    )
    SELECT
        ti.run_id,
        dr.run_date,
        ti.task_name,
        dt.task_sk,
        (SELECT ec.error_class_sk
         FROM dag_medallion.dim_error_class ec
         WHERE ti.error_text ~* ec.error_pattern
         ORDER BY ec.error_class_sk  
         LIMIT 1),
        ti.attempt,
        ti.status,
        ti.duration_ms,
        EXTRACT(EPOCH FROM (ti.start_ts - dr.start_ts)) * 1000,
        ti.attempt > 1,
        ti.status = 'UPSTREAM_FAILED',
        dr.run_type,
        ti.rows_processed,
        ti.error_text,
        ti.start_ts,
        ti.end_ts
    FROM dag_engine.task_instances ti
    JOIN dag_engine.dag_runs        dr ON dr.run_id   = ti.run_id
    LEFT JOIN dag_medallion.dim_task dt ON dt.task_name = ti.task_name
    WHERE ti.run_id = p_run_id
    ON CONFLICT (run_id, task_name) DO UPDATE SET
        attempt            = EXCLUDED.attempt,
        final_status       = EXCLUDED.final_status,
        duration_ms        = EXCLUDED.duration_ms,
        queue_wait_ms      = EXCLUDED.queue_wait_ms,
        had_retry          = EXCLUDED.had_retry,
        is_upstream_victim = EXCLUDED.is_upstream_victim,
        error_class_sk     = EXCLUDED.error_class_sk,
        error_text         = EXCLUDED.error_text,
        run_type           = EXCLUDED.run_type,
        rows_processed     = EXCLUDED.rows_processed,
        end_ts             = EXCLUDED.end_ts;
END;
$$;


-- BITMAP & ARRAY HEALTH TRACKING (DATINT)
-- As tabelas de saúde (task_health_cumulative, task_health_array_mensal) e as
-- procedures que as alimentam (proc_upsert_health_cumulative, proc_upsert_health_array)
-- são criadas na seção 8.2, após dag_runs.version_id existir, respeitando a FK.

-- ============================================================
-- SILVER: dim_run (Dimensão de Execução Enriquecida)
-- ============================================================
-- Promove dag_runs de tabela operacional para dimensão Silver com atributos derivados.
-- Habilita: detecção de sazonalidade (segunda-feira acumula final de semana),
-- comparação incremental vs backfill, e join temporal em qualquer Gold view.
CREATE TABLE IF NOT EXISTS dag_medallion.dim_run (
    run_sk       SERIAL PRIMARY KEY,
    run_id       INT UNIQUE NOT NULL,
    run_date     DATE NOT NULL,
    dag_name     VARCHAR(100),
    version_id   INT,
    version_tag  VARCHAR(50),
    run_type     VARCHAR(20),
    day_of_week  INT,          -- 0=Domingo .. 6=Sábado (EXTRACT(DOW))
    day_of_month INT,
    mes_ref      DATE,
    week_segment VARCHAR(20)   -- 'Monday' | 'Friday' | 'Midweek'
);

CREATE OR REPLACE PROCEDURE dag_medallion.proc_upsert_dim_run(p_run_id INT)
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO dag_medallion.dim_run (
        run_id, run_date, dag_name, version_id, version_tag,
        run_type, day_of_week, day_of_month, mes_ref, week_segment
    )
    SELECT
        dr.run_id,
        dr.run_date,
        dr.dag_name,
        dr.version_id,
        dv.version_tag,
        dr.run_type,
        EXTRACT(DOW  FROM dr.run_date)::INT,
        EXTRACT(DAY  FROM dr.run_date)::INT,
        DATE_TRUNC('month', dr.run_date)::DATE,
        CASE EXTRACT(DOW FROM dr.run_date)::INT
            WHEN 1 THEN 'Monday'
            WHEN 5 THEN 'Friday'
            ELSE       'Midweek'
        END
    FROM dag_engine.dag_runs dr
    LEFT JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
    WHERE dr.run_id = p_run_id
    ON CONFLICT (run_id) DO UPDATE SET
        version_tag  = EXCLUDED.version_tag,
        run_type     = EXCLUDED.run_type,
        week_segment = EXCLUDED.week_segment;
END;
$$;

-- ============================================================
-- SILVER: dim_dependency_edge (Grafo de Dependências Normalizado)
-- ============================================================
-- Materializa dependências por versão como tabela de arestas (parent → child).
-- Habilita: betweenness centrality, fan-out/fan-in por task,
-- diff estrutural completo entre versões além do fn_diff_versions atual.
CREATE TABLE IF NOT EXISTS dag_medallion.dim_dependency_edge (
    parent_task  VARCHAR(100) NOT NULL,
    child_task   VARCHAR(100) NOT NULL,
    dag_name     VARCHAR(100) NOT NULL,
    version_id   INT          NOT NULL,
    PRIMARY KEY  (parent_task, child_task, version_id)
);

CREATE OR REPLACE PROCEDURE dag_medallion.proc_populate_dim_dependency_edge()
LANGUAGE plpgsql AS $$
DECLARE
    v_version_id INT;
    v_dag_name   VARCHAR(100);
BEGIN
    SELECT version_id, dag_name INTO v_version_id, v_dag_name
    FROM dag_engine.dag_versions WHERE is_active = TRUE
    ORDER BY version_id DESC LIMIT 1;
    IF v_version_id IS NULL THEN RETURN; END IF;

    INSERT INTO dag_medallion.dim_dependency_edge (parent_task, child_task, dag_name, version_id)
    SELECT
        dep         AS parent_task,
        t.task_name AS child_task,
        v_dag_name,
        v_version_id
    FROM dag_engine.tasks t
    CROSS JOIN LATERAL UNNEST(t.dependencies) AS dep
    WHERE t.dag_name = v_dag_name
      AND array_length(t.dependencies, 1) > 0
    ON CONFLICT (parent_task, child_task, version_id) DO NOTHING;
END;
$$;

-- ============================================================
-- SILVER ANALYTICS: Custo de Retries (Tempo Desperdiçado)
-- ============================================================
-- Para tasks com retry, estima ms desperdiçados em tentativas frustradas.
-- Baseline: avg_ms de execuções limpas (sem retry).
-- estimated_wasted_ms = (num_retries) × baseline — quanto wall time a instabilidade custou.
CREATE OR REPLACE VIEW dag_medallion.silver_retry_waste AS
WITH task_baseline AS (
    SELECT
        task_name,
        ROUND(AVG(duration_ms) FILTER (WHERE NOT had_retry), 2) AS clean_avg_ms
    FROM dag_medallion.fato_task_exec
    WHERE final_status = 'SUCCESS'
    GROUP BY task_name
)
SELECT
    f.run_date,
    f.task_name,
    f.run_id,
    f.attempt - 1                                                         AS num_retries,
    f.duration_ms                                                         AS final_attempt_ms,
    tb.clean_avg_ms                                                       AS baseline_ms,
    ROUND((f.attempt - 1) * COALESCE(tb.clean_avg_ms, f.duration_ms), 2) AS estimated_wasted_ms,
    ROUND(f.duration_ms - COALESCE(tb.clean_avg_ms, f.duration_ms), 2)   AS final_overhead_ms,
    f.error_text
FROM dag_medallion.fato_task_exec f
LEFT JOIN task_baseline tb ON tb.task_name = f.task_name
WHERE f.had_retry = TRUE;

-- ============================================================
-- SILVER ANALYTICS: Eficiência de Paralelismo por Wave
-- ============================================================
-- dispatch_spread_ms = gap entre 1º e último início na mesma wave (≈0 = paralelo perfeito).
-- parallelism_pct = max_task_ms / sum_task_ms × 100 (100% = totalmente paralelo, baixo = serializado).
CREATE OR REPLACE VIEW dag_medallion.silver_wave_efficiency AS
SELECT
    f.run_id,
    f.run_date,
    dt.topological_layer                                                   AS wave,
    COUNT(*)                                                               AS tasks_in_wave,
    MIN(f.start_ts)                                                        AS first_task_started,
    MAX(f.start_ts)                                                        AS last_task_started,
    ROUND(EXTRACT(EPOCH FROM (MAX(f.start_ts) - MIN(f.start_ts))) * 1000, 2)
                                                                           AS dispatch_spread_ms,
    ROUND(AVG(f.duration_ms), 2)                                           AS avg_task_ms,
    ROUND(MAX(f.duration_ms), 2)                                           AS max_task_ms,
    ROUND(SUM(f.duration_ms), 2)                                           AS sum_task_ms,
    ROUND(
        MAX(f.duration_ms) / NULLIF(SUM(f.duration_ms), 0) * 100
    , 2)                                                                   AS parallelism_pct
FROM dag_medallion.fato_task_exec f
JOIN dag_medallion.dim_task dt ON dt.task_sk = f.task_sk
WHERE f.final_status = 'SUCCESS'
  AND f.start_ts IS NOT NULL
GROUP BY f.run_id, f.run_date, dt.topological_layer
HAVING COUNT(*) > 1;

-- ============================================================
-- SILVER ANALYTICS: Recorrência de Erros (Impressão Digital)
-- ============================================================
-- Janela deslizante de 7 runs por (task, error_class).
-- recurrence_last_7_runs ≥ 5 → sistemático (infra); < 3 → transiente (dado).
CREATE OR REPLACE VIEW dag_medallion.silver_error_recurrence AS
SELECT
    f.run_date,
    f.task_name,
    ec.error_class_name,
    COUNT(*) OVER (
        PARTITION BY f.task_name, f.error_class_sk
        ORDER BY f.run_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                                                                       AS recurrence_last_7_runs,
    COUNT(*) OVER (
        PARTITION BY f.task_name, f.error_class_sk
    )                                                                       AS total_occurrences,
    CASE
        WHEN COUNT(*) OVER (
                 PARTITION BY f.task_name, f.error_class_sk
                 ORDER BY f.run_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
             ) >= 5 THEN '🔴 SISTEMÁTICO'
        WHEN COUNT(*) OVER (
                 PARTITION BY f.task_name, f.error_class_sk
                 ORDER BY f.run_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
             ) >= 3 THEN '🟡 RECORRENTE'
        ELSE              '🟢 ESPORÁDICO'
    END                                                                     AS recurrence_label
FROM dag_medallion.fato_task_exec f
JOIN dag_medallion.dim_error_class ec ON ec.error_class_sk = f.error_class_sk
WHERE f.final_status IN ('FAILED', 'UPSTREAM_FAILED')
  AND f.error_class_sk IS NOT NULL;

-- ============================================================
-- SILVER ANALYTICS: Atribuição de Culpa Upstream
-- ============================================================
-- Para cada task raiz de falha, conta: runs afetadas, vítimas downstream, last incident.
-- Permite priorizar hardening: falhar X custa N tasks e M ms de fila perdida.
CREATE OR REPLACE VIEW dag_medallion.silver_upstream_blame AS
WITH blamed AS (
    SELECT
        v.run_id,
        v.run_date,
        v.task_name   AS victim_task,
        dep           AS blamed_task
    FROM dag_medallion.fato_task_exec v
    JOIN dag_medallion.dim_task dt ON dt.task_sk = v.task_sk
    CROSS JOIN LATERAL UNNEST(dt.dependencies) AS dep
    WHERE v.is_upstream_victim = TRUE
      AND EXISTS (
          SELECT 1 FROM dag_medallion.fato_task_exec f2
          WHERE f2.run_id     = v.run_id
            AND f2.task_name  = dep
            AND f2.final_status = 'FAILED'
      )
)
SELECT
    blamed_task,
    COUNT(DISTINCT run_id)                                                  AS runs_as_root_cause,
    COUNT(*)                                                                AS total_downstream_victims,
    COUNT(DISTINCT victim_task)                                             AS distinct_victims,
    MAX(run_date)                                                           AS last_incident,
    STRING_AGG(DISTINCT victim_task, ', ' ORDER BY victim_task)            AS victim_tasks
FROM blamed
GROUP BY blamed_task
ORDER BY total_downstream_victims DESC;

-- ============================================================
-- SILVER ANALYTICS: Throughput e Anomalia de Volume
-- ============================================================
-- rows_per_ms = throughput operacional da task.
-- volume_zscore > 2σ = sinal de qualidade de dado upstream, não problema de performance.
CREATE OR REPLACE VIEW dag_medallion.silver_throughput AS
WITH stats AS (
    SELECT
        task_name,
        AVG(rows_processed)    AS avg_rows,
        STDDEV(rows_processed) AS stddev_rows
    FROM dag_medallion.fato_task_exec
    WHERE rows_processed IS NOT NULL
      AND final_status = 'SUCCESS'
    GROUP BY task_name
)
SELECT
    f.run_date,
    f.task_name,
    f.rows_processed,
    f.duration_ms,
    ROUND(f.rows_processed::NUMERIC / NULLIF(f.duration_ms, 0), 4)        AS rows_per_ms,
    ROUND(
        (f.rows_processed - s.avg_rows) / NULLIF(s.stddev_rows, 0)
    , 2)                                                                   AS volume_zscore,
    CASE
        WHEN ABS((f.rows_processed - s.avg_rows) / NULLIF(s.stddev_rows, 0)) > 3
            THEN '🔴 ANOMALIA DE VOLUME (>3σ)'
        WHEN ABS((f.rows_processed - s.avg_rows) / NULLIF(s.stddev_rows, 0)) > 2
            THEN '🟡 VOLUME ALTO (>2σ)'
        ELSE '🟢 NORMAL'
    END                                                                    AS volume_status
FROM dag_medallion.fato_task_exec f
JOIN stats s ON s.task_name = f.task_name
WHERE f.rows_processed IS NOT NULL
  AND f.final_status = 'SUCCESS';

-- ============================================================
-- GOLD 1: Pipeline Health Score Z-Score Composto
-- ============================================================
-- DROP CASCADE pois CREATE OR REPLACE VIEW não pode renomear colunas
DROP VIEW IF EXISTS dag_medallion.gold_pipeline_health CASCADE;
CREATE OR REPLACE VIEW dag_medallion.gold_pipeline_health AS
WITH base AS (
    SELECT
        f.task_name,
        MAX(dt.topological_layer)                                        AS topological_layer,
        COUNT(*)                                                         AS total_runs,
        ROUND(100.0 * SUM(CASE WHEN f.final_status = 'SUCCESS' THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                         AS success_rate,
        ROUND(100.0 * SUM(CASE WHEN f.had_retry THEN 1 ELSE 0 END) / COUNT(*), 2)
                                                                         AS retry_rate,
        ROUND(AVG(f.duration_ms), 2)                                     AS avg_ms,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY f.duration_ms)::NUMERIC, 2)
                                                                         AS p50_ms,
        ROUND(STDDEV(f.duration_ms), 2)                                  AS stddev_ms,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY f.duration_ms)::NUMERIC, 2)
                                                                         AS p95_ms
    FROM dag_medallion.fato_task_exec f
    JOIN dag_medallion.dim_task dt ON dt.task_sk = f.task_sk
    GROUP BY f.task_name
),
scored AS (
    SELECT *,
        ROUND(
            ( (success_rate * 0.60)
              + ((100 - retry_rate) * 0.20)
              + CASE 
                  WHEN COALESCE(stddev_ms, 0) > (p50_ms * 0.5) THEN 
                      (100 * 0.20 / (1 + (stddev_ms - (p50_ms * 0.5)) / NULLIF(p50_ms, 0)))
                  ELSE 20.00 
                END
            ) * CASE WHEN total_runs < 10 THEN (total_runs::NUMERIC / 10) ELSE 1.0 END
        , 2) AS health_score
    FROM base
)
SELECT *,
    CASE
        WHEN total_runs < 10    THEN '🔵 CALIBRANDO'
        WHEN health_score >= 95 THEN '🟢 SAUDÁVEL'
        WHEN health_score >= 70 THEN '🟡 ATENÇÃO'
        ELSE                        '🔴 CRÍTICO'
    END AS health_label
FROM scored
ORDER BY topological_layer ASC;

-- ============================================================
-- GOLD 2: SLA Breach Detection (Contrato de P95)
-- ============================================================
DROP VIEW IF EXISTS dag_medallion.gold_sla_breach CASCADE;
CREATE OR REPLACE VIEW dag_medallion.gold_sla_breach AS
WITH sla_calc AS (
    SELECT
        task_name,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS sla_p95_ms
    FROM dag_medallion.fato_task_exec
    WHERE final_status = 'SUCCESS'
    GROUP BY task_name
),
sla_final AS (
    SELECT 
        dt.task_sk,
        dt.task_name,
        COALESCE(t.sla_ms_override, sc.sla_p95_ms) AS sla_limit_ms,
        CASE WHEN t.sla_ms_override IS NOT NULL THEN TRUE ELSE FALSE END AS is_manual
    FROM dag_medallion.dim_task dt
    LEFT JOIN dag_engine.tasks t ON t.task_name = dt.task_name
    LEFT JOIN sla_calc sc ON sc.task_name = dt.task_name
)
SELECT
    f.run_date,
    f.task_name,
    f.duration_ms           AS actual_ms,
    sf.sla_limit_ms         AS sla_target_ms,
    sf.is_manual            AS is_manual_sla,
    ROUND(f.duration_ms - sf.sla_limit_ms, 2)              AS breach_ms,
    ROUND((f.duration_ms / NULLIF(sf.sla_limit_ms, 0) - 1) * 100, 2) AS breach_pct,
    CASE
        WHEN f.duration_ms > sf.sla_limit_ms * 2.0 THEN '🔴 SLA CRÍTICO (>2x)'
        WHEN f.duration_ms > sf.sla_limit_ms * 1.5 THEN '🟠 SLA SEVERO (>1.5x)'
        WHEN f.duration_ms > sf.sla_limit_ms       THEN '🟡 SLA BREACH'
        ELSE                                           '🟢 DENTRO DO SLA'
    END AS sla_status
FROM dag_medallion.fato_task_exec f
JOIN sla_final sf ON sf.task_name = f.task_name
WHERE f.final_status = 'SUCCESS'
ORDER BY f.run_date DESC, breach_pct DESC;

-- ============================================================
-- GOLD 3: Taxonomia de Erros (Blast Analysis)
-- ============================================================
CREATE OR REPLACE VIEW dag_medallion.gold_error_taxonomy AS
SELECT
    ec.error_class_name,
    ec.description,
    COUNT(*)                                                   AS total_failures,
    COUNT(DISTINCT f.task_name)                                AS tasks_afetadas,
    COUNT(DISTINCT f.run_date)                                 AS runs_afetadas,
    MAX(f.run_date)                                            AS ultima_ocorrencia,
    STRING_AGG(DISTINCT f.task_name, ', ' ORDER BY f.task_name) AS tasks_lista,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)         AS pct_do_total
FROM dag_medallion.fato_task_exec f
JOIN dag_medallion.dim_error_class ec ON ec.error_class_sk = f.error_class_sk
WHERE f.final_status IN ('FAILED', 'UPSTREAM_FAILED')
GROUP BY ec.error_class_name, ec.description
ORDER BY total_failures DESC;

-- ============================================================
-- GOLD 4: Critical Path Analysis
-- ============================================================
DROP VIEW IF EXISTS dag_medallion.gold_critical_path CASCADE;
CREATE OR REPLACE VIEW dag_medallion.gold_critical_path AS
WITH per_run AS (
    SELECT
        f.run_date,
        f.task_name,
        dt.topological_layer,
        dt.is_leaf,
        f.duration_ms,
        f.queue_wait_ms,
        f.queue_wait_ms + f.duration_ms                         AS cumulative_ms
    FROM dag_medallion.fato_task_exec f
    JOIN dag_medallion.dim_task dt ON dt.task_sk = f.task_sk
    WHERE f.final_status = 'SUCCESS'
),
aggregated AS (
    SELECT
        task_name,
        topological_layer,
        is_leaf,
        ROUND(AVG(duration_ms), 2)       AS avg_duration_ms,
        ROUND(AVG(queue_wait_ms), 2)     AS avg_queue_wait_ms,
        ROUND(AVG(cumulative_ms), 2)     AS avg_cumulative_ms,
        ROUND(AVG(duration_ms) / NULLIF(SUM(AVG(duration_ms)) OVER (), 0) * 100, 2)
                                         AS pct_pipeline_time
    FROM per_run
    GROUP BY task_name, topological_layer, is_leaf
)
SELECT *,
    CASE WHEN pct_pipeline_time = MAX(pct_pipeline_time) OVER ()
         THEN '⭐ CRITICAL PATH' ELSE '' END AS critical_flag
FROM aggregated
ORDER BY topological_layer, avg_duration_ms DESC;

-- ============================================================
-- GOLD 5: Blast Radius cascades down (Topologia Corrente)
-- NOTA: Esta visão utiliza a topologia de tarefas ATUAL (viva) do motor para
-- calcular o impacto, não retrata necessariamente a topologia histórica da época de falha.
CREATE OR REPLACE VIEW dag_medallion.gold_blast_radius AS
WITH RECURSIVE downstream AS (
    SELECT
        t_root.task_name  AS source_task,
        t_child.task_name AS affected_task,
        1                 AS hops
    FROM dag_engine.tasks t_root
    JOIN dag_engine.tasks t_child ON t_root.task_name = ANY(t_child.dependencies)
    UNION ALL
    SELECT ds.source_task, t.task_name, ds.hops + 1
    FROM downstream ds
    JOIN dag_engine.tasks t ON ds.affected_task = ANY(t.dependencies)
    WHERE ds.hops < 100
)
SELECT
    source_task,
    COUNT(DISTINCT affected_task)                               AS downstream_count,
    STRING_AGG(DISTINCT affected_task, ' → ' ORDER BY affected_task) AS downstream_chain,
    MAX(hops)                                                   AS max_cascade_depth,
    COALESCE(f.total_failures, 0)                               AS historical_failures,
    ROUND(COALESCE(f.total_failures, 0) * COUNT(DISTINCT affected_task), 2)
                                                                AS risk_score
FROM downstream
LEFT JOIN (
    SELECT task_name, COUNT(*) AS total_failures
    FROM dag_medallion.fato_task_exec
    WHERE final_status = 'FAILED'
    GROUP BY task_name
) f ON f.task_name = source_task
GROUP BY source_task, f.total_failures
ORDER BY risk_score DESC;

-- ============================================================
-- GOLD 8: Step Duration Timelapse (Trend Analítico Preditivo)
-- ============================================================
CREATE OR REPLACE VIEW dag_medallion.gold_performance_timelapse AS
SELECT 
    f.run_date,
    f.task_name,
    dt.topological_layer,
    f.duration_ms AS actual_duration,
    -- Média móvel dos últimos 7 dias para visualizar a linha de tendência suave
    ROUND(AVG(f.duration_ms) OVER(
        PARTITION BY f.task_name 
        ORDER BY f.run_date::TIMESTAMP
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_7d,
    -- Variação percentual versus a execução de ontem (Rápida detecção de saltos em tempo real)
    ROUND((f.duration_ms / NULLIF(LAG(f.duration_ms) OVER(PARTITION BY f.task_name ORDER BY f.run_date), 0) - 1) * 100, 2) AS day_over_day_pct
FROM dag_medallion.fato_task_exec f
JOIN dag_medallion.dim_task dt ON dt.task_sk = f.task_sk
WHERE f.final_status = 'SUCCESS';

-- ============================================================
-- GOLD 9: Regressão de Performance por Versão (Detecção Estatística)
-- ============================================================
-- Compara avg_ms entre versões consecutivas da mesma task usando σ como régua.
-- regression_sigma > 2 após deploy = regressão crítica: rollback candidato.
-- regression_sigma < -2 = melhoria estatisticamente significativa: validação de otimização.
CREATE OR REPLACE VIEW dag_medallion.gold_version_regression AS
WITH per_version AS (
    SELECT
        dv.version_tag,
        dv.version_id,
        f.task_name,
        COUNT(*)                          AS run_count,
        ROUND(AVG(f.duration_ms),    2)   AS avg_ms,
        ROUND(STDDEV(f.duration_ms), 2)   AS stddev_ms
    FROM dag_medallion.fato_task_exec f
    JOIN dag_engine.dag_runs     dr ON dr.run_id     = f.run_id
    JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
    WHERE f.final_status  = 'SUCCESS'
      AND dr.version_id  IS NOT NULL
    GROUP BY dv.version_tag, dv.version_id, f.task_name
),
versioned AS (
    SELECT *,
        LAG(version_tag) OVER (PARTITION BY task_name ORDER BY version_id) AS prev_version_tag,
        LAG(avg_ms)      OVER (PARTITION BY task_name ORDER BY version_id) AS prev_avg_ms,
        LAG(stddev_ms)   OVER (PARTITION BY task_name ORDER BY version_id) AS prev_stddev_ms
    FROM per_version
)
SELECT
    task_name,
    prev_version_tag                                                        AS prev_version,
    version_tag                                                             AS curr_version,
    run_count                                                               AS curr_run_count,
    prev_avg_ms,
    avg_ms                                                                  AS curr_avg_ms,
    ROUND((avg_ms - prev_avg_ms) / NULLIF(prev_avg_ms,    0) * 100, 2)     AS delta_pct,
    ROUND((avg_ms - prev_avg_ms) / NULLIF(prev_stddev_ms, 0),       2)     AS regression_sigma,
    CASE
        WHEN prev_version_tag IS NULL                                       THEN '🔵 PRIMEIRA VERSÃO'
        WHEN run_count < 3                                                  THEN '🔵 CALIBRANDO'
        WHEN (avg_ms - prev_avg_ms) / NULLIF(prev_stddev_ms, 0) >  2.0    THEN '🔴 REGRESSÃO CRÍTICA (>2σ)'
        WHEN (avg_ms - prev_avg_ms) / NULLIF(prev_stddev_ms, 0) >  1.0    THEN '🟡 REGRESSÃO LEVE (>1σ)'
        WHEN (avg_ms - prev_avg_ms) / NULLIF(prev_stddev_ms, 0) < -2.0    THEN '🚀 MELHORIA SIGNIFICATIVA'
        WHEN (avg_ms - prev_avg_ms) / NULLIF(prev_stddev_ms, 0) < -1.0    THEN '🟢 MELHORIA'
        ELSE                                                                     '⬜ ESTÁVEL'
    END                                                                     AS regression_label
FROM versioned
WHERE prev_version_tag IS NOT NULL
ORDER BY ABS(COALESCE((avg_ms - prev_avg_ms) / NULLIF(prev_stddev_ms, 0), 0)) DESC;

-- ============================================================
-- GOLD 10: Aderência ao Agendamento (Schedule Adherence)
-- ============================================================
-- Compara start_ts real vs horário esperado pelo cron (parseado de dag_versions.schedule).
-- delay_minutes > 30 = contention no servidor de BD, não problema de lógica de pipeline.
CREATE OR REPLACE VIEW dag_medallion.gold_schedule_adherence AS
WITH scheduled AS (
    SELECT
        dr.run_id,
        dr.run_date,
        dr.dag_name,
        dr.start_ts,
        dv.schedule,
        NULLIF(TRIM(SPLIT_PART(dv.schedule, ' ', 2)), '*')::INT AS sched_hour,
        NULLIF(TRIM(SPLIT_PART(dv.schedule, ' ', 1)), '*')::INT AS sched_minute
    FROM dag_engine.dag_runs dr
    JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
    WHERE dr.start_ts IS NOT NULL
      AND dv.schedule  IS NOT NULL
),
with_planned AS (
    SELECT *,
        CASE WHEN sched_hour IS NOT NULL THEN
            (run_date + INTERVAL '1 day'
                + (sched_hour::TEXT                      || ' hours'  )::INTERVAL
                + (COALESCE(sched_minute, 0)::TEXT       || ' minutes')::INTERVAL
            )::TIMESTAMP
        END AS planned_start_ts
    FROM scheduled
)
SELECT
    run_date,
    dag_name,
    start_ts                                                                AS actual_start,
    planned_start_ts,
    schedule,
    ROUND(EXTRACT(EPOCH FROM (start_ts - planned_start_ts)) / 60, 2)       AS delay_minutes,
    CASE
        WHEN planned_start_ts IS NULL               THEN '🔵 SEM AGENDAMENTO'
        WHEN start_ts < planned_start_ts            THEN '⚡ INÍCIO ANTECIPADO'
        WHEN start_ts <= planned_start_ts + INTERVAL '5 minutes'  THEN '🟢 NO PRAZO (≤5min)'
        WHEN start_ts <= planned_start_ts + INTERVAL '30 minutes' THEN '🟡 ATRASADO (≤30min)'
        ELSE                                             '🔴 ATRASADO CRÍTICO (>30min)'
    END                                                                     AS adherence_status
FROM with_planned
ORDER BY run_date DESC;

-- ============================================================
-- GOLD 11: Frescor dos Dados (Data Freshness SLA)
-- ============================================================
-- data_ready_ts = end_ts da última leaf task = momento em que o dado ficou disponível.
-- sla_deadline_ts = horário agendado (cron) + janela de 2h de processamento.
-- sla_breach_minutes > 0 = dado ficou disponível depois do prazo contratual.
CREATE OR REPLACE VIEW dag_medallion.gold_data_freshness AS
WITH leaf_done AS (
    SELECT
        f.run_id,
        f.run_date,
        dr.dag_name,
        dr.start_ts                                                         AS run_start_ts,
        MAX(f.end_ts)                                                       AS data_ready_ts,
        dv.schedule
    FROM dag_medallion.fato_task_exec f
    JOIN dag_medallion.dim_task      dt ON dt.task_sk    = f.task_sk
    JOIN dag_engine.dag_runs         dr ON dr.run_id     = f.run_id
    JOIN dag_engine.dag_versions     dv ON dv.version_id = dr.version_id
    WHERE dt.is_leaf = TRUE
      AND f.final_status = 'SUCCESS'
      AND f.end_ts IS NOT NULL
    GROUP BY f.run_id, f.run_date, dr.dag_name, dr.start_ts, dv.schedule
),
with_sla AS (
    SELECT *,
        NULLIF(TRIM(SPLIT_PART(schedule, ' ', 2)), '*')::INT                AS sched_hour,
        CASE WHEN NULLIF(TRIM(SPLIT_PART(schedule, ' ', 2)), '*') IS NOT NULL THEN
            (run_date + INTERVAL '1 day'
                + (NULLIF(TRIM(SPLIT_PART(schedule, ' ', 2)), '*')::INT || ' hours')::INTERVAL
                + INTERVAL '2 hours'
            )::TIMESTAMP
        END                                                                 AS sla_deadline_ts
    FROM leaf_done
)
SELECT
    run_date,
    dag_name,
    data_ready_ts,
    run_start_ts,
    ROUND(EXTRACT(EPOCH FROM (data_ready_ts - run_start_ts)) / 60, 2)      AS pipeline_duration_minutes,
    sla_deadline_ts,
    ROUND(EXTRACT(EPOCH FROM (data_ready_ts - sla_deadline_ts)) / 60, 2)   AS sla_breach_minutes,
    CASE
        WHEN sla_deadline_ts IS NULL               THEN '🔵 SEM SLA'
        WHEN data_ready_ts  <= sla_deadline_ts     THEN '🟢 DENTRO DO SLA'
        ELSE                                            '🔴 SLA VIOLADO'
    END                                                                     AS freshness_status
FROM with_sla
ORDER BY run_date DESC;

-- ============================================================
-- GOLD 12: Correlação Cross-Run entre Tasks (Inteligência Preditiva)
-- ============================================================
-- pearson_r > 0.8 entre task A e B = gargalo de infra compartilhado (CPU/IO/lock).
-- r_vs_pipeline_total > 0.7 = indicador antecipado: task lenta → run toda será lenta.
-- Requer ≥ 5 runs em comum para calcular correlação estável.
CREATE OR REPLACE VIEW dag_medallion.gold_cross_run_correlation AS
WITH run_matrix AS (
    SELECT
        f.run_date,
        f.task_name,
        f.duration_ms
    FROM dag_medallion.fato_task_exec f
    WHERE f.final_status = 'SUCCESS'
      AND f.duration_ms  IS NOT NULL
),
pipeline_total AS (
    SELECT run_date, SUM(duration_ms) AS total_ms
    FROM run_matrix
    GROUP BY run_date
),
pairs AS (
    SELECT
        a.task_name                                                         AS task_a,
        b.task_name                                                         AS task_b,
        ROUND(CORR(a.duration_ms, b.duration_ms)::NUMERIC, 4)              AS pearson_r,
        COUNT(*)                                                            AS shared_runs
    FROM run_matrix a
    JOIN run_matrix b
      ON b.run_date   = a.run_date
     AND b.task_name  > a.task_name
    GROUP BY a.task_name, b.task_name
    HAVING COUNT(*) >= 5
),
task_vs_total AS (
    SELECT
        rm.task_name,
        ROUND(CORR(rm.duration_ms, pt.total_ms)::NUMERIC, 4)               AS r_vs_pipeline_total
    FROM run_matrix rm
    JOIN pipeline_total pt ON pt.run_date = rm.run_date
    GROUP BY rm.task_name
    HAVING COUNT(*) >= 5
)
SELECT
    p.task_a,
    p.task_b,
    p.pearson_r,
    p.shared_runs,
    tvt_a.r_vs_pipeline_total                                               AS task_a_r_vs_total,
    tvt_b.r_vs_pipeline_total                                               AS task_b_r_vs_total,
    CASE
        WHEN p.pearson_r >  0.8 THEN '🔴 ACOPLAMENTO FORTE (r>0.8)'
        WHEN p.pearson_r >  0.5 THEN '🟡 CORRELAÇÃO MODERADA (r>0.5)'
        WHEN p.pearson_r < -0.5 THEN '🔵 CORRELAÇÃO NEGATIVA (r<-0.5)'
        ELSE                         '⬜ SEM CORRELAÇÃO'
    END                                                                     AS correlation_label
FROM pairs p
LEFT JOIN task_vs_total tvt_a ON tvt_a.task_name = p.task_a
LEFT JOIN task_vs_total tvt_b ON tvt_b.task_name = p.task_b
ORDER BY ABS(p.pearson_r) DESC NULLS LAST;

-- ============================================================
-- MATERIALIZED VIEWS GOLD (Melhoria 6: refresh incremental por run)
-- CONCURRENTLY não bloqueia leituras durante refresh.
-- Unique index obrigatório para CONCURRENTLY funcionar.
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS dag_medallion.mgold_pipeline_health CASCADE;
CREATE MATERIALIZED VIEW dag_medallion.mgold_pipeline_health AS
    SELECT * FROM dag_medallion.gold_pipeline_health
WITH DATA;
CREATE UNIQUE INDEX ON dag_medallion.mgold_pipeline_health (task_name);

DROP MATERIALIZED VIEW IF EXISTS dag_medallion.mgold_sla_breach CASCADE;
CREATE MATERIALIZED VIEW dag_medallion.mgold_sla_breach AS
    SELECT * FROM dag_medallion.gold_sla_breach
WITH DATA;
CREATE UNIQUE INDEX ON dag_medallion.mgold_sla_breach (run_date, task_name);

DROP MATERIALIZED VIEW IF EXISTS dag_medallion.mgold_version_regression CASCADE;
CREATE MATERIALIZED VIEW dag_medallion.mgold_version_regression AS
    SELECT * FROM dag_medallion.gold_version_regression
WITH DATA;
CREATE UNIQUE INDEX ON dag_medallion.mgold_version_regression (task_name, curr_version);

DROP MATERIALIZED VIEW IF EXISTS dag_medallion.mgold_cross_run_correlation CASCADE;
CREATE MATERIALIZED VIEW dag_medallion.mgold_cross_run_correlation AS
    SELECT * FROM dag_medallion.gold_cross_run_correlation
WITH DATA;
CREATE UNIQUE INDEX ON dag_medallion.mgold_cross_run_correlation (task_a, task_b);

CREATE OR REPLACE PROCEDURE dag_medallion.proc_refresh_gold_views()
LANGUAGE plpgsql AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY dag_medallion.mgold_pipeline_health;
    REFRESH MATERIALIZED VIEW CONCURRENTLY dag_medallion.mgold_sla_breach;
    REFRESH MATERIALIZED VIEW CONCURRENTLY dag_medallion.mgold_version_regression;
    REFRESH MATERIALIZED VIEW CONCURRENTLY dag_medallion.mgold_cross_run_correlation;
END;
$$;

-- ============================================================
-- PARTICIONAMENTO (Melhoria 4: fato_task_exec por run_date)
-- Em produção, migrar com:
--   INSERT INTO dag_medallion.fato_task_exec_partitioned SELECT * FROM dag_medallion.fato_task_exec;
--   DROP TABLE dag_medallion.fato_task_exec;
--   ALTER TABLE dag_medallion.fato_task_exec_partitioned RENAME TO fato_task_exec;
-- ============================================================
CREATE TABLE IF NOT EXISTS dag_medallion.fato_task_exec_partitioned (
    exec_sk             SERIAL,
    run_id              INT NOT NULL,
    run_date            DATE NOT NULL,
    task_name           VARCHAR(100) NOT NULL,
    task_sk             INT REFERENCES dag_medallion.dim_task(task_sk),
    error_class_sk      INT REFERENCES dag_medallion.dim_error_class(error_class_sk),
    attempt             INT,
    final_status        VARCHAR(20),
    duration_ms         NUMERIC,
    queue_wait_ms       NUMERIC,
    had_retry           BOOLEAN,
    is_upstream_victim  BOOLEAN,
    run_type            VARCHAR(20),
    rows_processed      BIGINT,
    error_text          TEXT,
    start_ts            TIMESTAMP,
    end_ts              TIMESTAMP,
    PRIMARY KEY (run_id, task_name, run_date)  -- partition key run_date deve fazer parte do PK
) PARTITION BY RANGE (run_date);

-- Procedure que cria a partição mensal automaticamente antes de cada run
CREATE OR REPLACE PROCEDURE dag_medallion.proc_ensure_partition(p_date DATE)
LANGUAGE plpgsql AS $$
DECLARE
    v_start      DATE := DATE_TRUNC('month', p_date)::DATE;
    v_end        DATE := (DATE_TRUNC('month', p_date) + INTERVAL '1 month')::DATE;
    v_part_name  TEXT := 'fato_task_exec_' || TO_CHAR(p_date, 'YYYY_MM');
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'dag_medallion' AND c.relname = v_part_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE dag_medallion.%I PARTITION OF dag_medallion.fato_task_exec_partitioned
             FOR VALUES FROM (%L) TO (%L)',
            v_part_name, v_start, v_end
        );
        RAISE NOTICE '📦 Partição % criada (% → %)', v_part_name, v_start, v_end;
    END IF;
END;
$$;

-- ============================================================
-- O META-DAG PROCESSOR (Chamado Autônomo pelo Motor Principal da DAG)
-- ============================================================
CREATE OR REPLACE PROCEDURE dag_medallion.proc_run_medallion(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_run_date DATE;
BEGIN
    SELECT run_date INTO v_run_date FROM dag_engine.dag_runs WHERE run_id = p_run_id;

    -- Melhoria 4: garante partição do mês antes de qualquer escrita
    CALL dag_medallion.proc_ensure_partition(v_run_date);
    CALL dag_medallion.proc_ingest_bronze(p_run_id);
    CALL dag_medallion.proc_upsert_dim_task();
    CALL dag_medallion.proc_upsert_fato_task_exec(p_run_id);
    CALL dag_medallion.proc_upsert_health_cumulative(p_run_id);  -- NOVO (DATINT)
    CALL dag_medallion.proc_upsert_health_array(p_run_id);       -- NOVO (DATINT)
    CALL dag_medallion.proc_upsert_dim_run(p_run_id);             -- dim_run Silver
    CALL dag_medallion.proc_populate_dim_dependency_edge();       -- dim_dependency_edge Silver
    CALL dag_medallion.proc_refresh_gold_views();                -- Melhoria 6: refresh mat views
    CALL dag_semantic.proc_emit_reports(p_run_id);               -- Semantic: emite relatório oncall
END;
$$;

-- ==============================================================================
-- 7b. CAMADA SEMÂNTICA (dag_semantic) — Relatórios de Negócio Prontos
-- ==============================================================================
-- Propósito: abstrair a complexidade do modelo medallion em 4 relatórios
-- consumíveis diretamente por times de operação, confiabilidade e liderança.
--
--   rpt_oncall_handoff      → status do último run + falhas críticas + SLA (oncall)
--   rpt_weekly_reliability  → confiabilidade semanal (8 semanas, trend vs 4w anteriores)
--   rpt_monthly_sla_delivery→ entrega mensal vs SLA + health score (diretoria)
--   rpt_version_impact      → impacto por versão deployada (engenharia/retrospectiva)
--   proc_emit_reports       → publica rpt_oncall_handoff em pg_notify('dag_reports')
-- ==============================================================================

CREATE SCHEMA IF NOT EXISTS dag_semantic;

-- ============================================================
-- SEMANTIC 1: Oncall Handoff — status do último run por DAG
-- ============================================================
-- Grain: 1 linha por dag_name (último run).
-- Uso: painel de plantonista / alerta pós-run.
CREATE OR REPLACE VIEW dag_semantic.rpt_oncall_handoff AS
WITH last_run AS (
    SELECT DISTINCT ON (dag_name)
        run_id, dag_name, run_date, status, run_type, start_ts, end_ts
    FROM dag_engine.dag_runs
    ORDER BY dag_name, run_date DESC, run_id DESC
),
task_summary AS (
    SELECT
        f.run_id,
        COUNT(*)                                                        AS total_tasks,
        COUNT(*) FILTER (WHERE f.final_status = 'SUCCESS')              AS ok_tasks,
        COUNT(*) FILTER (WHERE f.final_status = 'FAILED')               AS failed_tasks,
        COUNT(*) FILTER (WHERE f.final_status = 'UPSTREAM_FAILED')      AS upstream_fail_tasks,
        COUNT(*) FILTER (WHERE f.had_retry = TRUE)                      AS retried_tasks,
        ROUND(AVG(f.duration_ms), 2)                                    AS avg_task_ms,
        ROUND(SUM(f.duration_ms), 2)                                    AS total_work_ms
    FROM dag_medallion.fato_task_exec f
    GROUP BY f.run_id
),
critical_failures AS (
    SELECT
        f.run_id,
        STRING_AGG(
            f.task_name || ': ' || COALESCE(LEFT(f.error_text, 120), 'sem detalhe'),
            E'\n' ORDER BY f.task_name
        ) AS failure_detail
    FROM dag_medallion.fato_task_exec f
    WHERE f.final_status = 'FAILED'
    GROUP BY f.run_id
),
sla_violations AS (
    SELECT
        f2.run_id,
        COUNT(*) FILTER (WHERE sb.sla_status != '🟢 DENTRO DO SLA')    AS sla_breach_count,
        STRING_AGG(
            sb.task_name || ' (' || sb.sla_status || ')',
            ', ' ORDER BY sb.breach_pct DESC
        ) FILTER (WHERE sb.sla_status != '🟢 DENTRO DO SLA')           AS sla_breach_detail
    FROM dag_medallion.gold_sla_breach sb
    JOIN dag_medallion.fato_task_exec f2
        ON f2.task_name = sb.task_name AND f2.run_date = sb.run_date
    GROUP BY f2.run_id
)
SELECT
    lr.dag_name,
    lr.run_date,
    lr.run_type,
    lr.status                                                    AS run_status,
    CASE
        WHEN lr.status = 'FAILED'                                  THEN '🔴 FALHOU'
        WHEN lr.status = 'SUCCESS'
         AND COALESCE(sv.sla_breach_count, 0) > 0                 THEN '🟡 OK c/ SLA BREACH'
        WHEN lr.status = 'SUCCESS'                                 THEN '✅ OK'
        ELSE                                                            '⏳ EM PROGRESSO'
    END                                                          AS status_icon,
    ts.total_tasks,
    ts.ok_tasks,
    ts.failed_tasks,
    ts.upstream_fail_tasks,
    ts.retried_tasks,
    ts.avg_task_ms,
    ts.total_work_ms,
    ROUND(EXTRACT(EPOCH FROM (lr.end_ts - lr.start_ts)) * 1000, 2) AS wall_clock_ms,
    COALESCE(sv.sla_breach_count, 0)                             AS sla_breach_count,
    sv.sla_breach_detail,
    cf.failure_detail,
    lr.start_ts                                                  AS run_start_ts,
    lr.end_ts                                                    AS run_end_ts
FROM last_run lr
LEFT JOIN task_summary ts       ON ts.run_id = lr.run_id
LEFT JOIN critical_failures cf  ON cf.run_id = lr.run_id
LEFT JOIN sla_violations sv     ON sv.run_id = lr.run_id
ORDER BY lr.dag_name;

-- ============================================================
-- SEMANTIC 2: Weekly Reliability — 8 semanas deslizantes
-- ============================================================
-- Grain: 1 linha por (dag_name, semana ISO).
-- Uso: revisão semanal de SRE / relatório de confiabilidade.
CREATE OR REPLACE VIEW dag_semantic.rpt_weekly_reliability AS
WITH weekly AS (
    SELECT
        dag_name,
        DATE_TRUNC('week', run_date)::DATE                              AS week_start,
        COUNT(*)                                                        AS total_runs,
        COUNT(*) FILTER (WHERE status = 'SUCCESS')                      AS ok_runs,
        COUNT(*) FILTER (WHERE status = 'FAILED')                       AS failed_runs,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE status = 'SUCCESS') / COUNT(*), 2
        )                                                               AS success_rate,
        ROUND(AVG(EXTRACT(EPOCH FROM (end_ts - start_ts)) * 1000), 2)  AS avg_wall_clock_ms
    FROM dag_engine.dag_runs
    WHERE run_type = 'INCREMENTAL'
      AND run_date >= (
          (SELECT MAX(run_date) FROM dag_engine.dag_runs WHERE run_type = 'INCREMENTAL')
          - INTERVAL '56 days'
      )
    GROUP BY dag_name, DATE_TRUNC('week', run_date)::DATE
),
trended AS (
    SELECT *,
        ROUND(AVG(success_rate) OVER (
            PARTITION BY dag_name
            ORDER BY week_start
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ), 2) AS prev_4w_avg_success_rate,
        ROUND(AVG(avg_wall_clock_ms) OVER (
            PARTITION BY dag_name
            ORDER BY week_start
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ), 2) AS prev_4w_avg_wall_ms
    FROM weekly
)
SELECT
    dag_name,
    week_start,
    total_runs,
    ok_runs,
    failed_runs,
    success_rate,
    avg_wall_clock_ms,
    prev_4w_avg_success_rate,
    prev_4w_avg_wall_ms,
    ROUND(success_rate - COALESCE(prev_4w_avg_success_rate, success_rate), 2)
                                                                        AS success_rate_delta,
    CASE
        WHEN success_rate >= 99 THEN '🟢 Excelente'
        WHEN success_rate >= 95 THEN '🟡 Aceitável'
        WHEN success_rate >= 85 THEN '🟠 Atenção'
        ELSE                         '🔴 Crítico'
    END                                                                 AS weekly_signal
FROM trended
ORDER BY dag_name, week_start DESC;

-- ============================================================
-- SEMANTIC 3: Monthly SLA Delivery — entrega mensal vs SLA
-- ============================================================
-- Grain: 1 linha por (dag_name, mês).
-- Uso: relatório mensal para diretoria / revisão de SLA contratual.
CREATE OR REPLACE VIEW dag_semantic.rpt_monthly_sla_delivery AS
WITH monthly_runs AS (
    SELECT
        dag_name,
        DATE_TRUNC('month', run_date)::DATE                             AS mes_referencia,
        COUNT(*)                                                        AS total_runs,
        COUNT(*) FILTER (WHERE status = 'SUCCESS')                      AS ok_runs,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE status = 'SUCCESS') / COUNT(*), 2
        )                                                               AS success_rate_mes
    FROM dag_engine.dag_runs
    WHERE run_type = 'INCREMENTAL'
    GROUP BY dag_name, DATE_TRUNC('month', run_date)::DATE
),
monthly_sla AS (
    SELECT
        DATE_TRUNC('month', run_date)::DATE                             AS mes_referencia,
        task_name,
        COUNT(*) FILTER (WHERE sla_status != '🟢 DENTRO DO SLA')       AS breach_count,
        COUNT(*)                                                        AS total_checked,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE sla_status != '🟢 DENTRO DO SLA') / COUNT(*), 2
        )                                                               AS breach_rate_pct
    FROM dag_medallion.gold_sla_breach
    GROUP BY DATE_TRUNC('month', run_date)::DATE, task_name
),
agg_sla AS (
    SELECT
        mes_referencia,
        COUNT(DISTINCT task_name) FILTER (WHERE breach_count > 0)       AS tasks_com_breach,
        ROUND(AVG(breach_rate_pct), 2)                                  AS avg_breach_rate_pct,
        ROUND(
            100.0 * COUNT(DISTINCT task_name) FILTER (WHERE breach_count = 0)
            / NULLIF(COUNT(DISTINCT task_name), 0), 2
        )                                                               AS tasks_dentro_sla_pct
    FROM monthly_sla
    -- Guarda: ignora tasks com baseline SLA ainda não estabilizado (< 3 runs)
    WHERE total_checked >= 3
    GROUP BY mes_referencia
),
agg_health AS (
    SELECT
        h.mes_referencia,
        ROUND(AVG(h.score_medio_mes), 2)                                AS avg_health_score,
        ROUND(MIN(h.score_minimo_mes), 2)                               AS min_health_score,
        COUNT(DISTINCT h.task_name)                                     AS tasks_monitoradas
    FROM dag_medallion.task_health_array_mensal h
    -- Exclui tasks ainda em período de calibração (histórico insuficiente)
    JOIN dag_medallion.gold_pipeline_health gph ON gph.task_name = h.task_name
    WHERE gph.health_label != '🔵 CALIBRANDO'
    GROUP BY h.mes_referencia
)
SELECT
    mr.dag_name,
    mr.mes_referencia,
    mr.total_runs,
    mr.ok_runs,
    mr.success_rate_mes,
    COALESCE(asl.tasks_dentro_sla_pct, 100)                            AS pct_tasks_dentro_sla,
    COALESCE(asl.tasks_com_breach, 0)                                  AS tasks_com_breach_sla,
    COALESCE(asl.avg_breach_rate_pct, 0)                               AS avg_breach_rate_pct,
    COALESCE(ah.avg_health_score, 0)                                   AS avg_health_score,
    COALESCE(ah.min_health_score, 0)                                   AS min_health_score,
    CASE
        WHEN mr.success_rate_mes >= 99.5
         AND COALESCE(asl.tasks_dentro_sla_pct, 100) >= 95             THEN '✅ SLA Cumprido'
        WHEN mr.success_rate_mes >= 97
         AND COALESCE(asl.tasks_dentro_sla_pct, 100) >= 80             THEN '⚠️ SLA Parcial'
        ELSE                                                                 '🔴 SLA Violado'
    END                                                                 AS commitment_status
FROM monthly_runs mr
LEFT JOIN agg_sla  asl ON asl.mes_referencia = mr.mes_referencia
LEFT JOIN agg_health ah ON ah.mes_referencia = mr.mes_referencia
ORDER BY mr.dag_name, mr.mes_referencia DESC;

-- ============================================================
-- SEMANTIC 4: Version Impact — impacto de cada deploy por task
-- ============================================================
-- Grain: 1 linha por (task_name, version_id).
-- Uso: post-mortem de deploy / validação de otimização / rollback candidato.
DROP VIEW IF EXISTS dag_semantic.rpt_version_impact;
CREATE OR REPLACE VIEW dag_semantic.rpt_version_impact AS
WITH version_stats AS (
    SELECT
        f.task_name,
        -- Rollup: chunks colapsam de volta para a task pai (fix: chunks como nova task)
        regexp_replace(f.task_name, '_chunk_\d+$', '')                  AS parent_task_name,
        dv.version_id,
        dv.version_tag,
        dv.deployed_at,
        dv.change_summary,
        COUNT(*)                                                        AS total_execs,
        COUNT(*) FILTER (WHERE f.final_status = 'SUCCESS')             AS ok_execs,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE f.final_status = 'SUCCESS') / COUNT(*), 2
        )                                                               AS success_rate,
        -- avg_ms inclui queue_wait; avg_exec_ms isola só o tempo de execução real
        ROUND(AVG(f.duration_ms) FILTER (WHERE f.final_status = 'SUCCESS'), 2)
                                                                        AS avg_ms,
        ROUND(AVG(f.queue_wait_ms) FILTER (WHERE f.final_status = 'SUCCESS'), 2)
                                                                        AS avg_queue_wait_ms,
        ROUND(AVG(f.duration_ms - COALESCE(f.queue_wait_ms, 0)) FILTER (WHERE f.final_status = 'SUCCESS'), 2)
                                                                        AS avg_exec_ms,
        ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY f.duration_ms
        ) FILTER (WHERE f.final_status = 'SUCCESS'))::NUMERIC, 2)      AS p95_ms,
        ROUND(STDDEV(f.duration_ms) FILTER (WHERE f.final_status = 'SUCCESS')::NUMERIC, 2)
                                                                        AS std_ms
    FROM dag_medallion.fato_task_exec f
    JOIN dag_engine.dag_runs dr ON dr.run_id = f.run_id
    JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
    WHERE f.run_type = 'INCREMENTAL'
    GROUP BY f.task_name, dv.version_id, dv.version_tag, dv.deployed_at, dv.change_summary
),
compared AS (
    SELECT *,
        LAG(avg_ms)           OVER (PARTITION BY task_name ORDER BY deployed_at) AS prev_avg_ms,
        LAG(avg_queue_wait_ms) OVER (PARTITION BY task_name ORDER BY deployed_at) AS prev_avg_queue_wait_ms,
        LAG(avg_exec_ms)      OVER (PARTITION BY task_name ORDER BY deployed_at) AS prev_avg_exec_ms,
        LAG(success_rate)     OVER (PARTITION BY task_name ORDER BY deployed_at) AS prev_success_rate,
        LAG(p95_ms)           OVER (PARTITION BY task_name ORDER BY deployed_at) AS prev_p95_ms,
        LAG(version_tag)      OVER (PARTITION BY task_name ORDER BY deployed_at) AS prev_version_tag
    FROM version_stats
)
SELECT
    task_name,
    parent_task_name,
    version_tag,
    prev_version_tag,
    deployed_at,
    change_summary,
    total_execs,
    ok_execs,
    success_rate,
    prev_success_rate,
    ROUND(success_rate - COALESCE(prev_success_rate, success_rate), 2)         AS success_rate_delta,
    avg_ms,
    prev_avg_ms,
    ROUND(avg_ms - COALESCE(prev_avg_ms, avg_ms), 2)                           AS avg_ms_delta,
    -- Deltas separados: execução pura vs tempo de fila
    avg_exec_ms,
    prev_avg_exec_ms,
    ROUND(avg_exec_ms - COALESCE(prev_avg_exec_ms, avg_exec_ms), 2)            AS avg_exec_ms_delta,
    avg_queue_wait_ms,
    prev_avg_queue_wait_ms,
    ROUND(avg_queue_wait_ms - COALESCE(prev_avg_queue_wait_ms, avg_queue_wait_ms), 2)
                                                                                AS avg_queue_wait_delta,
    p95_ms,
    prev_p95_ms,
    ROUND(p95_ms - COALESCE(prev_p95_ms, p95_ms), 2)                           AS p95_ms_delta,
    -- Veredicto usa avg_exec_ms para não penalizar tasks que melhoram no scheduler
    CASE
        WHEN prev_avg_exec_ms IS NULL
                                                                                THEN '🆕 Primeira Versão'
        WHEN success_rate < COALESCE(prev_success_rate, success_rate) - 5
                                                                                THEN '🟠 Queda de Confiabilidade'
        WHEN avg_exec_ms > COALESCE(prev_avg_exec_ms, avg_exec_ms) * 1.2
                                                                                THEN '🔴 Regressão de Performance'
        WHEN avg_exec_ms < COALESCE(prev_avg_exec_ms, avg_exec_ms) * 0.8
                                                                                THEN '🟢 Melhoria Significativa'
        ELSE                                                                         '✅ Estável'
    END                                                                         AS impact_signal
FROM compared
ORDER BY task_name, deployed_at DESC;

-- ============================================================
-- SEMANTIC PROC: Emite relatório oncall via pg_notify
-- ============================================================
-- Chamado automaticamente por proc_run_medallion ao final de cada run.
-- Consumidores: webhooks, alertas de Slack, dashboards reativos.
-- Canal: 'dag_reports' — payload JSON com event='oncall_handoff'.
CREATE OR REPLACE PROCEDURE dag_semantic.proc_emit_reports(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_run_date DATE;
    v_dag_name VARCHAR(100);
    v_payload  JSONB;
BEGIN
    SELECT run_date, dag_name
    INTO v_run_date, v_dag_name
    FROM dag_engine.dag_runs
    WHERE run_id = p_run_id;

    SELECT jsonb_agg(row_to_json(r)::JSONB)
    INTO v_payload
    FROM dag_semantic.rpt_oncall_handoff r
    WHERE r.run_date = v_run_date
      AND r.dag_name = v_dag_name;

    PERFORM pg_notify(
        'dag_reports',
        jsonb_build_object(
            'event',    'oncall_handoff',
            'run_id',   p_run_id,
            'run_date', v_run_date,
            'dag_name', v_dag_name,
            'payload',  COALESCE(v_payload, '[]'::JSONB)
        )::TEXT
    );
END;
$$;

-- ==============================================================================
-- 8. BACKLOG: DAG VERSIONING (Rastreabilidade Completa de Topologia)
-- ==============================================================================
-- Resolve os 3 problemas centrais que o Airflow sofre até hoje:
--   • Qual spec gerou qual run? (join histórico por version_id)
--   • Como comparar DAGs de épocas diferentes? (fn_diff_versions)
--   • Como fazer catchup fiel à topologia original? (snapshot replay)
-- ==============================================================================

-- ==============================================================================
-- 8.0 SEMANTIC VERSIONING ENFORCEMENT
-- ==============================================================================
-- Garante que nenhum deploy aceite tags livres como 'hotfix' ou 'final_v3'.
-- Três guardas embutidas em proc_deploy_dag:
--   S1 — Formato    → vMAJOR.MINOR.PATCH obrigatório
--   S2 — Monotonia  → novo tag deve ser estritamente maior que o ativo
--   S3 — Adequação  → under-bump BLOQUEADO | over-bump permitido com aviso
-- ==============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- fn_parse_semver: valida e decompõe um tag no formato vMAJOR.MINOR.PATCH
-- Retorna (major, minor, patch) como INTs ou levanta exceção se inválido.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION dag_engine.fn_parse_semver(
    p_tag TEXT
) RETURNS TABLE (major INT, minor INT, patch INT)
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    v_parts TEXT[];
BEGIN
    -- Aceita tanto 'v1.2.3' quanto '1.2.3'
    v_parts := regexp_match(
        lower(trim(p_tag)),
        '^v?(\d+)\.(\d+)\.(\d+)$'
    );

    IF v_parts IS NULL THEN
        RAISE EXCEPTION
            'Semver Error: tag "%" inválido. Formato esperado: vMAJOR.MINOR.PATCH (ex: v1.2.3)',
            p_tag;
    END IF;

    major := v_parts[1]::INT;
    minor := v_parts[2]::INT;
    patch := v_parts[3]::INT;
    RETURN NEXT;
END;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- fn_compare_semver: comparação total entre dois tags semver
-- Retorna -1 (v1 < v2), 0 (iguais), 1 (v1 > v2)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION dag_engine.fn_compare_semver(
    p_v1 TEXT,
    p_v2 TEXT
) RETURNS INT
LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN ROW(v1.major, v1.minor, v1.patch) < ROW(v2.major, v2.minor, v2.patch) THEN -1
        WHEN ROW(v1.major, v1.minor, v1.patch) > ROW(v2.major, v2.minor, v2.patch) THEN  1
        ELSE 0
    END
    FROM dag_engine.fn_parse_semver(p_v1) v1,
         dag_engine.fn_parse_semver(p_v2) v2;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- fn_next_semver: dado o tag atual e o tipo de bump, retorna o próximo tag
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION dag_engine.fn_next_semver(
    p_current_tag TEXT,
    p_bump        TEXT   -- 'MAJOR' | 'MINOR' | 'PATCH'
) RETURNS TEXT
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    v RECORD;
BEGIN
    IF p_bump NOT IN ('MAJOR', 'MINOR', 'PATCH') THEN
        RAISE EXCEPTION 'Bump inválido: %. Use MAJOR, MINOR ou PATCH.', p_bump;
    END IF;
    SELECT * INTO v FROM dag_engine.fn_parse_semver(p_current_tag);
    RETURN 'v' || CASE p_bump
        WHEN 'MAJOR' THEN (v.major + 1)::TEXT || '.0.0'
        WHEN 'MINOR' THEN  v.major::TEXT || '.' || (v.minor + 1)::TEXT || '.0'
        WHEN 'PATCH' THEN  v.major::TEXT || '.' || v.minor::TEXT || '.' || (v.patch + 1)::TEXT
    END;
END;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- fn_higher_bump: retorna o bump de maior precedência entre dois valores.
-- Hierarquia: MAJOR > MINOR > PATCH > NONE
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION dag_engine.fn_higher_bump(
    p_current   TEXT,
    p_candidate TEXT
) RETURNS TEXT LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN 'MAJOR' IN (p_current, p_candidate) THEN 'MAJOR'
        WHEN 'MINOR' IN (p_current, p_candidate) THEN 'MINOR'
        WHEN 'PATCH' IN (p_current, p_candidate) THEN 'PATCH'
        ELSE 'NONE'
    END;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- fn_suggest_semver_bump: analisa o diff entre o spec ativo e o novo spec
-- e retorna o bump recomendado + o próximo tag sugerido + detalhes do diff.
-- Pode ser chamado como dry-run antes de qualquer deploy (útil em CI/CD).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION dag_engine.fn_suggest_semver_bump(
    p_dag_name TEXT,
    p_new_spec JSONB
) RETURNS TABLE (
    recommended_bump TEXT,
    suggested_tag    TEXT,
    current_tag      TEXT,
    diff_summary     TEXT
)
LANGUAGE plpgsql AS $$
DECLARE
    v_active_version_id INT;
    v_current_tag       TEXT;
    v_active_spec       JSONB;
    v_bump              TEXT   := 'NONE';
    v_diff_lines        TEXT[] := '{}';
    v_rec               RECORD;
BEGIN
    SELECT version_id, version_tag, spec
    INTO v_active_version_id, v_current_tag, v_active_spec
    FROM dag_engine.dag_versions
    WHERE dag_name = p_dag_name AND is_active = TRUE;

    IF NOT FOUND THEN
        recommended_bump := 'INITIAL';
        suggested_tag    := 'v1.0.0';
        current_tag      := NULL;
        diff_summary     := 'Primeiro deploy.';
        RETURN NEXT; RETURN;
    END IF;

    -- ── Campos raiz do manifest ────────────────────────────────────────────

    IF (v_active_spec->>'description') IS DISTINCT FROM (p_new_spec->>'description') THEN
        v_bump       := dag_engine.fn_higher_bump(v_bump, 'PATCH');
        v_diff_lines := array_append(v_diff_lines, format(
            '  [PATCH] description: "%s" → "%s"',
            COALESCE(v_active_spec->>'description', 'null'),
            COALESCE(p_new_spec->>'description',    'null')));
    END IF;

    IF (v_active_spec->>'schedule') IS DISTINCT FROM (p_new_spec->>'schedule') THEN
        v_bump       := dag_engine.fn_higher_bump(v_bump, 'MINOR');
        v_diff_lines := array_append(v_diff_lines, format(
            '  [MINOR] schedule: "%s" → "%s"',
            COALESCE(v_active_spec->>'schedule', 'null'),
            COALESCE(p_new_spec->>'schedule',    'null')));
    END IF;

    -- ── Diff de tasks ──────────────────────────────────────────────────────
    FOR v_rec IN
        WITH
        v1 AS (
            SELECT
                elem->>'task_name'                                AS task_name,
                elem->>'call_fn'                                  AS call_fn,
                elem->'dependencies'                              AS deps,
                COALESCE((elem->>'max_retries')::INT,        0)  AS max_retries,
                COALESCE((elem->>'retry_delay_seconds')::INT, 5) AS retry_delay,
                (elem->>'sla_ms_override')::BIGINT               AS sla_ms,
                elem->>'layer'                                    AS layer,
                elem->'chunk_config'                              AS chunk_cfg
            FROM jsonb_array_elements(
                COALESCE(v_active_spec->'tasks', v_active_spec)
            ) AS elem
        ),
        v2 AS (
            SELECT
                elem->>'task_name'                                AS task_name,
                elem->>'call_fn'                                  AS call_fn,
                elem->'dependencies'                              AS deps,
                COALESCE((elem->>'max_retries')::INT,        0)  AS max_retries,
                COALESCE((elem->>'retry_delay_seconds')::INT, 5) AS retry_delay,
                (elem->>'sla_ms_override')::BIGINT               AS sla_ms,
                elem->>'layer'                                    AS layer,
                elem->'chunk_config'                              AS chunk_cfg
            FROM jsonb_array_elements(
                COALESCE(p_new_spec->'tasks', p_new_spec)
            ) AS elem
        )
        SELECT 'MAJOR'::TEXT AS bump, 'REMOVED'::TEXT AS change_type,
               v1.task_name, 'task removida'::TEXT AS detail
        FROM v1 WHERE NOT EXISTS (SELECT 1 FROM v2 WHERE v2.task_name = v1.task_name)
        UNION ALL
        SELECT 'MAJOR', 'ADDED', v2.task_name, 'task adicionada'
        FROM v2 WHERE NOT EXISTS (SELECT 1 FROM v1 WHERE v1.task_name = v2.task_name)
        UNION ALL
        SELECT 'MAJOR', 'DEPS_CHANGED', v1.task_name,
               'deps: ' || v1.deps::TEXT || ' → ' || v2.deps::TEXT
        FROM v1 JOIN v2 USING (task_name) WHERE v1.deps::TEXT IS DISTINCT FROM v2.deps::TEXT
        UNION ALL
        SELECT 'MAJOR', 'CALL_FN_CHANGED', v1.task_name,
               'call_fn: ' || COALESCE(v1.call_fn,'null') || ' → ' || COALESCE(v2.call_fn,'null')
        FROM v1 JOIN v2 USING (task_name) WHERE v1.call_fn IS DISTINCT FROM v2.call_fn
        UNION ALL
        SELECT 'MINOR', 'CHUNK_CHANGED', v1.task_name,
               'chunk_config: ' || COALESCE(v1.chunk_cfg::TEXT,'null')
               || ' → ' || COALESCE(v2.chunk_cfg::TEXT,'null')
        FROM v1 JOIN v2 USING (task_name)
        WHERE v1.chunk_cfg::TEXT IS DISTINCT FROM v2.chunk_cfg::TEXT
        UNION ALL
        SELECT 'PATCH', 'RETRY_CHANGED', v1.task_name,
               'max_retries: ' || v1.max_retries || ' → ' || v2.max_retries
        FROM v1 JOIN v2 USING (task_name) WHERE v1.max_retries != v2.max_retries
        UNION ALL
        SELECT 'PATCH', 'DELAY_CHANGED', v1.task_name,
               'retry_delay_seconds: ' || v1.retry_delay || ' → ' || v2.retry_delay
        FROM v1 JOIN v2 USING (task_name) WHERE v1.retry_delay != v2.retry_delay
        UNION ALL
        SELECT 'PATCH', 'SLA_CHANGED', v1.task_name,
               'sla_ms_override: ' || COALESCE(v1.sla_ms::TEXT,'null')
               || ' → ' || COALESCE(v2.sla_ms::TEXT,'null')
        FROM v1 JOIN v2 USING (task_name) WHERE v1.sla_ms IS DISTINCT FROM v2.sla_ms
        UNION ALL
        SELECT 'PATCH', 'LAYER_CHANGED', v1.task_name,
               'layer: ' || COALESCE(v1.layer,'null') || ' → ' || COALESCE(v2.layer,'null')
        FROM v1 JOIN v2 USING (task_name) WHERE v1.layer IS DISTINCT FROM v2.layer
    LOOP
        v_bump       := dag_engine.fn_higher_bump(v_bump, v_rec.bump);
        v_diff_lines := array_append(v_diff_lines, format(
            '  [%s] [%s] %s: %s',
            v_rec.bump, v_rec.change_type, v_rec.task_name, v_rec.detail));
    END LOOP;

    IF v_bump = 'NONE' THEN
        recommended_bump := 'NONE';
        suggested_tag    := v_current_tag;
        current_tag      := v_current_tag;
        diff_summary     := 'Spec idêntico ao ativo — nenhum bump necessário.';
        RETURN NEXT; RETURN;
    END IF;

    recommended_bump := v_bump;
    suggested_tag    := dag_engine.fn_next_semver(v_current_tag, v_bump);
    current_tag      := v_current_tag;
    diff_summary     := array_to_string(v_diff_lines, E'\n');
    RETURN NEXT;
END;
$$;

-- 8.1 Tabela de versões do spec (snapshot imutável de cada deploy)
CREATE TABLE IF NOT EXISTS dag_engine.dag_versions (
    version_id      SERIAL PRIMARY KEY,
    dag_name        VARCHAR(100) NOT NULL,   -- extraído do campo "name" do manifest
    description     TEXT,                    -- extraído do campo "description" do manifest
    schedule        TEXT,                    -- extraído do campo "schedule" do manifest (cron expr)
    version_tag     VARCHAR(50) NOT NULL,
    spec            JSONB NOT NULL,          -- o manifest completo (inclui "tasks")
    spec_hash       TEXT GENERATED ALWAYS AS (md5(spec::text)) STORED,
    deployed_at     TIMESTAMP DEFAULT clock_timestamp(),
    deployed_by     TEXT DEFAULT current_user,
    is_active       BOOLEAN DEFAULT FALSE,
    change_summary  TEXT,
    parent_version  INT REFERENCES dag_engine.dag_versions(version_id)
);

-- Uma versão ativa por DAG (índice parcial por dag_name — suporta múltiplos DAGs)
-- Migração: garante novas colunas e recria índice com a assinatura correta
ALTER TABLE dag_engine.dag_versions ADD COLUMN IF NOT EXISTS dag_name    VARCHAR(100);
ALTER TABLE dag_engine.dag_versions ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE dag_engine.dag_versions ADD COLUMN IF NOT EXISTS schedule    TEXT;
DROP INDEX IF EXISTS idx_dag_one_active;
CREATE UNIQUE INDEX IF NOT EXISTS idx_dag_one_active
ON dag_engine.dag_versions(dag_name, is_active)
WHERE is_active = TRUE;

-- 8.2 Carimba cada run com a versão que a gerou (rastreabilidade de proveniência)
ALTER TABLE dag_engine.dag_runs
    ADD COLUMN IF NOT EXISTS version_id INT REFERENCES dag_engine.dag_versions(version_id);

-- 8.2.1 Tabelas de Saúde Cumulativa (requerem dag_versions como FK)
-- DROP necessário apenas se a PK mudou (migração de schema):
-- a PK agora é tripla (task_name, version_id, data_snapshot)
CREATE TABLE IF NOT EXISTS dag_medallion.task_health_cumulative (
    task_name           VARCHAR(100) NOT NULL,
    version_id          INT          NOT NULL REFERENCES dag_engine.dag_versions(version_id),
    data_snapshot       DATE         NOT NULL,
    run_type            VARCHAR(20)  NOT NULL DEFAULT 'INCREMENTAL',
    health_bitmap_32d   BIGINT       NOT NULL DEFAULT 0,
    datas_saudavel      DATE[]       NOT NULL DEFAULT '{}',
    total_dias_saudavel INT GENERATED ALWAYS AS (CARDINALITY(datas_saudavel)) STORED,
    PRIMARY KEY (task_name, version_id, data_snapshot)
);

CREATE INDEX IF NOT EXISTS idx_health_cumul_version_snapshot
    ON dag_medallion.task_health_cumulative (version_id, data_snapshot);

CREATE TABLE IF NOT EXISTS dag_medallion.task_health_array_mensal (
    task_name        VARCHAR(100) NOT NULL,
    version_id       INT          NOT NULL REFERENCES dag_engine.dag_versions(version_id),
    mes_referencia   DATE         NOT NULL,
    scores_diarios   NUMERIC[]    NOT NULL DEFAULT '{}',
    score_minimo_mes NUMERIC      NOT NULL DEFAULT 100,
    score_medio_mes  NUMERIC      NOT NULL DEFAULT 0,
    PRIMARY KEY (task_name, version_id, mes_referencia)
);

-- 8.2.2 Views Gold de Saúde Cumulativa (dependem das tabelas acima)
-- Criadas aqui para garantir que task_health_cumulative já existe.
CREATE OR REPLACE VIEW dag_medallion.gold_health_streak AS
SELECT
    task_name,
    version_id,
    data_snapshot,
    total_dias_saudavel,
    dag_medallion.was_healthy_on_day(health_bitmap_32d, 0)  AS healthy_d0,
    dag_medallion.was_healthy_on_day(health_bitmap_32d, 1)  AS healthy_d1,
    dag_medallion.was_healthy_on_day(health_bitmap_32d, 7)  AS healthy_d7,
    dag_medallion.was_healthy_on_day(health_bitmap_32d, 14) AS healthy_d14,
    dag_medallion.was_healthy_on_day(health_bitmap_32d, 30) AS healthy_d30,
    health_bitmap_32d::BIT(32)                              AS bitmap_visual
FROM dag_medallion.task_health_cumulative
WHERE run_type = 'INCREMENTAL'
ORDER BY task_name, version_id, data_snapshot DESC;

CREATE OR REPLACE VIEW dag_medallion.gold_degradation_streaks AS
WITH daily_health AS (
    SELECT
        task_name,
        version_id,
        data_snapshot,
        CASE WHEN dag_medallion.was_healthy_on_day(health_bitmap_32d, 0)
             THEN 'SAUDAVEL' ELSE 'DEGRADADO'
        END AS health_state
    FROM dag_medallion.task_health_cumulative
    WHERE run_type = 'INCREMENTAL'
),
streak_started AS (
    SELECT
        task_name,
        version_id,
        data_snapshot,
        health_state,
        LAG(health_state) OVER w IS DISTINCT FROM health_state AS did_change
    FROM daily_health
    WINDOW w AS (PARTITION BY task_name, version_id ORDER BY data_snapshot)
),
streak_identified AS (
    SELECT *,
        SUM(CASE WHEN did_change THEN 1 ELSE 0 END)
            OVER (PARTITION BY task_name, version_id ORDER BY data_snapshot) AS streak_id
    FROM streak_started
)
SELECT
    task_name,
    version_id,
    health_state,
    MIN(data_snapshot) AS streak_start,
    MAX(data_snapshot) AS streak_end,
    COUNT(*)           AS streak_days,
    MAX(data_snapshot) = (
        SELECT MAX(data_snapshot)
        FROM dag_medallion.task_health_cumulative t
        WHERE t.task_name  = si.task_name
          AND t.version_id = si.version_id
          AND t.run_type   = 'INCREMENTAL'
    )                  AS is_current
FROM streak_identified si
GROUP BY task_name, version_id, health_state, streak_id
ORDER BY task_name, version_id, streak_start DESC;

-- 8.2.3 Procedures de Saúde Cumulativa (dependem de dag_runs.version_id e dag_versions)
-- Movidas para cá (após 8.2) para garantir que dag_runs.version_id já existe quando
-- a procedure tenta ler essa coluna em runtime — seguro mesmo em execução do zero.

-- Etapa 3: Upsert de Saúde Cumulativa (Bitmap)
-- Duas guardas de integridade — não existe GUARDA 3:
--   o reset de versão é emergente: yesterday filtra por version_id,
--   então a primeira run de uma versão nova começa com bitmap zero.
CREATE OR REPLACE PROCEDURE dag_medallion.proc_upsert_health_cumulative(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_run_date   DATE;
    v_run_type   VARCHAR(20);
    v_version_id INT;
    v_schedule   TEXT;
BEGIN
    SELECT dr.run_date, dr.run_type, dr.version_id, dv.schedule
    INTO v_run_date, v_run_type, v_version_id, v_schedule
    FROM dag_engine.dag_runs dr
    LEFT JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
    WHERE dr.run_id = p_run_id;

    -- ── GUARDA 0: version_id NULL = DAG sem versão registrada ───────────────
    -- Ocorre quando dag_versions está vazia (deploy sem proc_deploy_dag prévio).
    -- Não há bitmap a acumular sem rastreabilidade de versão.
    IF v_version_id IS NULL THEN
        RAISE NOTICE '⏭️ [health_cumulative] Ignorado: version_id NULL (run_id=%).', p_run_id;
        RETURN;
    END IF;

    -- ── GUARDA 1: Backfill nunca alimenta o bitmap operacional ──────────────
    -- Runs de backfill representam dados históricos, não o comportamento atual
    -- do pipeline em produção. Misturar os dois regimes corromperia o streak.
    IF v_run_type = 'BACKFILL' THEN
        RAISE NOTICE '⏭️ [health_cumulative] Ignorado: run BACKFILL (%).', v_run_date;
        RETURN;
    END IF;

    -- ── GUARDA 2: Mesmo (version_id, data_snapshot) = sobrescreve sem shift ──
    -- Garante idempotência: reprocessar o mesmo dia não desloca o bitmap.
    -- Apenas corrige o bit 31 (hoje) para refletir o score atual.
    IF EXISTS (
        SELECT 1 FROM dag_medallion.task_health_cumulative
        WHERE version_id    = v_version_id
          AND data_snapshot = v_run_date
    ) THEN
        RAISE NOTICE '⏭️ [health_cumulative] Reprocessamento de % (v%). Sobrescrevendo sem shift.',
            v_run_date, v_version_id;
        UPDATE dag_medallion.task_health_cumulative thc
        SET health_bitmap_32d = CASE
                WHEN g.health_label != '🔵 CALIBRANDO' AND g.health_score >= 95
                THEN (thc.health_bitmap_32d |  (1::BIGINT << 31))
                ELSE (thc.health_bitmap_32d & ~(1::BIGINT << 31))
            END
        FROM dag_medallion.gold_pipeline_health g
        WHERE thc.task_name     = g.task_name
          AND thc.version_id    = v_version_id
          AND thc.data_snapshot = v_run_date;
        RETURN;
    END IF;

    -- ── Acumulação principal ─────────────────────────────────────────────────
    -- yesterday é scoped por version_id: se esta é a primeira run da versão,
    -- o CTE retorna zero linhas e o FULL OUTER JOIN preenche tudo com zero.
    -- O reset de versão é emergente — não há lógica especial necessária.
    WITH yesterday AS (
        SELECT task_name, health_bitmap_32d, datas_saudavel
        FROM dag_medallion.task_health_cumulative
        WHERE version_id    = v_version_id
          AND data_snapshot = dag_engine.fn_prev_scheduled_date(
                                  COALESCE(v_schedule, '0 0 * * *'),
                                  v_run_date
                              )
    ),
    today AS (
        SELECT task_name,
               health_score >= 95
               AND health_label != '🔵 CALIBRANDO' AS is_healthy
        FROM dag_medallion.gold_pipeline_health
    ),
    merged AS (
        SELECT
            COALESCE(y.task_name, t.task_name)   AS task_name,
            v_version_id                          AS version_id,
            v_run_date                            AS data_snapshot,
            'INCREMENTAL'::VARCHAR(20)            AS run_type,
            dag_medallion.shift_health_bitmap(
                y.health_bitmap_32d,
                COALESCE(t.is_healthy, FALSE)
            )                                     AS health_bitmap_32d,
            COALESCE(y.datas_saudavel, ARRAY[]::DATE[])
            || CASE WHEN COALESCE(t.is_healthy, FALSE)
                    THEN ARRAY[v_run_date]
                    ELSE ARRAY[]::DATE[]
               END                                AS datas_saudavel
        FROM yesterday y
        FULL OUTER JOIN today t ON y.task_name = t.task_name
    )
    INSERT INTO dag_medallion.task_health_cumulative
        (task_name, version_id, data_snapshot, run_type, health_bitmap_32d, datas_saudavel)
    SELECT task_name, version_id, data_snapshot, run_type, health_bitmap_32d, datas_saudavel
    FROM merged
    ON CONFLICT (task_name, version_id, data_snapshot) DO UPDATE SET
        health_bitmap_32d = EXCLUDED.health_bitmap_32d,
        datas_saudavel    = EXCLUDED.datas_saudavel;
END;
$$;

-- Etapa 5: Upsert de Saúde Mensal (Array)
CREATE OR REPLACE PROCEDURE dag_medallion.proc_upsert_health_array(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_run_date     DATE;
    v_run_type     VARCHAR(20);
    v_version_id   INT;
    v_primeiro_dia DATE;
    v_dia_do_mes   INT;
BEGIN
    SELECT run_date, run_type, version_id
    INTO v_run_date, v_run_type, v_version_id
    FROM dag_engine.dag_runs WHERE run_id = p_run_id;

    -- Espelha GUARDA 0: sem version_id não há rastreabilidade de versão
    IF v_version_id IS NULL THEN
        RAISE NOTICE '⏭️ [health_array] Ignorado: version_id NULL (run_id=%).', p_run_id;
        RETURN;
    END IF;

    -- Espelha GUARDA 1: arrays mensais também ignoram BACKFILL
    IF v_run_type = 'BACKFILL' THEN
        RAISE NOTICE '⏭️ [health_array] Ignorado: run BACKFILL (%).', v_run_date;
        RETURN;
    END IF;

    v_primeiro_dia := DATE_TRUNC('month', v_run_date)::DATE;
    v_dia_do_mes   := EXTRACT(DAY FROM v_run_date)::INT;

    WITH yesterday AS (
        SELECT task_name, scores_diarios
        FROM dag_medallion.task_health_array_mensal
        WHERE version_id     = v_version_id
          AND mes_referencia = v_primeiro_dia
    ),
    today AS (
        SELECT task_name, health_score
        FROM dag_medallion.gold_pipeline_health
    ),
    merged AS (
        SELECT
            COALESCE(y.task_name, t.task_name) AS task_name,
            v_version_id                        AS version_id,
            v_primeiro_dia                      AS mes_referencia,
            -- Padding de dias sem execução + append do score de hoje
            -- Posição no array = dia do mês (espelho do varejo)
            COALESCE(y.scores_diarios, '{}'::NUMERIC[])
            || CASE
                 WHEN v_dia_do_mes - 1
                      - COALESCE(array_length(y.scores_diarios, 1), 0) > 0
                 THEN array_fill(
                        NULL::NUMERIC,
                        ARRAY[v_dia_do_mes - 1
                              - COALESCE(array_length(y.scores_diarios, 1), 0)]
                      )
                 ELSE '{}'::NUMERIC[]
               END
            || ARRAY[t.health_score]           AS scores_diarios
        FROM yesterday y
        FULL OUTER JOIN today t ON y.task_name = t.task_name
    )
    INSERT INTO dag_medallion.task_health_array_mensal
        (task_name, version_id, mes_referencia, scores_diarios, score_minimo_mes, score_medio_mes)
    SELECT
        task_name,
        version_id,
        mes_referencia,
        scores_diarios,
        (SELECT MIN(v)   FROM unnest(scores_diarios) AS v WHERE v IS NOT NULL),
        ROUND((SELECT AVG(v) FROM unnest(scores_diarios) AS v WHERE v IS NOT NULL)::NUMERIC, 2)
    FROM merged
    ON CONFLICT (task_name, version_id, mes_referencia) DO UPDATE SET
        scores_diarios   = EXCLUDED.scores_diarios,
        score_minimo_mes = EXCLUDED.score_minimo_mes,
        score_medio_mes  = EXCLUDED.score_medio_mes;
END;
$$;

-- 8.3 proc_deploy_dag: ponto de entrada versionado com semver automático
-- Detecta redeploys silenciosos via hash e mantém a cadeia de versões pai → filho.
-- p_spec aceita o manifest envelope: { "name": ..., "description": ..., "schedule": ..., "tasks": [...] }
-- O bump semver é inferido automaticamente por fn_suggest_semver_bump — não é mais necessário
-- informar p_tag manualmente. Guarda S0 garante grafo único antes de qualquer escrita.
DROP PROCEDURE IF EXISTS dag_engine.proc_deploy_dag(JSONB, VARCHAR, TEXT);

CREATE OR REPLACE PROCEDURE dag_engine.proc_deploy_dag(
    p_spec    JSONB,
    p_summary TEXT DEFAULT NULL
)
LANGUAGE plpgsql AS $$
DECLARE
    v_new_version_id INT;
    v_current_hash   TEXT;
    v_parent_id      INT;
    v_dag_name       TEXT   := COALESCE(p_spec->>'name', 'default');
    v_new_hash       TEXT   := md5(p_spec::text);
    v_suggestion     RECORD;
    v_line           TEXT;
BEGIN
    IF p_spec->>'name' IS NULL THEN
        RAISE EXCEPTION 'Manifest Error: campo "name" é obrigatório no manifest.';
    END IF;

    -- ── GUARDA S0: Grafo único ─────────────────────────────────────────────
    PERFORM dag_engine.fn_validate_single_graph(p_spec);

    SELECT spec_hash, version_id
    INTO v_current_hash, v_parent_id
    FROM dag_engine.dag_versions
    WHERE dag_name = v_dag_name AND is_active = TRUE;

    IF v_current_hash = v_new_hash THEN
        RAISE WARNING '⚠️  Spec idêntico ao ativo para "%" (hash: %). Nenhuma versão criada.',
            v_dag_name, v_new_hash;
        RETURN;
    END IF;

    -- ── Inferência automática do bump ──────────────────────────────────────
    SELECT * INTO v_suggestion
    FROM dag_engine.fn_suggest_semver_bump(v_dag_name, p_spec);

    UPDATE dag_engine.dag_versions SET is_active = FALSE
    WHERE dag_name = v_dag_name AND is_active = TRUE;

    INSERT INTO dag_engine.dag_versions (
        dag_name, description, schedule,
        version_tag, spec, is_active,
        change_summary, parent_version
    ) VALUES (
        v_dag_name,
        p_spec->>'description',
        p_spec->>'schedule',
        v_suggestion.suggested_tag,
        p_spec,
        TRUE,
        COALESCE(p_summary, v_suggestion.diff_summary),
        v_parent_id
    )
    RETURNING version_id INTO v_new_version_id;

    CALL dag_engine.proc_load_dag_spec(p_spec);

    RAISE NOTICE '✅ DAG "%" deployada como "%" (bump: %) — Version ID: % (parent: %)',
        v_dag_name,
        v_suggestion.suggested_tag,
        v_suggestion.recommended_bump,
        v_new_version_id,
        v_parent_id;

    FOREACH v_line IN ARRAY string_to_array(v_suggestion.diff_summary, E'\n') LOOP
        IF trim(v_line) != '' THEN RAISE NOTICE '   %', v_line; END IF;
    END LOOP;
END;
$$;

-- 8.4 fn_diff_versions: compara a topologia de duas versões (como git diff para DAGs)
-- Detecta tarefas removidas, adicionadas, procedures alteradas e deps modificadas
CREATE OR REPLACE FUNCTION dag_engine.fn_diff_versions(
    p_v1 INT,
    p_v2 INT
) RETURNS TABLE (
    change_type TEXT,
    task_name   TEXT,
    detail      TEXT
) LANGUAGE sql AS $$
    WITH
    v1_tasks AS (
        SELECT elem->>'task_name'      AS task_name,
               elem->>'procedure_call' AS proc,
               elem->'dependencies'    AS deps
        FROM dag_engine.dag_versions,
             jsonb_array_elements(COALESCE(spec->'tasks', spec)) AS elem
        WHERE version_id = p_v1
    ),
    v2_tasks AS (
        SELECT elem->>'task_name'      AS task_name,
               elem->>'procedure_call' AS proc,
               elem->'dependencies'    AS deps
        FROM dag_engine.dag_versions,
             jsonb_array_elements(COALESCE(spec->'tasks', spec)) AS elem
        WHERE version_id = p_v2
    )
    SELECT 'REMOVED'::TEXT, v1.task_name, v1.proc
    FROM v1_tasks v1 WHERE NOT EXISTS (SELECT 1 FROM v2_tasks v2 WHERE v2.task_name = v1.task_name)
    UNION ALL
    SELECT 'ADDED'::TEXT, v2.task_name, v2.proc
    FROM v2_tasks v2 WHERE NOT EXISTS (SELECT 1 FROM v1_tasks v1 WHERE v1.task_name = v2.task_name)
    UNION ALL
    SELECT 'PROC_CHANGED'::TEXT, v1.task_name,
           'era: ' || v1.proc || ' → agora: ' || v2.proc
    FROM v1_tasks v1 JOIN v2_tasks v2 ON v1.task_name = v2.task_name
    WHERE v1.proc != v2.proc
    UNION ALL
    SELECT 'DEPS_CHANGED'::TEXT, v1.task_name,
           'era: ' || v1.deps::text || ' → agora: ' || v2.deps::text
    FROM v1_tasks v1 JOIN v2_tasks v2 ON v1.task_name = v2.task_name
    WHERE v1.deps::text != v2.deps::text;
$$;

-- 8.5 vw_version_lineage: árvore genealógica do pipeline com contagem de runs por versão
-- Fecha o loop de observabilidade: mostra quando cada versão foi deployada, o que mudou,
-- e quantas runs ela acumulou — permitindo correlacionar mudanças de topologia com
-- mudanças de performance nos dashboards gold do Medallion.
-- DROP CASCADE pois CREATE OR REPLACE VIEW não pode renomear colunas (ex: generation → dag_name)
DROP VIEW IF EXISTS dag_engine.vw_version_lineage CASCADE;
CREATE OR REPLACE VIEW dag_engine.vw_version_lineage AS
WITH RECURSIVE lineage AS (
    SELECT version_id, dag_name, version_tag, parent_version, deployed_at,
           change_summary, 0 AS generation
    FROM dag_engine.dag_versions
    WHERE parent_version IS NULL
    UNION ALL
    SELECT v.version_id, v.dag_name, v.version_tag, v.parent_version, v.deployed_at,
           v.change_summary, l.generation + 1
    FROM dag_engine.dag_versions v
    JOIN lineage l ON l.version_id = v.parent_version
)
SELECT
    dag_name,
    generation,
    repeat('  ', generation) || '└─ ' || version_tag  AS version_tree,
    version_id,
    deployed_at,
    change_summary,
    (SELECT COUNT(*) FROM dag_engine.dag_runs WHERE version_id = lineage.version_id) AS total_runs
FROM lineage
ORDER BY dag_name, deployed_at;

-- 8.5.0 Utilitários de Linhagem para o Mermaid
-- Extrai tabelas referenciadas por uma procedure
CREATE OR REPLACE FUNCTION dag_engine.fn_extract_tables_from_proc(
    p_proc_name TEXT
) RETURNS TABLE (table_schema TEXT, table_name TEXT)
LANGUAGE plpgsql AS $$
DECLARE
    v_body TEXT;
BEGIN
    SELECT pg_get_functiondef(oid) INTO v_body
    FROM pg_proc WHERE oid = p_proc_name::regproc;

    IF v_body IS NULL THEN
        RAISE WARNING 'Procedure % não encontrada no catálogo.', p_proc_name;
        RETURN;
    END IF;

    RETURN QUERY
    SELECT DISTINCT
        CASE WHEN m[1] LIKE '%.%' THEN split_part(m[1], '.', 1) ELSE 'public' END AS table_schema,
        CASE WHEN m[1] LIKE '%.%' THEN split_part(m[1], '.', 2) ELSE m[1]     END AS table_name
    FROM regexp_matches(v_body,
        '\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)\s+(?:WHERE|SET|FROM|INTO|JOIN)',
        'gi') AS m
    WHERE m[1] NOT LIKE 'pg_%' AND m[1] NOT IN ('public', 'information_schema');
END;
$$;

-- Mapeia linhagem e classifica direção (READ/WRITE) com validação contra pg_class
DROP FUNCTION IF EXISTS dag_engine.fn_table_lineage(TEXT);
CREATE OR REPLACE FUNCTION dag_engine.fn_table_lineage(
    p_proc_name TEXT
) RETURNS TABLE (direction TEXT, table_schema TEXT, table_name TEXT)
LANGUAGE plpgsql AS $$
DECLARE
    v_body TEXT;
BEGIN
    BEGIN
        SELECT pg_get_functiondef(oid) INTO v_body
        FROM pg_proc WHERE oid = p_proc_name::regproc;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'fn_table_lineage: não foi possível resolver "%".', p_proc_name;
        RETURN;
    END;

    IF v_body IS NULL THEN
        RAISE NOTICE 'fn_table_lineage: "%" não encontrada no catálogo.', p_proc_name;
        RETURN;
    END IF;

    -- WRITE: INSERT INTO, UPDATE, DELETE FROM, TRUNCATE
    RETURN QUERY
    WITH candidates AS (
        SELECT DISTINCT
            CASE WHEN m[1] LIKE '%.%' THEN split_part(m[1], '.', 1) ELSE 'public' END AS s,
            CASE WHEN m[1] LIKE '%.%' THEN split_part(m[1], '.', 2) ELSE m[1]     END AS t
        FROM regexp_matches(v_body,
            '(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|TRUNCATE(?:\s+TABLE)?)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)',
            'gi') AS m
    )
    SELECT 'WRITE'::TEXT, c.s, c.t
    FROM candidates c
    WHERE EXISTS (
        SELECT 1 FROM pg_class pc
        JOIN pg_namespace pn ON pn.oid = pc.relnamespace
        WHERE pn.nspname = c.s
          AND pc.relname = c.t
          AND pc.relkind IN ('r','p','v','m')
    );

    -- READ: FROM, JOIN
    RETURN QUERY
    WITH candidates AS (
        SELECT DISTINCT
            CASE WHEN m[1] LIKE '%.%' THEN split_part(m[1], '.', 1) ELSE 'public' END AS s,
            CASE WHEN m[1] LIKE '%.%' THEN split_part(m[1], '.', 2) ELSE m[1]     END AS t
        FROM regexp_matches(v_body,
            '(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)?)',
            'gi') AS m
        WHERE m[1] NOT ILIKE 'pg_%'
          AND lower(m[1]) NOT IN (
              'public', 'information_schema', 'lateral', 'unnest',
              'jsonb_array_elements', 'jsonb_array_elements_text',
              'generate_series', 'regexp_matches', 'unnest'
          )
    )
    SELECT 'READ'::TEXT, c.s, c.t
    FROM candidates c
    WHERE EXISTS (
        SELECT 1 FROM pg_class pc
        JOIN pg_namespace pn ON pn.oid = pc.relnamespace
        WHERE pn.nspname = c.s
          AND pc.relname = c.t
          AND pc.relkind IN ('r','p','v','m')
    );
END;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- fn_resolve_task_layer: infere a camada medallion pelo alvo de escrita (WRITE)
-- Hierarquia: Gold > Silver > Bronze > Semantic
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION dag_engine.fn_resolve_task_layer(
    p_call_fn TEXT
) RETURNS TEXT LANGUAGE sql AS $$
    WITH writes AS (
        SELECT tl.table_schema, tl.table_name
        FROM dag_engine.fn_table_lineage(p_call_fn) tl
        WHERE tl.direction = 'WRITE'
    )
    SELECT CASE
        WHEN 'Gold'   IN (
            SELECT r.layer FROM dag_engine.table_layer_registry r
            JOIN writes w USING (table_schema, table_name)) THEN 'Gold'
        WHEN 'Silver' IN (
            SELECT r.layer FROM dag_engine.table_layer_registry r
            JOIN writes w USING (table_schema, table_name)) THEN 'Silver'
        WHEN 'Bronze' IN (
            SELECT r.layer FROM dag_engine.table_layer_registry r
            JOIN writes w USING (table_schema, table_name)) THEN 'Bronze'
        ELSE NULL
    END;
$$;

-- 8.5.2 vw_topological_sort: versão final com resolução automática de layer
DROP VIEW IF EXISTS dag_engine.vw_topological_sort CASCADE;
CREATE OR REPLACE VIEW dag_engine.vw_topological_sort AS
WITH RECURSIVE topo_sort AS (
    SELECT task_name, dag_name, call_fn, call_args, procedure_call,
           dependencies, layer, 0 AS execution_level
    FROM dag_engine.tasks
    WHERE array_length(dependencies, 1) IS NULL OR array_length(dependencies, 1) = 0
    UNION ALL
    SELECT t.task_name, t.dag_name, t.call_fn, t.call_args, t.procedure_call,
           t.dependencies, t.layer, ts.execution_level + 1
    FROM dag_engine.tasks t
    JOIN topo_sort ts ON ts.task_name = ANY(t.dependencies)
    WHERE ts.execution_level < 100
)
SELECT
    task_name,
    dag_name,
    call_fn,
    call_args,
    procedure_call,
    dependencies,
    -- Override manual no spec tem precedência; fallback via WRITE target
    COALESCE(
        layer,
        dag_engine.fn_resolve_task_layer(call_fn)
    ) AS layer,
    MAX(execution_level) AS topological_layer
FROM topo_sort
GROUP BY task_name, dag_name, call_fn, call_args, procedure_call, dependencies, layer
ORDER BY dag_name, topological_layer, task_name;

-- 8.5.1 fn_dag_to_mermaid: gera representação Mermaid da topologia de um DAG
DROP FUNCTION IF EXISTS dag_engine.fn_dag_to_mermaid(TEXT);
DROP FUNCTION IF EXISTS dag_engine.fn_dag_to_mermaid(TEXT, INT);
DROP FUNCTION IF EXISTS dag_engine.fn_dag_to_mermaid(TEXT, INT, BOOLEAN);
DROP FUNCTION IF EXISTS dag_engine.fn_dag_to_mermaid(TEXT, INT, BOOLEAN, BOOLEAN);
DROP FUNCTION IF EXISTS dag_engine.fn_dag_to_mermaid(TEXT, INT, BOOLEAN, BOOLEAN, BOOLEAN);

CREATE OR REPLACE FUNCTION dag_engine.fn_dag_to_mermaid(
    p_dag_name     TEXT,
    p_version_id   INT     DEFAULT NULL,
    p_show_health  BOOLEAN DEFAULT TRUE,
    p_show_tables  BOOLEAN DEFAULT FALSE,   -- modo lineage puro (comportamento anterior)
    p_show_lineage BOOLEAN DEFAULT FALSE    -- NOVO: overlay de lineage sobre topologia
) RETURNS TEXT
LANGUAGE plpgsql AS $$
DECLARE
    v_lines       TEXT[] := '{}';
    v_layer       INT;
    v_max_layer   INT;
    v_dep         TEXT;
    v_node_id     TEXT;
    v_dep_id      TEXT;
    v_class_list  TEXT;
    v_version_tag TEXT;
    v_title       TEXT;
    v_rec         RECORD;
    v_tbl_rec     RECORD;
    v_layer_rec   RECORD;
    v_lin_rec     RECORD;
    v_colors TEXT[] := ARRAY[
        'fill:#4a90d9,stroke:#2c6ea5,color:#fff',
        'fill:#7db87d,stroke:#4a8c4a,color:#fff',
        'fill:#d4af37,stroke:#b8960e,color:#000',
        'fill:#e07b54,stroke:#b85a34,color:#fff',
        'fill:#9b59b6,stroke:#7d3c98,color:#fff',
        'fill:#a8a9ad,stroke:#7a7b7f,color:#000'
    ];
BEGIN
    IF p_version_id IS NOT NULL THEN
        SELECT version_tag INTO v_version_tag FROM dag_engine.dag_versions WHERE version_id = p_version_id;
        v_title := p_dag_name || ' · ' || COALESCE(v_version_tag, 'v?') || ' (snapshot)';
    ELSE
        SELECT version_tag INTO v_version_tag FROM dag_engine.dag_versions WHERE dag_name = p_dag_name AND is_active = TRUE;
        v_title := p_dag_name || ' · ' || COALESCE(v_version_tag, 'live');
    END IF;

    v_lines := array_append(v_lines, '%%{init: {"theme": "base"}}%%');
    v_lines := array_append(v_lines, 'graph LR');
    v_lines := array_append(v_lines, '    %% ' || v_title);

    DROP TABLE IF EXISTS _dag_mermaid_topo;
    CREATE TEMP TABLE _dag_mermaid_topo (
        task_name TEXT,
        layer     INT,
        sem_layer TEXT,
        call_fn   TEXT,
        proc_call TEXT,
        deps      TEXT[]
    ) ON COMMIT DROP;

    IF p_version_id IS NULL THEN
        INSERT INTO _dag_mermaid_topo (task_name, layer, sem_layer, call_fn, proc_call, deps)
        SELECT v.task_name, v.topological_layer,
               COALESCE(v.layer, 'ORCHESTRATION'),
               v.call_fn,
               v.procedure_call, v.dependencies
        FROM dag_engine.vw_topological_sort v
        WHERE v.dag_name = p_dag_name;
    ELSE
        INSERT INTO _dag_mermaid_topo (task_name, layer, sem_layer, proc_call, deps)
        WITH RECURSIVE spec_tasks AS (
            SELECT elem->>'task_name'      AS task_name,
                   elem->>'procedure_call' AS proc_call,
                   elem->>'medallion_layer' AS medallion_layer,
                   ARRAY(SELECT jsonb_array_elements_text(elem->'dependencies')) AS deps
            FROM dag_engine.dag_versions,
                 jsonb_array_elements(COALESCE(spec->'tasks', spec)) AS elem
            WHERE version_id = p_version_id
        ),
        topo AS (
            SELECT task_name, medallion_layer, proc_call, deps, 0 AS lvl
            FROM spec_tasks
            WHERE array_length(deps, 1) IS NULL OR array_length(deps, 1) = 0
            UNION ALL
            SELECT t.task_name, t.medallion_layer, t.proc_call, t.deps, ts.lvl + 1
            FROM spec_tasks t
            JOIN topo ts ON ts.task_name = ANY(t.deps)
            WHERE ts.lvl < 100
        )
        SELECT task_name, MAX(lvl) AS layer, COALESCE(medallion_layer, 'ORCHESTRATION'), proc_call, deps
        FROM topo
        GROUP BY task_name, medallion_layer, proc_call, deps;
    END IF;

    SELECT MAX(layer) INTO v_max_layer FROM _dag_mermaid_topo;
    IF v_max_layer IS NULL THEN
        RETURN '-- Nenhuma tarefa encontrada para o DAG "' || p_dag_name || '"';
    END IF;

    FOR v_layer IN 0..v_max_layer LOOP
        v_lines := array_append(v_lines, '    subgraph WAVE_' || v_layer || '["⚡ Wave ' || v_layer || CASE WHEN v_layer = 0 THEN ' · Roots"' ELSE '"' END || ']');
        FOR v_rec IN SELECT task_name FROM _dag_mermaid_topo WHERE layer = v_layer ORDER BY task_name LOOP
            v_node_id := replace(replace(v_rec.task_name, '-', '_'), '.', '_');
            v_lines := array_append(v_lines, '        ' || v_node_id || '["' || v_rec.task_name || '"]');
        END LOOP;
        v_lines := array_append(v_lines, '    end');
    END LOOP;

    FOR v_rec IN SELECT task_name, deps FROM _dag_mermaid_topo WHERE array_length(deps, 1) > 0 ORDER BY task_name LOOP
        v_node_id := replace(replace(v_rec.task_name, '-', '_'), '.', '_');
        FOREACH v_dep IN ARRAY v_rec.deps LOOP
            v_dep_id := replace(replace(v_dep, '-', '_'), '.', '_');
            v_lines := array_append(v_lines, '    ' || v_dep_id || ' --> ' || v_node_id);
        END LOOP;
    END LOOP;

    IF p_show_tables THEN
        DROP TABLE IF EXISTS _dag_mermaid_lineage;
        CREATE TEMP TABLE _dag_mermaid_lineage (
            task_name TEXT, direction TEXT, tbl_fqn TEXT, tbl_nid TEXT, sem_layer TEXT
        ) ON COMMIT DROP;

        INSERT INTO _dag_mermaid_lineage (task_name, direction, tbl_fqn, tbl_nid, sem_layer)
        SELECT t.task_name, tl.direction,
               tl.table_schema || '.' || tl.table_name,
               replace(tl.table_schema || '_' || tl.table_name, '.', '_'),
               t.sem_layer
        FROM _dag_mermaid_topo t
        CROSS JOIN LATERAL dag_engine.fn_table_lineage(
            regexp_replace(t.proc_call, '^\s*(?:CALL|call)\s+([^\s(]+)\s*\(.*$', '\1', 'i')
        ) tl
        WHERE t.proc_call ~* '^\s*CALL\s+';

        FOR v_layer_rec IN SELECT DISTINCT sem_layer FROM _dag_mermaid_lineage WHERE sem_layer IS NOT NULL ORDER BY sem_layer LOOP
            v_lines := array_append(v_lines, '');
            v_lines := array_append(v_lines, '    subgraph ART_' || v_layer_rec.sem_layer || '["🥈 ' || v_layer_rec.sem_layer || '"]');
            FOR v_tbl_rec IN SELECT tbl_nid, tbl_fqn FROM _dag_mermaid_lineage WHERE sem_layer = v_layer_rec.sem_layer LOOP
                v_lines := array_append(v_lines, '        ' || v_tbl_rec.tbl_nid || '[["' || v_tbl_rec.tbl_fqn || '"]]');
            END LOOP;
            v_lines := array_append(v_lines, '    end');
        END LOOP;
        
        FOR v_rec IN SELECT task_name, tbl_nid, direction FROM _dag_mermaid_lineage LOOP
            v_node_id := replace(replace(v_rec.task_name, '-', '_'), '.', '_');
            IF v_rec.direction = 'WRITE' THEN
                v_lines := array_append(v_lines, '    ' || v_node_id || ' -.-> ' || v_rec.tbl_nid);
            ELSE
                v_lines := array_append(v_lines, '    ' || v_rec.tbl_nid || ' -.-> ' || v_node_id);
            END IF;
        END LOOP;
    END IF;

    IF p_show_lineage AND NOT p_show_tables THEN

        -- Subgraphs de fontes agrupadas por schema
        FOR v_layer_rec IN
            SELECT DISTINCT tl.table_schema,
                   'SRC_' || upper(tl.table_schema) AS sg_id
            FROM _dag_mermaid_topo tp
            CROSS JOIN LATERAL dag_engine.fn_table_lineage(tp.call_fn) tl
            WHERE tl.direction = 'READ'
            ORDER BY tl.table_schema
        LOOP
            v_lines := array_append(v_lines,
                '    subgraph ' || v_layer_rec.sg_id ||
                '["\U0001f4e5 ' || v_layer_rec.table_schema || '"]');
            FOR v_tbl_rec IN
                SELECT DISTINCT
                    tl.table_schema || '.' || tl.table_name AS fqn,
                    replace(tl.table_schema || '_' || tl.table_name, '.', '_') AS nid
                FROM _dag_mermaid_topo tp
                CROSS JOIN LATERAL dag_engine.fn_table_lineage(tp.call_fn) tl
                WHERE tl.direction = 'READ'
                  AND tl.table_schema = v_layer_rec.table_schema
                ORDER BY fqn
            LOOP
                v_lines := array_append(v_lines,
                    '        ' || v_tbl_rec.nid || '[("' || v_tbl_rec.fqn || '")]');
            END LOOP;
            v_lines := array_append(v_lines, '    end');
            v_lines := array_append(v_lines, '');
        END LOOP;

        -- Subgraphs de destinos agrupados por schema
        FOR v_layer_rec IN
            SELECT DISTINCT tl.table_schema,
                   'TGT_' || upper(tl.table_schema) AS sg_id
            FROM _dag_mermaid_topo tp
            CROSS JOIN LATERAL dag_engine.fn_table_lineage(tp.call_fn) tl
            WHERE tl.direction = 'WRITE'
            ORDER BY tl.table_schema
        LOOP
            v_lines := array_append(v_lines,
                '    subgraph ' || v_layer_rec.sg_id ||
                '["\U0001f4e4 ' || v_layer_rec.table_schema || '"]');
            FOR v_tbl_rec IN
                SELECT DISTINCT
                    tl.table_schema || '.' || tl.table_name AS fqn,
                    replace(tl.table_schema || '_' || tl.table_name, '.', '_') AS nid
                FROM _dag_mermaid_topo tp
                CROSS JOIN LATERAL dag_engine.fn_table_lineage(tp.call_fn) tl
                WHERE tl.direction = 'WRITE'
                  AND tl.table_schema = v_layer_rec.table_schema
                ORDER BY fqn
            LOOP
                v_lines := array_append(v_lines,
                    '        ' || v_tbl_rec.nid || '[["' || v_tbl_rec.fqn || '"]]');
            END LOOP;
            v_lines := array_append(v_lines, '    end');
            v_lines := array_append(v_lines, '');
        END LOOP;

        -- Arestas pontilhadas de lineage
        v_lines := array_append(v_lines, '    %% Lineage edges');
        FOR v_lin_rec IN
            SELECT DISTINCT
                replace(tl.table_schema || '_' || tl.table_name, '.', '_') AS tbl_nid,
                replace(replace(tp.task_name, '-', '_'), '.', '_')          AS task_nid,
                tl.direction
            FROM _dag_mermaid_topo tp
            CROSS JOIN LATERAL dag_engine.fn_table_lineage(tp.call_fn) tl
            ORDER BY tl.direction, task_nid, tbl_nid
        LOOP
            IF v_lin_rec.direction = 'READ' THEN
                v_lines := array_append(v_lines,
                    '    ' || v_lin_rec.tbl_nid || ' -. r .-> ' || v_lin_rec.task_nid);
            ELSE
                v_lines := array_append(v_lines,
                    '    ' || v_lin_rec.task_nid || ' -. w .-> ' || v_lin_rec.tbl_nid);
            END IF;
        END LOOP;

        -- Estilos dos nós de tabela
        v_lines := array_append(v_lines,
            '    classDef src_tbl fill:#f8f9fa,stroke:#6c757d,color:#555,stroke-dasharray:4');
        v_lines := array_append(v_lines,
            '    classDef tgt_tbl fill:#e8f5e9,stroke:#4caf50,color:#1b5e20,stroke-width:2px');

    END IF;

    FOR v_layer IN 0..v_max_layer LOOP
        v_lines := array_append(v_lines, '    classDef wave' || v_layer || ' ' || v_colors[LEAST(v_layer + 1, array_length(v_colors, 1))]);
        SELECT string_agg(replace(replace(task_name, '-', '_'), '.', '_'), ',') INTO v_class_list FROM _dag_mermaid_topo WHERE layer = v_layer;
        IF v_class_list IS NOT NULL THEN v_lines := array_append(v_lines, '    class ' || v_class_list || ' wave' || v_layer); END IF;
    END LOOP;

    IF p_show_health THEN
        v_lines := array_append(v_lines, '    classDef health_ok fill:#2ecc71,stroke:#27ae60,color:#000');
        v_lines := array_append(v_lines, '    classDef health_warn fill:#f39c12,stroke:#d68910,color:#000');
        v_lines := array_append(v_lines, '    classDef health_critical fill:#e74c3c,stroke:#cb4335,color:#fff');
        FOR v_rec IN SELECT t.task_name, g.health_label FROM _dag_mermaid_topo t JOIN dag_medallion.gold_pipeline_health g ON g.task_name = t.task_name LOOP
            v_node_id := replace(replace(v_rec.task_name, '-', '_'), '.', '_');
            v_lines := array_append(v_lines, '    class ' || v_node_id || ' ' || CASE WHEN v_rec.health_label LIKE '%ANOMALIA%' THEN 'health_critical' WHEN v_rec.health_label LIKE '%MODERADO%' THEN 'health_warn' ELSE 'health_ok' END);
        END LOOP;
    END IF;

    RETURN array_to_string(v_lines, E'\n');
END;
$$;

-- 8.6 proc_run_dag: versão intermediária que carimba version_id no dag_run
-- (será sobrescrita novamente na seção 9 com dispatch assíncrono completo)
DROP PROCEDURE IF EXISTS dag_engine.proc_run_dag(DATE);
CREATE OR REPLACE PROCEDURE dag_engine.proc_run_dag(
    p_dag_name TEXT, 
    p_data DATE, 
    p_verbose BOOLEAN DEFAULT TRUE,
    p_run_type VARCHAR(20) DEFAULT 'INCREMENTAL'
)
LANGUAGE plpgsql AS $$
DECLARE
    v_run_id        INT;
    v_task          RECORD;
    v_pending_count INT;
    v_running_count INT;
    v_sql           TEXT;
BEGIN
    IF p_verbose THEN
        RAISE NOTICE '=================================================';
        RAISE NOTICE '🚀 Iniciando DAG [%] para: %', p_dag_name, p_data;
    END IF;

    BEGIN
        INSERT INTO dag_engine.dag_runs (dag_name, run_date, version_id)
        VALUES (p_dag_name, p_data, (
            SELECT version_id FROM dag_engine.dag_versions
            WHERE dag_name = p_dag_name AND is_active = TRUE
        ))
        RETURNING run_id INTO v_run_id;
    EXCEPTION WHEN unique_violation THEN
        IF p_verbose THEN RAISE WARNING 'Já existe run para "%" em %. Use proc_clear_run.', p_dag_name, p_data; END IF;
        RETURN;
    END;

    INSERT INTO dag_engine.task_instances (run_id, task_name)
    SELECT v_run_id, task_name FROM dag_engine.tasks WHERE dag_name = p_dag_name;

    LOOP
        v_task := NULL;
        SELECT ti.task_name, t.procedure_call, t.call_fn, t.call_args INTO v_task
        FROM dag_engine.task_instances ti
        JOIN dag_engine.tasks t ON ti.task_name = t.task_name
        WHERE ti.run_id = v_run_id AND ti.status = 'PENDING'
          AND (ti.retry_after_ts IS NULL OR ti.retry_after_ts <= clock_timestamp())
          AND NOT EXISTS (
              SELECT 1 FROM unnest(t.dependencies) AS dep
              JOIN dag_engine.task_instances dep_ti ON dep_ti.run_id = v_run_id AND dep_ti.task_name = dep
              WHERE dep_ti.status != 'SUCCESS'
          )
        ORDER BY t.task_id
        FOR UPDATE OF ti SKIP LOCKED
        LIMIT 1;

        IF v_task IS NOT NULL THEN
            UPDATE dag_engine.task_instances SET status = 'RUNNING', start_ts = clock_timestamp()
            WHERE run_id = v_run_id AND task_name = v_task.task_name;
            COMMIT;
            BEGIN
                v_sql := dag_engine.fn_build_call_sql(v_task.call_fn, v_task.call_args, p_data);
                IF p_verbose THEN RAISE NOTICE '  --> 🔄 [%] %', v_task.task_name, v_sql; END IF;
                EXECUTE v_sql;
                UPDATE dag_engine.task_instances
                SET status = 'SUCCESS', end_ts = clock_timestamp(),
                    duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - start_ts)) * 1000
                WHERE run_id = v_run_id AND task_name = v_task.task_name;
            EXCEPTION WHEN OTHERS THEN
                DECLARE
                    v_cur_att INT; v_delay INT; v_max INT;
                BEGIN
                    SELECT ti.attempt, t.retry_delay_seconds, t.max_retries
                    INTO v_cur_att, v_delay, v_max
                    FROM dag_engine.task_instances ti JOIN dag_engine.tasks t ON t.task_name = ti.task_name
                    WHERE ti.run_id = v_run_id AND ti.task_name = v_task.task_name;
                    IF v_cur_att < v_max + 1 THEN
                        UPDATE dag_engine.task_instances
                        SET status = 'PENDING', attempt = attempt + 1,
                            retry_after_ts = clock_timestamp() + (v_delay * (v_cur_att + 1)) * INTERVAL '1 second',
                            error_text = 'Retry | ' || SQLERRM
                        WHERE run_id = v_run_id AND task_name = v_task.task_name;
                        IF p_verbose THEN RAISE WARNING '🔄 [%] Retry agendado.', v_task.task_name; END IF;
                    ELSE
                        UPDATE dag_engine.task_instances
                        SET status = 'FAILED', end_ts = clock_timestamp(),
                            duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - start_ts)) * 1000,
                            error_text = SQLERRM
                        WHERE run_id = v_run_id AND task_name = v_task.task_name;
                        WITH RECURSIVE fail_cascade AS (
                            SELECT t.task_name, 1 AS depth FROM dag_engine.tasks t WHERE v_task.task_name = ANY(t.dependencies)
                            UNION ALL
                            SELECT t.task_name, fc.depth + 1 FROM dag_engine.tasks t
                            JOIN fail_cascade fc ON fc.task_name = ANY(t.dependencies)
                            WHERE fc.depth < 100
                        )
                        UPDATE dag_engine.task_instances
                        SET status = 'UPSTREAM_FAILED', end_ts = clock_timestamp(),
                            error_text = 'Propagado de: ' || v_task.task_name
                        WHERE run_id = v_run_id AND task_name IN (SELECT task_name FROM fail_cascade) AND status = 'PENDING';
                    END IF;
                END;
            END;
            COMMIT;
        ELSE
            SELECT COUNT(*) INTO v_pending_count FROM dag_engine.task_instances WHERE run_id = v_run_id AND status = 'PENDING';
            SELECT COUNT(*) INTO v_running_count FROM dag_engine.task_instances WHERE run_id = v_run_id AND status = 'RUNNING';
            IF v_running_count > 0 THEN
                PERFORM pg_sleep(1);
            ELSIF v_pending_count > 0 THEN
                IF EXISTS (SELECT 1 FROM dag_engine.task_instances WHERE run_id = v_run_id AND status = 'PENDING' AND retry_after_ts > clock_timestamp()) THEN
                    PERFORM pg_sleep(1);
                ELSE
                    UPDATE dag_engine.dag_runs SET status = 'DEADLOCK', end_ts = clock_timestamp() WHERE run_id = v_run_id;
                    COMMIT;
                    IF p_verbose THEN RAISE WARNING '💀 Deadlock Topológico!'; END IF;
                    CALL dag_medallion.proc_run_medallion(v_run_id);  -- DEADLOCK também alimenta o medallion
                    EXIT;
                END IF;
            ELSE
                IF EXISTS (SELECT 1 FROM dag_engine.task_instances WHERE run_id = v_run_id AND status IN ('FAILED', 'UPSTREAM_FAILED')) THEN
                    UPDATE dag_engine.dag_runs SET status = 'FAILED', end_ts = clock_timestamp() WHERE run_id = v_run_id;
                    IF p_verbose THEN RAISE WARNING '❌ DAG % finalizada com falhas.', p_data; END IF;
                ELSE
                    UPDATE dag_engine.dag_runs SET status = 'SUCCESS', end_ts = clock_timestamp() WHERE run_id = v_run_id;
                    IF p_verbose THEN RAISE NOTICE '✅ DAG % finalizada com sucesso!', p_data; END IF;
                END IF;
                COMMIT;
                EXIT;
            END IF;
        END IF;
    END LOOP;

    CALL dag_medallion.proc_run_medallion(v_run_id);
END;
$$;

-- 8.6 proc_catchup: extende com p_version opcional para replay fiel de topologia histórica
-- NULL = current HEAD (comportamento original); INT = congela motor na versão antiga e restaura no fim
DROP PROCEDURE IF EXISTS dag_engine.proc_catchup(DATE, DATE);
DROP PROCEDURE IF EXISTS dag_engine.proc_catchup(DATE, DATE, BOOLEAN);
-- Remove a versão 4.1 (TEXT, DATE, DATE) para evitar ambiguidade com a versão estendida abaixo
DROP PROCEDURE IF EXISTS dag_engine.proc_catchup(TEXT, DATE, DATE, BOOLEAN);
CREATE OR REPLACE PROCEDURE dag_engine.proc_catchup(
    p_dag_name TEXT,
    p_from     DATE,
    p_to       DATE,
    p_verbose  BOOLEAN DEFAULT TRUE,
    p_version  INT     DEFAULT NULL
)
LANGUAGE plpgsql AS $$
DECLARE
    v_date   DATE := p_from;
    v_spec   JSONB;
    v_status VARCHAR(20);
BEGIN
    IF p_version IS NOT NULL THEN
        SELECT spec INTO v_spec FROM dag_engine.dag_versions
        WHERE version_id = p_version AND dag_name = p_dag_name;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'Versão % não encontrada para o DAG "%" em dag_versions.', p_version, p_dag_name;
        END IF;
        RAISE NOTICE '📌 Catchup histórico de "%" fixado na versão %', p_dag_name, p_version;
        CALL dag_engine.proc_load_dag_spec(v_spec);
    END IF;

    WHILE v_date <= p_to LOOP
        v_status := NULL;
        SELECT status INTO v_status FROM dag_engine.dag_runs
        WHERE dag_name = p_dag_name AND run_date = v_date;

        IF v_status = 'SUCCESS' THEN
            IF p_verbose THEN RAISE NOTICE '⏭️ Pulando % — já processado com sucesso.', v_date; END IF;
        ELSIF v_status = 'RUNNING' THEN
            RAISE WARNING '⚠️ Run de % está RUNNING (fantasma). Catchup interrompido — resolva manualmente.', v_date;
            EXIT;
        ELSE
            IF v_status IS NOT NULL THEN CALL dag_engine.proc_clear_run(p_dag_name, v_date, p_verbose); END IF;
            IF p_verbose THEN RAISE NOTICE '📅 Catch-up: rodando % (Regime BACKFILL)', v_date; END IF;
            COMMIT;
            CALL dag_engine.proc_run_dag(p_dag_name, v_date, p_verbose, 'BACKFILL');
        END IF;

        v_date := v_date + 1;
        COMMIT;
    END LOOP;

    IF p_version IS NOT NULL THEN
        SELECT spec INTO v_spec FROM dag_engine.dag_versions
        WHERE dag_name = p_dag_name AND is_active = TRUE;
        IF FOUND THEN
            CALL dag_engine.proc_load_dag_spec(v_spec);
            RAISE NOTICE '🔄 Versão ativa de "%" restaurada após catchup histórico.', p_dag_name;
        END IF;
    END IF;
END;
$$;

-- ==============================================================================
-- 9. BACKLOG: ASYNC EXECUTION (Non-Blocking dblink Dispatch + Chunking Temporal)
-- ==============================================================================
-- Elimina poll-blocking de conexão sem infraestrutura externa.
--   Estratégia 1: fire-and-forget via pg_background_launch — workers nativos do
--                 processo principal, sem TCP, autenticação ou handshake.
--   Estratégia 4: chunking temporal automático — tasks com chunk_config no spec
--                 são expandidas em sub-tasks paralelas por janela de tempo.
-- ==============================================================================

-- Habilita pg_background (requer max_worker_processes ≥ número de tasks paralelas)
-- Fallback: compatibilidade mantida com dblink (extensao já carregada na abertura)
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_background;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '⚠️  pg_background não disponível (%). Fallback dblink inativo — use engine síncrona.', SQLERRM;
END $$;

-- 9.1 Colunas de rastreio assíncrono em task_instances
ALTER TABLE dag_engine.task_instances
    ADD COLUMN IF NOT EXISTS worker_conn TEXT,          -- nome da conexão dblink ativa (legado)
    ADD COLUMN IF NOT EXISTS bgw_pid     INT,           -- PID do background worker (pg_background)
    ADD COLUMN IF NOT EXISTS is_chunk    BOOLEAN DEFAULT FALSE,  -- é sub-task de chunking?
    ADD COLUMN IF NOT EXISTS chunk_index INT,           -- índice do bucket (0-based)
    ADD COLUMN IF NOT EXISTS parent_task VARCHAR(100);  -- task original que gerou este chunk

-- Remove FK restritiva em task_instances.task_name para que a remoção da task pai durante
-- chunk expansion não seja bloqueada por runs históricas (SUCCESS/FAILED).
-- O valor VARCHAR é preservado como string — analytics históricas continuam funcionando.
DO $$ BEGIN
    ALTER TABLE dag_engine.task_instances DROP CONSTRAINT task_instances_task_name_fkey;
EXCEPTION WHEN undefined_object THEN NULL; END $$;

-- 9.2 Hint de chunking no spec de tarefas
-- Formato: {"column": "data_venda", "buckets": 4} | NULL = sem chunking
ALTER TABLE dag_engine.tasks
    ADD COLUMN IF NOT EXISTS chunk_config JSONB DEFAULT NULL;

-- 9.3 Tabela de workers ativos (debug e housekeeping de órfãos)
CREATE TABLE IF NOT EXISTS dag_engine.async_workers (
    conn_name   TEXT PRIMARY KEY,   -- PID do BGW (como TEXT para compatibilidade)
    run_id      INT  REFERENCES dag_engine.dag_runs(run_id),
    task_name   VARCHAR(100),
    launched_at TIMESTAMP DEFAULT clock_timestamp()
);

-- 9.4 fn_bgw_task: wrapper executado no background worker (pg_background)
-- Executa o SQL, persiste SUCCESS/PENDING(retry)/FAILED e cascade UPSTREAM_FAILED.
CREATE OR REPLACE FUNCTION dag_engine.fn_bgw_task(
    p_run_id    INT,
    p_task_name VARCHAR(100),
    p_sql       TEXT
) RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    v_err         TEXT;
    v_cur_att     INT;
    v_retry_delay INT;
    v_max_retries INT;
BEGIN
    EXECUTE p_sql;

    UPDATE dag_engine.task_instances
    SET status      = 'SUCCESS',
        end_ts      = clock_timestamp(),
        duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - start_ts)) * 1000,
        bgw_pid     = NULL
    WHERE run_id = p_run_id AND task_name = p_task_name;

    DELETE FROM dag_engine.async_workers
    WHERE conn_name = p_run_id::TEXT || '_' || replace(p_task_name, '.', '_');

EXCEPTION WHEN OTHERS THEN
    v_err := SQLERRM;

    SELECT ti.attempt, t.retry_delay_seconds, t.max_retries
    INTO v_cur_att, v_retry_delay, v_max_retries
    FROM dag_engine.task_instances ti
    JOIN dag_engine.tasks t ON t.task_name = ti.task_name
    WHERE ti.run_id = p_run_id AND ti.task_name = p_task_name;

    IF v_cur_att < v_max_retries + 1 THEN
        UPDATE dag_engine.task_instances
        SET status         = 'PENDING',
            attempt        = attempt + 1,
            bgw_pid        = NULL,
            retry_after_ts = clock_timestamp() + (v_retry_delay * (v_cur_att + 1)) * INTERVAL '1 second',
            error_text     = 'Retry | Último erro: ' || v_err
        WHERE run_id = p_run_id AND task_name = p_task_name;
    ELSE
        UPDATE dag_engine.task_instances
        SET status      = 'FAILED',
            end_ts      = clock_timestamp(),
            error_text  = v_err,
            bgw_pid     = NULL,
            duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - start_ts)) * 1000
        WHERE run_id = p_run_id AND task_name = p_task_name;

        WITH RECURSIVE fail_cascade AS (
            SELECT t.task_name, 1 AS depth FROM dag_engine.tasks t
            WHERE p_task_name = ANY(t.dependencies)
            UNION ALL
            SELECT t.task_name, fc.depth + 1 FROM dag_engine.tasks t
            JOIN fail_cascade fc ON fc.task_name = ANY(t.dependencies)
            WHERE fc.depth < 100
        )
        UPDATE dag_engine.task_instances
        SET status     = 'UPSTREAM_FAILED',
            end_ts     = clock_timestamp(),
            error_text = 'Propagado de: ' || p_task_name
        WHERE run_id = p_run_id
          AND task_name IN (SELECT task_name FROM fail_cascade)
          AND status = 'PENDING';
    END IF;

    DELETE FROM dag_engine.async_workers
    WHERE conn_name = p_run_id::TEXT || '_' || replace(p_task_name, '.', '_');
END;
$$;

-- 9.4b fn_exec_task: wrapper legado compatível com dblink (mantém retrocompatibilidade)
-- Retorna NULL (sucesso) ou SQLERRM (falha) como TEXT
DROP FUNCTION IF EXISTS dag_engine.fn_exec_task(TEXT, BOOLEAN);
CREATE OR REPLACE FUNCTION dag_engine.fn_exec_task(p_sql TEXT)
RETURNS TEXT LANGUAGE plpgsql AS $$
BEGIN
    EXECUTE p_sql;
    RETURN NULL;
EXCEPTION WHEN OTHERS THEN
    RETURN SQLERRM;
END;
$$;

-- 9.5 fn_build_chunk_ranges: divide um dia em N buckets de tempo uniformes (method=range)
-- Retorna ranges [range_start, range_end] prontos para injeção na procedure_call
CREATE OR REPLACE FUNCTION dag_engine.fn_build_chunk_ranges(
    p_schema  TEXT,
    p_table   TEXT,
    p_column  TEXT,
    p_date    DATE,
    p_buckets INT
) RETURNS TABLE (
    chunk_index INT,
    range_start TEXT,
    range_end   TEXT
) LANGUAGE plpgsql AS $$
DECLARE
    v_min  TIMESTAMP := p_date::TIMESTAMP;
    v_max  TIMESTAMP := p_date::TIMESTAMP + INTERVAL '1 day' - INTERVAL '1 second';
    v_step INTERVAL  := (v_max - v_min) / p_buckets;
    i      INT;
BEGIN
    FOR i IN 0..(p_buckets - 1) LOOP
        chunk_index := i;
        range_start := (v_min + (v_step * i))::TEXT;
        range_end   := CASE
            WHEN i = p_buckets - 1 THEN v_max::TEXT
            ELSE (v_min + (v_step * (i + 1)) - INTERVAL '1 second')::TEXT
        END;
        RETURN NEXT;
    END LOOP;
END;
$$;

-- 9.6 fn_expand_chunk_tasks: expande uma task com chunk_config em N sub-tasks paralelas
-- Suporta dois métodos:
--   method=range (padrão): divide por intervalo de tempo; injeta $range_start/$range_end nos call_args.
--   method=hash:  bucketing por hashtext(col::text) % buckets; injeta $chunk_filter nos call_args.
--     Cada chunk processa exatamente 1/N dos registros por construção — variância de volume zero.
CREATE OR REPLACE FUNCTION dag_engine.fn_expand_chunk_tasks(
    p_task_name    VARCHAR(100),
    p_procedure    TEXT,
    p_dependencies VARCHAR(100)[],
    p_chunk_config JSONB,
    p_date         DATE
) RETURNS TABLE (
    task_name      VARCHAR(100),
    procedure_call TEXT,
    dependencies   VARCHAR(100)[]
) LANGUAGE plpgsql AS $$
DECLARE
    v_column  TEXT := p_chunk_config->>'column';
    v_buckets INT  := COALESCE((p_chunk_config->>'buckets')::INT, 4);
    v_method  TEXT := COALESCE(p_chunk_config->>'method', 'range');
    v_range   RECORD;
    v_proc    TEXT;
    i         INT;
BEGIN
    IF v_column IS NULL THEN
        RAISE EXCEPTION 'chunk_config requer campo "column". Recebido: %', p_chunk_config;
    END IF;

    IF v_method = 'hash' THEN
        -- Hash modular: WHERE hashtext(col::text) % buckets = i
        -- Garante distribuição uniforme independente dos dados (sem skew de range).
        -- Injeta o predicado como token $chunk_filter nos call_args da task.
        FOR i IN 0..(v_buckets - 1) LOOP
            task_name      := p_task_name || '_chunk_' || i;
            procedure_call := NULL;   -- não usado; call_args via fn_build_call_sql
            dependencies   := p_dependencies;
            -- Armazena o índice do bucket; fn_build_call_sql substituirá $chunk_filter
            -- com o predicado concreto: hashtext(col::text) % N = i
            RETURN NEXT;
        END LOOP;
    ELSE
        -- Range por tempo (comportamento original)
        FOR v_range IN
            SELECT * FROM dag_engine.fn_build_chunk_ranges('public', p_task_name, v_column, p_date, v_buckets)
        LOOP
            task_name      := p_task_name || '_chunk_' || v_range.chunk_index;
            v_proc         := REPLACE(p_procedure, '$1',           quote_literal(p_date));
            v_proc         := REPLACE(v_proc,      '$range_start', quote_literal(v_range.range_start));
            v_proc         := REPLACE(v_proc,      '$range_end',   quote_literal(v_range.range_end));
            procedure_call := v_proc;
            dependencies   := p_dependencies;
            RETURN NEXT;
        END LOOP;
    END IF;
END;
$$;

-- 9.8 proc_load_dag_spec: sobrescreve versão anterior adicionando Passo 3 — chunking automático
-- Passo 3 expande tasks com chunk_config em N sub-tasks paralelas antes de finalizar o load
CREATE OR REPLACE PROCEDURE dag_engine.proc_load_dag_spec(p_spec JSONB)
LANGUAGE plpgsql AS $$
DECLARE
    v_task        JSONB;
    v_deps        VARCHAR(100)[];
    v_missing_dep TEXT;
    v_expanded    RECORD;
    v_chunk_names VARCHAR(100)[];
    v_tasks       JSONB  := COALESCE(p_spec->'tasks', p_spec);  -- suporta manifest envelope E array legado
    v_dag_name    TEXT   := COALESCE(p_spec->>'name', 'default');
BEGIN
    IF jsonb_array_length(v_tasks) = 0 THEN
        RAISE EXCEPTION 'DAG Spec Error: spec vazio. Operação abortada.';
    END IF;

    -- ── GUARDA L0: Grafo único ─────────────────────────────────────────────
    PERFORM dag_engine.fn_validate_single_graph(p_spec);

    DELETE FROM dag_engine.tasks
    WHERE dag_name = v_dag_name
      AND task_name NOT IN (
        SELECT t->>'task_name' FROM jsonb_array_elements(v_tasks) AS t
    );

    -- PASSO 1: Insere todas as tarefas sem dependências (Forward References safe)
    FOR v_task IN SELECT * FROM jsonb_array_elements(v_tasks)
    LOOP
        IF v_task->>'task_name' IS NULL OR v_task->>'call_fn' IS NULL THEN
            RAISE EXCEPTION 'DAG Spec Error: "task_name" e "call_fn" são obrigatórios. Payload: %', v_task;
        END IF;

        -- ── GUARDA C0: call_fn deve existir no catálogo pg_proc ──────────────────
        -- Resolve por (schema, name, nargs) para suportar sobrecargas.
        -- Aceita pronargs >= n_args (parâmetros extras podem ter DEFAULT),
        -- desde que os obrigatórios (pronargs - pronargdefaults) <= n_args.
        IF NOT EXISTS (
            SELECT 1 FROM pg_catalog.pg_proc
            WHERE proname = split_part(v_task->>'call_fn', '.', 2)
              AND pronamespace = (SELECT oid FROM pg_namespace
                                  WHERE nspname = split_part(v_task->>'call_fn', '.', 1))
              AND pronargs >= jsonb_array_length(
                    COALESCE(v_task->'call_args', '["$1"]'::JSONB))
              AND (pronargs - pronargdefaults) <= jsonb_array_length(
                    COALESCE(v_task->'call_args', '["$1"]'::JSONB))
        ) THEN
            RAISE EXCEPTION 'DAG Spec Error: call_fn "%" com % args não existe no catálogo pg_proc.',
                v_task->>'call_fn',
                jsonb_array_length(COALESCE(v_task->'call_args', '["$1"]'::JSONB));
        END IF;

        INSERT INTO dag_engine.tasks (
            task_name, dag_name,
            call_fn, call_args,
            medallion_layer, layer,
            dependencies, max_retries, retry_delay_seconds,
            chunk_config, sla_ms_override
        ) VALUES (
            v_task->>'task_name',
            v_dag_name,
            v_task->>'call_fn',
            COALESCE(v_task->'call_args', '["$1"]'::JSONB),
            v_task->>'medallion_layer',
            v_task->>'layer',
            '{}',
            COALESCE((v_task->>'max_retries')::INT, 0),
            COALESCE((v_task->>'retry_delay_seconds')::INT, 5),
            v_task->'chunk_config',
            (v_task->>'sla_ms_override')::BIGINT
        )
        ON CONFLICT (task_name) DO UPDATE SET
            dag_name            = EXCLUDED.dag_name,
            call_fn             = EXCLUDED.call_fn,
            call_args           = EXCLUDED.call_args,
            medallion_layer     = EXCLUDED.medallion_layer,
            layer               = EXCLUDED.layer,
            dependencies        = '{}',
            max_retries         = EXCLUDED.max_retries,
            retry_delay_seconds = EXCLUDED.retry_delay_seconds,
            chunk_config        = EXCLUDED.chunk_config,
            sla_ms_override     = EXCLUDED.sla_ms_override;
    END LOOP;

    -- PASSO 2: Aplica dependências (trigger de ciclo protege a integridade)
    FOR v_task IN SELECT * FROM jsonb_array_elements(v_tasks)
    LOOP
        SELECT array_agg(d::VARCHAR) INTO v_deps
        FROM jsonb_array_elements_text(v_task->'dependencies') d;
        v_deps := COALESCE(v_deps, '{}'::VARCHAR(100)[]);

        SELECT d INTO v_missing_dep
        FROM unnest(v_deps) AS d
        WHERE NOT EXISTS (SELECT 1 FROM dag_engine.tasks WHERE task_name = d)
        LIMIT 1;

        IF FOUND THEN
            RAISE EXCEPTION 'DAG Spec Error: dep "%" declarada em "%" não existe no engine.',
                v_missing_dep, v_task->>'task_name';
        END IF;

        UPDATE dag_engine.tasks SET dependencies = v_deps WHERE task_name = v_task->>'task_name';
    END LOOP;

    -- PASSO 3 (NOVO): Expande tasks com chunk_config em sub-tasks paralelas
    -- A task original é removida e substituída por N chunks que herdam suas dependências.
    -- Se há task_instances históricas finalizadas ligadas à task pai, a expansão é ADITIVA
    -- (pai coexiste com chunks, mas não é instanciada em novas runs pois não está no tasks table).
    -- Se não há histórico, a task pai é removida e os chunks a substituem completamente.
    -- Após expansão, atualiza deps de tasks downstream que referenciam o parent pelo nome.
    FOR v_task IN SELECT * FROM jsonb_array_elements(v_tasks) WHERE (value->'chunk_config') IS NOT NULL
    LOOP
        v_chunk_names := '{}';
        -- Só bloqueia se há uma run RUNNING agora (condição de corrida).
        -- Runs históricas (SUCCESS/FAILED) não bloqueiam: FK foi removida em 9.1.
        IF EXISTS (
            SELECT 1
            FROM dag_engine.task_instances ti
            JOIN dag_engine.dag_runs dr ON dr.run_id = ti.run_id
            WHERE ti.task_name = v_task->>'task_name'
              AND dr.status    = 'RUNNING'
            LIMIT 1
        ) THEN
            RAISE EXCEPTION 'Chunk expansion para "%" bloqueada: run RUNNING ativa. Aguarde ou use proc_clear_run.',
                v_task->>'task_name';
        END IF;
        DELETE FROM dag_engine.tasks
        WHERE task_name = v_task->>'task_name' AND dag_name = v_dag_name;

        FOR v_expanded IN
            -- Passa NULL para p_procedure: não usamos procedure_call herdado no novo formato.
            -- fn_expand_chunk_tasks apenas gera os nomes e indexes dos chunks.
            SELECT * FROM dag_engine.fn_expand_chunk_tasks(
                v_task->>'task_name',
                NULL,
                ARRAY(SELECT jsonb_array_elements_text(v_task->'dependencies')),
                v_task->'chunk_config',
                CURRENT_DATE
            )
        LOOP
            -- Usa call_fn/call_args do parent — trigger trg_sync_procedure_call deriva procedure_call.
            -- Em runtime, fn_build_call_sql(call_fn, call_args, p_data) constrói o SQL correto.
            INSERT INTO dag_engine.tasks (
                task_name, dag_name,
                call_fn, call_args, medallion_layer,
                dependencies, max_retries, retry_delay_seconds,
                chunk_config, sla_ms_override
            ) VALUES (
                v_expanded.task_name,
                v_dag_name,
                v_task->>'call_fn',
                COALESCE(v_task->'call_args', '["$1"]'::JSONB),
                v_task->>'medallion_layer',
                v_expanded.dependencies,
                COALESCE((v_task->>'max_retries')::INT, 0),
                COALESCE((v_task->>'retry_delay_seconds')::INT, 5),
                v_task->'chunk_config',
                (v_task->>'sla_ms_override')::BIGINT
            )
            ON CONFLICT (task_name) DO UPDATE SET
                call_fn         = EXCLUDED.call_fn,
                call_args       = EXCLUDED.call_args,
                medallion_layer = EXCLUDED.medallion_layer,
                dependencies    = EXCLUDED.dependencies,
                chunk_config    = EXCLUDED.chunk_config,
                sla_ms_override = EXCLUDED.sla_ms_override;
            v_chunk_names := v_chunk_names || v_expanded.task_name::VARCHAR(100);
        END LOOP;

        -- Substitui a referência ao parent pelo array de chunks em tasks downstream
        UPDATE dag_engine.tasks
        SET dependencies = array_remove(dependencies, (v_task->>'task_name')::VARCHAR(100)) || v_chunk_names
        WHERE dag_name = v_dag_name
          AND (v_task->>'task_name')::VARCHAR(100) = ANY(dependencies);
    END LOOP;

    RAISE NOTICE '✅ DAG Spec carregada! % tasks interpretadas.', jsonb_array_length(v_tasks);

    -- Melhoria 7: registra schedule pg_cron automaticamente no deploy
    CALL dag_engine.proc_register_cron_job(v_dag_name);
END;
$$;

-- 8.4 proc_register_cron_job: registra schedule do DAG como job pg_cron no deploy
-- Idempotente: remove job anterior antes de recriar (redeploy seguro).
-- Requer pg_cron no shared_preload_libraries. Silencia graciosamente se não disponível.
CREATE OR REPLACE PROCEDURE dag_engine.proc_register_cron_job(p_dag_name TEXT)
LANGUAGE plpgsql AS $$
DECLARE
    v_schedule TEXT;
    v_job_name TEXT := 'dag_run_' || p_dag_name;
    v_sql      TEXT;
BEGIN
    -- Verifica se pg_cron está disponível
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'pg_cron') THEN
        RAISE NOTICE '⚠️  pg_cron não disponível. Job "dag_run_%" não agendado.', p_dag_name;
        RETURN;
    END IF;

    SELECT schedule INTO v_schedule
    FROM dag_engine.dag_versions
    WHERE dag_name = p_dag_name AND is_active = TRUE;

    IF v_schedule IS NULL THEN
        RAISE NOTICE '⚠️  DAG "%" sem schedule definido. Job pg_cron não registrado.', p_dag_name;
        RETURN;
    END IF;

    -- Remove job anterior (redeploy idempotente)
    BEGIN
        PERFORM cron.unschedule(v_job_name);
    EXCEPTION WHEN OTHERS THEN NULL;
    END;

    -- Agenda: schedule do manifest é literalmente a cron expression
    -- Convenção D-1: passa CURRENT_DATE - 1 para processar o dia anterior
    -- Aspas simples duplicadas (''''%s'''') porque format() processa %s dentro de uma
    -- string delimitada por aspas simples. Não usar dollar-quoting aqui (fecharia o corpo).
    v_sql := format(
        'CALL dag_engine.proc_run_dag(''%s'', CURRENT_DATE - 1)',
        p_dag_name
    );

    PERFORM cron.schedule(v_job_name, v_schedule, v_sql);

    RAISE NOTICE '⏰ pg_cron job "%" registrado: %', v_job_name, v_schedule;
END;
$$;

-- 9.9 proc_dispatch_task: fire-and-forget via dblink quando disponível, fallback síncrono caso contrário.
-- Tenta abrir conexão dblink real (DSN configurável via dag_engine.worker_dsn).
-- Se o DSN não é acessível, executa inline e registra worker_conn='SYNC_FALLBACK' para auditoria.
DROP PROCEDURE IF EXISTS dag_engine.proc_dispatch_task(INT, TEXT, TEXT);
DROP PROCEDURE IF EXISTS dag_engine.proc_dispatch_task(INT, TEXT, TEXT, BOOLEAN);
DROP PROCEDURE IF EXISTS dag_engine.proc_dispatch_task(INT, VARCHAR, TEXT);
DROP PROCEDURE IF EXISTS dag_engine.proc_dispatch_task(INT, VARCHAR, TEXT, BOOLEAN);
CREATE OR REPLACE PROCEDURE dag_engine.proc_dispatch_task(
    p_run_id    INT,
    p_task_name VARCHAR(100),
    p_sql       TEXT
)
LANGUAGE plpgsql AS $$
DECLARE
    v_conn_name    TEXT := 'dag_worker_' || p_run_id || '_'
                           || replace(p_task_name, '.', '_');
    v_dsn          TEXT;
    v_sync_fallback BOOLEAN := FALSE;
    v_final_status TEXT;
BEGIN
    -- DSN configurável: SET dag_engine.worker_dsn = 'host=... dbname=... user=...';
    -- Se não configurado, constrói DSN via socket Unix local (caminho explícito
    -- necessário pois libpq no processo servidor não usa o default de socket).
    v_dsn := current_setting('dag_engine.worker_dsn', true);
    IF v_dsn IS NULL OR v_dsn = '' THEN
        v_dsn := format('host=%s dbname=%s user=%s',
            split_part(current_setting('unix_socket_directories'), ',', 1),
            current_database(),
            current_user);
    END IF;

    -- Tenta abrir conexão dblink real.
    -- COMMIT não pode estar dentro de um bloco EXCEPTION (viola "invalid transaction
    -- termination"), por isso usamos flag e tratamos o fallback fora do handler.
    BEGIN
        PERFORM dblink_connect(v_conn_name, v_dsn);
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '⚠️  dblink indisponível (%). Executando [%] de forma síncrona.',
            SQLERRM, p_task_name;
        v_sync_fallback := TRUE;
    END;

    -- Fallback síncrono: COMMIT aqui é legal porque estamos fora do bloco EXCEPTION.
    IF v_sync_fallback THEN
        UPDATE dag_engine.task_instances
        SET status      = 'RUNNING',
            start_ts    = clock_timestamp(),
            worker_conn = 'SYNC_FALLBACK'
        WHERE run_id = p_run_id AND task_name = p_task_name;
        COMMIT;

        PERFORM dag_engine.fn_bgw_task(p_run_id, p_task_name, p_sql);

        SELECT status INTO v_final_status
        FROM dag_engine.task_instances
        WHERE run_id = p_run_id AND task_name = p_task_name;
        IF v_final_status = 'SUCCESS' THEN
            RAISE NOTICE '  ✅ [%] concluído (sync).', p_task_name;
        ELSE
            RAISE WARNING '  ❌ [%] falhou (sync): %', p_task_name,
                (SELECT error_text FROM dag_engine.task_instances
                 WHERE run_id = p_run_id AND task_name = p_task_name);
        END IF;
        RETURN;
    END IF;

    -- dblink disponível: fire-and-forget real
    PERFORM dblink_send_query(v_conn_name,
        format('SELECT dag_engine.fn_bgw_task(%s, %L, %L)', p_run_id, p_task_name, p_sql));

    UPDATE dag_engine.task_instances
    SET status      = 'RUNNING',
        start_ts    = clock_timestamp(),
        worker_conn = v_conn_name
    WHERE run_id = p_run_id AND task_name = p_task_name;

    INSERT INTO dag_engine.async_workers (conn_name, run_id, task_name)
    VALUES (v_conn_name, p_run_id, p_task_name)
    ON CONFLICT (conn_name) DO UPDATE SET launched_at = clock_timestamp();
END;
$$;

-- 9.10 proc_collect_workers: crash-recovery de workers dblink (timeout > 30min)
-- A coleta normal acontece inline no loop de proc_run_dag (FASE 1).
-- Este procedimento só intervém em workers que excederam o timeout — cancela a
-- query remota, drena o resultado e propaga UPSTREAM_FAILED para dependentes.
CREATE OR REPLACE PROCEDURE dag_engine.proc_collect_workers(
    p_run_id  INT,
    p_verbose BOOLEAN DEFAULT FALSE
)
LANGUAGE plpgsql AS $$
DECLARE
    v_worker RECORD;
BEGIN
    FOR v_worker IN
        SELECT aw.conn_name, aw.task_name, ti.start_ts
        FROM dag_engine.async_workers aw
        JOIN dag_engine.task_instances ti
          ON ti.run_id = aw.run_id AND ti.task_name = aw.task_name
        WHERE aw.run_id = p_run_id
    LOOP
        -- Apenas intervém em workers com timeout (> 30 min)
        CONTINUE WHEN (clock_timestamp() - v_worker.start_ts) <= INTERVAL '30 minutes';

        IF p_verbose THEN
            RAISE WARNING '⚠️ Worker [%] timeout (>30min). Cancelando e aplicando FAILED.', v_worker.task_name;
        END IF;

        BEGIN
            PERFORM dblink_cancel_query(v_worker.conn_name);
            PERFORM dblink_get_result(v_worker.conn_name);
            PERFORM dblink_disconnect(v_worker.conn_name);
        EXCEPTION WHEN OTHERS THEN NULL; END;

        DELETE FROM dag_engine.async_workers WHERE conn_name = v_worker.conn_name;

        UPDATE dag_engine.task_instances
        SET status      = 'FAILED',
            end_ts      = clock_timestamp(),
            worker_conn = NULL,
            error_text  = 'Worker timeout: cancelado após 30min'
        WHERE run_id = p_run_id AND task_name = v_worker.task_name AND status = 'RUNNING';

        WITH RECURSIVE fail_cascade AS (
            SELECT t.task_name, 1 AS depth FROM dag_engine.tasks t
            WHERE v_worker.task_name = ANY(t.dependencies)
            UNION ALL
            SELECT t.task_name, fc.depth + 1 FROM dag_engine.tasks t
            JOIN fail_cascade fc ON fc.task_name = ANY(t.dependencies)
            WHERE fc.depth < 100
        )
        UPDATE dag_engine.task_instances
        SET status     = 'UPSTREAM_FAILED',
            end_ts     = clock_timestamp(),
            error_text = 'Propagado de: ' || v_worker.task_name
        WHERE run_id = p_run_id
          AND task_name IN (SELECT task_name FROM fail_cascade)
          AND status = 'PENDING';

        COMMIT;
    END LOOP;
END;
$$;

-- 9.10b fn_wait_for_task_done: substitui pg_sleep(0.5) com LISTEN/NOTIFY + polling compacto
-- LISTEN registra o canal para monitors externos (libpq, pgAdmin, ferramentas MLOps).
-- Internamente usa 100ms de granularidade em vez de 500ms para maior responsividade.
CREATE OR REPLACE FUNCTION dag_engine.fn_wait_for_task_done(
    p_run_id     INT,
    p_timeout_ms INT DEFAULT 2000
) RETURNS BOOLEAN LANGUAGE plpgsql AS $$
DECLARE
    v_deadline TIMESTAMP := clock_timestamp() + (p_timeout_ms || 'ms')::INTERVAL;
BEGIN
    -- Registra no canal dag_task_done: monitors externos recebem push quando task termina
    EXECUTE 'LISTEN dag_task_done';

    LOOP
        -- Sai se não há mais tasks RUNNING no run (todas terminaram ou estão em backoff)
        IF NOT EXISTS (
            SELECT 1 FROM dag_engine.task_instances
            WHERE run_id = p_run_id AND status = 'RUNNING' AND bgw_pid IS NOT NULL
        ) THEN
            EXECUTE 'UNLISTEN dag_task_done';
            RETURN TRUE;
        END IF;

        EXIT WHEN clock_timestamp() >= v_deadline;
        PERFORM pg_sleep(0.1);  -- 100ms de granularidade vs 500ms original
    END LOOP;

    EXECUTE 'UNLISTEN dag_task_done';
    RETURN FALSE;  -- timeout — loop principal re-avalia o estado
END;
$$;

-- 9.10c fn_try_acquire_task: advisory lock por (run_id, task_name)
-- Garante exclusividade mesmo com múltiplas sessões externas reiniciando a DAG.
-- Complementa SKIP LOCKED (que protege apenas dentro da mesma sessão).
CREATE OR REPLACE FUNCTION dag_engine.fn_try_acquire_task(
    p_run_id    INT,
    p_task_name VARCHAR(100)
) RETURNS BOOLEAN LANGUAGE sql AS $$
    -- pg_try_advisory_xact_lock(integer, integer): dois INTs de 32-bit como chave
    -- p_run_id é INT nativo; hashtext() retorna INT — sem cast necessário
    SELECT pg_try_advisory_xact_lock(p_run_id, hashtext(p_task_name));
$$;

-- 9.11 proc_run_dag: substituição final — async dispatch + versionamento combinados
-- Loop de 4 fases: COLETA workers que terminaram → aguarda se ainda em voo →
-- DESPACHA tasks elegíveis → avalia estado final (deadlock/success/fail).
DROP PROCEDURE IF EXISTS dag_engine.proc_run_dag(DATE, BOOLEAN);
DROP PROCEDURE IF EXISTS dag_engine.proc_run_dag(TEXT, DATE, BOOLEAN);
CREATE OR REPLACE PROCEDURE dag_engine.proc_run_dag(
    p_dag_name TEXT,
    p_data     DATE,
    p_verbose  BOOLEAN     DEFAULT TRUE,
    p_run_type VARCHAR(20) DEFAULT 'INCREMENTAL'
)
LANGUAGE plpgsql AS $$
DECLARE
    v_run_id        INT;
    v_task          RECORD;
    v_worker        RECORD;
    v_sql           TEXT;
    v_pending_count INT;
    v_dispatched    INT;
    v_collected     INT;
    v_status        TEXT;
    v_error         TEXT;
BEGIN
    IF p_verbose THEN
        RAISE NOTICE '=================================================';
        RAISE NOTICE '🚀 Iniciando DAG Async [%] para: %', p_dag_name, p_data;
    END IF;

    BEGIN
        INSERT INTO dag_engine.dag_runs (dag_name, run_date, run_type, version_id)
        VALUES (p_dag_name, p_data, p_run_type, (
            SELECT version_id FROM dag_engine.dag_versions
            WHERE dag_name = p_dag_name AND is_active = TRUE
        ))
        RETURNING run_id INTO v_run_id;
    EXCEPTION WHEN unique_violation THEN
        RAISE WARNING 'Run já existe para "%" em %. Use proc_clear_run.', p_dag_name, p_data;
        RETURN;
    END;

    INSERT INTO dag_engine.task_instances (run_id, task_name)
    SELECT v_run_id, task_name FROM dag_engine.tasks WHERE dag_name = p_dag_name;
    COMMIT;

    LOOP
        -- FASE 1: Coleta workers dblink que terminaram (dblink_is_busy = FALSE)
        -- Obrigatório chamar dblink_get_result antes de dblink_disconnect para drenar
        -- o buffer de resultado; sem isso a conexão fica aberta e a task fica RUNNING.
        v_collected := 0;
        FOR v_worker IN
            SELECT conn_name, task_name FROM dag_engine.async_workers WHERE run_id = v_run_id
        LOOP
            IF dblink_is_busy(v_worker.conn_name) = 0 THEN
                BEGIN
                    PERFORM dblink_get_result(v_worker.conn_name);
                EXCEPTION WHEN OTHERS THEN NULL; END;
                PERFORM dblink_disconnect(v_worker.conn_name);
                DELETE FROM dag_engine.async_workers WHERE conn_name = v_worker.conn_name;
                v_collected := v_collected + 1;

                IF p_verbose THEN
                    SELECT status, error_text INTO v_status, v_error
                    FROM dag_engine.task_instances
                    WHERE run_id = v_run_id AND task_name = v_worker.task_name;

                    IF v_status = 'SUCCESS' THEN
                        RAISE NOTICE '  ✅ [%] concluído (async).', v_worker.task_name;
                    ELSE
                        RAISE WARNING '  ❌ [%] falhou (async): %', v_worker.task_name, v_error;
                    END IF;
                END IF;
            END IF;
        END LOOP;
        IF v_collected > 0 THEN COMMIT; END IF;

        -- FASE 2: Há workers ainda em voo mas nenhum terminou — aguarda 50ms
        IF v_collected = 0 AND EXISTS (
            SELECT 1 FROM dag_engine.async_workers WHERE run_id = v_run_id
        ) THEN
            PERFORM pg_sleep(0.05);
            CONTINUE;
        END IF;

        -- FASE 3: Despacha todas as tasks elegíveis da wave atual
        -- (deps são SUCCESS pois workers terminados foram coletados na FASE 1)
        v_dispatched := 0;
        FOR v_task IN
            SELECT ti.task_name, t.call_fn, t.call_args, t.chunk_config
            FROM dag_engine.task_instances ti
            JOIN dag_engine.tasks t ON ti.task_name = t.task_name
            WHERE ti.run_id = v_run_id
              AND ti.status = 'PENDING'
              AND (ti.retry_after_ts IS NULL OR ti.retry_after_ts <= clock_timestamp())
              AND NOT EXISTS (
                  SELECT 1 FROM unnest(t.dependencies) AS dep
                  JOIN dag_engine.task_instances dep_ti
                      ON dep_ti.run_id = v_run_id AND dep_ti.task_name = dep
                  WHERE dep_ti.status != 'SUCCESS'
              )
              AND dag_engine.fn_try_acquire_task(v_run_id, ti.task_name)
            ORDER BY t.task_id
            FOR UPDATE OF ti SKIP LOCKED
        LOOP
            -- Deriva chunk_index do sufixo _chunk_N no task_name (se aplicável)
            v_sql := dag_engine.fn_build_call_sql(
                v_task.call_fn, v_task.call_args, p_data,
                CASE WHEN v_task.task_name ~ '_chunk_\d+$'
                     THEN (regexp_match(v_task.task_name, '_chunk_(\d+)$'))[1]::INT
                     ELSE NULL END,
                v_task.chunk_config
            );
            IF p_verbose THEN
                RAISE NOTICE '  --> 📤 Despachando: [%]', v_task.task_name;
            END IF;
            CALL dag_engine.proc_dispatch_task(v_run_id, v_task.task_name, v_sql);
            v_dispatched := v_dispatched + 1;
        END LOOP;

        IF v_dispatched > 0 THEN
            COMMIT;
            CONTINUE;
        END IF;

        -- FASE 4: Nada coletado, nada despachado — avalia estado final
        IF EXISTS (SELECT 1 FROM dag_engine.async_workers WHERE run_id = v_run_id) THEN
            -- Workers ainda em voo (busy); aguarda próxima iteração
            PERFORM pg_sleep(0.05);
            CONTINUE;
        END IF;

        SELECT COUNT(*) INTO v_pending_count FROM dag_engine.task_instances
        WHERE run_id = v_run_id AND status = 'PENDING';

        IF v_pending_count > 0 THEN
            IF EXISTS (
                SELECT 1 FROM dag_engine.task_instances
                WHERE run_id = v_run_id AND status = 'PENDING'
                  AND retry_after_ts > clock_timestamp()
            ) THEN
                PERFORM pg_sleep(1);
            ELSE
                UPDATE dag_engine.dag_runs SET status = 'DEADLOCK', end_ts = clock_timestamp()
                WHERE run_id = v_run_id;
                COMMIT;
                RAISE WARNING '💀 Deadlock Topológico: tasks pendentes irresolvíveis.';
                EXIT;
            END IF;
        ELSE
            IF EXISTS (
                SELECT 1 FROM dag_engine.task_instances
                WHERE run_id = v_run_id AND status IN ('FAILED', 'UPSTREAM_FAILED')
            ) THEN
                UPDATE dag_engine.dag_runs SET status = 'FAILED', end_ts = clock_timestamp()
                WHERE run_id = v_run_id;
                RAISE WARNING '❌ DAG % finalizada com falhas.', p_data;
            ELSE
                UPDATE dag_engine.dag_runs SET status = 'SUCCESS', end_ts = clock_timestamp()
                WHERE run_id = v_run_id;
                IF p_verbose THEN RAISE NOTICE '✅ DAG % finalizada com sucesso!', p_data; END IF;
            END IF;
            COMMIT;
            EXIT;
        END IF;
    END LOOP;

    CALL dag_medallion.proc_run_medallion(v_run_id);
END;
$$;

-- 9.12 proc_cleanup_orphan_workers: drena e fecha conexões dblink órfãs
-- Chamado por proc_clear_run antes de deletar metadados do run.
CREATE OR REPLACE PROCEDURE dag_engine.proc_cleanup_orphan_workers(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_worker RECORD;
BEGIN
    FOR v_worker IN
        SELECT conn_name FROM dag_engine.async_workers WHERE run_id = p_run_id
    LOOP
        BEGIN
            PERFORM dblink_get_result(v_worker.conn_name);
            PERFORM dblink_disconnect(v_worker.conn_name);
        EXCEPTION WHEN OTHERS THEN NULL; END;
    END LOOP;

    DELETE FROM dag_engine.async_workers WHERE run_id = p_run_id;

    UPDATE dag_engine.task_instances
    SET status      = 'FAILED',
        worker_conn = NULL,
        error_text  = 'Worker órfão: conexão terminada pelo cleanup'
    WHERE run_id = p_run_id AND status = 'RUNNING';

    RAISE NOTICE '🧹 Workers órfãos do run_id % limpos.', p_run_id;
END;
$$;

-- 9.13 proc_clear_run: sobrescreve versão anterior — inclui cleanup de workers órfãos
DROP PROCEDURE IF EXISTS dag_engine.proc_clear_run(DATE);
DROP PROCEDURE IF EXISTS dag_engine.proc_clear_run(DATE, BOOLEAN);
CREATE OR REPLACE PROCEDURE dag_engine.proc_clear_run(p_dag_name TEXT, p_date DATE, p_verbose BOOLEAN DEFAULT TRUE)
LANGUAGE plpgsql AS $$
DECLARE v_run_id INT;
BEGIN
    SELECT run_id INTO v_run_id FROM dag_engine.dag_runs
    WHERE dag_name = p_dag_name AND run_date = p_date;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Nenhuma execução encontrada para o DAG "%" na data %', p_dag_name, p_date;
    END IF;

    IF EXISTS (SELECT 1 FROM dag_engine.dag_runs WHERE run_id = v_run_id AND status = 'RUNNING') THEN
        RAISE EXCEPTION '🚫 Run de % está RUNNING. Interrompa o worker antes de limpar.', p_date;
    END IF;

    -- Fecha conexões dblink órfãs antes de deletar metadados
    CALL dag_engine.proc_cleanup_orphan_workers(v_run_id);

    DELETE FROM dag_medallion.brnz_state_transitions_snap WHERE run_id = v_run_id;
    DELETE FROM dag_medallion.brnz_task_instances_snap    WHERE run_id = v_run_id;
    DELETE FROM dag_medallion.fato_task_exec              WHERE run_id = v_run_id;

    DELETE FROM dag_engine.state_transitions WHERE run_id = v_run_id;
    DELETE FROM dag_engine.task_instances     WHERE run_id = v_run_id;
    DELETE FROM dag_engine.dag_runs           WHERE run_id = v_run_id;

    IF p_verbose THEN RAISE NOTICE '🗑️ Run de % limpada. Pronto para re-execução.', p_date; END IF;
END;
$$;

-- ==============================================================================
-- 10. BACKLOG: OBSERVABILIDADE PROATIVA (Queue Depth + Throughput + Alertas)
-- ==============================================================================
-- Três gaps que completam o ciclo de observabilidade passiva → ativa:
--   Gap 1 — Queue Depth Timeline: quantas tasks estavam PENDING/RUNNING em cada
--            instante de uma run? Identifica gargalos de paralelismo.
--   Gap 2 — Throughput Metrics: tasks/hora e runs/dia normalizados por janela
--            de tempo real — v_task_percentiles conta, mas não normaliza.
--   Gap 3 — Alertas Proativos: pg_cron + pg_notify dispara alertas quando
--            health_score < 70 ou SLA breach ocorre — a infraestrutura já
--            existia (extensões instaladas, canal dag_events ativo), faltava
--            o conector.
-- ==============================================================================

-- ============================================================
-- 10.1 GAP 1: Queue Depth Timeline
-- ============================================================
-- Reconstrói, evento a evento via state_transitions, quantas tasks estavam
-- PENDING (fila esperando slot) e RUNNING (consumindo paralelismo) em cada
-- instante de cada run. Cada linha representa um momento de mudança de estado.
--
-- Como ler: concurrent_running = 3 significa que naquele instante havia 3
-- workers ativos simultâneos. queued_pending = 5 significa 5 tasks prontas
-- para rodar mas sem deps bloqueando — puro gargalo de capacidade de workers.
-- Ação 3: reconstruída com snapshot de estado absoluto por transição, eliminando
-- o bug de acumulação de deltas que produzia pct_queued = 88% constante.
DROP VIEW IF EXISTS dag_engine.v_queue_depth_timeline CASCADE;
CREATE VIEW dag_engine.v_queue_depth_timeline AS
WITH snapshots AS (
    -- Para cada transição, reconstrói o estado atual de todas as tasks deste run
    -- no instante imediatamente após a mudança
    SELECT
        st.run_id,
        dr.run_date,
        st.task_name  AS trigger_task,
        st.transition_ts,
        -- "estava RUNNING em T" = iniciou antes/em T e terminou depois de T
        COUNT(*) FILTER (
            WHERE ti.start_ts IS NOT NULL
              AND ti.start_ts <= st.transition_ts
              AND (ti.end_ts IS NULL OR ti.end_ts > st.transition_ts)
        ) AS concurrent_running,
        -- "estava PENDING e elegível em T" = ainda não iniciou, sem deps ativas
        COUNT(*) FILTER (
            WHERE (ti.start_ts IS NULL OR ti.start_ts > st.transition_ts)
              AND (ti.retry_after_ts IS NULL OR ti.retry_after_ts <= st.transition_ts)
              AND NOT EXISTS (
                  SELECT 1
                  FROM dag_engine.tasks t2
                  JOIN dag_engine.task_instances dep_ti
                      ON dep_ti.run_id = st.run_id
                     AND dep_ti.task_name = ANY(t2.dependencies)
                  WHERE t2.task_name = ti.task_name
                    AND (dep_ti.end_ts IS NULL OR dep_ti.end_ts > st.transition_ts)
              )
        ) AS queued_pending,
        -- total ainda não iniciado em T (inclui bloqueados por deps)
        COUNT(*) FILTER (
            WHERE ti.start_ts IS NULL OR ti.start_ts > st.transition_ts
        ) AS total_pending
    FROM dag_engine.state_transitions st
    JOIN dag_engine.dag_runs dr ON dr.run_id = st.run_id
    JOIN dag_engine.task_instances ti ON ti.run_id = st.run_id
    WHERE st.task_name IS NOT NULL
    GROUP BY st.run_id, dr.run_date, st.task_name, st.transition_ts
)
SELECT
    run_id,
    run_date,
    trigger_task,
    transition_ts,
    concurrent_running,
    queued_pending,
    total_pending,
    concurrent_running + queued_pending                             AS total_active,
    ROUND(
        100.0 * queued_pending
        / NULLIF(concurrent_running + queued_pending, 0)
    , 2)                                                           AS pct_queued,
    total_pending - queued_pending                                 AS blocked_by_deps
FROM snapshots
ORDER BY run_id, transition_ts;

-- ============================================================
-- 10.2 GAP 2: Throughput Metrics
-- ============================================================
-- Normaliza contagens por janela de tempo real de execução (wall time),
-- produzindo taxa de throughput comparável entre runs de tamanhos diferentes.
--
-- tasks_per_run_hour: tasks completadas por hora de pipeline wall time.
-- throughput_7d_avg: média móvel de 7 dias — tendência suave de capacidade.
-- throughput_dod_pct: variação percentual dia-a-dia — detecta degradação abrupta.
CREATE OR REPLACE VIEW dag_engine.v_throughput_metrics AS
WITH per_run AS (
    SELECT
        f.run_date,
        COUNT(*)                                                              AS tasks_total,
        COUNT(*) FILTER (WHERE f.final_status = 'SUCCESS')                   AS tasks_succeeded,
        COUNT(*) FILTER (WHERE f.final_status = 'FAILED')                    AS tasks_failed,
        COUNT(*) FILTER (WHERE f.final_status = 'UPSTREAM_FAILED')           AS tasks_upstream_failed,
        COUNT(*) FILTER (WHERE f.had_retry)                                   AS tasks_with_retry,
        -- CPU total consumido pelas tasks (soma de durações individuais)
        ROUND(SUM(f.duration_ms) / 60000.0, 4)                               AS total_task_cpu_min,
        -- Wall time real da run (do primeiro start ao último end)
        ROUND(EXTRACT(EPOCH FROM (MAX(f.end_ts) - MIN(f.start_ts))) / 60.0, 4) AS wall_min
    FROM dag_medallion.fato_task_exec f
    WHERE f.start_ts IS NOT NULL AND f.end_ts IS NOT NULL
    GROUP BY f.run_date
)
SELECT
    run_date,
    tasks_total,
    tasks_succeeded,
    tasks_failed,
    tasks_upstream_failed,
    tasks_with_retry,
    ROUND(100.0 * tasks_succeeded / NULLIF(tasks_total, 0), 2)                AS success_rate_pct,
    total_task_cpu_min,
    wall_min,
    -- Paralelismo médio efetivo: CPU / wall — quanto do tempo paralelo foi aproveitado
    ROUND(total_task_cpu_min / NULLIF(wall_min, 0), 2)                        AS avg_parallelism,
    -- Throughput normalizado: tasks concluídas por hora de wall time
    ROUND(tasks_succeeded / NULLIF(wall_min / 60.0, 0), 2)                    AS tasks_per_run_hour,
    -- Tendência de throughput: média móvel 7 dias
    ROUND(AVG(tasks_succeeded / NULLIF(wall_min / 60.0, 0))
          OVER (ORDER BY run_date::TIMESTAMP RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW), 2) AS throughput_7d_avg,
    -- Variação diária de throughput (detecta regressões de capacidade)
    ROUND(
        (  (tasks_succeeded / NULLIF(wall_min / 60.0, 0))
         / NULLIF(LAG(tasks_succeeded / NULLIF(wall_min / 60.0, 0))
                  OVER (ORDER BY run_date), 0) - 1
        ) * 100
    , 2)                                                                       AS throughput_dod_pct
FROM per_run
ORDER BY run_date DESC;

-- ============================================================
-- 10.3 GAP 3: Alertas Proativos (pg_cron + pg_notify)
-- ============================================================
-- A infraestrutura já existia (pg_cron carregado, pg_notify no motor de
-- estado, gold_pipeline_health e gold_sla_breach no Medallion) — faltava
-- o elo de conexão entre observabilidade passiva e disparo ativo.
--
-- Canal pg_notify: 'dag_alerts' (separado de 'dag_events' que é infra)
-- Formato do payload: JSON com alert_type, task, métrica, ts
-- Clientes externos (Python, Node, etc.) fazem LISTEN 'dag_alerts' para
-- receber as notificações em tempo real sem polling.

-- Tabela de log de alertas: histórico auditável e deduplicação por cooldown
CREATE TABLE IF NOT EXISTS dag_engine.alert_log (
    alert_id   SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,   -- 'HEALTH_DEGRADED' | 'SLA_BREACH'
    task_name  VARCHAR(100),
    run_date   DATE,
    payload    JSONB NOT NULL,
    fired_at   TIMESTAMP DEFAULT clock_timestamp()
);

-- Índice para consulta de cooldown: não re-disparar o mesmo alerta dentro de 1h
CREATE INDEX IF NOT EXISTS idx_alert_log_dedup
ON dag_engine.alert_log (alert_type, task_name, fired_at);

-- fn_check_alerts: verifica condições de alerta e dispara pg_notify + log
-- Retorna o número de novos alertas disparados nesta invocação.
-- Deduplicação: ignora combinação (tipo, task) que já foi alertada na última hora.
CREATE OR REPLACE FUNCTION dag_engine.fn_check_alerts(
    p_health_threshold  NUMERIC DEFAULT 70,   -- dispara se health_score < threshold
    p_cooldown_minutes  INT     DEFAULT 60    -- não re-dispara o mesmo alerta neste intervalo
) RETURNS INT LANGUAGE plpgsql AS $$
DECLARE
    v_rec     RECORD;
    v_payload JSONB;
    v_count   INT := 0;
    v_cutoff  TIMESTAMP := clock_timestamp() - (p_cooldown_minutes * INTERVAL '1 minute');
BEGIN
    -- --------------------------------------------------------
    -- CONDIÇÃO 1: Health Score abaixo do threshold
    -- --------------------------------------------------------
    FOR v_rec IN
        SELECT task_name, health_score, health_label, total_runs, success_rate, retry_rate
        FROM dag_medallion.gold_pipeline_health
        WHERE health_score < p_health_threshold
          AND health_label != '🔵 CALIBRANDO'  -- NOVO: ignora tasks ainda sem histórico suficiente
    LOOP
        -- Deduplicação: só dispara se não houve alerta do mesmo tipo para esta task recentemente
        CONTINUE WHEN EXISTS (
            SELECT 1 FROM dag_engine.alert_log
            WHERE alert_type = 'HEALTH_DEGRADED'
              AND task_name  = v_rec.task_name
              AND fired_at   > v_cutoff
        );

        v_payload := jsonb_build_object(
            'alert_type',   'HEALTH_DEGRADED',
            'task',         v_rec.task_name,
            'health_score', v_rec.health_score,
            'label',        v_rec.health_label,
            'success_rate', v_rec.success_rate,
            'retry_rate',   v_rec.retry_rate,
            'total_runs',   v_rec.total_runs,
            'ts',           clock_timestamp()
        );

        PERFORM pg_notify('dag_alerts', v_payload::TEXT);

        INSERT INTO dag_engine.alert_log (alert_type, task_name, payload)
        VALUES ('HEALTH_DEGRADED', v_rec.task_name, v_payload);

        v_count := v_count + 1;
    END LOOP;

    -- --------------------------------------------------------
    -- CONDIÇÃO 2: SLA Breach nas últimas 24h
    -- --------------------------------------------------------
    FOR v_rec IN
        SELECT run_date, task_name, sla_status, actual_ms, sla_target_ms, breach_pct
        FROM dag_medallion.gold_sla_breach
        WHERE run_date >= CURRENT_DATE - 1
          AND sla_status NOT LIKE '%DENTRO%'   -- qualquer nível de breach
        ORDER BY breach_pct DESC
    LOOP
        CONTINUE WHEN EXISTS (
            SELECT 1 FROM dag_engine.alert_log
            WHERE alert_type = 'SLA_BREACH'
              AND task_name  = v_rec.task_name
              AND run_date   = v_rec.run_date   -- um alerta por (run, task), não por hora
        );

        v_payload := jsonb_build_object(
            'alert_type', 'SLA_BREACH',
            'run_date',   v_rec.run_date,
            'task',       v_rec.task_name,
            'status',     v_rec.sla_status,
            'actual_ms',  v_rec.actual_ms,
            'sla_target_ms', v_rec.sla_target_ms,
            'breach_pct', v_rec.breach_pct,
            'ts',         clock_timestamp()
        );

        PERFORM pg_notify('dag_alerts', v_payload::TEXT);

        INSERT INTO dag_engine.alert_log (alert_type, task_name, run_date, payload)
        VALUES ('SLA_BREACH', v_rec.task_name, v_rec.run_date, v_payload);

        v_count := v_count + 1;
    END LOOP;

    -- ──────────────────────────────────────────────────────────────────────────────
    -- CONDIÇÃO 3: Degradação contínua por N dias (não detectável por cooldown)
    -- ──────────────────────────────────────────────────────────────────────────────
    FOR v_rec IN
        SELECT task_name, version_id, streak_days, streak_start
        FROM dag_medallion.gold_degradation_streaks
        WHERE health_state = 'DEGRADADO'
          AND is_current   = TRUE
          AND streak_days  >= 3
          -- Correlaciona version_id com a versão mais recente que rodou esta task
          -- (correto em ambientes multi-DAG: cada task aponta para o seu próprio DAG)
          AND version_id = (
              SELECT dr.version_id
              FROM dag_engine.dag_runs dr
              JOIN dag_engine.task_instances ti ON ti.run_id = dr.run_id
              WHERE ti.task_name  = gold_degradation_streaks.task_name
                AND dr.version_id IS NOT NULL
              ORDER BY dr.run_date DESC
              LIMIT 1
          )
    LOOP
        CONTINUE WHEN EXISTS (
            SELECT 1 FROM dag_engine.alert_log
            WHERE alert_type = 'DEGRADATION_STREAK'
              AND task_name  = v_rec.task_name
              AND fired_at   > v_cutoff
        );

        v_payload := jsonb_build_object(
            'alert_type',   'DEGRADATION_STREAK',
            'task',         v_rec.task_name,
            'version_id',   v_rec.version_id,
            'streak_days',  v_rec.streak_days,
            'streak_start', v_rec.streak_start,
            'ts',           clock_timestamp()
        );

        PERFORM pg_notify('dag_alerts', v_payload::TEXT);

        INSERT INTO dag_engine.alert_log (alert_type, task_name, payload)
        VALUES ('DEGRADATION_STREAK', v_rec.task_name, v_payload);

        v_count := v_count + 1;
    END LOOP;


    RETURN v_count;
END;
$$;

-- Agendamento via pg_cron: verifica alertas a cada 5 minutos
-- (Requer pg_cron no shared_preload_libraries — habilitado se disponível na seção 1)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'pg_cron') THEN
        -- Idempotente: remove job anterior se existir, depois re-registra
        BEGIN PERFORM cron.unschedule('dag_alert_check'); EXCEPTION WHEN OTHERS THEN NULL; END;
        PERFORM cron.schedule('dag_alert_check', '*/5 * * * *', 'SELECT dag_engine.fn_check_alerts()');
        RAISE NOTICE '✅ pg_cron job "dag_alert_check" registrado (a cada 5 min).';
    ELSE
        RAISE NOTICE '⚠️  pg_cron não disponível — job "dag_alert_check" não agendado. Chame SELECT dag_engine.fn_check_alerts() manualmente.';
    END IF;
END $$;

-- View de inspeção do log de alertas — últimos alertas disparados
CREATE OR REPLACE VIEW dag_engine.v_alert_log AS
SELECT
    alert_id,
    alert_type,
    task_name,
    run_date,
    payload->>'label'       AS health_label,
    (payload->>'health_score')::NUMERIC AS health_score,
    (payload->>'breach_pct')::NUMERIC   AS breach_pct,
    payload->>'status'      AS sla_status,
    fired_at,
    -- Agrupamento temporal para contagem de frequência (alertas/hora)
    date_trunc('hour', fired_at) AS fired_hour
FROM dag_engine.alert_log
ORDER BY fired_at DESC;

-- ==============================================================================
-- 6. DAG INITIALIZATION (Definição do Pipeline no Motor via JSON)
-- ==============================================================================
-- Transformamos a inserção crua num formato declarativo JSONB. 
-- Imagine ler isso tudo diretamente de um arquivo yaml ou json externo no backend!

-- Reset completo ANTES do deploy inicial: garante que version_id=1 seja carimbado
-- nas runs 7.1–7.4 e a demo de rastreabilidade da seção 7.6 funcione corretamente.
-- (Se dag_versions fosse truncada DEPOIS do deploy, as runs ficariam com version_id=NULL)
TRUNCATE dag_medallion.brnz_state_transitions_snap CASCADE;
TRUNCATE dag_medallion.brnz_task_instances_snap    CASCADE;
TRUNCATE dag_medallion.fato_task_exec              CASCADE;
TRUNCATE dag_engine.async_workers                  CASCADE;
TRUNCATE dag_engine.state_transitions              CASCADE;
TRUNCATE dag_engine.task_instances                 CASCADE;
TRUNCATE dag_engine.dag_runs                       CASCADE;
TRUNCATE dag_engine.dag_versions                   CASCADE;
TRUNCATE dag_engine.alert_log                      CASCADE;

CALL dag_engine.proc_deploy_dag('{
    "name": "daily_varejo_dw",
    "description": "Carga D-1 completa das Fatos e Dimensões do Varejo",
    "schedule": "0 2 * * *",
    "tasks": [
    {
        "task_name": "1_snapshot_clientes",
        "call_fn": "varejo.proc_snapshot_clientes",
        "call_args": ["$1"],
        "medallion_layer": "BRONZE",
        "dependencies": [],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "2_upsert_clientes_scd1",
        "call_fn": "varejo.proc_upsert_clientes_scd1",
        "call_args": [],
        "medallion_layer": "BRONZE",
        "dependencies": [],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "3_upsert_clientes_scd2",
        "call_fn": "varejo.proc_upsert_clientes_scd2",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": ["1_snapshot_clientes", "2_upsert_clientes_scd1"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "4_upsert_produtos_scd3",
        "call_fn": "varejo.proc_upsert_produtos_scd3",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": [],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "5_ingestao_fato_vendas",
        "call_fn": "varejo.proc_ingestao_fato_vendas",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": ["3_upsert_clientes_scd2", "4_upsert_produtos_scd3"],
        "max_retries": 1,
        "retry_delay_seconds": 10
    },
    {
        "task_name": "6_acumular_atividade",
        "call_fn": "varejo.proc_acumular_atividade",
        "call_args": ["($1::DATE - 1)", "$1"],
        "medallion_layer": "SILVER",
        "dependencies": ["5_ingestao_fato_vendas"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "7_acumular_vendas_mes",
        "call_fn": "varejo.proc_acumular_vendas_mensal",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": ["6_acumular_atividade"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "8_ingestao_gold_diaria",
        "call_fn": "varejo.proc_ingestao_gold_diaria",
        "call_args": ["$1"],
        "medallion_layer": "GOLD",
        "dependencies": ["7_acumular_vendas_mes"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    }
    ]
}'::JSONB,
'Spec inicial — 8 tasks do pipeline varejo');

-- Seed do registry para o pipeline de exemplo
INSERT INTO dag_engine.table_layer_registry VALUES
    ('varejo', 'origem_cliente',              'Bronze'),
    ('varejo', 'origem_produto',              'Bronze'),
    ('varejo', 'origem_venda',                'Bronze'),
    ('varejo', 'cliente_snapshot_diario',     'Silver'),
    ('varejo', 'dim_cliente_type1',           'Silver'),
    ('varejo', 'dim_cliente_type2',           'Silver'),
    ('varejo', 'dim_produto_type3',           'Silver'),
    ('varejo', 'fato_vendas',                 'Silver'),
    ('varejo', 'fct_atividade_acum',          'Silver'),
    ('varejo', 'fct_vendas_mensal_array',     'Silver'),
    ('varejo', 'fct_metricas_diarias',        'Gold'),
    ('dag_semantic', 'rpt_oncall_handoff',       'Semantic'),
    ('dag_semantic', 'rpt_weekly_reliability',   'Semantic'),
    ('dag_semantic', 'rpt_monthly_sla_delivery', 'Semantic'),
    ('dag_semantic', 'rpt_version_impact',       'Semantic')
ON CONFLICT DO NOTHING;

-- ==============================================================================
-- 7. ÁREA DE TESTES E INTERAÇÃO (Hands-on Standalone Demonstração)
-- ==============================================================================
-- Aqui nós fundimos toda a lógica de negócio do "apresentacao.sql" sendo orquestrada
-- de verdade e nativamente apenas por este motor, sem precisar de for loops improvisados!

-- 7.0 Tuning da Sessão Local e Reset Completo do DW
SET synchronous_commit = off;       
SET work_mem = '256MB';            
SET maintenance_work_mem = '256MB'; 

DO $$ BEGIN RAISE NOTICE '🔄 Resetando o estado do OLTP e limpando o DW...'; END $$;

-- Cria Índices pro Engine voar baixo processando 2 meses de dados na CPU local
CREATE INDEX IF NOT EXISTS idx_fato_data ON varejo.fato_vendas(data_venda);
CREATE INDEX IF NOT EXISTS idx_ativ_acum_data ON varejo.fct_atividade_acum(data_snapshot);
CREATE INDEX IF NOT EXISTS idx_vendas_arr_mes ON varejo.fct_vendas_mensal_array(mes_referencia);
CREATE INDEX IF NOT EXISTS idx_snap_diario_data ON varejo.cliente_snapshot_diario(data_snapshot);
CREATE INDEX IF NOT EXISTS idx_origem_venda_data ON varejo.origem_venda(data_venda);
CREATE INDEX IF NOT EXISTS idx_dim_cli2_ativo ON varejo.dim_cliente_type2(cliente_id) WHERE ativo = TRUE;

-- Estado OLTP do Início do Curso
UPDATE varejo.origem_cliente SET estado = 'SP' WHERE cliente_id = 101;
UPDATE varejo.origem_produto SET categoria = 'Informática' WHERE produto_id = 'PROD001';

-- Reset das tabelas de negócio (o engine já foi resetado na seção 6)
TRUNCATE varejo.dim_cliente_type1 CASCADE;
TRUNCATE varejo.dim_cliente_type2 RESTART IDENTITY CASCADE;
TRUNCATE varejo.dim_produto_type3 CASCADE;
TRUNCATE varejo.fato_vendas RESTART IDENTITY CASCADE;
TRUNCATE varejo.fct_atividade_acum CASCADE;
TRUNCATE varejo.fct_vendas_mensal_array CASCADE;
TRUNCATE varejo.cliente_snapshot_diario CASCADE;
TRUNCATE varejo.fct_metricas_diarias CASCADE;

-- 7.1. Carga ETL Inicial (Primeiro dia da camada crua para ODS/SCDs)
DO $$ BEGIN RAISE NOTICE '📥 1. Executando Carga Inicial do DW via DAG Engine (D-0, 2024-05-04)...'; END $$;
CALL dag_engine.proc_run_dag('daily_varejo_dw', '2024-05-04');

-- 7.2. Mudança OLTP Simples (Para observar as chaves substitutas em ação amanhã)
UPDATE varejo.origem_cliente SET estado = 'PR' WHERE cliente_id = 101;
UPDATE varejo.origem_produto SET categoria = 'Gamer' WHERE produto_id = 'PROD001';

-- 7.3. Carga Incremental Seguindo Alterações Temporais
DO $$ BEGIN RAISE NOTICE '📥 2. Executando Carga Incremental via DAG (D-1, 2024-05-05)...'; END $$;
CALL dag_engine.proc_run_dag('daily_varejo_dw', '2024-05-05');

-- 7.4. Fast-Forward Temporário Robusto
-- Com 2 meses de backfill rodando massivamente com tolerância a deadlock na arquitetura
DO $$ 
DECLARE
    ts_start TIMESTAMP := clock_timestamp();
    dur_backfill INTERVAL;
BEGIN 
    RAISE NOTICE '🚀 3. Iniciando MLOps Fast-Forward de 2 meses contínuos...'; 
    CALL dag_engine.proc_catchup('daily_varejo_dw', '2024-05-06'::DATE, '2024-07-04'::DATE);
    
    dur_backfill := clock_timestamp() - ts_start;
    RAISE NOTICE '✅ Fast-Forward concluído com sucesso via Motor Nativo em %!', dur_backfill;
END $$;

-- 7.5. Observabilidade - Métrica Estatística de Saúde e Meta-Medallion no PostgreSQL
DO $$ BEGIN RAISE NOTICE '========================================================='; END $$;
DO $$ BEGIN RAISE NOTICE '📊 RESULTADO FINAL DA ENGINE DE AUTOMAÇÃO DE DAG NO BANCO'; END $$;
DO $$ BEGIN RAISE NOTICE '========================================================='; END $$;

SELECT * FROM dag_engine.v_task_percentiles ORDER BY p99_ms DESC;

-- Use the Medallion Output Analytics Dashboards here:
SELECT * 
FROM dag_medallion.gold_pipeline_health;

SELECT * 
FROM dag_medallion.gold_critical_path 
ORDER BY topological_layer ASC, pct_pipeline_time DESC;

SELECT * 
FROM dag_medallion.gold_performance_timelapse 
WHERE task_name = '7_acumular_vendas_mes' ORDER BY run_date DESC;

-- ==============================================================================
-- 7.6. BACKLOG DEMO: VERSIONAMENTO
-- ==============================================================================

-- Árvore genealógica do pipeline: geração, quando deployou, o que mudou, quantas runs acumulou
SELECT dag_name, version_tree, deployed_at, change_summary, total_runs
FROM dag_engine.vw_version_lineage;

-- Inspecionar a versão ativa e o histórico de deploys (visão tabular detalhada)
SELECT version_id, dag_name, version_tag, description, schedule, deployed_at, change_summary, is_active, spec_hash
FROM dag_engine.dag_versions
ORDER BY dag_name, deployed_at;

-- Simula um redeploy com mudança: adiciona task de envio de email pós-gold
-- (Em produção: altere o JSONB real do pipeline)
CALL dag_engine.proc_deploy_dag('{
    "name": "daily_varejo_dw",
    "description": "Carga D-1 completa das Fatos e Dimensões do Varejo",
    "schedule": "0 2 * * *",
    "tasks": [
    {"task_name": "1_snapshot_clientes", "call_fn": "varejo.proc_snapshot_clientes", "call_args": ["$1"], "medallion_layer": "BRONZE", "dependencies": []},
    {"task_name": "2_upsert_clientes_scd1", "call_fn": "varejo.proc_upsert_clientes_scd1", "call_args": [], "medallion_layer": "BRONZE", "dependencies": []},
    {"task_name": "3_upsert_clientes_scd2", "call_fn": "varejo.proc_upsert_clientes_scd2", "call_args": ["$1"], "medallion_layer": "SILVER", "dependencies": ["1_snapshot_clientes", "2_upsert_clientes_scd1"]},
    {"task_name": "4_upsert_produtos_scd3", "call_fn": "varejo.proc_upsert_produtos_scd3", "call_args": ["$1"], "medallion_layer": "SILVER", "dependencies": []},
    {"task_name": "5_ingestao_fato_vendas", "call_fn": "varejo.proc_ingestao_fato_vendas", "call_args": ["$1"], "medallion_layer": "SILVER", "dependencies": ["3_upsert_clientes_scd2", "4_upsert_produtos_scd3"], "max_retries": 1, "retry_delay_seconds": 10},
    {"task_name": "6_acumular_atividade", "call_fn": "varejo.proc_acumular_atividade", "call_args": ["($1::DATE - 1)", "$1"], "medallion_layer": "SILVER", "dependencies": ["5_ingestao_fato_vendas"]},
    {"task_name": "7_acumular_vendas_mes", "call_fn": "varejo.proc_acumular_vendas_mensal", "call_args": ["$1"], "medallion_layer": "SILVER", "dependencies": ["6_acumular_atividade"]},
    {"task_name": "8_ingestao_gold_diaria", "call_fn": "varejo.proc_ingestao_gold_diaria", "call_args": ["$1"], "medallion_layer": "GOLD", "dependencies": ["7_acumular_vendas_mes"]},
    {"task_name": "9_envio_relatorio", "call_fn": "dag_engine.proc_noop", "call_args": ["$1"], "medallion_layer": "ORCHESTRATION", "dependencies": ["8_ingestao_gold_diaria"]}
    ]
}'::JSONB,
'Adicionada task 9_envio_relatorio no downstream do gold'
);

-- Inspeciona o diff entre a penúltima e a última versão ativa do DAG
-- (subquery dinâmica: funciona em qualquer ambiente, sem hardcode de IDs)
SELECT * FROM dag_engine.fn_diff_versions(
    (SELECT version_id FROM dag_engine.dag_versions
     WHERE dag_name = 'daily_varejo_dw' ORDER BY deployed_at DESC LIMIT 1 OFFSET 1),
    (SELECT version_id FROM dag_engine.dag_versions
     WHERE dag_name = 'daily_varejo_dw' ORDER BY deployed_at DESC LIMIT 1)
);

-- Confirma que runs estão carimbadas com version_id
SELECT dr.run_date, dr.status, dv.version_tag, dv.change_summary
FROM dag_engine.dag_runs dr
LEFT JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
ORDER BY dr.run_date DESC
LIMIT 10;

-- Análise de performance por versão (rastreabilidade: separar variáveis de estrutura de variáveis de volume)
SELECT dv.version_tag, f.task_name, ROUND(AVG(f.duration_ms), 2) AS avg_ms
FROM dag_medallion.fato_task_exec f
JOIN dag_engine.dag_runs dr ON dr.run_id = f.run_id
LEFT JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
GROUP BY dv.version_tag, f.task_name
ORDER BY f.task_name, dv.version_tag;

-- Diagrama Mermaid da topologia ativa (com health overlay)
-- Copie a saída e cole em https://mermaid.live para visualizar interativamente
SELECT dag_engine.fn_dag_to_mermaid('daily_varejo_dw'::TEXT);

-- Diagrama da topologia pura (sem health overlay — só estrutura e waves)
SELECT dag_engine.fn_dag_to_mermaid('daily_varejo_dw'::TEXT, NULL::INT, FALSE);

-- Diagrama da v1.0.0 a partir do snapshot histórico frozen (antes da task 9_envio_relatorio)
SELECT dag_engine.fn_dag_to_mermaid(
    'daily_varejo_dw'::TEXT,
    (SELECT version_id FROM dag_engine.dag_versions
     WHERE dag_name = 'daily_varejo_dw' ORDER BY deployed_at ASC LIMIT 1)::INT
);

-- Topologia com lineage sobreposto (arestas pontilhadas com rótulo r/w)
-- Copie a saída e cole em https://mermaid.live para visualizar interativamente
SELECT dag_engine.fn_dag_to_mermaid('daily_varejo_dw', NULL, TRUE, FALSE, TRUE);

-- ==============================================================================
-- 7.7. BACKLOG DEMO: EXECUÇÃO ASSÍNCRONA
-- ==============================================================================

-- Testa expansão de chunks manualmente
SELECT * FROM dag_engine.fn_expand_chunk_tasks(
    '5_ingestao_fato_vendas',
    'CALL varejo.proc_ingestao_fato_vendas($1, $range_start::TIMESTAMP, $range_end::TIMESTAMP)',
    ARRAY['3_upsert_clientes_scd2', '4_upsert_produtos_scd3'],
    '{"column": "data_venda", "buckets": 4}'::JSONB,
    '2024-05-04'::DATE
);

-- Testa extração de tabelas via catálogo
SELECT * FROM dag_engine.fn_extract_tables_from_proc('varejo.proc_ingestao_fato_vendas');

-- Confirma que não há workers dblink órfãos após as runs
SELECT * FROM dag_engine.async_workers;  -- deve retornar 0 linhas

-- Topologia vw_topological_sort: confirma layers e camadas de paralelismo
SELECT dag_name, task_name, topological_layer, dependencies
FROM dag_engine.vw_topological_sort
WHERE dag_name = 'daily_varejo_dw'
ORDER BY topological_layer, task_name;

-- ==============================================================================
-- 7.8. BACKLOG DEMO: OBSERVABILIDADE PROATIVA
-- ==============================================================================

-- Queue Depth Timeline
-- Seleciona a última run e mostra o perfil de concorrência ao longo do tempo.
-- concurrent_running > 1 = paralelismo real em ação.
-- queued_pending alto com concurrent_running travado = gargalo de capacidade.
SELECT
    trigger_task,
    transition_ts,
    concurrent_running,
    queued_pending,
    total_active,
    pct_queued
FROM dag_engine.v_queue_depth_timeline
WHERE run_id = (SELECT MAX(run_id) FROM dag_engine.dag_runs)
ORDER BY transition_ts;

-- Pico de paralelismo e gargalo máximo por run
SELECT
    run_id,
    run_date,
    MAX(concurrent_running) AS peak_parallelism,
    MAX(queued_pending)     AS max_queue_depth,
    ROUND(AVG(pct_queued), 2) AS avg_pct_queued
FROM dag_engine.v_queue_depth_timeline
GROUP BY run_id, run_date
ORDER BY run_date DESC;

-- Throughput Metrics
-- tasks/hora normalizado por wall time real — comparar runs de dias diferentes.
-- avg_parallelism < 1 = pipeline seqüencial na prática (desperdício de dblink).
-- throughput_dod_pct negativo = regressão de capacidade no dia.
SELECT
    run_date,
    tasks_succeeded,
    tasks_failed,
    ROUND(wall_min, 2)             AS wall_min,
    avg_parallelism,
    tasks_per_run_hour,
    throughput_7d_avg,
    throughput_dod_pct
FROM dag_engine.v_throughput_metrics
LIMIT 20;

-- Alertas Proativos
-- Dispara manualmente para verificar a lógica (em produção o pg_cron chama automaticamente)
SELECT dag_engine.fn_check_alerts();  -- retorna número de alertas disparados

-- Inspeciona o log de alertas
SELECT alert_type, task_name, run_date, health_label, health_score, breach_pct, sla_status, fired_at
FROM dag_engine.v_alert_log
LIMIT 20;

-- Confirma que o job pg_cron foi registrado (só executa se pg_cron estiver disponível)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'pg_cron') THEN
        RAISE NOTICE 'pg_cron jobs ativos: %',
            (SELECT string_agg(jobname || ' [' || schedule || ']', ', ')
             FROM cron.job WHERE jobname = 'dag_alert_check');
    ELSE
        RAISE NOTICE '⚠️  pg_cron não disponível neste ambiente.';
    END IF;
END $$;

-- Para ouvir alertas em tempo real em outro cliente:
-- (Execute dag_engine.fn_check_alerts() numa sessão e o LISTEN recebe o JSON)

-- ==============================================================================
-- 7.9. AÇÃO 2 — Redeploy com Chunking do Critical Path
-- 1_snapshot_clientes (35%) e 6_acumular_atividade (35%) particionadas em 4 chunks
-- paralelos. Bump automático para v3.0.0 (MAJOR: tasks removidas e substituídas).
-- ==============================================================================
CALL dag_engine.proc_deploy_dag('{
    "name": "daily_varejo_dw",
    "description": "Carga D-1 completa das Fatos e Dimensões do Varejo",
    "schedule": "0 2 * * *",
    "tasks": [
    {
        "task_name": "1_snapshot_clientes",
        "call_fn": "varejo.proc_snapshot_clientes",
        "call_args": ["$1", "$chunk_filter"],
        "medallion_layer": "BRONZE",
        "dependencies": [],
        "chunk_config": {"column": "cliente_id", "buckets": 4, "method": "hash"}
    },
    {
        "task_name": "2_upsert_clientes_scd1",
        "call_fn": "varejo.proc_upsert_clientes_scd1",
        "call_args": [],
        "medallion_layer": "BRONZE",
        "dependencies": []
    },
    {
        "task_name": "3_upsert_clientes_scd2",
        "call_fn": "varejo.proc_upsert_clientes_scd2",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": ["1_snapshot_clientes", "2_upsert_clientes_scd1"]
    },
    {
        "task_name": "4_upsert_produtos_scd3",
        "call_fn": "varejo.proc_upsert_produtos_scd3",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": []
    },
    {
        "task_name": "5_ingestao_fato_vendas",
        "call_fn": "varejo.proc_ingestao_fato_vendas",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": ["3_upsert_clientes_scd2", "4_upsert_produtos_scd3"],
        "max_retries": 1,
        "retry_delay_seconds": 10
    },
    {
        "task_name": "6_acumular_atividade",
        "call_fn": "varejo.proc_acumular_atividade",
        "call_args": ["($1::DATE - 1)", "$1", "$chunk_filter"],
        "medallion_layer": "SILVER",
        "dependencies": ["5_ingestao_fato_vendas"],
        "chunk_config": {"column": "cliente_id", "method": "hash", "buckets": 4}
    },
    {
        "task_name": "7_acumular_vendas_mes",
        "call_fn": "varejo.proc_acumular_vendas_mensal",
        "call_args": ["$1"],
        "medallion_layer": "SILVER",
        "dependencies": ["6_acumular_atividade"]
    },
    {
        "task_name": "8_ingestao_gold_diaria",
        "call_fn": "varejo.proc_ingestao_gold_diaria",
        "call_args": ["$1"],
        "medallion_layer": "GOLD",
        "dependencies": ["7_acumular_vendas_mes"]
    },
    {
        "task_name": "9_envio_relatorio",
        "call_fn": "dag_engine.proc_noop",
        "call_args": ["$1"],
        "medallion_layer": "ORCHESTRATION",
        "dependencies": ["8_ingestao_gold_diaria"]
    }
    ]
}'::JSONB,
'v3: chunking hash 4x em 1_snapshot_clientes e 6_acumular_atividade por cliente_id (critical path 70%)'
);

-- Confirma as sub-tasks geradas pelo chunking
SELECT task_name, topological_layer, dependencies
FROM dag_engine.vw_topological_sort
WHERE dag_name = 'daily_varejo_dw'
ORDER BY topological_layer, task_name;

-- Roda um dia com o novo motor e spec v3
CALL dag_engine.proc_clear_run('daily_varejo_dw', '2024-07-04');
CALL dag_engine.proc_run_dag('daily_varejo_dw', '2024-07-04');

-- Validação: paralelismo real na nova run
SELECT
    MAX(concurrent_running)  AS peak_parallelism,
    MIN(pct_queued)          AS min_pct_queued,
    MAX(blocked_by_deps)     AS max_blocked_by_deps
FROM dag_engine.v_queue_depth_timeline
WHERE run_id = (SELECT MAX(run_id) FROM dag_engine.dag_runs);

-- Chunks instanciados e suas durações
SELECT task_name, status, start_ts, end_ts, duration_ms
FROM dag_engine.task_instances
WHERE run_id = (SELECT MAX(run_id) FROM dag_engine.dag_runs)
  AND task_name LIKE '%chunk%'
ORDER BY start_ts;

-- Wall time por versão: antes vs depois do chunking
SELECT
    dr.run_date,
    dv.version_tag,
    ROUND(EXTRACT(EPOCH FROM (dr.end_ts - dr.start_ts)) * 1000, 2) AS wall_ms
FROM dag_engine.dag_runs dr
JOIN dag_engine.dag_versions dv ON dv.version_id = dr.version_id
WHERE dr.dag_name = 'daily_varejo_dw'
  AND dr.status   = 'SUCCESS'
ORDER BY dr.run_date DESC;

-- ==============================================================================
-- QUERIES DE VALIDAÇÃO FINAL (DATINT HEALTH BOOST)
-- ==============================================================================
-- 1. Confirma que BACKFILL não contaminou as tabelas novas
-- SELECT COUNT(*) AS deve_ser_zero FROM dag_medallion.task_health_cumulative WHERE run_type = 'BACKFILL';

-- 2. Bitmap visual por task no snapshot mais recente
-- SELECT task_name, data_snapshot, total_dias_saudavel, bitmap_visual, healthy_d0, healthy_d1
-- FROM dag_medallion.gold_health_streak
-- WHERE data_snapshot = (SELECT MAX(data_snapshot) FROM dag_medallion.task_health_cumulative);

-- 3. Array posicional do mês corrente
-- SELECT task_name, mes_referencia, scores_diarios, score_minimo_mes, score_medio_mes
-- FROM dag_medallion.task_health_array_mensal ORDER BY task_name;

-- 4. Streaks correntes (Gap-and-Island)
-- SELECT task_name, health_state, streak_days, streak_start, version_id
-- FROM dag_medallion.gold_degradation_streaks WHERE is_current = TRUE ORDER BY task_name;

-- ==============================================================================
-- CAMADA SEMÂNTICA — Resultados Finais
-- ==============================================================================

-- ► Oncall Handoff: status do último run por DAG (visão do plantonista)
DO $$ BEGIN RAISE NOTICE ''; END $$;
DO $$ BEGIN RAISE NOTICE '══════════════════════════════════════════════'; END $$;
DO $$ BEGIN RAISE NOTICE ' SEMANTIC LAYER — Resultados Finais'; END $$;
DO $$ BEGIN RAISE NOTICE '══════════════════════════════════════════════'; END $$;

\echo '► 1. rpt_oncall_handoff — Último run por DAG'
SELECT
    dag_name,
    run_date,
    run_type,
    status_icon,
    total_tasks,
    ok_tasks,
    failed_tasks,
    retried_tasks,
    sla_breach_count,
    wall_clock_ms,
    COALESCE(failure_detail, '—') AS failure_detail
FROM dag_semantic.rpt_oncall_handoff
ORDER BY dag_name;

\echo '► 2. rpt_weekly_reliability — Confiabilidade semanal (8 semanas)'
SELECT
    dag_name,
    week_start,
    total_runs,
    ok_runs,
    failed_runs,
    success_rate          AS "success%",
    prev_4w_avg_success_rate AS "prev_4w_avg%",
    success_rate_delta    AS "delta%",
    weekly_signal
FROM dag_semantic.rpt_weekly_reliability
ORDER BY dag_name, week_start DESC;

\echo '► 3. rpt_monthly_sla_delivery — Entrega mensal vs SLA'
SELECT
    dag_name,
    mes_referencia,
    total_runs,
    ok_runs,
    success_rate_mes      AS "success%",
    pct_tasks_dentro_sla  AS "tasks_ok_sla%",
    tasks_com_breach_sla,
    avg_health_score,
    min_health_score,
    commitment_status
FROM dag_semantic.rpt_monthly_sla_delivery
ORDER BY dag_name, mes_referencia DESC;

\echo '► 4. rpt_version_impact — Impacto por versão deployada'
SELECT
    task_name,
    parent_task_name,
    version_tag,
    prev_version_tag,
    total_execs,
    success_rate          AS "success%",
    success_rate_delta    AS "sr_delta%",
    avg_exec_ms,
    avg_exec_ms_delta,
    avg_queue_wait_ms,
    avg_queue_wait_delta,
    p95_ms,
    p95_ms_delta,
    impact_signal
FROM dag_semantic.rpt_version_impact
ORDER BY task_name, deployed_at DESC;

-- ==============================================================================
