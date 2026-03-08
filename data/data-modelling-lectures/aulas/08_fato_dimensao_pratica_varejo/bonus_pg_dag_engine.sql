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
    sla_ms_override BIGINT DEFAULT NULL       -- NOVO: Âncora externa de SLA (manual)
);

-- Migração: garante coluna dag_name em tabelas que possam ter sido criadas antes deste campo
ALTER TABLE dag_engine.tasks ADD COLUMN IF NOT EXISTS dag_name VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_tasks_dag_name ON dag_engine.tasks(dag_name);
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
DROP VIEW IF EXISTS dag_engine.vw_topological_sort CASCADE;
CREATE OR REPLACE VIEW dag_engine.vw_topological_sort AS
WITH RECURSIVE topo_sort AS (
    SELECT task_name, dag_name, procedure_call, dependencies, 0 AS execution_level
    FROM dag_engine.tasks
    WHERE array_length(dependencies, 1) IS NULL OR array_length(dependencies, 1) = 0
    UNION ALL
    SELECT t.task_name, t.dag_name, t.procedure_call, t.dependencies, ts.execution_level + 1
    FROM dag_engine.tasks t
    JOIN topo_sort ts ON ts.task_name = ANY(t.dependencies)
    WHERE ts.execution_level < 100 -- Safety Valve: quebra infinite loop em DAGs adulteradas manuais
)
SELECT task_name, dag_name, procedure_call, dependencies, MAX(execution_level) as topological_layer
FROM topo_sort
GROUP BY task_name, dag_name, procedure_call, dependencies
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
        
        -- BROADCAST DE EVENTOS REAL-TIME PARA MLOPS ASSÍNCRONO!
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
        SELECT ti.task_name, t.procedure_call
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
                -- Interpola p_data via Replace Dinâmico (Magia pura injetando o equivalente ao Jinja no SQL)
                v_sql := REPLACE(v_task.procedure_call, '$1', quote_literal(p_data));
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
-- Se o servidor cair por 3 dias, o Cron só chamaria uma vez. Isso preenche o Gap:
DROP PROCEDURE IF EXISTS dag_engine.proc_catchup(DATE, DATE);
CREATE OR REPLACE PROCEDURE dag_engine.proc_catchup(
    p_dag_name TEXT, 
    p_from DATE, 
    p_to DATE, 
    p_verbose BOOLEAN DEFAULT TRUE,
    p_version INT DEFAULT NULL
)
LANGUAGE plpgsql AS $$
DECLARE
    v_date DATE := p_from;
    v_status VARCHAR(20);
BEGIN
    WHILE v_date <= p_to LOOP
        v_status := NULL;
        SELECT status INTO v_status FROM dag_engine.dag_runs
        WHERE dag_name = p_dag_name AND run_date = v_date;

        IF v_status = 'SUCCESS' THEN
            IF p_verbose THEN RAISE NOTICE '⏭️ Pulando % — já processado com sucesso.', v_date; END IF;
        ELSIF v_status = 'RUNNING' THEN
            -- Run fantasma: banco reiniciou sem finalizar
            IF p_verbose THEN RAISE WARNING '⚠️ Run de % está como RUNNING (fantasma). Catchup interrompido — resolva manualmente antes de continuar.', v_date; END IF;
            EXIT; -- paralisa o loop de recovery até que o ghost RUNNING seja limpo
        ELSE
            -- NULL (nunca rodou), FAILED, DEADLOCK — tenta/retenta
            IF v_status IS NOT NULL THEN
                CALL dag_engine.proc_clear_run(p_dag_name, v_date, p_verbose);
            END IF;
            IF p_verbose THEN RAISE NOTICE '📅 Catch-up: rodando %', v_date; END IF;
            CALL dag_engine.proc_run_dag(p_dag_name, v_date, p_verbose);
        END IF;

        v_date := v_date + 1;
    END LOOP;
END;
$$;

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
-- 4.4 PROCEDURES DE MANUTENÇÃO (CLEAR RUN)
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
-- 4.5 PROCEDURES DE DEPLOYMENT (DAG SPEC AS JSON)
-- ==============================================================================
-- Permite carregar a topologia do DAG via um payload JSON (estilo dbt/Airflow configs).
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
-- GOLD 6: Step Duration Timelapse (Trend Analítico Preditivo)
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
-- O META-DAG PROCESSOR (Chamado Autônomo pelo Motor Principal da DAG)
-- ============================================================
CREATE OR REPLACE PROCEDURE dag_medallion.proc_run_medallion(p_run_id INT)
LANGUAGE plpgsql AS $$
BEGIN
    CALL dag_medallion.proc_ingest_bronze(p_run_id);
    CALL dag_medallion.proc_upsert_dim_task();
    CALL dag_medallion.proc_upsert_fato_task_exec(p_run_id);
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

-- 8.3 proc_deploy_dag: ponto de entrada versionado que substitui proc_load_dag_spec
-- Detecta redeploys silenciosos via hash e mantém a cadeia de versões pai → filho
-- p_spec agora aceita o manifest envelope completo:
--   { "name": "...", "description": "...", "schedule": "...", "tasks": [...] }
CREATE OR REPLACE PROCEDURE dag_engine.proc_deploy_dag(
    p_spec    JSONB,
    p_tag     VARCHAR(50),
    p_summary TEXT DEFAULT NULL
)
LANGUAGE plpgsql AS $$
DECLARE
    v_new_version_id  INT;
    v_current_hash    TEXT;
    v_new_hash        TEXT := md5(p_spec::text);
    v_parent_id       INT;
    v_dag_name        TEXT := COALESCE(p_spec->>'name', 'default');
    v_description     TEXT := p_spec->>'description';
    v_schedule        TEXT := p_spec->>'schedule';
BEGIN
    -- Valida que o manifest tem o campo "name" — obrigatório para suporte multi-DAG
    IF p_spec->>'name' IS NULL THEN
        RAISE EXCEPTION 'Manifest Error: campo "name" é obrigatório no manifest. Ex: {"name": "daily_varejo_dw", "tasks": [...]}';
    END IF;

    -- Verifica se já existe uma versão ativa para ESTE dag_name (não global)
    SELECT spec_hash, version_id
    INTO v_current_hash, v_parent_id
    FROM dag_engine.dag_versions
    WHERE dag_name = v_dag_name AND is_active = TRUE;

    IF v_current_hash = v_new_hash THEN
        RAISE WARNING '⚠️ Spec idêntico ao ativo para "%" (hash: %). Nenhuma versão criada.', v_dag_name, v_new_hash;
        RETURN;
    END IF;

    -- Desativa apenas a versão ativa deste DAG (não afeta outros DAGs)
    UPDATE dag_engine.dag_versions SET is_active = FALSE
    WHERE dag_name = v_dag_name AND is_active = TRUE;

    INSERT INTO dag_engine.dag_versions (
        dag_name, description, schedule,
        version_tag, spec, is_active, change_summary, parent_version
    )
    VALUES (
        v_dag_name, v_description, v_schedule,
        p_tag, p_spec, TRUE, p_summary, v_parent_id
    )
    RETURNING version_id INTO v_new_version_id;

    -- Carrega a topologia no motor de tarefas (proc_load_dag_spec já sabe lidar com manifest)
    CALL dag_engine.proc_load_dag_spec(p_spec);

    RAISE NOTICE '✅ DAG "%" deployada como "%" — Version ID: % (parent: %)', v_dag_name, p_tag, v_new_version_id, v_parent_id;
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
        SELECT ti.task_name, t.procedure_call INTO v_task
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
                v_sql := REPLACE(v_task.procedure_call, '$1', quote_literal(p_data));
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
--   Estratégia 1: fire-and-forget via dblink_send_query — o loop principal
--                 despacha N tasks em paralelo sem esperar cada uma terminar.
--   Estratégia 4: chunking temporal automático — tasks com chunk_config no spec
--                 são expandidas em sub-tasks paralelas por janela de tempo.
-- ==============================================================================

-- Habilita dblink (extensão built-in do PostgreSQL)
CREATE EXTENSION IF NOT EXISTS dblink;

-- 9.1 Colunas de rastreio assíncrono em task_instances
ALTER TABLE dag_engine.task_instances
    ADD COLUMN IF NOT EXISTS worker_conn TEXT,          -- nome da conexão dblink ativa
    ADD COLUMN IF NOT EXISTS is_chunk    BOOLEAN DEFAULT FALSE,  -- é sub-task de chunking?
    ADD COLUMN IF NOT EXISTS chunk_index INT,           -- índice do bucket (0-based)
    ADD COLUMN IF NOT EXISTS parent_task VARCHAR(100);  -- task original que gerou este chunk

-- 9.2 Hint de chunking no spec de tarefas
-- Formato: {"column": "data_venda", "buckets": 4} | NULL = sem chunking
ALTER TABLE dag_engine.tasks
    ADD COLUMN IF NOT EXISTS chunk_config JSONB DEFAULT NULL;

-- 9.3 Tabela de workers dblink ativos (debug e housekeeping de órfãos)
CREATE TABLE IF NOT EXISTS dag_engine.async_workers (
    conn_name   TEXT PRIMARY KEY,
    run_id      INT  REFERENCES dag_engine.dag_runs(run_id),
    task_name   VARCHAR(100),
    launched_at TIMESTAMP DEFAULT clock_timestamp()
);

-- 9.4 fn_exec_task: wrapper que executa SQL dinâmico e retorna NULL (sucesso) ou SQLERRM (falha)
-- Permite ao dblink_get_result inspecionar o resultado como valor, não como exceção
-- Remove overloads antigos (ex: versão com p_verbose BOOLEAN) que causam ambiguidade no dblink
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

-- 9.5 fn_build_chunk_ranges: divide um dia em N buckets de tempo uniformes
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
-- Convenção: procedure_call deve usar tokens $1 (data), $range_start, $range_end
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
    v_range   RECORD;
    v_proc    TEXT;
BEGIN
    IF v_column IS NULL THEN
        RAISE EXCEPTION 'chunk_config requer campo "column". Recebido: %', p_chunk_config;
    END IF;

    FOR v_range IN
        SELECT * FROM dag_engine.fn_build_chunk_ranges('public', p_task_name, v_column, p_date, v_buckets)
    LOOP
        task_name := p_task_name || '_chunk_' || v_range.chunk_index;
        v_proc    := REPLACE(p_procedure, '$1',           quote_literal(p_date));
        v_proc    := REPLACE(v_proc,      '$range_start', quote_literal(v_range.range_start));
        v_proc    := REPLACE(v_proc,      '$range_end',   quote_literal(v_range.range_end));
        procedure_call := v_proc;
        dependencies   := p_dependencies;
        RETURN NEXT;
    END LOOP;
END;
$$;

-- 9.7 fn_extract_tables_from_proc: extrai tabelas referenciadas por uma procedure
-- Usa regex sobre pg_get_functiondef — não requer pg_query (extensão opcional)
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

-- 9.8 proc_load_dag_spec: sobrescreve versão anterior adicionando Passo 3 — chunking automático
-- Passo 3 expande tasks com chunk_config em N sub-tasks paralelas antes de finalizar o load
CREATE OR REPLACE PROCEDURE dag_engine.proc_load_dag_spec(p_spec JSONB)
LANGUAGE plpgsql AS $$
DECLARE
    v_task        JSONB;
    v_deps        VARCHAR(100)[];
    v_missing_dep TEXT;
    v_expanded    RECORD;
    v_tasks       JSONB  := COALESCE(p_spec->'tasks', p_spec);  -- suporta manifest envelope E array legado
    v_dag_name    TEXT   := COALESCE(p_spec->>'name', 'default');
BEGIN
    IF jsonb_array_length(v_tasks) = 0 THEN
        RAISE EXCEPTION 'DAG Spec Error: spec vazio. Operação abortada.';
    END IF;

    DELETE FROM dag_engine.tasks
    WHERE dag_name = v_dag_name
      AND task_name NOT IN (
        SELECT t->>'task_name' FROM jsonb_array_elements(v_tasks) AS t
    );

    -- PASSO 1: Insere todas as tarefas sem dependências (Forward References safe)
    FOR v_task IN SELECT * FROM jsonb_array_elements(v_tasks)
    LOOP
        IF v_task->>'task_name' IS NULL OR v_task->>'procedure_call' IS NULL THEN
            RAISE EXCEPTION 'DAG Spec Error: "task_name" e "procedure_call" são obrigatórios. Payload: %', v_task;
        END IF;

        INSERT INTO dag_engine.tasks (
            task_name, dag_name, procedure_call, dependencies, max_retries, retry_delay_seconds, chunk_config,
            sla_ms_override
        ) VALUES (
            v_task->>'task_name',
            v_dag_name,
            v_task->>'procedure_call',
            '{}',
            COALESCE((v_task->>'max_retries')::INT, 0),
            COALESCE((v_task->>'retry_delay_seconds')::INT, 5),
            v_task->'chunk_config',
            (v_task->>'sla_ms_override')::BIGINT
        )
        ON CONFLICT (task_name) DO UPDATE SET
            dag_name            = EXCLUDED.dag_name,
            procedure_call      = EXCLUDED.procedure_call,
            dependencies        = '{}',
            max_retries         = EXCLUDED.max_retries,
            retry_delay_seconds = EXCLUDED.retry_delay_seconds,
            chunk_config        = EXCLUDED.chunk_config,
            sla_ms_override      = EXCLUDED.sla_ms_override;
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
    -- A task original é removida e substituída por N chunks que herdam suas dependências
    FOR v_task IN SELECT * FROM jsonb_array_elements(v_tasks) WHERE (value->'chunk_config') IS NOT NULL
    LOOP
        DELETE FROM dag_engine.tasks WHERE task_name = v_task->>'task_name';

        FOR v_expanded IN
            SELECT * FROM dag_engine.fn_expand_chunk_tasks(
                v_task->>'task_name',
                v_task->>'procedure_call',
                ARRAY(SELECT jsonb_array_elements_text(v_task->'dependencies')),
                v_task->'chunk_config',
                CURRENT_DATE  -- placeholder; tokens reais injetados em runtime pelo proc_run_dag
            )
        LOOP
            INSERT INTO dag_engine.tasks (
                task_name, dag_name, procedure_call, dependencies,
                max_retries, retry_delay_seconds, chunk_config,
                sla_ms_override
            ) VALUES (
                v_expanded.task_name,
                v_dag_name,
                v_expanded.procedure_call,
                v_expanded.dependencies,
                COALESCE((v_task->>'max_retries')::INT, 0),
                COALESCE((v_task->>'retry_delay_seconds')::INT, 5),
                v_task->'chunk_config',
                (v_task->>'sla_ms_override')::BIGINT
            )
            ON CONFLICT (task_name) DO UPDATE SET
                procedure_call      = EXCLUDED.procedure_call,
                dependencies        = EXCLUDED.dependencies,
                sla_ms_override      = EXCLUDED.sla_ms_override;
        END LOOP;
    END LOOP;

    RAISE NOTICE '✅ DAG Spec carregada! % tasks interpretadas.', jsonb_array_length(v_tasks);
END;
$$;

-- 9.9 proc_dispatch_task: abre conexão dblink dedicada e despacha tarefa fire-and-forget
-- Encapsula a procedure_call em fn_exec_task para retornar erro como TEXT (não exceção)
--
-- DSN configurável: defina dag_engine.worker_dsn em postgresql.conf ou via ALTER SYSTEM
-- para suportar SSL, pgBouncer, réplicas ou qualquer variação de topologia de rede:
--   ALTER SYSTEM SET "dag_engine.worker_dsn" = 'host=myhost port=5432 sslmode=require';
-- Se não configurado, usa a conexão local padrão como fallback.
-- Remove overloads antigos que causam ambiguidade no PostgreSQL (ex: task_name era TEXT antes de ser VARCHAR, ou havia parâmetro p_verbose extra)
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
    v_conn_name TEXT := 'dag_worker_' || p_run_id || '_' || replace(p_task_name, '.', '_');
    v_dsn       TEXT;
    v_query     TEXT;
BEGIN
    -- DSN: usa configuração customizada se disponível, senão fallback local
    v_dsn := current_setting('dag_engine.worker_dsn', true);  -- true = retorna NULL se não existe
    IF v_dsn IS NULL OR v_dsn = '' THEN
        v_dsn := format(
            'dbname=%s host=localhost port=%s user=%s',
            current_database(), current_setting('port'), current_user
        );
    END IF;

    -- Encapsula em fn_exec_task: retorna NULL=sucesso ou mensagem de erro
    -- Cast explícito para TEXT evita ambiguidade se houver overloads residuais no catalógo
    v_query := format('SELECT dag_engine.fn_exec_task(%s::TEXT)', quote_literal(p_sql));

    PERFORM dblink_connect(v_conn_name, v_dsn);
    PERFORM dblink_send_query(v_conn_name, v_query);  -- fire-and-forget: retorna imediatamente

    UPDATE dag_engine.task_instances
    SET status      = 'RUNNING',
        start_ts    = clock_timestamp(),
        worker_conn = v_conn_name
    WHERE run_id = p_run_id AND task_name = p_task_name;

    INSERT INTO dag_engine.async_workers (conn_name, run_id, task_name)
    VALUES (v_conn_name, p_run_id, p_task_name)
    ON CONFLICT (conn_name) DO NOTHING;
END;
$$;

-- 9.10 proc_collect_workers: polling não-bloqueante dos workers ativos
-- Para cada worker que terminou, coleta resultado e aplica retry ou cascade UPSTREAM_FAILED
-- ATENÇÃO: COMMIT deve ficar FORA de blocos BEGIN...EXCEPTION...END (que criam subtransações);
-- chamá-lo dentro resultaria em "cannot commit while a subtransaction is active".
CREATE OR REPLACE PROCEDURE dag_engine.proc_collect_workers(
    p_run_id  INT,
    p_verbose BOOLEAN DEFAULT FALSE
)
LANGUAGE plpgsql AS $$
DECLARE
    v_worker        RECORD;
    v_err           TEXT;
    v_cur_att       INT;
    v_retry_delay   INT;
    v_max_retries   INT;
    v_needs_commit  BOOLEAN;
BEGIN
    FOR v_worker IN
        SELECT ti.task_name, ti.worker_conn, ti.start_ts
        FROM dag_engine.task_instances ti
        WHERE ti.run_id = p_run_id AND ti.status = 'RUNNING' AND ti.worker_conn IS NOT NULL
    LOOP
        v_needs_commit := FALSE;

        BEGIN
            -- Não-bloqueante: se worker ainda processa, skip sem esperar
            IF dblink_is_busy(v_worker.worker_conn) THEN
                -- NOVO: Proteção contra Zombie Workers (congelamentos infinitos)
                IF (clock_timestamp() - v_worker.start_ts) > INTERVAL '30 minutes' THEN
                    v_err := 'Zombie Worker Timeout (> 30 min sem resposta)';
                    IF p_verbose THEN RAISE WARNING '⚠️ Worker [%] zumbi detectado. Forçando falha.', v_worker.task_name; END IF;
                    BEGIN PERFORM dblink_cancel_query(v_worker.worker_conn); EXCEPTION WHEN OTHERS THEN NULL; END;
                ELSE
                    CONTINUE;
                END IF;
            END IF;

            -- Drena completamente o result set de fn_exec_task: NULL=sucesso, texto=erro
            -- CRÍTICO: dblink_get_result deve ser chamado até não retornar mais linhas;
            -- parar no LIMIT 1 deixa a conexão em estado "busy" e dblink_is_busy
            -- retornará TRUE para sempre, causando Deadlock Topológico.
            -- Drena completamente o result set de fn_exec_task: NULL=sucesso, texto=erro
            -- Se v_err já foi preenchido pelo Timeout, pulamos o dblink_get_result para evitar hang
            IF v_err IS NULL THEN
                LOOP
                    DECLARE v_row TEXT;
                    BEGIN
                        SELECT r.result INTO v_row
                        FROM dblink_get_result(v_worker.worker_conn) AS r(result TEXT)
                        LIMIT 1;
                        EXIT WHEN NOT FOUND;
                        IF v_row IS NOT NULL THEN v_err := v_row; END IF;
                    END;
                END LOOP;
            END IF;

            -- Fecha conexão independente do resultado
            -- NOTA: dblink_cancel_query sinaliza o cancelamento mas não drena o result set.
            -- dblink_disconnect falhará silenciosamente (conexão ainda "busy").
            -- O handle vaza até o fim da sessão. Cleanup via proc_cleanup_orphan_workers
            -- resolve na próxima execução, pois o processo backend já terá encerrado.
            BEGIN
                PERFORM dblink_disconnect(v_worker.worker_conn);
                DELETE FROM dag_engine.async_workers WHERE conn_name = v_worker.worker_conn;
            EXCEPTION WHEN OTHERS THEN NULL;
            END;

            IF v_err IS NULL THEN
                -- SUCESSO
                UPDATE dag_engine.task_instances
                SET status      = 'SUCCESS',
                    end_ts      = clock_timestamp(),
                    worker_conn = NULL,
                    duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - v_worker.start_ts)) * 1000
                WHERE run_id = p_run_id AND task_name = v_worker.task_name;
                IF p_verbose THEN RAISE NOTICE '  ✅ [%] concluído (async).', v_worker.task_name; END IF;
            ELSE
                -- FALHA: retry ou cascade (espelho da lógica do motor síncrono)
                SELECT ti.attempt, t.retry_delay_seconds, t.max_retries
                INTO v_cur_att, v_retry_delay, v_max_retries
                FROM dag_engine.task_instances ti
                JOIN dag_engine.tasks t ON t.task_name = ti.task_name
                WHERE ti.run_id = p_run_id AND ti.task_name = v_worker.task_name;

                IF v_cur_att < v_max_retries + 1 THEN
                    UPDATE dag_engine.task_instances
                    SET status         = 'PENDING',
                        attempt        = attempt + 1,
                        worker_conn    = NULL,
                        retry_after_ts = clock_timestamp() + (v_retry_delay * (v_cur_att + 1)) * INTERVAL '1 second',
                        error_text     = 'Retry | Último erro: ' || v_err
                    WHERE run_id = p_run_id AND task_name = v_worker.task_name;
                    IF p_verbose THEN RAISE WARNING '🔄 [%] Retry agendado.', v_worker.task_name; END IF;
                ELSE
                    UPDATE dag_engine.task_instances
                    SET status      = 'FAILED',
                        end_ts      = clock_timestamp(),
                        worker_conn = NULL,
                        duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - v_worker.start_ts)) * 1000,
                        error_text  = v_err
                    WHERE run_id = p_run_id AND task_name = v_worker.task_name;

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
                    IF p_verbose THEN RAISE WARNING '❌ [%] Falha definitiva: %', v_worker.task_name, v_err; END IF;
                END IF;
            END IF;
            v_needs_commit := TRUE;

        EXCEPTION WHEN OTHERS THEN
            -- Erro de infraestrutura dblink (conexão morreu, timeout, etc.)
            BEGIN
                PERFORM dblink_disconnect(v_worker.worker_conn);
                DELETE FROM dag_engine.async_workers WHERE conn_name = v_worker.worker_conn;
            EXCEPTION WHEN OTHERS THEN NULL;
            END;
            UPDATE dag_engine.task_instances
            SET status = 'FAILED', worker_conn = NULL,
                error_text = 'dblink infra error: ' || SQLERRM, end_ts = clock_timestamp()
            WHERE run_id = p_run_id AND task_name = v_worker.task_name;
            -- Propaga UPSTREAM_FAILED para dependentes, evitando Deadlock Topológico
            WITH RECURSIVE fail_cascade AS (
                SELECT t.task_name FROM dag_engine.tasks t
                WHERE v_worker.task_name = ANY(t.dependencies)
                UNION ALL
                SELECT t.task_name FROM dag_engine.tasks t
                JOIN fail_cascade fc ON fc.task_name = ANY(t.dependencies)
            )
            UPDATE dag_engine.task_instances
            SET status     = 'UPSTREAM_FAILED',
                end_ts     = clock_timestamp(),
                error_text = 'Propagado de: ' || v_worker.task_name
            WHERE run_id = p_run_id
              AND task_name IN (SELECT task_name FROM fail_cascade)
              AND status = 'PENDING';
            v_needs_commit := TRUE;
        END;

        -- COMMIT fora do bloco BEGIN...EXCEPTION...END: PL/pgSQL cria savepoint (subtransação)
        -- para blocos com EXCEPTION, e COMMIT dentro de subtransações levanta exceção.
        IF v_needs_commit THEN
            COMMIT;
        END IF;
    END LOOP;
END;
$$;

-- 9.11 proc_run_dag: substituição final — async dispatch + versionamento combinados
-- Integra os dois backlogs: version_id no dag_run + 3 branches não-bloqueantes no loop
DROP PROCEDURE IF EXISTS dag_engine.proc_run_dag(DATE, BOOLEAN);
DROP PROCEDURE IF EXISTS dag_engine.proc_run_dag(TEXT, DATE, BOOLEAN);
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
    v_sql           TEXT;
    v_pending_count INT;
    v_running_count INT;
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
        RAISE WARNING 'Run já existe para "%" em %. Use proc_clear_run para re-executar.', p_dag_name, p_data;
        RETURN;
    END;

    INSERT INTO dag_engine.task_instances (run_id, task_name)
    SELECT v_run_id, task_name FROM dag_engine.tasks WHERE dag_name = p_dag_name;
    COMMIT;

    -- ================================================================
    -- LOOP PRINCIPAL: três branches não-bloqueantes
    --   Branch B: coleta workers que terminaram (sempre primeiro, não-bloqueante)
    --   Branch A: despacha próxima task elegível (fire-and-forget via dblink)
    --   Branch C: avalia término ou aguarda backoff / polling
    -- ================================================================
    LOOP
        -- Branch B: polling de coleta (não-bloqueante — skipa workers ainda rodando)
        CALL dag_engine.proc_collect_workers(v_run_id, p_verbose);
        COMMIT;

        -- Branch A: despacha próxima task PENDING com todas as deps resolvidas
        SELECT ti.task_name, t.procedure_call
        INTO v_task
        FROM dag_engine.task_instances ti
        JOIN dag_engine.tasks t ON ti.task_name = t.task_name
        WHERE ti.run_id = v_run_id
          AND ti.status = 'PENDING'
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
            v_sql := REPLACE(v_task.procedure_call, '$1', quote_literal(p_data));
            IF p_verbose THEN RAISE NOTICE '  --> 📤 Despachando: [%]', v_task.task_name; END IF;
            CALL dag_engine.proc_dispatch_task(v_run_id, v_task.task_name, v_sql);
            COMMIT;
            CONTINUE;  -- volta ao topo imediatamente para despachar mais tasks em paralelo
        END IF;

        -- Branch C: nenhuma task elegível — avalia estado geral
        SELECT COUNT(*) INTO v_pending_count FROM dag_engine.task_instances
        WHERE run_id = v_run_id AND status = 'PENDING';

        SELECT COUNT(*) INTO v_running_count FROM dag_engine.task_instances
        WHERE run_id = v_run_id AND status = 'RUNNING';

        IF v_running_count > 0 THEN
            PERFORM pg_sleep(0.5);  -- polling interval — workers ainda ativos

        ELSIF v_pending_count > 0 THEN
            -- Só existem tasks em backoff de retry
            IF EXISTS (
                SELECT 1 FROM dag_engine.task_instances
                WHERE run_id = v_run_id AND status = 'PENDING' AND retry_after_ts > clock_timestamp()
            ) THEN
                PERFORM pg_sleep(1);
            ELSE
                -- Deadlock topológico (tasks pendentes mas nenhuma pode rodar)
                UPDATE dag_engine.dag_runs SET status = 'DEADLOCK', end_ts = clock_timestamp()
                WHERE run_id = v_run_id;
                COMMIT;
                RAISE WARNING '💀 Deadlock Topológico: tasks pendentes irresolvíveis.';
                EXIT;
            END IF;

        ELSE
            -- Pipeline completo: avalia resultado final
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

-- 9.12 proc_cleanup_orphan_workers: housekeeping de conexões dblink perdidas em crashes
-- Fecha conexões abertas, remove da tabela de workers e marca tasks RUNNING como FAILED
CREATE OR REPLACE PROCEDURE dag_engine.proc_cleanup_orphan_workers(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_conn RECORD;
BEGIN
    FOR v_conn IN SELECT conn_name FROM dag_engine.async_workers WHERE run_id = p_run_id
    LOOP
        BEGIN
            PERFORM dblink_disconnect(v_conn.conn_name);
        EXCEPTION WHEN OTHERS THEN NULL;
        END;
    END LOOP;

    DELETE FROM dag_engine.async_workers WHERE run_id = p_run_id;

    UPDATE dag_engine.task_instances
    SET status      = 'FAILED',
        worker_conn = NULL,
        error_text  = 'Worker órfão: conexão dblink perdida'
    WHERE run_id = p_run_id AND status = 'RUNNING' AND worker_conn IS NOT NULL;

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
CREATE OR REPLACE VIEW dag_engine.v_queue_depth_timeline AS
WITH events AS (
    SELECT
        st.run_id,
        dr.run_date,
        st.task_name,
        st.transition_ts,
        -- Cada transição contribui com deltas: +1 ao entrar no estado, -1 ao sair
        CASE WHEN st.new_state = 'RUNNING'                               THEN  1
             WHEN st.old_state = 'RUNNING' AND st.new_state != 'RUNNING' THEN -1
             ELSE 0 END AS delta_running,
        CASE WHEN st.new_state = 'PENDING'                               THEN  1
             WHEN st.old_state = 'PENDING' AND st.new_state != 'PENDING' THEN -1
             ELSE 0 END AS delta_pending
    FROM dag_engine.state_transitions st
    JOIN dag_engine.dag_runs dr ON dr.run_id = st.run_id
    WHERE st.task_name IS NOT NULL  -- exclui eventos dag-level (task_name NULL)
)
SELECT
    run_id,
    run_date,
    task_name          AS trigger_task,   -- task cuja transição causou esta leitura
    transition_ts,
    -- Acumulado até este instante via window sem frame limite superior
    SUM(delta_running) OVER w AS concurrent_running,
    SUM(delta_pending) OVER w AS queued_pending,
    SUM(delta_running + delta_pending) OVER w AS total_active,
    -- Saturação de paralelismo: quanto da fila estava bloqueada esperando workers
    ROUND(
        100.0 * SUM(delta_pending) OVER w
        / NULLIF(SUM(delta_running + delta_pending) OVER w, 0)
    , 2) AS pct_queued
FROM events
WINDOW w AS (PARTITION BY run_id ORDER BY transition_ts ROWS UNBOUNDED PRECEDING)
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
        "procedure_call": "CALL varejo.proc_snapshot_clientes($1)",
        "dependencies": [],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "2_upsert_clientes_scd1",
        "procedure_call": "CALL varejo.proc_upsert_clientes_scd1()",
        "dependencies": [],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "3_upsert_clientes_scd2",
        "procedure_call": "CALL varejo.proc_upsert_clientes_scd2($1)",
        "dependencies": ["1_snapshot_clientes", "2_upsert_clientes_scd1"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "4_upsert_produtos_scd3",
        "procedure_call": "CALL varejo.proc_upsert_produtos_scd3($1)",
        "dependencies": [],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "5_ingestao_fato_vendas",
        "procedure_call": "CALL varejo.proc_ingestao_fato_vendas($1)",
        "dependencies": ["3_upsert_clientes_scd2", "4_upsert_produtos_scd3"],
        "max_retries": 1,
        "retry_delay_seconds": 10
    },
    {
        "task_name": "6_acumular_atividade",
        "procedure_call": "CALL varejo.proc_acumular_atividade(($1::DATE - 1), $1)",
        "dependencies": ["5_ingestao_fato_vendas"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "7_acumular_vendas_mes",
        "procedure_call": "CALL varejo.proc_acumular_vendas_mensal($1)",
        "dependencies": ["6_acumular_atividade"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    },
    {
        "task_name": "8_ingestao_gold_diaria",
        "procedure_call": "CALL varejo.proc_ingestao_gold_diaria($1)",
        "dependencies": ["7_acumular_vendas_mes"],
        "max_retries": 0,
        "retry_delay_seconds": 5
    }
    ]
}'::JSONB,
'v1.0',
'Spec inicial — 8 tasks do pipeline varejo');

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
CREATE INDEX IF NOT EXISTS idx_ativ_acum_data ON varejo.cliente_atividade_acumulada(data_snapshot);
CREATE INDEX IF NOT EXISTS idx_vendas_arr_mes ON varejo.cliente_vendas_array_mensal(mes_referencia);
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
TRUNCATE varejo.cliente_atividade_acumulada CASCADE;
TRUNCATE varejo.cliente_vendas_array_mensal CASCADE;
TRUNCATE varejo.cliente_snapshot_diario CASCADE;
TRUNCATE varejo.gold_metricas_diarias CASCADE;

-- 7.1. Carga ETL Inicial (Primeiro dia da camada crua para ODS/SCDs)
DO $$ BEGIN RAISE NOTICE '📥 1. Executando Carga Inicial do DW via DAG Engine (D-0, 2024-05-04)...'; END $$;
CALL dag_engine.proc_run_dag('daily_varejo_dw', '2024-05-04');

-- 7.2. Mudança OLTP Simples (Para observar as chaves substitutas em ação amanhã)
UPDATE varejo.origem_cliente SET estado = 'PR' WHERE cliente_id = 101;
UPDATE varejo.origem_produto SET categoria = 'Gamer' WHERE produto_id = 'PROD001';

-- 7.3. Carga Incremental Seguindo Alterações Temporais
DO $$ BEGIN RAISE NOTICE '📥 2. Executando Carga Incremental via DAG (D-1, 2024-05-05)...'; END $$;
CALL dag_engine.proc_run_dag('daily_varejo_dw', '2024-05-05');

-- 7.4. Fast-Forward Temporário Robusto (Cat-Chup MLOps nativo!)
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
    {"task_name": "1_snapshot_clientes", "procedure_call": "CALL varejo.proc_snapshot_clientes($1)", "dependencies": []},
    {"task_name": "2_upsert_clientes_scd1", "procedure_call": "CALL varejo.proc_upsert_clientes_scd1()", "dependencies": []},
    {"task_name": "3_upsert_clientes_scd2", "procedure_call": "CALL varejo.proc_upsert_clientes_scd2($1)", "dependencies": ["1_snapshot_clientes", "2_upsert_clientes_scd1"]},
    {"task_name": "4_upsert_produtos_scd3", "procedure_call": "CALL varejo.proc_upsert_produtos_scd3($1)", "dependencies": []},
    {"task_name": "5_ingestao_fato_vendas", "procedure_call": "CALL varejo.proc_ingestao_fato_vendas($1)", "dependencies": ["3_upsert_clientes_scd2", "4_upsert_produtos_scd3"], "max_retries": 1, "retry_delay_seconds": 10},
    {"task_name": "6_acumular_atividade", "procedure_call": "CALL varejo.proc_acumular_atividade(($1::DATE - 1), $1)", "dependencies": ["5_ingestao_fato_vendas"]},
    {"task_name": "7_acumular_vendas_mes", "procedure_call": "CALL varejo.proc_acumular_vendas_mensal($1)", "dependencies": ["6_acumular_atividade"]},
    {"task_name": "8_ingestao_gold_diaria", "procedure_call": "CALL varejo.proc_ingestao_gold_diaria($1)", "dependencies": ["7_acumular_vendas_mes"]},
    {"task_name": "9_envio_relatorio", "procedure_call": "SELECT 1", "dependencies": ["8_ingestao_gold_diaria"]}
    ]
}'::JSONB,
'v1.1',
'Adicionada task 9_envio_relatorio no downstream do gold'
);

-- Inspeciona o diff entre v1.0 (version_id=1) e v1.1 (version_id=2)
SELECT * FROM dag_engine.fn_diff_versions(1, 2);

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

-- ==============================================================================
-- 7.7. BACKLOG DEMO: EXECUÇÃO ASSÍNCRONA
-- ==============================================================================

-- Verifica que extensão dblink está disponível
SELECT extname, extversion FROM pg_extension WHERE extname = 'dblink';

-- Confirma colunas async nas tabelas
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'dag_engine'
  AND table_name   = 'task_instances'
  AND column_name  IN ('worker_conn', 'is_chunk', 'chunk_index', 'parent_task')
ORDER BY column_name;

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
SELECT task_name, dag_name, topological_layer, dependencies
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
-- LISTEN dag_alerts;
-- (Execute dag_engine.fn_check_alerts() numa sessão e o LISTEN recebe o JSON)
-- ==============================================================================
