-- ==============================================================================
-- BÔNUS: THE POSTGRES DAG ENGINE (Airflow + dbt inside Postgres)
-- ==============================================================================
-- Cansado de pagar caro no Airflow ou de manter infraestrutura Python injetando queries?
-- Vamos construir uma topologia de DAG completa com dependências, catch-up de erros
-- e telemetria nativa puramente em PostgreSQL, usando pg_cron como nosso Scheduler!
-- ==============================================================================

-- 1. Habilitamos o pgcron (Requer que a extensão esteja instalada/habilitada no postgres.conf shared_preload_libraries='pg_cron')
CREATE EXTENSION IF NOT EXISTS pg_cron;

CREATE SCHEMA IF NOT EXISTS dag_engine;

-- 2. TABELA DE TAREFAS (A definição do nosso DAG Topológico)
CREATE TABLE IF NOT EXISTS dag_engine.tasks (
    task_id SERIAL PRIMARY KEY,
    task_name VARCHAR(100) UNIQUE NOT NULL,
    procedure_call TEXT NOT NULL, 
    dependencies VARCHAR(100)[] DEFAULT '{}', -- Array de dependências topológicas (Quais tarefas precisam rodar antes?)
    max_retries INT DEFAULT 0,
    retry_delay_seconds INT DEFAULT 5
);

-- 2.1 VALIDAÇÃO DE DEPENDÊNCIAS (Cycle Detection)
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
CREATE OR REPLACE VIEW dag_engine.vw_topological_sort AS
WITH RECURSIVE topo_sort AS (
    SELECT task_name, procedure_call, dependencies, 0 AS execution_level
    FROM dag_engine.tasks
    WHERE array_length(dependencies, 1) IS NULL OR array_length(dependencies, 1) = 0
    UNION ALL
    SELECT t.task_name, t.procedure_call, t.dependencies, ts.execution_level + 1
    FROM dag_engine.tasks t
    JOIN topo_sort ts ON ts.task_name = ANY(t.dependencies)
    WHERE ts.execution_level < 100 -- Safety Valve: quebra infinite loop em DAGs adulteradas manuais
)
SELECT task_name, procedure_call, dependencies, MAX(execution_level) as topological_layer
FROM topo_sort
GROUP BY task_name, procedure_call, dependencies
ORDER BY topological_layer, task_name;

-- 3. TABELAS DE METADADOS E TELEMETRIA DE EXECUÇÃO
CREATE TABLE IF NOT EXISTS dag_engine.dag_runs (
    run_id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'RUNNING',
    start_ts TIMESTAMP DEFAULT clock_timestamp(),
    end_ts TIMESTAMP,
    UNIQUE(run_date)
);

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
CREATE OR REPLACE PROCEDURE dag_engine.proc_run_dag(p_data DATE, p_verbose BOOLEAN DEFAULT TRUE)
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
        RAISE NOTICE '🚀 Iniciando DAG Topológica para a data: %', p_data;
    END IF;
    
    -- Cria nova execução (ou avisa se já existe pra data, necessitando intervenção de re-run)
    BEGIN
        INSERT INTO dag_engine.dag_runs (run_date) VALUES (p_data) RETURNING run_id INTO v_run_id;
    EXCEPTION WHEN unique_violation THEN
        IF p_verbose THEN RAISE WARNING 'Já existe execução do DAG para %! Faça clear manual se quiser rodar de novo.', p_data; END IF;
        RETURN;
    END;
    
    -- Instancia todas as tarefas do DAG base como PENDING
    INSERT INTO dag_engine.task_instances (run_id, task_name)
    SELECT v_run_id, task_name FROM dag_engine.tasks;

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
CREATE OR REPLACE PROCEDURE dag_engine.proc_catchup(p_from DATE, p_to DATE, p_verbose BOOLEAN DEFAULT TRUE)
LANGUAGE plpgsql AS $$
DECLARE
    v_date DATE := p_from;
    v_status VARCHAR(20);
BEGIN
    WHILE v_date <= p_to LOOP
        v_status := NULL;
        SELECT status INTO v_status FROM dag_engine.dag_runs WHERE run_date = v_date;

        IF v_status = 'SUCCESS' THEN
            IF p_verbose THEN RAISE NOTICE '⏭️ Pulando % — já processado com sucesso.', v_date; END IF;
        ELSIF v_status = 'RUNNING' THEN
            -- Run fantasma: banco reiniciou sem finalizar
            IF p_verbose THEN RAISE WARNING '⚠️ Run de % está como RUNNING (fantasma). Catchup interrompido — resolva manualmente antes de continuar.', v_date; END IF;
            EXIT; -- paralisa o loop de recovery até que o ghost RUNNING seja limpo
        ELSE
            -- NULL (nunca rodou), FAILED, DEADLOCK — tenta/retenta
            IF v_status IS NOT NULL THEN
                CALL dag_engine.proc_clear_run(v_date, p_verbose);
            END IF;
            IF p_verbose THEN RAISE NOTICE '📅 Catch-up: rodando %', v_date; END IF;
            CALL dag_engine.proc_run_dag(v_date, p_verbose);
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
CREATE OR REPLACE VIEW dag_engine.v_task_percentiles AS
SELECT 
    'daily_varejo_dw' AS pipeline_name,
    COALESCE(task_name, '--- TOTAL DAG (Soma) ---') AS step_name,
    COUNT(*) as num_execucoes,
    ROUND(SUM(duration_ms), 2) as sum_ms,
    ROUND((SUM(duration_ms) / (NULLIF(SUM(SUM(duration_ms)) OVER(), 0) / 2)) * 100, 2) as pct_total,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS p25_ms,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS p50_mediana_ms,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS p75_ms,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS p90_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS p95_ms,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS p99_ms
FROM dag_engine.task_instances
WHERE status = 'SUCCESS'
GROUP BY ROLLUP(task_name)
ORDER BY step_name;

-- ==============================================================================
-- 4.4 PROCEDURES DE MANUTENÇÃO (CLEAR RUN)
-- ==============================================================================
-- Limpa completamente o rastro de uma RUN passada, permitindo reexecução do marco zero.
DROP PROCEDURE IF EXISTS dag_engine.proc_clear_run(DATE);
CREATE OR REPLACE PROCEDURE dag_engine.proc_clear_run(p_date DATE, p_verbose BOOLEAN DEFAULT TRUE)
LANGUAGE plpgsql AS $$
DECLARE v_run_id INT;
BEGIN
    SELECT run_id INTO v_run_id FROM dag_engine.dag_runs WHERE run_date = p_date;
    IF NOT FOUND THEN 
        RAISE EXCEPTION 'Nenhuma execução encontrada para a data %', p_date; 
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
BEGIN
    -- PASSO 0: Remove tarefas que não estão mais no spec (deprecação limpa)
    IF jsonb_array_length(p_spec) = 0 THEN
        RAISE EXCEPTION 'DAG Spec Error: spec vazio recebido. Operação abortada para proteger o engine.';
    END IF;

    -- Tasks que tem histórico acoplado falharão com Foreign Key Violation, exigindo a limpeza consciente do DBA
    DELETE FROM dag_engine.tasks
    WHERE task_name NOT IN (
        SELECT t->>'task_name' FROM jsonb_array_elements(p_spec) AS t
    );

    -- PASSO 1: Insere todas as tarefas sem validar deps (usando array vazio temporário para permitir Forward References sem quebras JSON de ordem)
    FOR v_task IN SELECT * FROM jsonb_array_elements(p_spec)
    LOOP
        IF v_task->>'task_name' IS NULL OR v_task->>'procedure_call' IS NULL THEN
            RAISE EXCEPTION 'DAG Spec Error: campos "task_name" e "procedure_call" são obrigatórios. Payload recebido: %', v_task;
        END IF;

        INSERT INTO dag_engine.tasks (
            task_name, 
            procedure_call, 
            dependencies, 
            max_retries, 
            retry_delay_seconds
        ) VALUES (
            v_task->>'task_name',
            v_task->>'procedure_call',
            '{}',
            COALESCE((v_task->>'max_retries')::INT, 0),
            COALESCE((v_task->>'retry_delay_seconds')::INT, 5)
        )
        ON CONFLICT (task_name) DO UPDATE SET 
            procedure_call = EXCLUDED.procedure_call,
            dependencies = '{}',   -- garante que o passo 2 parte do zero em caso de re-deploy
            max_retries = EXCLUDED.max_retries,
            retry_delay_seconds = EXCLUDED.retry_delay_seconds;
    END LOOP;

    -- PASSO 2: Agora que todos existem no motor, aplica as dependências (trigger de ciclo e validação protegem integridade da malha)
    FOR v_task IN SELECT * FROM jsonb_array_elements(p_spec)
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
    
    RAISE NOTICE '✅ DAG Specification declarativa carregada em Engine! % tarefas interpretadas.', jsonb_array_length(p_spec);
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
        had_retry, is_upstream_victim, error_text, start_ts, end_ts
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
        end_ts             = EXCLUDED.end_ts;
END;
$$;

-- ============================================================
-- GOLD 1: Pipeline Health Score Z-Score Composto
-- ============================================================
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
            (success_rate * 0.60)
            + ((100 - retry_rate) * 0.20)
            + (100 * 0.20 / (1 + COALESCE(stddev_ms, 0) / NULLIF(avg_ms, 0)))
        , 2) AS health_score
    FROM base
)
SELECT *,
    CASE
        -- Baixamos para 85 porque uma variância alta de runtime devido à carga inicial (D0) vs incremental (D1+) é um comportamento perfeitamente normal da esteira Medallion.
        WHEN health_score >= 85 THEN '🟢 SAUDÁVEL'
        WHEN health_score >= 70 THEN '🟡 ATENÇÃO'
        ELSE                        '🔴 CRÍTICO'
    END AS health_label
FROM scored
ORDER BY topological_layer ASC;

-- ============================================================
-- GOLD 2: SLA Breach Detection (Contrato de P95)
-- ============================================================
CREATE OR REPLACE VIEW dag_medallion.gold_sla_breach AS
WITH sla AS (
    SELECT
        task_name,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC, 2) AS sla_ms
    FROM dag_medallion.fato_task_exec
    WHERE final_status = 'SUCCESS'
    GROUP BY task_name
)
SELECT
    f.run_date,
    f.task_name,
    f.duration_ms           AS actual_ms,
    s.sla_ms                AS sla_p95_ms,
    ROUND(f.duration_ms - s.sla_ms, 2)              AS breach_ms,
    ROUND((f.duration_ms / NULLIF(s.sla_ms, 0) - 1) * 100, 2) AS breach_pct,
    CASE
        WHEN f.duration_ms > s.sla_ms * 2.0 THEN '🔴 SLA CRÍTICO (>2x)'
        WHEN f.duration_ms > s.sla_ms * 1.5 THEN '🟠 SLA SEVERO (>1.5x)'
        WHEN f.duration_ms > s.sla_ms       THEN '🟡 SLA BREACH (>P95)'
        ELSE                                     '🟢 DENTRO DO SLA'
    END AS sla_status
FROM dag_medallion.fato_task_exec f
JOIN sla s ON s.task_name = f.task_name
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
ORDER BY topological_layer ASC, avg_duration_ms DESC;

-- ============================================================
-- GOLD 5: Blast Radius cascades down
-- ============================================================
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
        ORDER BY f.run_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
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
-- 6. DAG INITIALIZATION (Definição do Pipeline no Motor via JSON)
-- ==============================================================================
-- Transformamos a inserção crua num formato declarativo JSONB. 
-- Imagine ler isso tudo diretamente de um arquivo yaml ou json externo no backend!
CALL dag_engine.proc_load_dag_spec('[
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
]'::JSONB);

-- ==============================================================================
-- 7. ÁREA DE TESTES E INTERAÇÃO (Hands-on Standalone Demonstração)
-- ==============================================================================
-- Aqui nós fundimos toda a lógica de negócio do "apresentacao.sql" sendo orquestrada
-- de verdade e nativamente apenas por este motor, sem precisar de for loops improvisados!

-- 7.0 Tuning da Sessão Local e Reset Completo do DW
SET synchronous_commit = off;       
SET work_mem = '256MB';            
SET maintenance_work_mem = '256MB'; 

DO $$ BEGIN RAISE NOTICE '🔄 Resetando o estado do OLTP e limpando o DW e DAG Engine...'; END $$;

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

TRUNCATE varejo.dim_cliente_type1 CASCADE;
TRUNCATE varejo.dim_cliente_type2 RESTART IDENTITY CASCADE;
TRUNCATE varejo.dim_produto_type3 CASCADE;
TRUNCATE varejo.fato_vendas RESTART IDENTITY CASCADE;
TRUNCATE varejo.cliente_atividade_acumulada CASCADE;
TRUNCATE varejo.cliente_vendas_array_mensal CASCADE;
TRUNCATE varejo.cliente_snapshot_diario CASCADE;
TRUNCATE varejo.gold_metricas_diarias CASCADE;
TRUNCATE dag_engine.state_transitions CASCADE;
TRUNCATE dag_engine.task_instances CASCADE;
TRUNCATE dag_engine.dag_runs CASCADE;

-- 7.1. Carga ETL Inicial (Primeiro dia da camada crua para ODS/SCDs)
DO $$ BEGIN RAISE NOTICE '📥 1. Executando Carga Inicial do DW via DAG Engine (D-0, 2024-05-04)...'; END $$;
CALL dag_engine.proc_run_dag('2024-05-04');

-- 7.2. Mudança OLTP Simples (Para observar as chaves substitutas em ação amanhã)
UPDATE varejo.origem_cliente SET estado = 'PR' WHERE cliente_id = 101;
UPDATE varejo.origem_produto SET categoria = 'Gamer' WHERE produto_id = 'PROD001';

-- 7.3. Carga Incremental Seguindo Alterações Temporais
DO $$ BEGIN RAISE NOTICE '📥 2. Executando Carga Incremental via DAG (D-1, 2024-05-05)...'; END $$;
CALL dag_engine.proc_run_dag('2024-05-05');

-- 7.4. Fast-Forward Temporário Robusto (Cat-Chup MLOps nativo!)
-- Com 2 meses de backfill rodando massivamente com tolerância a deadlock na arquitetura
DO $$ 
DECLARE
    ts_start TIMESTAMP := clock_timestamp();
    dur_backfill INTERVAL;
BEGIN 
    RAISE NOTICE '🚀 3. Iniciando MLOps Fast-Forward de 2 meses contínuos...'; 
    CALL dag_engine.proc_catchup('2024-05-06'::DATE, '2024-07-04'::DATE);
    
    dur_backfill := clock_timestamp() - ts_start;
    RAISE NOTICE '✅ Fast-Forward concluído com sucesso via Motor Nativo em %!', dur_backfill;
END $$;

-- 7.5. Observabilidade - Métrica Estatística de Saúde e Meta-Medallion no PostgreSQL
DO $$ BEGIN RAISE NOTICE '========================================================='; END $$;
DO $$ BEGIN RAISE NOTICE '📊 RESULTADO FINAL DA ENGINE DE AUTOMAÇÃO DE DAG NO BANCO'; END $$;
DO $$ BEGIN RAISE NOTICE '========================================================='; END $$;

SELECT * FROM dag_engine.v_task_percentiles ORDER BY p99_ms DESC;

-- Use the Medallion Output Analytics Dashboards here:
SELECT * FROM dag_medallion.gold_pipeline_health;
SELECT * FROM dag_medallion.gold_critical_path ORDER BY topological_layer ASC, pct_pipeline_time DESC;
SELECT * FROM dag_medallion.gold_performance_timelapse WHERE task_name = '7_acumular_vendas_mes' ORDER BY topological_layer ASC, run_date DESC;
-- ==============================================================================
