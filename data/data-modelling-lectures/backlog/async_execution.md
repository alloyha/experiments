# DAG Engine — Refatoração: Non-Blocking Execution (Estratégias 1 + 4)

## Objetivo

Refatorar o `dag_engine` para eliminar o problema de **poll-blocking de conexão** sem introduzir infraestrutura externa.

A abordagem combina duas estratégias complementares:

- **Estratégia 1 — `dblink` Assíncrono:** O loop principal deixa de `EXECUTE` tarefas diretamente. Ele despacha cada tarefa em uma conexão separada via `dblink_send_query` (fire-and-forget) e continua imediatamente para a próxima tarefa disponível. Um segundo branch do loop faz polling de coleta nos workers ativos.
- **Estratégia 4 — Chunking Temporal Automático via AST:** Tarefas elegíveis são automaticamente expandidas em sub-tasks paralelas durante o `proc_load_dag_spec`, usando `pg_query` para extrair tabelas e `pg_stats` para dimensionar os buckets. A hint de chunking é declarada no spec JSON.

---

## Pré-requisitos

Antes de iniciar, verifique e instale as extensões necessárias:

```sql
-- Requeridas
CREATE EXTENSION IF NOT EXISTS dblink;      -- async dispatch
CREATE EXTENSION IF NOT EXISTS pg_query;    -- AST parser para chunking automático

-- Já presente no motor base
CREATE EXTENSION IF NOT EXISTS pg_cron;
```

Confirme que o usuário do banco tem permissão para criar conexões dblink:

```sql
GRANT USAGE ON SCHEMA public TO <seu_usuario>;
-- dblink_connect requer que o role possa se conectar ao próprio banco
```

---

## Parte 1 — Modificações no Schema

### 1.1 Adicionar colunas de rastreio assíncrono em `task_instances`

```sql
ALTER TABLE dag_engine.task_instances
    ADD COLUMN IF NOT EXISTS worker_conn  TEXT,     -- nome da conexão dblink ativa
    ADD COLUMN IF NOT EXISTS is_chunk     BOOLEAN DEFAULT FALSE,  -- é sub-task de chunking?
    ADD COLUMN IF NOT EXISTS chunk_index  INT,      -- índice do bucket (0-based)
    ADD COLUMN IF NOT EXISTS parent_task  VARCHAR(100); -- task original que gerou este chunk
```

### 1.2 Adicionar hint de chunking no spec de tarefas

```sql
ALTER TABLE dag_engine.tasks
    ADD COLUMN IF NOT EXISTS chunk_config JSONB DEFAULT NULL;
-- Formato esperado:
-- {"column": "data_venda", "buckets": 4}
-- NULL = sem chunking (comportamento padrão)
```

### 1.3 Criar tabela de conexões dblink ativas (opcional mas recomendado para debug)

```sql
CREATE TABLE IF NOT EXISTS dag_engine.async_workers (
    conn_name    TEXT PRIMARY KEY,
    run_id       INT  REFERENCES dag_engine.dag_runs(run_id),
    task_name    VARCHAR(100),
    launched_at  TIMESTAMP DEFAULT clock_timestamp()
);
```

---

## Parte 2 — Chunking Automático via AST

### 2.1 Função de extração de tabelas via AST

Esta função recebe o nome de uma procedure e retorna as tabelas que ela referencia, usando `pg_query` para parsear o corpo PL/pgSQL em JSON navegável.

```sql
CREATE OR REPLACE FUNCTION dag_engine.fn_extract_tables_from_proc(
    p_proc_name TEXT
) RETURNS TABLE (table_schema TEXT, table_name TEXT)
LANGUAGE plpgsql AS $$
DECLARE
    v_body      TEXT;
    v_ast       JSONB;
BEGIN
    -- Extrai o corpo da procedure do catálogo
    SELECT pg_get_functiondef(p_proc_name::regproc) INTO v_body;

    IF v_body IS NULL THEN
        RAISE WARNING 'Procedure % não encontrada no catálogo.', p_proc_name;
        RETURN;
    END IF;

    -- Parseia em AST JSON via pg_query
    SELECT pg_query.parse(v_body)::jsonb INTO v_ast;

    -- Navega o AST extraindo nós do tipo RangeVar (referências a tabelas)
    RETURN QUERY
    SELECT DISTINCT
        COALESCE(node->>'schemaname', 'public') AS table_schema,
        node->>'relname'                        AS table_name
    FROM jsonb_path_query(
        v_ast,
        '$.**.RangeVar'   -- JSONPath recursivo — encontra todos os nós RangeVar
    ) AS node
    WHERE node->>'relname' IS NOT NULL;
END;
$$;
```

### 2.2 Função de dimensionamento de buckets via `pg_stats`

Dado que sabemos a tabela e a coluna candidata, usamos as estatísticas do planner para calcular os ranges de cada bucket com distribuição uniforme.

```sql
CREATE OR REPLACE FUNCTION dag_engine.fn_build_chunk_ranges(
    p_schema     TEXT,
    p_table      TEXT,
    p_column     TEXT,
    p_date       DATE,
    p_buckets    INT
) RETURNS TABLE (
    chunk_index  INT,
    range_start  TEXT,
    range_end    TEXT
)
LANGUAGE plpgsql AS $$
DECLARE
    v_min   DATE;
    v_max   DATE;
    v_step  INTERVAL;
    i       INT;
BEGIN
    -- Usa os limites reais do dia alvo (não pg_stats, para precisão diária)
    -- pg_stats seria usado para particionamento de ranges maiores (meses/anos)
    v_min := p_date;
    v_max := p_date + INTERVAL '1 day' - INTERVAL '1 second';

    -- Divide o intervalo do dia em N buckets iguais
    v_step := (v_max - v_min) / p_buckets;

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
```

### 2.3 Função de expansão de tasks com chunking

Esta é a peça central da Estratégia 4. Recebe uma task com `chunk_config` definido e retorna N tasks derivadas com `procedure_call` parametrizado por range.

```sql
CREATE OR REPLACE FUNCTION dag_engine.fn_expand_chunk_tasks(
    p_task_name     VARCHAR(100),
    p_procedure     TEXT,
    p_dependencies  VARCHAR(100)[],
    p_chunk_config  JSONB,
    p_date          DATE
) RETURNS TABLE (
    task_name       VARCHAR(100),
    procedure_call  TEXT,
    dependencies    VARCHAR(100)[]
)
LANGUAGE plpgsql AS $$
DECLARE
    v_column   TEXT    := p_chunk_config->>'column';
    v_buckets  INT     := COALESCE((p_chunk_config->>'buckets')::INT, 4);
    v_range    RECORD;
    v_proc     TEXT;
BEGIN
    -- Valida que a hint é minimamente coerente
    IF v_column IS NULL THEN
        RAISE EXCEPTION 'chunk_config requer campo "column". Recebido: %', p_chunk_config;
    END IF;

    FOR v_range IN
        SELECT * FROM dag_engine.fn_build_chunk_ranges(
            'public', split_part(p_task_name, '_', 2),  -- heurística de schema
            v_column, p_date, v_buckets
        )
    LOOP
        task_name := p_task_name || '_chunk_' || v_range.chunk_index;

        -- Injeta os parâmetros de range na procedure_call via replace de tokens
        -- Convenção: procedure deve aceitar ($1, $range_start, $range_end)
        v_proc := REPLACE(p_procedure, '$1',          quote_literal(p_date));
        v_proc := REPLACE(v_proc,      '$range_start', quote_literal(v_range.range_start));
        v_proc := REPLACE(v_proc,      '$range_end',   quote_literal(v_range.range_end));

        procedure_call := v_proc;
        dependencies   := p_dependencies;  -- todos os chunks herdam as mesmas deps
        RETURN NEXT;
    END LOOP;
END;
$$;
```

### 2.4 Modificar `proc_load_dag_spec` para expandir chunks automaticamente

Adicione este bloco **após o Passo 2** (aplicação de dependências) dentro de `proc_load_dag_spec`:

```sql
-- PASSO 3 (NOVO): Expande tasks com chunk_config em sub-tasks paralelas
FOR v_task IN
    SELECT * FROM jsonb_array_elements(p_spec)
    WHERE value->'chunk_config' IS NOT NULL
LOOP
    DECLARE
        v_expanded RECORD;
    BEGIN
        -- Remove a task original (será substituída pelos chunks)
        DELETE FROM dag_engine.tasks
        WHERE task_name = v_task->>'task_name';

        -- Insere os chunks expandidos
        FOR v_expanded IN
            SELECT * FROM dag_engine.fn_expand_chunk_tasks(
                v_task->>'task_name',
                v_task->>'procedure_call',
                ARRAY(SELECT jsonb_array_elements_text(v_task->'dependencies')),
                v_task->'chunk_config',
                CURRENT_DATE  -- será sobrescrito em runtime; serve para validação estrutural
            )
        LOOP
            INSERT INTO dag_engine.tasks (
                task_name, procedure_call, dependencies,
                max_retries, retry_delay_seconds, chunk_config
            ) VALUES (
                v_expanded.task_name,
                v_expanded.procedure_call,
                v_expanded.dependencies,
                COALESCE((v_task->>'max_retries')::INT, 0),
                COALESCE((v_task->>'retry_delay_seconds')::INT, 5),
                v_task->'chunk_config'   -- preservado para reexpansão futura
            )
            ON CONFLICT (task_name) DO UPDATE SET
                procedure_call = EXCLUDED.procedure_call,
                dependencies   = EXCLUDED.dependencies;
        END LOOP;
    END;
END LOOP;
```

---

## Parte 3 — Refatorar o Loop Principal (`proc_run_dag`)

Esta é a mudança mais cirúrgica. O loop original tinha um único branch de execução. Agora terá **três branches**:

```
LOOP
  ├── BRANCH A: há task PENDING com deps resolvidas?
  │     └── dblink_send_query → registra RUNNING com worker_conn → COMMIT → continua
  │
  ├── BRANCH B: há workers RUNNING ativos?
  │     └── poll cada worker_conn com dblink_is_busy
  │           ├── ainda rodando → skip
  │           └── terminou → dblink_get_result → SUCCESS ou FAILED + cascade
  │
  └── BRANCH C: nenhuma task disponível nem workers ativos
        └── avalia término (tudo SUCCESS/FAILED) ou aguarda retry backoff
```

### 3.1 Procedure auxiliar de dispatch assíncrono

```sql
CREATE OR REPLACE PROCEDURE dag_engine.proc_dispatch_task(
    p_run_id    INT,
    p_task_name VARCHAR(100),
    p_sql       TEXT
)
LANGUAGE plpgsql AS $$
DECLARE
    v_conn_name TEXT := 'dag_worker_' || p_run_id || '_' || replace(p_task_name, ' ', '_');
    v_dsn       TEXT := 'dbname=' || current_database()
                     || ' host=localhost'
                     || ' user=' || current_user;
BEGIN
    -- Abre conexão assíncrona dedicada para esta task
    PERFORM dblink_connect(v_conn_name, v_dsn);

    -- Fire-and-forget: retorna imediatamente
    PERFORM dblink_send_query(v_conn_name, p_sql);

    -- Registra worker ativo
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
```

### 3.2 Procedure auxiliar de coleta de resultados

```sql
CREATE OR REPLACE PROCEDURE dag_engine.proc_collect_workers(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_worker  RECORD;
    v_result  RECORD;
    v_err     TEXT;
BEGIN
    FOR v_worker IN
        SELECT ti.task_name, ti.worker_conn, ti.start_ts
        FROM dag_engine.task_instances ti
        WHERE ti.run_id = p_run_id AND ti.status = 'RUNNING'
          AND ti.worker_conn IS NOT NULL
    LOOP
        BEGIN
            -- Não bloqueia: se ainda está rodando, pula para o próximo
            IF dblink_is_busy(v_worker.worker_conn) THEN
                CONTINUE;
            END IF;

            -- Coleta resultado (lança exceção se a task falhou)
            PERFORM * FROM dblink_get_result(v_worker.worker_conn) AS t(result TEXT);

            -- Sucesso
            UPDATE dag_engine.task_instances
            SET status      = 'SUCCESS',
                end_ts      = clock_timestamp(),
                duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - v_worker.start_ts)) * 1000,
                worker_conn = NULL
            WHERE run_id = p_run_id AND task_name = v_worker.task_name;

        EXCEPTION WHEN OTHERS THEN
            v_err := SQLERRM;

            -- Falha: aplica lógica de retry ou cascade (igual ao motor original)
            DECLARE
                v_attempt     INT;
                v_max_retries INT;
                v_delay       INT;
            BEGIN
                SELECT ti.attempt, t.max_retries, t.retry_delay_seconds
                INTO v_attempt, v_max_retries, v_delay
                FROM dag_engine.task_instances ti
                JOIN dag_engine.tasks t ON t.task_name = ti.task_name
                WHERE ti.run_id = p_run_id AND ti.task_name = v_worker.task_name;

                IF v_attempt < v_max_retries + 1 THEN
                    UPDATE dag_engine.task_instances
                    SET status         = 'PENDING',
                        attempt        = attempt + 1,
                        retry_after_ts = clock_timestamp() + (v_delay * (v_attempt + 1)) * INTERVAL '1 second',
                        worker_conn    = NULL,
                        error_text     = 'Retry | Último erro: ' || v_err
                    WHERE run_id = p_run_id AND task_name = v_worker.task_name;
                ELSE
                    -- Fatal: marca FAILED e propaga UPSTREAM_FAILED
                    UPDATE dag_engine.task_instances
                    SET status      = 'FAILED',
                        end_ts      = clock_timestamp(),
                        duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - v_worker.start_ts)) * 1000,
                        worker_conn = NULL,
                        error_text  = v_err
                    WHERE run_id = p_run_id AND task_name = v_worker.task_name;

                    WITH RECURSIVE fail_cascade AS (
                        SELECT t.task_name FROM dag_engine.tasks t
                        WHERE v_worker.task_name = ANY(t.dependencies)
                        UNION ALL
                        SELECT t.task_name FROM dag_engine.tasks t
                        JOIN fail_cascade fc ON fc.task_name = ANY(t.dependencies)
                    )
                    UPDATE dag_engine.task_instances
                    SET status    = 'UPSTREAM_FAILED',
                        end_ts    = clock_timestamp(),
                        error_text = 'Propagado de: ' || v_worker.task_name
                    WHERE run_id = p_run_id
                      AND task_name IN (SELECT task_name FROM fail_cascade)
                      AND status = 'PENDING';
                END IF;
            END;
        END;

        -- Libera a conexão dblink independente do resultado
        BEGIN
            PERFORM dblink_disconnect(v_worker.worker_conn);
            DELETE FROM dag_engine.async_workers WHERE conn_name = v_worker.worker_conn;
        EXCEPTION WHEN OTHERS THEN NULL; END;  -- ignora se já foi fechada
    END LOOP;
END;
$$;
```

### 3.3 Novo `proc_run_dag` — substituição completa

```sql
DROP PROCEDURE IF EXISTS dag_engine.proc_run_dag(DATE, BOOLEAN);
CREATE OR REPLACE PROCEDURE dag_engine.proc_run_dag(
    p_data    DATE,
    p_verbose BOOLEAN DEFAULT TRUE
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
        RAISE NOTICE '🚀 Iniciando DAG Async para: %', p_data;
    END IF;

    -- Cria run (mantém proteção de unique_violation original)
    BEGIN
        INSERT INTO dag_engine.dag_runs (run_date)
        VALUES (p_data)
        RETURNING run_id INTO v_run_id;
    EXCEPTION WHEN unique_violation THEN
        RAISE WARNING 'Run já existe para %. Use proc_clear_run para re-executar.', p_data;
        RETURN;
    END;

    -- Instancia todas as tasks como PENDING
    INSERT INTO dag_engine.task_instances (run_id, task_name)
    SELECT v_run_id, task_name FROM dag_engine.tasks;
    COMMIT;

    -- ================================================================
    -- LOOP PRINCIPAL DO SCHEDULER
    -- ================================================================
    LOOP

        -- ------------------------------------------------------------
        -- BRANCH B: Coleta workers que terminaram (não-bloqueante)
        -- ------------------------------------------------------------
        CALL dag_engine.proc_collect_workers(v_run_id);
        COMMIT;

        -- ------------------------------------------------------------
        -- BRANCH A: Despacha próxima task PENDING elegível
        -- ------------------------------------------------------------
        SELECT ti.task_name, t.procedure_call
        INTO v_task
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
        ORDER BY t.task_id
        FOR UPDATE OF ti SKIP LOCKED
        LIMIT 1;

        IF v_task IS NOT NULL THEN
            -- Interpola p_data nos tokens da procedure_call
            v_sql := REPLACE(v_task.procedure_call, '$1', quote_literal(p_data));

            IF p_verbose THEN
                RAISE NOTICE '  --> 📤 Despachando async: [%] %', v_task.task_name, v_sql;
            END IF;

            -- Despacha sem bloquear (Estratégia 1)
            CALL dag_engine.proc_dispatch_task(v_run_id, v_task.task_name, v_sql);
            COMMIT;

            -- Volta ao topo do loop imediatamente para despachar mais tasks paralelas
            CONTINUE;
        END IF;

        -- ------------------------------------------------------------
        -- BRANCH C: Nenhuma task disponível — avalia estado geral
        -- ------------------------------------------------------------
        SELECT COUNT(*) INTO v_pending_count
        FROM dag_engine.task_instances
        WHERE run_id = v_run_id AND status = 'PENDING';

        SELECT COUNT(*) INTO v_running_count
        FROM dag_engine.task_instances
        WHERE run_id = v_run_id AND status = 'RUNNING';

        IF v_running_count > 0 THEN
            -- Workers ainda ativos — aguarda polling interval
            PERFORM pg_sleep(0.5);

        ELSIF v_pending_count > 0 THEN
            -- Só resta tasks em backoff de retry
            IF EXISTS (
                SELECT 1 FROM dag_engine.task_instances
                WHERE run_id = v_run_id
                  AND status = 'PENDING'
                  AND retry_after_ts > clock_timestamp()
            ) THEN
                PERFORM pg_sleep(1);
            ELSE
                -- Deadlock topológico
                UPDATE dag_engine.dag_runs
                SET status = 'DEADLOCK', end_ts = clock_timestamp()
                WHERE run_id = v_run_id;
                COMMIT;
                RAISE WARNING '💀 Deadlock Topológico: tasks pendentes irresolvíveis.';
                EXIT;
            END IF;

        ELSE
            -- Pipeline completo — avalia resultado final
            IF EXISTS (
                SELECT 1 FROM dag_engine.task_instances
                WHERE run_id = v_run_id
                  AND status IN ('FAILED', 'UPSTREAM_FAILED')
            ) THEN
                UPDATE dag_engine.dag_runs
                SET status = 'FAILED', end_ts = clock_timestamp()
                WHERE run_id = v_run_id;
                RAISE WARNING '❌ DAG % finalizada com falhas.', p_data;
            ELSE
                UPDATE dag_engine.dag_runs
                SET status = 'SUCCESS', end_ts = clock_timestamp()
                WHERE run_id = v_run_id;
                IF p_verbose THEN
                    RAISE NOTICE '✅ DAG % finalizada com sucesso.', p_data;
                END IF;
            END IF;
            COMMIT;
            EXIT;
        END IF;

    END LOOP;

    -- Meta-DAG do Medallion (inalterado)
    CALL dag_medallion.proc_run_medallion(v_run_id);
END;
$$;
```

---

## Parte 4 — Uso do Chunking no Spec JSON

Para habilitar chunking em uma task, adicione `chunk_config` e adapte a `procedure_call` para aceitar os tokens `$range_start` e `$range_end`:

```sql
CALL dag_engine.proc_deploy_dag(
'[
    {
        "task_name": "5_ingestao_fato_vendas",
        "procedure_call": "CALL varejo.proc_ingestao_fato_vendas($1, $range_start, $range_end)",
        "dependencies": ["3_upsert_clientes_scd2", "4_upsert_produtos_scd3"],
        "max_retries": 1,
        "retry_delay_seconds": 10,
        "chunk_config": {
            "column": "data_venda",
            "buckets": 4
        }
    }
]'::JSONB,
'v2.0-async-chunked',
'Migração para dispatch assíncrono + chunking temporal em fato_vendas'
);
```

O `proc_load_dag_spec` expandirá automaticamente em:

```
5_ingestao_fato_vendas_chunk_0   (00:00 → 05:59)  ──┐
5_ingestao_fato_vendas_chunk_1   (06:00 → 11:59)  ──┤─ todos paralelos
5_ingestao_fato_vendas_chunk_2   (12:00 → 17:59)  ──┤─ via SKIP LOCKED
5_ingestao_fato_vendas_chunk_3   (18:00 → 23:59)  ──┘
```

A procedure alvo deve ser adaptada para aceitar os parâmetros de range:

```sql
CREATE OR REPLACE PROCEDURE varejo.proc_ingestao_fato_vendas(
    p_data        DATE,
    p_range_start TIMESTAMP DEFAULT NULL,
    p_range_end   TIMESTAMP DEFAULT NULL
)
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO varejo.fato_vendas (...)
    SELECT ...
    FROM varejo.origem_venda
    WHERE data_venda = p_data
      AND (p_range_start IS NULL OR hora_venda >= p_range_start)
      AND (p_range_end   IS NULL OR hora_venda <= p_range_end);
END;
$$;
```

Parâmetros `NULL` preservam o comportamento original, garantindo retrocompatibilidade.

---

## Parte 5 — Limpeza de Conexões Órfãs

Em crashes ou interrupções, conexões dblink podem ficar abertas. Adicione esta procedure de housekeeping:

```sql
CREATE OR REPLACE PROCEDURE dag_engine.proc_cleanup_orphan_workers(p_run_id INT)
LANGUAGE plpgsql AS $$
DECLARE
    v_conn RECORD;
BEGIN
    FOR v_conn IN
        SELECT conn_name FROM dag_engine.async_workers WHERE run_id = p_run_id
    LOOP
        BEGIN
            PERFORM dblink_disconnect(v_conn.conn_name);
        EXCEPTION WHEN OTHERS THEN NULL; END;
    END LOOP;

    DELETE FROM dag_engine.async_workers WHERE run_id = p_run_id;

    -- Marca como FAILED tasks que ficaram presas em RUNNING sem worker
    UPDATE dag_engine.task_instances
    SET status     = 'FAILED',
        error_text = 'Worker órfão: conexão dblink perdida'
    WHERE run_id = p_run_id
      AND status  = 'RUNNING'
      AND worker_conn IS NOT NULL;

    RAISE NOTICE '🧹 Workers órfãos do run_id % limpos.', p_run_id;
END;
$$;

-- Integrar no proc_clear_run existente:
-- Adicione antes das DELETEs:
-- CALL dag_engine.proc_cleanup_orphan_workers(v_run_id);
```

---

## Parte 6 — Checklist de Validação

Execute esta sequência para confirmar que a refatoração está funcional:

```sql
-- 1. Verifica extensões
SELECT extname FROM pg_extension WHERE extname IN ('dblink', 'pg_query');

-- 2. Confirma novas colunas
SELECT column_name FROM information_schema.columns
WHERE table_schema = 'dag_engine'
  AND table_name   = 'task_instances'
  AND column_name IN ('worker_conn', 'is_chunk', 'chunk_index', 'parent_task');

-- 3. Testa extração de tabelas via AST (substitua pelo nome de uma procedure real)
SELECT * FROM dag_engine.fn_extract_tables_from_proc(
    'varejo.proc_ingestao_fato_vendas'
);

-- 4. Testa expansão de chunks manualmente
SELECT * FROM dag_engine.fn_expand_chunk_tasks(
    '5_ingestao_fato_vendas',
    'CALL varejo.proc_ingestao_fato_vendas($1, $range_start, $range_end)',
    ARRAY['3_upsert_clientes_scd2'],
    '{"column": "data_venda", "buckets": 4}'::JSONB,
    '2024-05-04'::DATE
);

-- 5. Executa uma run de smoke test
CALL dag_engine.proc_run_dag('2024-05-04', TRUE);

-- 6. Confirma que não há conexões dblink abertas após conclusão
SELECT * FROM dag_engine.async_workers;  -- deve retornar 0 linhas

-- 7. Verifica paralelismo real — durante a run, em outra sessão:
SELECT task_name, status, worker_conn
FROM dag_engine.task_instances
WHERE status = 'RUNNING';
```

---

## Resumo das Mudanças por Arquivo/Objeto

| Objeto | Tipo de Mudança | Impacto |
|---|---|---|
| `dag_engine.task_instances` | `ALTER TABLE` — 4 colunas novas | Não-destrutivo |
| `dag_engine.tasks` | `ALTER TABLE` — coluna `chunk_config` | Não-destrutivo |
| `dag_engine.async_workers` | `CREATE TABLE` nova | Aditivo |
| `fn_extract_tables_from_proc` | `CREATE FUNCTION` nova | Aditivo |
| `fn_build_chunk_ranges` | `CREATE FUNCTION` nova | Aditivo |
| `fn_expand_chunk_tasks` | `CREATE FUNCTION` nova | Aditivo |
| `proc_load_dag_spec` | Adição do Passo 3 (chunking) | Retrocompatível |
| `proc_dispatch_task` | `CREATE PROCEDURE` nova | Aditivo |
| `proc_collect_workers` | `CREATE PROCEDURE` nova | Aditivo |
| `proc_run_dag` | **Substituição completa** | Breaking — requer re-deploy |
| `proc_cleanup_orphan_workers` | `CREATE PROCEDURE` nova | Aditivo |
| `proc_clear_run` | Adição de chamada de cleanup | Retrocompatível |