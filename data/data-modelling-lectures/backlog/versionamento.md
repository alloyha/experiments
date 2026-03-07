# DAG Versioning — O Problema é Mais Fundo do que Parece

## Por que é não-trivial

O Airflow sofre com isso até hoje. O problema central é que **uma DAG é código que muda, mas os dados que ela produziu não mudam** — e você precisa correlacionar os dois para sempre.

Três cenários que quebram sem versionamento:

```
v1 do DAG (maio):  A → B → C → D
v2 do DAG (junho): A → B → E → D  (C removida, E adicionada)

Pergunta: quando você faz catchup de maio com a v2 rodando,
qual topologia deve ser usada? A de maio ou a atual?

Pergunta 2: o gold_critical_path comparando maio vs junho
está comparando estruturas incompatíveis — C não existe em junho.

Pergunta 3: se você fez clear_run de maio para reprocessar,
qual spec você restaura?
```

---

## O que precisa ser versionado

São três artefatos distintos com ciclos de vida diferentes:

```
┌─────────────────────────────────────────────────┐
│  DAG SPEC VERSION                                │
│  • topologia (nós + arestas)                    │
│  • procedure_calls                              │  ← muda com refatorações
│  • retry policies                               │
├─────────────────────────────────────────────────┤
│  EXECUTION CONTEXT                              │
│  • qual versão rodou em qual run_id             │  ← join histórico
│  • qual versão está ativa agora                 │
├─────────────────────────────────────────────────┤
│  SCHEMA VERSION (das tabelas tocadas)           │
│  • a procedure mudou por causa do schema?       │  ← dependência implícita
│  • qual migration estava ativa naquela data?    │
└─────────────────────────────────────────────────┘
```

---

## Design para o Motor

### Tabela de versões do spec

```sql
CREATE TABLE dag_engine.dag_versions (
    version_id      SERIAL PRIMARY KEY,
    version_tag     VARCHAR(50) NOT NULL,        -- 'v1.0', 'v1.2-hotfix', semver
    spec            JSONB NOT NULL,              -- snapshot completo do JSON carregado
    spec_hash       TEXT GENERATED ALWAYS AS    -- fingerprint para detectar diff silencioso
                    (md5(spec::text)) STORED,
    deployed_at     TIMESTAMP DEFAULT clock_timestamp(),
    deployed_by     TEXT DEFAULT current_user,
    is_active       BOOLEAN DEFAULT FALSE,
    change_summary  TEXT,                        -- "Adicionada task 9_envio_email"
    parent_version  INT REFERENCES dag_engine.dag_versions(version_id)
);

-- Apenas uma versão ativa por vez
CREATE UNIQUE INDEX idx_dag_one_active 
ON dag_engine.dag_versions(is_active) 
WHERE is_active = TRUE;
```

### Linkar runs à versão que os executou

```sql
ALTER TABLE dag_engine.dag_runs 
ADD COLUMN version_id INT REFERENCES dag_engine.dag_versions(version_id);
```

### Modificar `proc_load_dag_spec` para versionar

```sql
CREATE OR REPLACE PROCEDURE dag_engine.proc_deploy_dag(
    p_spec       JSONB,
    p_tag        VARCHAR(50),
    p_summary    TEXT DEFAULT NULL
)
LANGUAGE plpgsql AS $$
DECLARE
    v_new_version_id  INT;
    v_current_hash    TEXT;
    v_new_hash        TEXT := md5(p_spec::text);
    v_parent_id       INT;
BEGIN
    -- Detecta redeploy silencioso (spec idêntico)
    SELECT spec_hash, version_id 
    INTO v_current_hash, v_parent_id
    FROM dag_engine.dag_versions 
    WHERE is_active = TRUE;

    IF v_current_hash = v_new_hash THEN
        RAISE WARNING '⚠️ Spec idêntico ao ativo (hash: %). Nenhuma versão criada.', v_new_hash;
        RETURN;
    END IF;

    -- Desativa versão anterior
    UPDATE dag_engine.dag_versions SET is_active = FALSE WHERE is_active = TRUE;

    -- Registra nova versão
    INSERT INTO dag_engine.dag_versions 
        (version_tag, spec, is_active, change_summary, parent_version)
    VALUES 
        (p_tag, p_spec, TRUE, p_summary, v_parent_id)
    RETURNING version_id INTO v_new_version_id;

    -- Carrega no motor (lógica atual do proc_load_dag_spec)
    CALL dag_engine.proc_load_dag_spec(p_spec);

    RAISE NOTICE '✅ DAG % deployada. Version ID: % (parent: %)', 
        p_tag, v_new_version_id, v_parent_id;
END;
$$;
```

### `proc_run_dag` passa a carimbar a versão ativa

```sql
-- Dentro do proc_run_dag, no INSERT do dag_runs:
INSERT INTO dag_engine.dag_runs (run_date, version_id)
SELECT p_data, version_id 
FROM dag_engine.dag_versions 
WHERE is_active = TRUE
RETURNING run_id INTO v_run_id;
```

---

## Diff entre versões

```sql
CREATE OR REPLACE FUNCTION dag_engine.fn_diff_versions(
    p_v1 INT, p_v2 INT
) RETURNS TABLE (
    change_type  TEXT,
    task_name    TEXT,
    detail       TEXT
) LANGUAGE sql AS $$
    WITH 
    v1_tasks AS (
        SELECT elem->>'task_name' AS task_name,
               elem->>'procedure_call' AS proc,
               elem->'dependencies' AS deps
        FROM dag_engine.dag_versions,
        jsonb_array_elements(spec) AS elem
        WHERE version_id = p_v1
    ),
    v2_tasks AS (
        SELECT elem->>'task_name' AS task_name,
               elem->>'procedure_call' AS proc,
               elem->'dependencies' AS deps
        FROM dag_engine.dag_versions,
        jsonb_array_elements(spec) AS elem
        WHERE version_id = p_v2
    )
    -- Tarefas removidas
    SELECT 'REMOVED', v1.task_name, v1.proc
    FROM v1_tasks v1 WHERE NOT EXISTS (SELECT 1 FROM v2_tasks v2 WHERE v2.task_name = v1.task_name)
    UNION ALL
    -- Tarefas adicionadas
    SELECT 'ADDED', v2.task_name, v2.proc
    FROM v2_tasks v2 WHERE NOT EXISTS (SELECT 1 FROM v1_tasks v1 WHERE v1.task_name = v2.task_name)
    UNION ALL
    -- Procedure alterada
    SELECT 'PROC_CHANGED', v1.task_name, 
           'era: ' || v1.proc || ' → agora: ' || v2.proc
    FROM v1_tasks v1 JOIN v2_tasks v2 ON v1.task_name = v2.task_name
    WHERE v1.proc != v2.proc
    UNION ALL
    -- Dependências alteradas
    SELECT 'DEPS_CHANGED', v1.task_name,
           'era: ' || v1.deps::text || ' → agora: ' || v2.deps::text
    FROM v1_tasks v1 JOIN v2_tasks v2 ON v1.task_name = v2.task_name
    WHERE v1.deps != v2.deps;
$$;

-- Uso:
SELECT * FROM dag_engine.fn_diff_versions(1, 2);
```

---

## O problema do Catchup com versões mistas

Esta é a decisão de design mais importante. Você tem duas filosofias:

```
┌──────────────────────┬─────────────────────────────────────┐
│ FILOSOFIA            │ COMPORTAMENTO                       │
├──────────────────────┼─────────────────────────────────────┤
│ "Snapshot fiel"      │ Catchup de maio usa spec de maio    │
│                      │ → reprodutibilidade total           │
│                      │ → requer restaurar versão antiga    │
├──────────────────────┼─────────────────────────────────────┤
│ "Current HEAD"       │ Catchup sempre usa spec ativo       │
│                      │ → mais simples operacionalmente     │  
│                      │ → pode falhar se task foi removida  │
└──────────────────────┴─────────────────────────────────────┘
```

O `proc_catchup` com suporte a versão ficaria assim:

```sql
CREATE OR REPLACE PROCEDURE dag_engine.proc_catchup(
    p_from    DATE, 
    p_to      DATE,
    p_version INT DEFAULT NULL  -- NULL = usa versão ativa (current HEAD)
)
LANGUAGE plpgsql AS $$
DECLARE
    v_date      DATE := p_from;
    v_spec      JSONB;
    v_version   INT;
BEGIN
    -- Resolve qual versão usar
    IF p_version IS NOT NULL THEN
        SELECT spec, version_id INTO v_spec, v_version
        FROM dag_engine.dag_versions WHERE version_id = p_version;
        
        RAISE NOTICE '📌 Catchup usando versão fixada: %', v_version;
        CALL dag_engine.proc_load_dag_spec(v_spec);  -- restaura temporariamente
    END IF;

    WHILE v_date <= p_to LOOP
        CALL dag_engine.proc_run_dag(v_date);
        v_date := v_date + 1;
    END LOOP;

    -- Restaura versão ativa se tínhamos fixado uma antiga
    IF p_version IS NOT NULL THEN
        SELECT spec INTO v_spec FROM dag_engine.dag_versions WHERE is_active = TRUE;
        CALL dag_engine.proc_load_dag_spec(v_spec);
        RAISE NOTICE '🔄 Versão ativa restaurada após catchup histórico.';
    END IF;
END;
$$;
```

---

## O que isso habilita no Medallion

Com `version_id` em `dag_runs`, o `dim_task` vira **SCD2 real** em vez de SCD1 — cada versão da topologia fica preservada, e o `gold_critical_path` consegue comparar performance isolando variáveis de estrutura:

```sql
-- "A task 5 ficou mais lenta depois da v1.2 ou já era assim antes?"
SELECT v.version_tag, f.task_name, ROUND(AVG(f.duration_ms), 2)
FROM dag_medallion.fato_task_exec f
JOIN dag_engine.dag_runs dr ON dr.run_id = f.run_id
JOIN dag_engine.dag_versions v ON v.version_id = dr.version_id
GROUP BY v.version_tag, f.task_name
ORDER BY f.task_name, v.deployed_at;
```

Isso é o que separa observabilidade de **rastreabilidade** — a capacidade de atribuir causas a mudanças de comportamento ao longo do tempo.# DAG Versioning — O Problema é Mais Fundo do que Parece

## Por que é não-trivial

O Airflow sofre com isso até hoje. O problema central é que **uma DAG é código que muda, mas os dados que ela produziu não mudam** — e você precisa correlacionar os dois para sempre.

Três cenários que quebram sem versionamento:

```
v1 do DAG (maio):  A → B → C → D
v2 do DAG (junho): A → B → E → D  (C removida, E adicionada)

Pergunta: quando você faz catchup de maio com a v2 rodando,
qual topologia deve ser usada? A de maio ou a atual?

Pergunta 2: o gold_critical_path comparando maio vs junho
está comparando estruturas incompatíveis — C não existe em junho.

Pergunta 3: se você fez clear_run de maio para reprocessar,
qual spec você restaura?
```

---

## O que precisa ser versionado

São três artefatos distintos com ciclos de vida diferentes:

```
┌─────────────────────────────────────────────────┐
│  DAG SPEC VERSION                                │
│  • topologia (nós + arestas)                    │
│  • procedure_calls                              │  ← muda com refatorações
│  • retry policies                               │
├─────────────────────────────────────────────────┤
│  EXECUTION CONTEXT                              │
│  • qual versão rodou em qual run_id             │  ← join histórico
│  • qual versão está ativa agora                 │
├─────────────────────────────────────────────────┤
│  SCHEMA VERSION (das tabelas tocadas)           │
│  • a procedure mudou por causa do schema?       │  ← dependência implícita
│  • qual migration estava ativa naquela data?    │
└─────────────────────────────────────────────────┘
```

---

## Design para o Motor

### Tabela de versões do spec

```sql
CREATE TABLE dag_engine.dag_versions (
    version_id      SERIAL PRIMARY KEY,
    version_tag     VARCHAR(50) NOT NULL,        -- 'v1.0', 'v1.2-hotfix', semver
    spec            JSONB NOT NULL,              -- snapshot completo do JSON carregado
    spec_hash       TEXT GENERATED ALWAYS AS    -- fingerprint para detectar diff silencioso
                    (md5(spec::text)) STORED,
    deployed_at     TIMESTAMP DEFAULT clock_timestamp(),
    deployed_by     TEXT DEFAULT current_user,
    is_active       BOOLEAN DEFAULT FALSE,
    change_summary  TEXT,                        -- "Adicionada task 9_envio_email"
    parent_version  INT REFERENCES dag_engine.dag_versions(version_id)
);

-- Apenas uma versão ativa por vez
CREATE UNIQUE INDEX idx_dag_one_active 
ON dag_engine.dag_versions(is_active) 
WHERE is_active = TRUE;
```

### Linkar runs à versão que os executou

```sql
ALTER TABLE dag_engine.dag_runs 
ADD COLUMN version_id INT REFERENCES dag_engine.dag_versions(version_id);
```

### Modificar `proc_load_dag_spec` para versionar

```sql
CREATE OR REPLACE PROCEDURE dag_engine.proc_deploy_dag(
    p_spec       JSONB,
    p_tag        VARCHAR(50),
    p_summary    TEXT DEFAULT NULL
)
LANGUAGE plpgsql AS $$
DECLARE
    v_new_version_id  INT;
    v_current_hash    TEXT;
    v_new_hash        TEXT := md5(p_spec::text);
    v_parent_id       INT;
BEGIN
    -- Detecta redeploy silencioso (spec idêntico)
    SELECT spec_hash, version_id 
    INTO v_current_hash, v_parent_id
    FROM dag_engine.dag_versions 
    WHERE is_active = TRUE;

    IF v_current_hash = v_new_hash THEN
        RAISE WARNING '⚠️ Spec idêntico ao ativo (hash: %). Nenhuma versão criada.', v_new_hash;
        RETURN;
    END IF;

    -- Desativa versão anterior
    UPDATE dag_engine.dag_versions SET is_active = FALSE WHERE is_active = TRUE;

    -- Registra nova versão
    INSERT INTO dag_engine.dag_versions 
        (version_tag, spec, is_active, change_summary, parent_version)
    VALUES 
        (p_tag, p_spec, TRUE, p_summary, v_parent_id)
    RETURNING version_id INTO v_new_version_id;

    -- Carrega no motor (lógica atual do proc_load_dag_spec)
    CALL dag_engine.proc_load_dag_spec(p_spec);

    RAISE NOTICE '✅ DAG % deployada. Version ID: % (parent: %)', 
        p_tag, v_new_version_id, v_parent_id;
END;
$$;
```

### `proc_run_dag` passa a carimbar a versão ativa

```sql
-- Dentro do proc_run_dag, no INSERT do dag_runs:
INSERT INTO dag_engine.dag_runs (run_date, version_id)
SELECT p_data, version_id 
FROM dag_engine.dag_versions 
WHERE is_active = TRUE
RETURNING run_id INTO v_run_id;
```

---

## Diff entre versões

```sql
CREATE OR REPLACE FUNCTION dag_engine.fn_diff_versions(
    p_v1 INT, p_v2 INT
) RETURNS TABLE (
    change_type  TEXT,
    task_name    TEXT,
    detail       TEXT
) LANGUAGE sql AS $$
    WITH 
    v1_tasks AS (
        SELECT elem->>'task_name' AS task_name,
               elem->>'procedure_call' AS proc,
               elem->'dependencies' AS deps
        FROM dag_engine.dag_versions,
        jsonb_array_elements(spec) AS elem
        WHERE version_id = p_v1
    ),
    v2_tasks AS (
        SELECT elem->>'task_name' AS task_name,
               elem->>'procedure_call' AS proc,
               elem->'dependencies' AS deps
        FROM dag_engine.dag_versions,
        jsonb_array_elements(spec) AS elem
        WHERE version_id = p_v2
    )
    -- Tarefas removidas
    SELECT 'REMOVED', v1.task_name, v1.proc
    FROM v1_tasks v1 WHERE NOT EXISTS (SELECT 1 FROM v2_tasks v2 WHERE v2.task_name = v1.task_name)
    UNION ALL
    -- Tarefas adicionadas
    SELECT 'ADDED', v2.task_name, v2.proc
    FROM v2_tasks v2 WHERE NOT EXISTS (SELECT 1 FROM v1_tasks v1 WHERE v1.task_name = v2.task_name)
    UNION ALL
    -- Procedure alterada
    SELECT 'PROC_CHANGED', v1.task_name, 
           'era: ' || v1.proc || ' → agora: ' || v2.proc
    FROM v1_tasks v1 JOIN v2_tasks v2 ON v1.task_name = v2.task_name
    WHERE v1.proc != v2.proc
    UNION ALL
    -- Dependências alteradas
    SELECT 'DEPS_CHANGED', v1.task_name,
           'era: ' || v1.deps::text || ' → agora: ' || v2.deps::text
    FROM v1_tasks v1 JOIN v2_tasks v2 ON v1.task_name = v2.task_name
    WHERE v1.deps != v2.deps;
$$;

-- Uso:
SELECT * FROM dag_engine.fn_diff_versions(1, 2);
```

---

## O problema do Catchup com versões mistas

Esta é a decisão de design mais importante. Você tem duas filosofias:

```
┌──────────────────────┬─────────────────────────────────────┐
│ FILOSOFIA            │ COMPORTAMENTO                       │
├──────────────────────┼─────────────────────────────────────┤
│ "Snapshot fiel"      │ Catchup de maio usa spec de maio    │
│                      │ → reprodutibilidade total           │
│                      │ → requer restaurar versão antiga    │
├──────────────────────┼─────────────────────────────────────┤
│ "Current HEAD"       │ Catchup sempre usa spec ativo       │
│                      │ → mais simples operacionalmente     │  
│                      │ → pode falhar se task foi removida  │
└──────────────────────┴─────────────────────────────────────┘
```

O `proc_catchup` com suporte a versão ficaria assim:

```sql
CREATE OR REPLACE PROCEDURE dag_engine.proc_catchup(
    p_from    DATE, 
    p_to      DATE,
    p_version INT DEFAULT NULL  -- NULL = usa versão ativa (current HEAD)
)
LANGUAGE plpgsql AS $$
DECLARE
    v_date      DATE := p_from;
    v_spec      JSONB;
    v_version   INT;
BEGIN
    -- Resolve qual versão usar
    IF p_version IS NOT NULL THEN
        SELECT spec, version_id INTO v_spec, v_version
        FROM dag_engine.dag_versions WHERE version_id = p_version;
        
        RAISE NOTICE '📌 Catchup usando versão fixada: %', v_version;
        CALL dag_engine.proc_load_dag_spec(v_spec);  -- restaura temporariamente
    END IF;

    WHILE v_date <= p_to LOOP
        CALL dag_engine.proc_run_dag(v_date);
        v_date := v_date + 1;
    END LOOP;

    -- Restaura versão ativa se tínhamos fixado uma antiga
    IF p_version IS NOT NULL THEN
        SELECT spec INTO v_spec FROM dag_engine.dag_versions WHERE is_active = TRUE;
        CALL dag_engine.proc_load_dag_spec(v_spec);
        RAISE NOTICE '🔄 Versão ativa restaurada após catchup histórico.';
    END IF;
END;
$$;
```

---

## O que isso habilita no Medallion

Com `version_id` em `dag_runs`, o `dim_task` vira **SCD2 real** em vez de SCD1 — cada versão da topologia fica preservada, e o `gold_critical_path` consegue comparar performance isolando variáveis de estrutura:

```sql
-- "A task 5 ficou mais lenta depois da v1.2 ou já era assim antes?"
SELECT v.version_tag, f.task_name, ROUND(AVG(f.duration_ms), 2)
FROM dag_medallion.fato_task_exec f
JOIN dag_engine.dag_runs dr ON dr.run_id = f.run_id
JOIN dag_engine.dag_versions v ON v.version_id = dr.version_id
GROUP BY v.version_tag, f.task_name
ORDER BY f.task_name, v.deployed_at;
```

Isso é o que separa observabilidade de **rastreabilidade** — a capacidade de atribuir causas a mudanças de comportamento ao longo do tempo.