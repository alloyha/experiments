-- ==============================================================================
-- Aula 8: Fato e Dimensão na Prática (Caso Varejo + SCDs Clássicos)
-- ==============================================================================
-- IMPORTANTE: Para executar este script, rode antes o setup do Varejo:
--   \i scripts/setup_varejo.sql
-- Ele cria as tabelas OLTP de origem (origem_cliente, origem_produto, 
-- origem_venda) que alimentam nosso pipeline.
-- ==============================================================================
-- Princípio de Design: Toda Procedure é INCREMENTAL por natureza.
-- Backfill = Fast-Forward de chamadas incrementais em sequência.
-- ==============================================================================

-- ==============================================================================
-- SCD TYPE 1: SOBRESCREVE (SEM HISTÓRICO)
-- ==============================================================================
-- A Procedure apenas toca nas linhas que divergem entre Origem e Destino.
-- Linhas iguais = zero I/O.

DROP TABLE IF EXISTS varejo.dim_cliente_type1;
CREATE TABLE varejo.dim_cliente_type1 (
    cliente_id INTEGER PRIMARY KEY,
    nome VARCHAR(100),
    estado VARCHAR(2),
    segmento VARCHAR(50)
);

CREATE OR REPLACE PROCEDURE varejo.proc_upsert_clientes_scd1()
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO varejo.dim_cliente_type1 AS tgt (cliente_id, nome, estado, segmento)
    SELECT src.cliente_id, src.nome, src.estado, src.segmento
    FROM varejo.origem_cliente src
    LEFT JOIN varejo.dim_cliente_type1 cur ON cur.cliente_id = src.cliente_id
    WHERE ROW(cur.nome, cur.estado, cur.segmento)
          IS DISTINCT FROM ROW(src.nome, src.estado, src.segmento)
    ON CONFLICT (cliente_id) 
    DO UPDATE SET 
        nome = EXCLUDED.nome,
        estado = EXCLUDED.estado,
        segmento = EXCLUDED.segmento;
END;
$$;

-- ==============================================================================
-- SCD TYPE 2: HISTÓRICO COMPLETO COM JSONB
-- ==============================================================================

DROP TABLE IF EXISTS varejo.dim_cliente_type2 CASCADE;
CREATE TABLE varejo.dim_cliente_type2 (
    cliente_sk SERIAL PRIMARY KEY,        -- Surrogate key (gerada pelo DW)
    cliente_id INTEGER,                   -- Natural key (vinda do OLTP)
    nome VARCHAR(100),
    properties JSONB,                     -- Atributos variáveis (ex: estado, segmento)
    properties_diff JSONB,                -- Registro de deltas: {"estado": {"from":"SP","to":"RJ"}}
    data_inicio DATE,                     -- Início validade
    data_fim DATE,                        -- NULL = registro atual
    versao INTEGER,                       -- Número da versão
    ativo BOOLEAN,                        -- Flag atual
    UNIQUE (cliente_id, versao)
);

-- Função Auxiliar para calcular a Diferença de Propriedades (JSONB Diff)
CREATE OR REPLACE FUNCTION varejo.get_jsonb_diff(old_json JSONB, new_json JSONB)
RETURNS JSONB AS $$
DECLARE
    result JSONB := '{}'::JSONB;
    k TEXT;
    v JSONB;
BEGIN
    IF old_json IS NULL THEN old_json := '{}'::JSONB; END IF;
    IF new_json IS NULL THEN new_json := '{}'::JSONB; END IF;

    FOR k, v IN SELECT * FROM jsonb_each(new_json) LOOP
        IF old_json->k IS DISTINCT FROM v THEN
            result := jsonb_set(result, ARRAY[k], jsonb_build_object('from', old_json->k, 'to', v));
        END IF;
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Procedure Incremental SCD2
-- Toca APENAS nos clientes que mudaram ou são novos.
CREATE OR REPLACE PROCEDURE varejo.proc_upsert_clientes_scd2(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    WITH deltas AS (
        SELECT 
            src.cliente_id,
            src.nome,
            jsonb_build_object('estado', src.estado, 'segmento', src.segmento) AS new_props,
            tgt.properties AS old_props,
            tgt.cliente_sk AS old_sk
        FROM varejo.origem_cliente src
        LEFT JOIN varejo.dim_cliente_type2 tgt 
            ON tgt.cliente_id = src.cliente_id AND tgt.ativo = TRUE
        WHERE tgt.cliente_sk IS NULL
           OR tgt.properties IS DISTINCT FROM jsonb_build_object('estado', src.estado, 'segmento', src.segmento)
    ),
    fechados AS (
        UPDATE varejo.dim_cliente_type2 AS tgt
        SET data_fim = p_data_processamento - 1,
            ativo = FALSE
        FROM deltas d
        WHERE tgt.cliente_id = d.cliente_id
          AND tgt.ativo = TRUE
          AND d.old_sk IS NOT NULL
        RETURNING tgt.cliente_id
    )
    INSERT INTO varejo.dim_cliente_type2 (cliente_id, nome, properties, properties_diff, data_inicio, data_fim, versao, ativo)
    SELECT 
        d.cliente_id,
        d.nome,
        d.new_props,
        varejo.get_jsonb_diff(d.old_props, d.new_props),
        p_data_processamento,
        NULL::DATE,
        COALESCE((SELECT MAX(versao) FROM varejo.dim_cliente_type2 WHERE cliente_id = d.cliente_id), 0) + 1,
        TRUE
    FROM deltas d
    ON CONFLICT (cliente_id, versao) 
    DO UPDATE SET 
        nome = EXCLUDED.nome,
        properties = EXCLUDED.properties,
        properties_diff = EXCLUDED.properties_diff;
END;
$$;

-- ==============================================================================
-- GAP-AND-ISLAND: RECONSTRUÇÃO SCD2 A PARTIR DE SNAPSHOTS INCREMENTAIS
-- ==============================================================================
-- Em produção, a origem dos dados para reconstruir um SCD2 é um log CDC ou
-- um dump diário dos atributos. Simulamos isso com uma tabela de snapshots
-- alimentada incrementalmente pela mesma filosofia "Fast-Forward" do pipeline.

DROP TABLE IF EXISTS varejo.cliente_snapshot_diario CASCADE;
CREATE TABLE varejo.cliente_snapshot_diario (
    cliente_id   INTEGER  NOT NULL,
    data_snapshot DATE    NOT NULL,
    nome         VARCHAR(100),
    estado       VARCHAR(2),
    segmento     VARCHAR(50),
    PRIMARY KEY (cliente_id, data_snapshot)
);

-- Procedure Incremental: Captura o estado atual do OLTP como uma "foto" do dia.
-- Idempotente via ON CONFLICT (reprocessar o dia = sobrescrever a foto).
CREATE OR REPLACE PROCEDURE varejo.proc_snapshot_clientes(p_data DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO varejo.cliente_snapshot_diario (cliente_id, data_snapshot, nome, estado, segmento)
    SELECT cliente_id, p_data, nome, estado, segmento
    FROM varejo.origem_cliente
    ON CONFLICT (cliente_id, data_snapshot) DO UPDATE
        SET nome     = EXCLUDED.nome,
            estado   = EXCLUDED.estado,
            segmento = EXCLUDED.segmento;
END;
$$;

-- View Gap-And-Island: Reconstrói o SCD2 a partir dos snapshots brutos acumulados.
-- Nenhuma dependência da dim_cliente_type2 — prova que o padrão funciona do zero.
CREATE OR REPLACE VIEW varejo.mart_reconstrucao_scd2 AS
WITH streak_started AS (
    -- 1. Detecta a "quebra" (Gap) comparando o estado de hoje com o de ontem
    SELECT 
        cliente_id, nome, estado, segmento, data_snapshot,
        LAG(ROW(estado, segmento), 1) OVER (PARTITION BY cliente_id ORDER BY data_snapshot) 
        IS DISTINCT FROM ROW(estado, segmento) AS did_change
    FROM varejo.cliente_snapshot_diario
),
streak_identified AS (
    -- 2. Acumula a flag de mudança para criar um "Agrupamento" (Island) único por versão contínua
    SELECT *,
        SUM(CASE WHEN did_change THEN 1 ELSE 0 END) OVER (PARTITION BY cliente_id ORDER BY data_snapshot) as streak_id
    FROM streak_started
)
-- 3. Reagrupa as "Ilhas" em datas de início e fim recriando o SCD2!
SELECT 
    cliente_id,
    MAX(nome) as nome,
    jsonb_build_object('estado', MAX(estado), 'segmento', MAX(segmento)) AS properties,
    MIN(data_snapshot) AS data_inicio,
    CASE WHEN MAX(data_snapshot) = (SELECT MAX(data_snapshot) FROM varejo.cliente_snapshot_diario) 
         THEN NULL 
         ELSE MAX(data_snapshot) END AS data_fim,
    streak_id AS versao,
    CASE WHEN MAX(data_snapshot) = (SELECT MAX(data_snapshot) FROM varejo.cliente_snapshot_diario) 
         THEN TRUE 
         ELSE FALSE END AS ativo
FROM streak_identified
GROUP BY cliente_id, streak_id;

-- ==============================================================================
-- SCD TYPE 3: HISTÓRICO LIMITADO (COLUNA ANTERIOR - PRODUTO)
-- ==============================================================================

DROP TABLE IF EXISTS varejo.dim_produto_type3;
CREATE TABLE varejo.dim_produto_type3 (
    produto_id VARCHAR(20) PRIMARY KEY,
    nome_produto VARCHAR(200),
    categoria_atual VARCHAR(50),
    categoria_anterior VARCHAR(50),
    data_mudanca_categoria DATE
);

-- Procedure Incremental SCD3
CREATE OR REPLACE PROCEDURE varejo.proc_upsert_produtos_scd3(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO varejo.dim_produto_type3 (
        produto_id, nome_produto, categoria_atual, categoria_anterior, data_mudanca_categoria
    )
    SELECT 
        src.produto_id, 
        src.nome_produto, 
        src.categoria, 
        NULL,
        p_data_processamento
    FROM varejo.origem_produto src
    LEFT JOIN varejo.dim_produto_type3 cur ON cur.produto_id = src.produto_id
    WHERE cur.produto_id IS NULL
       OR cur.categoria_atual IS DISTINCT FROM src.categoria
    ON CONFLICT (produto_id) 
    DO UPDATE SET 
        categoria_anterior = CASE 
            WHEN varejo.dim_produto_type3.categoria_atual <> EXCLUDED.categoria_atual 
            THEN varejo.dim_produto_type3.categoria_atual 
            ELSE varejo.dim_produto_type3.categoria_anterior 
        END,
        data_mudanca_categoria = CASE 
            WHEN varejo.dim_produto_type3.categoria_atual <> EXCLUDED.categoria_atual 
            THEN p_data_processamento 
            ELSE varejo.dim_produto_type3.data_mudanca_categoria 
        END,
        categoria_atual = EXCLUDED.categoria_atual,
        nome_produto = EXCLUDED.nome_produto;
END;
$$;

-- ==============================================================================
-- TABELA FATO: PROCEDURE IDEMPOTENTE DE INGESTÃO
-- ==============================================================================
-- Padrão clássico para fatos de evento (vendas, cliques, transações):
-- Cada dia é uma "partição lógica". Reprocessar = deletar o dia e reinserir.
-- A Fato resolve as Surrogate Keys no momento da ingestão (Point-In-Time).

DROP TABLE IF EXISTS varejo.fato_vendas;
CREATE TABLE varejo.fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    data_venda DATE NOT NULL,
    produto_id VARCHAR(20),     -- NK do Produto (SCD3 usa NK como PK)
    cliente_sk INTEGER,         -- SK do Cliente (SCD2 gera Surrogate Keys)
    quantidade INTEGER NOT NULL,
    valor_total DECIMAL(10, 2) NOT NULL
);

CREATE OR REPLACE PROCEDURE varejo.proc_ingestao_fato_vendas(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    -- 1. Idempotência: Limpa a janela do dia
    DELETE FROM varejo.fato_vendas WHERE data_venda = p_data_processamento;

    -- 2. Insere os fatos do dia resolvendo a SK da Dimensão SCD2 via Point-In-Time
    INSERT INTO varejo.fato_vendas (data_venda, produto_id, cliente_sk, quantidade, valor_total)
    SELECT 
        src.data_venda,
        src.produto_id,            -- NK direta (SCD3 não gera SK)
        c.cliente_sk,              -- Surrogate Key resolvida via Point-In-Time!
        src.quantidade,
        src.valor_total
    FROM varejo.origem_venda src
    JOIN varejo.dim_cliente_type2 c
        ON c.cliente_id = src.cliente_id
       AND c.data_inicio <= src.data_venda
       AND (c.data_fim >= src.data_venda OR c.data_fim IS NULL)
    WHERE src.data_venda = p_data_processamento;
END;
$$;

-- ==============================================================================
-- FATO ACUMULADA (Cumulative Table - Yesterday + Today Pattern)
-- ==============================================================================
-- Comprime N linhas de evento em 1 array por cliente/snapshot.
-- Encapsulamos o pipeline Yesterday+Today numa Procedure parametrizada.

-- ==============================================================================
-- DATINT UTILITIES: Funções para Manipulação do Bitmap de 32 dias O(1)
-- ==============================================================================
CREATE OR REPLACE FUNCTION varejo.shift_activity_bitmap(current_bitmap BIGINT, is_active BOOLEAN)
RETURNS BIGINT LANGUAGE sql IMMUTABLE AS $$
    -- Desloca toda a máscara 1 bit para a direita usando Shift Bitwise genuíno ( >> 1)
    -- E acende o 31º bit (mais significativo) se houve atividade hoje.
    SELECT (COALESCE(current_bitmap, 0::BIGINT) >> 1) | CASE WHEN is_active THEN (1::BIGINT << 31) ELSE 0::BIGINT END;
$$;

CREATE OR REPLACE FUNCTION varejo.is_active_on_day(activity_bitmap_32d BIGINT, days_ago INT)
RETURNS BOOLEAN LANGUAGE sql IMMUTABLE AS $$
    -- days_ago: 0 = hoje (D0), 1 = ontem (D1), 7 = semana passada (D7)
    -- O bit mais recente fica na posição 31, recuando para a direita com o tempo.
    SELECT (activity_bitmap_32d & (1::BIGINT << (31 - days_ago))) > 0;
$$;

DROP TABLE IF EXISTS varejo.cliente_atividade_acumulada CASCADE;
CREATE TABLE varejo.cliente_atividade_acumulada (
    cliente_id        INTEGER  NOT NULL,
    data_snapshot     DATE     NOT NULL,
    datas_atividade   DATE[]   NOT NULL DEFAULT '{}',
    activity_bitmap_32d BIGINT NOT NULL DEFAULT 0,
    activity_bitmap_32m BIGINT NOT NULL DEFAULT 0,
    total_dias_ativos INTEGER  GENERATED ALWAYS AS (CARDINALITY(datas_atividade)) STORED,
    PRIMARY KEY (cliente_id, data_snapshot)
);

CREATE OR REPLACE PROCEDURE varejo.proc_acumular_atividade(p_data_ontem DATE, p_data_hoje DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    WITH yesterday AS (
        SELECT cliente_id, datas_atividade, activity_bitmap_32d, activity_bitmap_32m
        FROM varejo.cliente_atividade_acumulada
        WHERE data_snapshot = p_data_ontem
    ),
    today AS (
        -- Atividade = cliente realizou compra neste dia (derivado da fato_vendas)
        SELECT DISTINCT c.cliente_id, f.data_venda AS data_evento
        FROM varejo.fato_vendas f
        JOIN varejo.dim_cliente_type2 c ON c.cliente_sk = f.cliente_sk
        WHERE f.data_venda = p_data_hoje
    ),
    merged AS (
        SELECT
            COALESCE(y.cliente_id, t.cliente_id) AS cliente_id,
            p_data_hoje                          AS data_snapshot,
            COALESCE(y.datas_atividade, ARRAY[]::DATE[])
                || CASE
                     WHEN t.cliente_id IS NOT NULL
                     THEN ARRAY[t.data_evento]
                     ELSE ARRAY[]::DATE[]
                   END                           AS datas_atividade,
            -- Máscara Diária (Desliza e atualiza todo dia)
            varejo.shift_activity_bitmap(y.activity_bitmap_32d, t.cliente_id IS NOT NULL) AS activity_bitmap_32d,
            -- Máscara Mensal (Desliza SOMENTE na virada do mês, mas absorve vendas diárias do mês atual nela)
            (CASE WHEN EXTRACT(month FROM p_data_hoje) != EXTRACT(month FROM p_data_ontem) 
                  THEN COALESCE(y.activity_bitmap_32m, 0::BIGINT) >> 1 
                  ELSE COALESCE(y.activity_bitmap_32m, 0::BIGINT) 
             END) | CASE WHEN t.cliente_id IS NOT NULL THEN (1::BIGINT << 31) ELSE 0::BIGINT END AS activity_bitmap_32m
        FROM yesterday y
        FULL OUTER JOIN today t ON y.cliente_id = t.cliente_id
    )
    INSERT INTO varejo.cliente_atividade_acumulada (cliente_id, data_snapshot, datas_atividade, activity_bitmap_32d, activity_bitmap_32m)
    SELECT cliente_id, data_snapshot, datas_atividade, activity_bitmap_32d, activity_bitmap_32m
    FROM merged
    ON CONFLICT (cliente_id, data_snapshot) DO UPDATE
        SET datas_atividade = EXCLUDED.datas_atividade,
            activity_bitmap_32d = EXCLUDED.activity_bitmap_32d,
            activity_bitmap_32m = EXCLUDED.activity_bitmap_32m;
END;
$$;

-- ==============================================================================
-- POSITIONAL ARRAY METRICS & BITWISE RETENTION (DATINT)
-- ==============================================================================
-- 1. Positional Array Form (Monthly Array Metrics)
-- Arrays index-aligned to the day of the month for fast aggregation:

DROP TABLE IF EXISTS varejo.cliente_vendas_array_mensal;
CREATE TABLE varejo.cliente_vendas_array_mensal (
    cliente_id INTEGER,
    mes_referencia DATE,
    valores_diarios DECIMAL(10,2)[],
    valor_acumulado_mes DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    atividade_bitmap BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (cliente_id, mes_referencia)
);

-- Procedure to populate the monthly array (Daily execution)
CREATE OR REPLACE PROCEDURE varejo.proc_acumular_vendas_mensal(p_data_hoje DATE)
LANGUAGE plpgsql
AS $$
DECLARE
    v_primeiro_dia DATE;
    v_dia_do_mes INTEGER;
BEGIN
    v_primeiro_dia := DATE_TRUNC('month', p_data_hoje)::DATE;
    v_dia_do_mes := CAST(EXTRACT(DAY FROM p_data_hoje) AS INTEGER);

    WITH yesterday AS (
        SELECT cliente_id, valores_diarios, valor_acumulado_mes
        FROM varejo.cliente_vendas_array_mensal
        WHERE mes_referencia = v_primeiro_dia
    ),
    today AS (
        SELECT c.cliente_id, SUM(f.valor_total) as valor_dia
        FROM varejo.fato_vendas f
        JOIN varejo.dim_cliente_type2 c ON c.cliente_sk = f.cliente_sk
        WHERE f.data_venda = p_data_hoje
        GROUP BY c.cliente_id
    ),
    activity AS (
        SELECT cliente_id, activity_bitmap_32m
        FROM varejo.cliente_atividade_acumulada
        WHERE data_snapshot = p_data_hoje
    )
    INSERT INTO varejo.cliente_vendas_array_mensal (cliente_id, mes_referencia, valores_diarios, valor_acumulado_mes, atividade_bitmap)
    SELECT
        COALESCE(y.cliente_id, t.cliente_id) AS cliente_id,
        v_primeiro_dia AS mes_referencia,
        -- Extend array up to yesterday, then append today's value
        -- We pad missing days with 0.00 to guarantee position = day
        COALESCE(y.valores_diarios, '{}'::DECIMAL(10,2)[]) 
        || 
        CASE WHEN v_dia_do_mes - 1 - COALESCE(array_length(y.valores_diarios, 1), 0) > 0
             THEN array_fill(0.00::DECIMAL(10,2), ARRAY[v_dia_do_mes - 1 - COALESCE(array_length(y.valores_diarios, 1), 0)])
             ELSE '{}'::DECIMAL(10,2)[]
        END
        || ARRAY[COALESCE(t.valor_dia, 0.00)] AS valores_diarios,
        COALESCE(y.valor_acumulado_mes, 0.00) + COALESCE(t.valor_dia, 0.00) AS valor_acumulado_mes,
        COALESCE(a.activity_bitmap_32m, 0::BIGINT) AS atividade_bitmap
    FROM yesterday y
    FULL OUTER JOIN today t ON y.cliente_id = t.cliente_id
    LEFT JOIN activity a ON a.cliente_id = COALESCE(y.cliente_id, t.cliente_id)
    ON CONFLICT (cliente_id, mes_referencia) DO UPDATE
        SET valores_diarios = EXCLUDED.valores_diarios,
            valor_acumulado_mes = EXCLUDED.valor_acumulado_mes,
            atividade_bitmap = EXCLUDED.atividade_bitmap;
END;
$$;

-- Bitwise History (Datint Pattern for Retention Analysis)
-- Convert date arrays into bitmaps representing rolling 32-day retention
CREATE OR REPLACE VIEW varejo.dataviz_retencao_bitwise AS
SELECT 
    cliente_id,
    data_snapshot,
    datas_atividade,
    activity_bitmap_32d,
    activity_bitmap_32m
FROM varejo.cliente_atividade_acumulada;

-- ==============================================================================
-- CAMADA GOLD: TABELA OBT INCREMENTAL PARA DATAVIZ
-- ==============================================================================
-- Ferramentas de DataViz (Tableau, PowerBI) brilham com "One Big Tables" (OBT).
-- Materializamos métricas diárias de forma idempotente, consumindo os passos
-- anteriores do pipeline como um DAG (fato_vendas + cliente_atividade_acumulada).

DROP TABLE IF EXISTS varejo.gold_metricas_diarias CASCADE;
CREATE TABLE varejo.gold_metricas_diarias (
    data_ref DATE,
    segmento VARCHAR(50),
    daily_active_users INTEGER DEFAULT 0,
    cumulative_active_users INTEGER DEFAULT 0,
    day_revenue DECIMAL(10, 2) DEFAULT 0.00,
    PRIMARY KEY (data_ref, segmento)
);

-- Procedure Incremental da Camada Gold (Dependente da Cumulative Table + Fato)
CREATE OR REPLACE PROCEDURE varejo.proc_ingestao_gold_diaria(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM varejo.gold_metricas_diarias WHERE data_ref = p_data_processamento;

    WITH vendas AS (
        SELECT 
            CASE c.properties->>'segmento'
                WHEN 'Ouro' THEN 'premium'
                WHEN 'Prata' THEN 'standard'
                WHEN 'Bronze' THEN 'trial'
                ELSE 'trial'
            END as segmento,
            SUM(f.valor_total) as day_revenue
        FROM varejo.fato_vendas f
        JOIN varejo.dim_cliente_type2 c ON c.cliente_sk = f.cliente_sk
        WHERE f.data_venda = p_data_processamento
        GROUP BY 1
    ),
    atividade_acumulada AS (
        -- Consome o Pipeline Acumulado (DAG Dependency)
        SELECT 
            CASE c.properties->>'segmento'
                WHEN 'Ouro' THEN 'premium'
                WHEN 'Prata' THEN 'standard'
                WHEN 'Bronze' THEN 'trial'
                ELSE 'trial'
            END as segmento,
            COUNT(*) FILTER (WHERE p_data_processamento = ANY(a.datas_atividade)) as daily_active_users,
            COUNT(*) as total_active_users_to_date
        FROM varejo.cliente_atividade_acumulada a
        JOIN varejo.dim_cliente_type2 c ON c.cliente_id = a.cliente_id 
             AND c.data_inicio <= p_data_processamento 
             AND (c.data_fim >= p_data_processamento OR c.data_fim IS NULL)
        WHERE a.data_snapshot = p_data_processamento
        GROUP BY 1
    )
    INSERT INTO varejo.gold_metricas_diarias (data_ref, segmento, daily_active_users, cumulative_active_users, day_revenue)
    SELECT 
        p_data_processamento,
        COALESCE(v.segmento, a.segmento),
        COALESCE(a.daily_active_users, 0),
        COALESCE(a.total_active_users_to_date, 0),
        COALESCE(v.day_revenue, 0)
    FROM vendas v
    FULL OUTER JOIN atividade_acumulada a ON v.segmento = a.segmento;
END;
$$;

-- View leve sobre a Gold (apenas acumula receita via Window Function)
CREATE OR REPLACE VIEW varejo.mart_metricas_acumuladas AS
SELECT 
    data_ref,
    segmento,
    daily_active_users,
    cumulative_active_users,
    day_revenue,
    SUM(day_revenue) OVER (PARTITION BY segmento ORDER BY data_ref) as cumulative_revenue
FROM varejo.gold_metricas_diarias
ORDER BY data_ref, segmento;

-- View Avançada para Dashboard: Curva de Retenção (J-Curve)
-- Totaliza as métricas de coorte (D1, D3, D7, D14, D30) em O(1) usando a abstração do Bitmap (bitshifts escalares)
CREATE OR REPLACE VIEW varejo.mart_jcurve_retencao AS
SELECT 
    data_snapshot as data_coorte,
    -- Quantos usuários da base estavam ativos hoje (D0)?
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0)) as safra_total,
    -- D1: Presente no D0 e no D-1
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0) AND varejo.is_active_on_day(activity_bitmap_32d, 1)) as retidos_d1,
    -- D3: Presente no D0 e no D-3
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0) AND varejo.is_active_on_day(activity_bitmap_32d, 3)) as retidos_d3,
    -- D7: Presente no D0 e no D-7
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0) AND varejo.is_active_on_day(activity_bitmap_32d, 7)) as retidos_d7,
    -- D14: Presente no D0 e no D-14
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0) AND varejo.is_active_on_day(activity_bitmap_32d, 14)) as retidos_d14,
    -- D21: Presente no D0 e no D-21
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0) AND varejo.is_active_on_day(activity_bitmap_32d, 21)) as retidos_d21,
    -- D28: Presente no D0 e no D-28
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0) AND varejo.is_active_on_day(activity_bitmap_32d, 28)) as retidos_d28,
    -- D30: Presente no D0 e no D-30 (limite inferior da nossa janela de 32 bits)
    COUNT(cliente_id) FILTER (WHERE varejo.is_active_on_day(activity_bitmap_32d, 0) AND varejo.is_active_on_day(activity_bitmap_32d, 30)) as retidos_d30
FROM varejo.dataviz_retencao_bitwise
GROUP BY data_snapshot;

-- ==============================================================================
-- ORQUESTRAÇÃO: FAST-FORWARD INCREMENTAL (SIMULAÇÃO PRÁTICA)
-- ==============================================================================

-- Performance Tuning para o Ambiente de Simulação (Sessão)
-- Em produção, esses valores seriam configurados no postgresql.conf ou pelo Airflow.
-- Aqui, otimizamos para o cenário de rebuild total + Docker/WSL.
SET synchronous_commit = off;      -- Não espera flush do WAL a cada COMMIT (seguro: dados são rebuilds)
SET work_mem = '256MB';            -- Hash joins nas FULL OUTER JOINs cabem na RAM (evita disk spill)
SET maintenance_work_mem = '256MB'; -- CREATE INDEX mais rápido

-- Reset do Estado Inicial (garantindo reproducibilidade)
DO $$ BEGIN RAISE NOTICE '🔄 Resetando o estado do OLTP e limpando o DW...'; END $$;

-- CRITICAL PERFORMANCE FIX: Índices B-Tree para os Joins e Filtros Temporais.
-- Sem isso, o laço temporal causa Table Scans massivos O(N^2) escalando para +10 minutos!
CREATE INDEX IF NOT EXISTS idx_fato_data ON varejo.fato_vendas(data_venda);
CREATE INDEX IF NOT EXISTS idx_ativ_acum_data ON varejo.cliente_atividade_acumulada(data_snapshot);
CREATE INDEX IF NOT EXISTS idx_vendas_arr_mes ON varejo.cliente_vendas_array_mensal(mes_referencia);
CREATE INDEX IF NOT EXISTS idx_snap_diario_data ON varejo.cliente_snapshot_diario(data_snapshot);
-- Hot-path: a fato_vendas filtra origem_venda por data 150k+ linhas sem índice = Seq Scan!
CREATE INDEX IF NOT EXISTS idx_origem_venda_data ON varejo.origem_venda(data_venda);
-- Hot-path: o JOIN Point-In-Time filtra dim_cliente_type2 pelo cliente ativo
CREATE INDEX IF NOT EXISTS idx_dim_cli2_ativo ON varejo.dim_cliente_type2(cliente_id) WHERE ativo = TRUE;

-- Tabela de metadados para armazenar os tempos de execução do pipeline em milisegundos
CREATE TABLE IF NOT EXISTS varejo.pipeline_telemetry (
    data_pipeline DATE,
    pipeline_name VARCHAR(100),
    step_name VARCHAR(100),
    duration_ms NUMERIC
);

UPDATE varejo.origem_cliente SET estado = 'SP' WHERE cliente_id = 101;
UPDATE varejo.origem_produto SET categoria = 'Informática' WHERE produto_id = 'PROD001';

-- Limpa tabelas DW de destino (nunca a origem OLTP!)
TRUNCATE varejo.dim_cliente_type1 CASCADE;
TRUNCATE varejo.dim_cliente_type2 RESTART IDENTITY CASCADE;
TRUNCATE varejo.dim_produto_type3 CASCADE;
TRUNCATE varejo.fato_vendas RESTART IDENTITY CASCADE;
TRUNCATE varejo.cliente_atividade_acumulada CASCADE;
TRUNCATE varejo.cliente_vendas_array_mensal CASCADE;
TRUNCATE varejo.cliente_snapshot_diario CASCADE;
TRUNCATE varejo.gold_metricas_diarias CASCADE;
TRUNCATE varejo.pipeline_telemetry CASCADE;

-- ==============================================================================
-- ⚙️ O MAESTRO DA ORQUESTRAÇÃO (THE AIRFLOW BOUNDARY)
-- ==============================================================================
-- Tudo que fizemos ATÉ AQUI (DDL, Views, INSERT ON CONFLICT, Arrays e Bitmaps)...
-- ... na Vida Real é o "DBT RESPONSIBILITY" (Aplicações limpas de transformações SQL).
-- 
-- A PROCEDURE ORQUESTRADORA ABAIXO E O SEU LOOP TEMPORAL DE DATAS...
-- ... na Vida Real são atributos do "AIRFLOW RESPONSIBILITY" (A Orquestração de DAGs).
-- Ele é quem chama o dbt passando a variável {{ ds }} e comanda a ordem de preenchimento.
-- Função auxiliar limpa e reaproveitável em todo o DW para registrar telemetria de performance
CREATE OR REPLACE PROCEDURE varejo.proc_log_telemetry(p_data DATE, p_pipeline VARCHAR, p_step VARCHAR, p_start_ts TIMESTAMP)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO varejo.pipeline_telemetry (data_pipeline, pipeline_name, step_name, duration_ms)
    VALUES (p_data, p_pipeline, p_step, EXTRACT(EPOCH FROM clock_timestamp() - p_start_ts) * 1000);
END;
$$;

-- Capsula todo o fluxo diário numa única transação procedural (simulando um DAG Run).
CREATE OR REPLACE PROCEDURE varejo.proc_executar_pipeline_diario(p_data DATE)
LANGUAGE plpgsql
AS $$
DECLARE
    ts TIMESTAMP;
    v_pipeline VARCHAR(100) := 'proc_executar_pipeline_diario';
BEGIN
    ts := clock_timestamp(); CALL varejo.proc_snapshot_clientes(p_data);
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '1_snapshot_clientes', ts);

    ts := clock_timestamp(); CALL varejo.proc_upsert_clientes_scd1();
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '2_upsert_clientes_scd1', ts);

    ts := clock_timestamp(); CALL varejo.proc_upsert_clientes_scd2(p_data);
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '3_upsert_clientes_scd2', ts);

    ts := clock_timestamp(); CALL varejo.proc_upsert_produtos_scd3(p_data);
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '4_upsert_produtos_scd3', ts);

    ts := clock_timestamp(); CALL varejo.proc_ingestao_fato_vendas(p_data);
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '5_ingestao_fato_vendas', ts);

    ts := clock_timestamp(); CALL varejo.proc_acumular_atividade(p_data - 1, p_data);
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '6_acumular_atividade', ts);

    ts := clock_timestamp(); CALL varejo.proc_acumular_vendas_mensal(p_data);
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '7_acumular_vendas_mes', ts);

    ts := clock_timestamp(); CALL varejo.proc_ingestao_gold_diaria(p_data);
    CALL varejo.proc_log_telemetry(p_data, v_pipeline, '8_ingestao_gold_diaria', ts);
END;
$$;

-- 1. Carga ETL Dia 1 (2024-05-04) — Estado inicial do OLTP
DO $$ BEGIN RAISE NOTICE '📥 1. Executando Carga Inicial do DW (Dia 2024-05-04)...'; END $$;

CALL varejo.proc_executar_pipeline_diario('2024-05-04');

-- 2. Mudança no OLTP: João muda de Estado, Notebook vira Gamer
DO $$ BEGIN RAISE NOTICE '📝 2. Simulando mudanças no OLTP (Ex: João muda de UF, Produto muda de Categoria)...'; END $$;
UPDATE varejo.origem_cliente SET estado = 'PR' WHERE cliente_id = 101;
UPDATE varejo.origem_produto SET categoria = 'Gamer' WHERE produto_id = 'PROD001';

-- 3. Carga Incremental Dia 2 (2024-05-05)
DO $$ BEGIN RAISE NOTICE '📥 3. Executando Carga Incremental (Dia 2024-05-05)...'; END $$;

CALL varejo.proc_executar_pipeline_diario('2024-05-05');

-- 4. Fast-Forward (Backfill) simulando 2 meses de carga
-- Criamos uma procedure envolvente para podermos realizar o COMMIT a cada dia.
-- Isso previne um estrangulamento de Transação (WAL Tracking) gigantesco no Postgres.
CREATE OR REPLACE PROCEDURE varejo.proc_run_fast_forward()
LANGUAGE plpgsql
AS $$
DECLARE
    dt DATE;
    ts_start TIMESTAMP;
    ts_dia TIMESTAMP;
    dur_backfill INTERVAL;
    dur_dia INTERVAL;
BEGIN
    RAISE NOTICE '🚀 4. Iniciando Fast-Forward de 2 meses (Simulação contínua rodando dia a dia)...';
    ts_start := clock_timestamp();
    FOR dt IN SELECT generate_series('2024-05-06'::DATE, '2024-07-04'::DATE, '1 day'::INTERVAL)::DATE LOOP
        ts_dia := clock_timestamp();
        CALL varejo.proc_executar_pipeline_diario(dt);
        COMMIT; -- Libera I/O e a transação log (WAL) para cada dia individualmente!
        dur_dia := clock_timestamp() - ts_dia;
        RAISE NOTICE '   📅 Dia % completado em %', dt, dur_dia;
    END LOOP;
    dur_backfill := clock_timestamp() - ts_start;
    RAISE NOTICE '✅ Fast-Forward concluído com sucesso em %!', dur_backfill;
    PERFORM set_config('varejo.dur_backfill', dur_backfill::text, false);
END;
$$;

CALL varejo.proc_run_fast_forward();


-- ==============================================================================
-- AUDITORIA: VERIFICANDO O COMPORTAMENTO "AS-WAS" E PROGRESSÃO DIMENSIONAL
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- PARTE A. AS BASES DIMENSIONAIS
-- ------------------------------------------------------------------------------
-- Parametrizando o cliente focado na demonstração:
\set p_cliente_id 101

-- A.1 O Cliente (SCD1 e SCD2: Observe a preservação x perda de histórico)
SELECT * FROM varejo.dim_cliente_type1 WHERE cliente_id = :p_cliente_id;
SELECT * FROM varejo.dim_cliente_type2 WHERE cliente_id = :p_cliente_id ORDER BY versao;

-- A.2 O Produto (SCD3: Histórico Colunar)
SELECT * FROM varejo.dim_produto_type3 WHERE produto_id = 'PROD001';

-- ------------------------------------------------------------------------------
-- PARTE B. TRANSAÇÕES E RESOLUÇÃO POINT-IN-TIME
-- ------------------------------------------------------------------------------
-- B.1 A Fato e a Mágica do Point-In-Time (SCD2)
-- Observe como as vendas do dia 04 e 05 se ligam a SKs diferentes, 
-- preservando o "estado_venda" correto da época, não o atual!
SELECT 
    f.venda_id, 
    f.data_venda, 
    c.nome, 
    c.properties->>'estado' as estado_venda
FROM varejo.fato_vendas f
JOIN varejo.dim_cliente_type2 c 
    ON c.cliente_sk = f.cliente_sk
WHERE f.data_venda IN ('2024-05-04', '2024-05-05')
  AND c.cliente_id = :p_cliente_id
ORDER BY f.venda_id;

-- ------------------------------------------------------------------------------
-- PARTE C. MODELAGEM COMPORTAMENTAL E COMPRESSÃO (ARRAYS E DATINT)
-- ------------------------------------------------------------------------------
-- C.1 A Cumulative Table (O Padrão Clássico: Dias Ativos em Array)
SELECT 
    cliente_id, 
    data_snapshot, 
    total_dias_ativos, 
    datas_atividade[1:5] as primeiros_5_dias
FROM varejo.cliente_atividade_acumulada 
WHERE cliente_id = :p_cliente_id
ORDER BY data_snapshot DESC ;

-- C.2 Positional Array Form (Evolução: Métricas indexadas ao dia do mês)
-- Enriquecido com a máscara mensal rolling de 32 meses e receita acumulada
SELECT 
    mes_referencia,
    valores_diarios, 
    atividade_bitmap::BIT(32) as mes_bitmap_visual,
    valor_acumulado_mes
FROM varejo.cliente_vendas_array_mensal 
WHERE cliente_id = :p_cliente_id
ORDER BY mes_referencia DESC;

-- C.3 Retenção Bitwise (Datint Pattern: Retenção rolling 32d como um bitmap)
-- Parte A: Visualizando o BitMap diário gerado pelo cruzamento (0s e 1s)
SELECT 
    cliente_id, 
    data_snapshot, 
    activity_bitmap_32d, 
    activity_bitmap_32d::BIT(32) as bitmap_visual
FROM varejo.dataviz_retencao_bitwise 
WHERE cliente_id = :p_cliente_id 
ORDER BY data_snapshot DESC;

-- Parte B: O "Pulo do Gato" - Usando Álgebra Booleana (Bitwise AND) para Retenção Analítica!
-- Pergunta de Negócio: "Quantos clientes acessaram HOJE (D0) e também há exatos 7 DIAS (D7)?"
-- Ao invés de fazer JOINs custosos com 7 dias atrás, calculamos a retenção varrendo os INTs em O(1):
CREATE OR REPLACE VIEW varejo.mart_retencao_d7 AS
SELECT 
    data_snapshot,
    COUNT(cliente_id) as dau_total,
    -- O & binário cruza a máscara; se o bit bater, ele retorna > 0
    COUNT(cliente_id) FILTER (
        WHERE varejo.is_active_on_day(activity_bitmap_32d, 0)
          AND varejo.is_active_on_day(activity_bitmap_32d, 7)
    ) as retidos_d7
FROM varejo.dataviz_retencao_bitwise
GROUP BY data_snapshot;

SELECT * FROM varejo.mart_retencao_d7 
ORDER BY data_snapshot DESC 
LIMIT 5;

-- ------------------------------------------------------------------------------
-- PARTE D. CAMADA AGREGADA (ONE BIG TABLE - OBT)
-- ------------------------------------------------------------------------------
-- D.1 Dashboard Ouro
SELECT * 
FROM varejo.mart_metricas_acumuladas
ORDER BY segmento, data_ref;

-- D.2 Visão de Produto (The J-Curve / Retention Dashboard)
-- Lendo a mart view e formatando as métricas para plotagem na ferramenta gráfica (em percentuais)
CREATE OR REPLACE VIEW varejo.mart_jcurve_formatado AS
SELECT 
    data_coorte,
    safra_total,
    ROUND(retidos_d1 * 100.0 / NULLIF(safra_total, 0), 2) as d1_pct,
    ROUND(retidos_d3 * 100.0 / NULLIF(safra_total, 0), 2) as d3_pct,
    ROUND(retidos_d7 * 100.0 / NULLIF(safra_total, 0), 2) as d7_pct,
    ROUND(retidos_d14 * 100.0 / NULLIF(safra_total, 0), 2) as d14_pct,
    ROUND(retidos_d21 * 100.0 / NULLIF(safra_total, 0), 2) as d21_pct,
    ROUND(retidos_d28 * 100.0 / NULLIF(safra_total, 0), 2) as d28_pct,
    ROUND(retidos_d30 * 100.0 / NULLIF(safra_total, 0), 2) as d30_pct
FROM varejo.mart_jcurve_retencao;

SELECT * FROM varejo.mart_jcurve_formatado 
ORDER BY data_coorte DESC 
LIMIT 5;

-- ==============================================================================
-- PARTE E. TELEMETRIA E DAG PROFILING
-- ==============================================================================
-- Resumo estatístico do comportamento das procedures ao longo dos dias, por Quartil.
WITH daily_totals AS (
    SELECT 
        data_pipeline,
        pipeline_name,
        NULL::VARCHAR as step_name,
        SUM(duration_ms) as duration_ms
    FROM varejo.pipeline_telemetry
    GROUP BY data_pipeline, pipeline_name
),
combined_telemetry AS (
    SELECT data_pipeline, pipeline_name, step_name, duration_ms
    FROM varejo.pipeline_telemetry
    UNION ALL
    SELECT data_pipeline, pipeline_name, step_name, duration_ms
    FROM daily_totals
),
aggregated AS (
    SELECT 
        pipeline_name,
        COALESCE(step_name, '--- TOTAL BACKFILL ---') as step_name,
        COUNT(*) as num_execucoes,
        SUM(duration_ms) as sum_ms,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC AS p25_ms,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC AS p50_mediana_ms,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC AS p75_ms,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC AS p90_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC AS p95_ms,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms)::NUMERIC AS p99_ms
    FROM combined_telemetry
    GROUP BY pipeline_name, COALESCE(step_name, '--- TOTAL BACKFILL ---')
)
SELECT 
    pipeline_name,
    step_name,
    num_execucoes,
    ROUND(sum_ms, 2) as sum_ms,
    ROUND((sum_ms / MAX(sum_ms) OVER(PARTITION BY pipeline_name)) * 100, 2) as pct_total,
    ROUND(p25_ms, 2) as p25_ms,
    ROUND(p50_mediana_ms, 2) as p50_mediana_ms,
    ROUND(p75_ms, 2) as p75_ms,
    ROUND(p90_ms, 2) as p90_ms,
    ROUND(p95_ms, 2) as p95_ms,
    ROUND(p99_ms, 2) as p99_ms
FROM aggregated
ORDER BY pipeline_name, step_name;

-- ==============================================================================
-- 6. FINAL DURATION REPORT: PROFILING DATA TRANSFORMATIONS
-- ==============================================================================
DO $$
DECLARE
    ts_start TIMESTAMP;
    dur_gap_island INTERVAL;
    dur_bitwise_d7 INTERVAL;
    dur_obt INTERVAL;
    dur_jcurve INTERVAL;
    dur_backfill INTERVAL;
BEGIN
    -- 0. Retrieve Backfill Pipeline Duration (Fast-Forward)
    dur_backfill := current_setting('varejo.dur_backfill', true)::interval;

    -- 1. Gap-And-Island Batch Regeneration (SCD2 do ZERO)
    ts_start := clock_timestamp();
    PERFORM * FROM varejo.mart_reconstrucao_scd2;
    dur_gap_island := clock_timestamp() - ts_start;

    -- 2. Bitwise D7 Retention Filter
    ts_start := clock_timestamp();
    PERFORM * FROM varejo.mart_retencao_d7;
    dur_bitwise_d7 := clock_timestamp() - ts_start;

    -- 3. OBT Dashboard
    ts_start := clock_timestamp();
    PERFORM * FROM varejo.mart_metricas_acumuladas;
    dur_obt := clock_timestamp() - ts_start;

    -- 4. J-Curve Cohort Dashboard
    ts_start := clock_timestamp();
    PERFORM * FROM varejo.mart_jcurve_formatado;
    dur_jcurve := clock_timestamp() - ts_start;

    RAISE NOTICE '=========================================================';
    RAISE NOTICE '📊 FINAL DURATION REPORT: DATA TRANSFORMATIONS 📊';
    RAISE NOTICE '=========================================================';
    RAISE NOTICE '0. Fast-Forward Pipeline (2 Months Load): %', COALESCE(dur_backfill, '00:00:00'::interval);
    RAISE NOTICE '1. Gap-And-Island (SCD2 Reconstruction): %', dur_gap_island;
    RAISE NOTICE '2. Bitwise Boolean Algebra (D7 O(1)):    %', dur_bitwise_d7;
    RAISE NOTICE '3. Cumulative Revenue (OBT Dashboard):   %', dur_obt;
    RAISE NOTICE '4. J-Curve (Percentage Cohorts):         %', dur_jcurve;
    RAISE NOTICE '=========================================================';
END $$;
