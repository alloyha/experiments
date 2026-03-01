-- ==============================================================================
-- Aula 5: Tabelas Fato vs. Tabelas Cumulativas (Big Data)
-- ==============================================================================

drop table fato_vendas;
drop table fato_estoque_diario;
drop table fato_pedido_pipeline;

-- ------------------------------------------------------------------------------
-- ABORDAGEM CLÁSSICA (KIMBALL) - FACT TABLES
-- ------------------------------------------------------------------------------
-- Padrão de Ouro para BI e Relatórios
-- Cada evento é uma linha, agregamos no Read-Time.

-- 1. Fato Transacional (Transaction Fact)
CREATE TABLE IF NOT EXISTS fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    data_venda DATE NOT NULL,
    cliente_id INTEGER NOT NULL,
    valor_total DECIMAL(10, 2)
);

-- ------------------------------------------------------------------------------
-- TIPOS DE FATO ADICIONAIS
-- ------------------------------------------------------------------------------

-- 4. Fato Snapshot Periódico (Periodic Snapshot)
-- Granularidade: estado diário/mensal (Estoque)
CREATE TABLE IF NOT EXISTS fato_estoque_diario (
    snapshot_id SERIAL PRIMARY KEY,
    data_snapshot DATE,
    produto_id INTEGER,
    quantidade_estoque INTEGER
);

-- 5. Fato Acumulativo (Accumulating Snapshot)
-- Acompanha um processo/pipeline (Pedido -> Entrega)
CREATE TABLE IF NOT EXISTS fato_pedido_pipeline (
    pedido_id INTEGER PRIMARY KEY,
    data_pedido DATE,
    data_pagamento DATE,
    data_entrega DATE
);

-- ------------------------------------------------------------------------------
-- DEMONSTRAÇÃO: SÍNTESE DE 100.000 VENDAS (Fato Transacional)
-- ------------------------------------------------------------------------------
-- Simulamos 100k vendas distribuídas aleatoriamente nos últimos 30 dias.

INSERT INTO fato_vendas (data_venda, cliente_id, valor_total)
SELECT 
    ('2024-06-03'::DATE - (random() * 30)::INT * INTERVAL '1 day')::DATE as data_venda,
    (random() * 10000 + 1)::INT as cliente_id,
    (random() * 500 + 10)::DECIMAL(10,2) as valor_total
FROM generate_series(1, 100000) AS id;


-- Problema em Escala: "Quantos dias o usuário X comprou no último ano?"
-- Exige SCAN na tabela inteira de fatos (potencialmente bilhões de linhas)
SELECT count(distinct data_venda)
FROM fato_vendas
WHERE cliente_id = 123 AND data_venda > '2023-01-01';

-- ------------------------------------------------------------------------------
-- ESTRATÉGIAS DE CARGA (MOVIMENTAÇÃO DE DADOS)
-- ------------------------------------------------------------------------------
-- Em Big Data, como o dado "entra" na tabela define a performance e idempotência.
--
-- 1. APPEND (Incremental): Apenas adiciona novos eventos. Padrão de Tabelas Fato.
-- 2. OVERWRITE (Full Refresh): DELETE + INSERT. Limpa e recarrega. Simples, mas caro.
-- 3. UPSERT (Update or Insert): INSERT ON CONFLICT. O coração das Cumulative Tables.
-- ------------------------------------------------------------------------------

-- ==============================================================================
-- 0. SETUP: LIMPEZA, EXTENSÕES E VARIÁVEIS
-- ==============================================================================

DROP TABLE IF EXISTS usuarios_atividade_fato;
DROP TABLE IF EXISTS usuarios_atividade_acumulada;
DROP TABLE IF EXISTS usuarios_dim;

-- Configuração de Datas (Parametrização para a aula)
-- Usamos sintaxe de psql para variáveis
\set data_ontem '''2024-05-05'''
\set data_hoje  '''2024-05-06'''

-- Garante reprodutibilidade do dataset (seed fixo)
-- Nota: Em Postgres, SET seed afeta random() na sessão inteira
SELECT setseed(0.42);


-- ==============================================================================
-- 1. DIMENSÃO DE USUÁRIOS (lookup table)
-- Representa 10.000 usuários únicos com perfis realistas
-- ==============================================================================

CREATE TABLE IF NOT EXISTS usuarios_dim (
    usuario_id   INTEGER      PRIMARY KEY,
    nome         TEXT         NOT NULL,
    segmento     TEXT         NOT NULL,   -- 'premium' | 'standard' | 'trial'
    data_cadastro DATE        NOT NULL
);

INSERT INTO usuarios_dim (usuario_id, nome, segmento, data_cadastro)
SELECT
    gs                                                          AS usuario_id,
    'Usuario_' || LPAD(gs::TEXT, 6, '0')                       AS nome,
    CASE
        WHEN random() < 0.15 THEN 'premium'
        WHEN random() < 0.55 THEN 'standard'
        ELSE 'trial'
    END                                                         AS segmento,
    -- Cadastros espalhados nos últimos 2 anos
    CURRENT_DATE - (random() * 730)::INT                        AS data_cadastro
FROM generate_series(1, 10000) AS gs;


-- ==============================================================================
-- 2. TABELA FATO — APPEND (Granularidade: 1 linha por evento/dia/usuário)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS usuarios_atividade_fato (
    usuario_id   INTEGER  NOT NULL,
    data_evento  DATE     NOT NULL,
    PRIMARY KEY (usuario_id, data_evento)
);

-- ------------------------------------------------------------------------------
-- CARGA HISTÓRICA: 30 dias de atividade (2024-05-05 a 2024-06-03)
-- Simula ~40% de DAU (Daily Active Users) sobre 10.000 usuários
-- Cada execução é idempotente graças ao ON CONFLICT DO NOTHING
-- ------------------------------------------------------------------------------

INSERT INTO usuarios_atividade_fato (usuario_id, data_evento)
SELECT DISTINCT
    (random() * 9999 + 1)::INT   AS usuario_id,
    (DATE '2024-05-05' + (gs % 30))::DATE  AS data_evento
FROM generate_series(1, 120000) AS gs   -- ~4.000 eventos únicos/dia × 30 dias
ON CONFLICT DO NOTHING;

-- Injeção determinística: garante usuário 1 ativo em todos os 30 dias
INSERT INTO usuarios_atividade_fato (usuario_id, data_evento)
SELECT 1, DATE '2024-05-05' + i
FROM generate_series(0, 29) AS i
ON CONFLICT DO NOTHING;

-- Injeção: usuário 2 inativo nos últimos 7 dias (churn detection test)
INSERT INTO usuarios_atividade_fato (usuario_id, data_evento)
SELECT 2, DATE '2024-05-05' + i
FROM generate_series(0, 22) AS i
ON CONFLICT DO NOTHING;

-- Injeção: usuário 3 apenas no dia de hoje (new user test)
INSERT INTO usuarios_atividade_fato (usuario_id, data_evento)
VALUES (3, '2024-06-03')
ON CONFLICT DO NOTHING;


-- ==============================================================================
-- 3. TABELA ACUMULADA — UPSERT (Cumulative Table Pattern)
-- Comprime N linhas de fato em 1 array por usuário/snapshot
-- Ideal para: contagem de dias ativos, retention, cohort analysis
-- ==============================================================================

CREATE TABLE IF NOT EXISTS usuarios_atividade_acumulada (
    usuario_id        INTEGER  NOT NULL,
    data_snapshot     DATE     NOT NULL,
    datas_atividade   DATE[]   NOT NULL DEFAULT '{}',
    -- Campos derivados para evitar recalculo em queries
    total_dias_ativos INTEGER  GENERATED ALWAYS AS (CARDINALITY(datas_atividade)) STORED,
    PRIMARY KEY (usuario_id, data_snapshot)
);


-- ------------------------------------------------------------------------------
-- PASSO 1: Carga do "estado de ontem" (2024-06-02)
-- Agrega todos os eventos históricos até 2024-06-02
-- ------------------------------------------------------------------------------

TRUNCATE TABLE usuarios_atividade_acumulada;

INSERT INTO usuarios_atividade_acumulada (usuario_id, data_snapshot, datas_atividade)
SELECT
    f.usuario_id,
    :data_ontem::DATE                                     AS data_snapshot,
    ARRAY_AGG(f.data_evento ORDER BY f.data_evento)       AS datas_atividade
FROM usuarios_atividade_fato f
WHERE f.data_evento <= :data_ontem::DATE
GROUP BY f.usuario_id
ON CONFLICT (usuario_id, data_snapshot) DO UPDATE
    SET datas_atividade = EXCLUDED.datas_atividade;


-- ------------------------------------------------------------------------------
-- PASSO 2: Pipeline incremental "Yesterday + Today" → snapshot 2024-06-03
-- Este é o coração do padrão: lê apenas o delta do dia, faz merge no array
-- ------------------------------------------------------------------------------

WITH yesterday AS (
    -- Estado comprimido do dia anterior (leitura de 1 linha por usuário)
    SELECT usuario_id, datas_atividade
    FROM usuarios_atividade_acumulada
    WHERE data_snapshot = :data_ontem::DATE
),
today AS (
    -- Delta: apenas eventos novos do dia corrente (APPEND da fato)
    SELECT usuario_id, data_evento
    FROM usuarios_atividade_fato
    WHERE data_evento = :data_hoje::DATE
),
merged AS (
    SELECT
        COALESCE(y.usuario_id, t.usuario_id)              AS usuario_id,
        :data_hoje::DATE                                  AS data_snapshot,
        COALESCE(y.datas_atividade, ARRAY[]::DATE[])
            || CASE
                 WHEN t.usuario_id IS NOT NULL
                 THEN ARRAY[t.data_evento]
                 ELSE ARRAY[]::DATE[]
               END                                        AS datas_atividade
    FROM yesterday y
    FULL OUTER JOIN today t ON y.usuario_id = t.usuario_id
)
INSERT INTO usuarios_atividade_acumulada (usuario_id, data_snapshot, datas_atividade)
SELECT usuario_id, data_snapshot, datas_atividade
FROM merged
ON CONFLICT (usuario_id, data_snapshot) DO UPDATE
    SET datas_atividade = EXCLUDED.datas_atividade;


-- ==============================================================================
-- 4. QUERIES DE VALIDAÇÃO E ANÁLISE
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- V1. Sanidade do dataset: volume por camada
-- ------------------------------------------------------------------------------
SELECT
    'usuarios_dim'               AS tabela,
    COUNT(*)                     AS total_linhas,
    NULL::NUMERIC                AS media_dias_ativos
FROM usuarios_dim
UNION ALL
SELECT
    'usuarios_atividade_fato',
    COUNT(*),
    NULL
FROM usuarios_atividade_fato
UNION ALL
SELECT
    'usuarios_atividade_acumulada (snapshot ' || :data_hoje || ')',
    COUNT(*),
    ROUND(AVG(total_dias_ativos), 2)
FROM usuarios_atividade_acumulada
WHERE data_snapshot = :data_hoje::DATE;


-- ------------------------------------------------------------------------------
-- V2. Consulta O(1): dias ativos de um usuário (sem histórico scan)
-- A coluna GENERATED evita CARDINALITY em runtime
-- ------------------------------------------------------------------------------
SELECT
    a.usuario_id,
    d.segmento,
    a.total_dias_ativos,
    a.datas_atividade[1]                                  AS primeira_atividade,
    a.datas_atividade[CARDINALITY(a.datas_atividade)]     AS ultima_atividade
FROM usuarios_atividade_acumulada a
JOIN usuarios_dim d ON d.usuario_id = a.usuario_id
WHERE a.data_snapshot = :data_hoje::DATE
  AND a.usuario_id IN (1, 2, 3);   -- casos determinísticos injetados


-- ------------------------------------------------------------------------------
-- V3. Distribuição de engajamento (retention bucket)
-- Classificação sem scan de histórico — lê só o snapshot mais recente
-- ------------------------------------------------------------------------------
SELECT
    d.segmento,
    CASE
        WHEN a.total_dias_ativos >= 25 THEN 'Power User (25-30d)'
        WHEN a.total_dias_ativos >= 14 THEN 'Regular (14-24d)'
        WHEN a.total_dias_ativos >= 7  THEN 'Casual (7-13d)'
        WHEN a.total_dias_ativos >= 1  THEN 'Dormant (1-6d)'
        ELSE                                'Inativo'
    END                                                   AS bucket,
    COUNT(*)                                              AS usuarios,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY d.segmento), 1) AS pct_segmento
FROM usuarios_atividade_acumulada a
JOIN usuarios_dim d ON d.usuario_id = a.usuario_id
WHERE a.data_snapshot = :data_hoje::DATE
GROUP BY d.segmento, bucket
ORDER BY d.segmento, usuarios DESC;


-- ------------------------------------------------------------------------------
-- V4. Churn Alert: usuários inativos nos últimos 7 dias
-- Usando array_position para checar última data sem explodir linhas
-- ------------------------------------------------------------------------------
SELECT
    a.usuario_id,
    d.segmento,
    a.total_dias_ativos,
    a.datas_atividade[CARDINALITY(a.datas_atividade)]     AS ultima_atividade,
    :data_hoje::DATE
        - a.datas_atividade[CARDINALITY(a.datas_atividade)] AS dias_sem_acesso
FROM usuarios_atividade_acumulada a
JOIN usuarios_dim d ON d.usuario_id = a.usuario_id
WHERE a.data_snapshot = :data_hoje::DATE
  AND a.datas_atividade[CARDINALITY(a.datas_atividade)] < :data_hoje::DATE - 6
ORDER BY dias_sem_acesso DESC
LIMIT 20;


-- ------------------------------------------------------------------------------
-- V5. Idempotência: reexecutar a carga não duplica dados
-- Resultado deve ser idêntico ao anterior
-- ------------------------------------------------------------------------------
WITH yesterday AS (
    SELECT usuario_id, datas_atividade
    FROM usuarios_atividade_acumulada
    WHERE data_snapshot = :data_ontem::DATE
),
today AS (
    SELECT usuario_id, data_evento
    FROM usuarios_atividade_fato
    WHERE data_evento = :data_hoje::DATE
)
INSERT INTO usuarios_atividade_acumulada (usuario_id, data_snapshot, datas_atividade)
SELECT
    COALESCE(y.usuario_id, t.usuario_id),
    :data_hoje::DATE,
    COALESCE(y.datas_atividade, ARRAY[]::DATE[])
        || CASE WHEN t.usuario_id IS NOT NULL THEN ARRAY[t.data_evento] ELSE ARRAY[]::DATE[] END
FROM yesterday y
FULL OUTER JOIN today t ON y.usuario_id = t.usuario_id
ON CONFLICT (usuario_id, data_snapshot) DO UPDATE
    SET datas_atividade = EXCLUDED.datas_atividade;

-- Contagem deve ser igual à execução anterior
SELECT COUNT(*), MAX(total_dias_ativos), MIN(total_dias_ativos)
FROM usuarios_atividade_acumulada
WHERE data_snapshot = :data_hoje::DATE;
