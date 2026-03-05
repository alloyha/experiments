-- ==============================================================================
-- Aula 5: Tabelas Fato vs. Tabelas Cumulativas (Big Data)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- ABORDAGEM CLÁSSICA (KIMBALL) - FACT TABLES
-- ------------------------------------------------------------------------------
-- Padrão de Ouro para BI e Relatórios
-- Cada evento é uma linha, agregamos no Read-Time.

-- NOTA PEDAGÓGICA:
-- Para simular o desafio de Big Data, o script de setup (entrypoint) 
-- já populou estas tabelas com volumes reais (100k+ registros).
-- O foco aqui não é o INSERT, mas sim como modelar para PERFORMANCE em escala.
-- Reflita: quanto tempo levaria para processar o scan total de bilhões de linhas?

-- 1. Fato Transacional (Transaction Fact)
-- Já criada no entrypoint: varejo.fato_vendas (100.000 records)
-- Vamos apenas verificar a estrutura e o volume:

SELECT count(*) as total_vendas FROM fato_vendas;

-- Problema em Escala: "Quantos dias o usuário X comprou no último ano?"
-- Exige SCAN na tabela inteira de fatos (potencialmente bilhões de linhas)
-- Tente rodar esta query em uma tabela real de Big Data:
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
-- 0. SETUP DE AULA: VARIÁVEIS
-- ==============================================================================

-- Configuração de Datas (Parametrização para a aula)
\set data_ontem 2024-05-08
\set data_hoje  2024-05-09

-- As tabelas usuarios_dim e usuarios_atividade_fato foram pré-populadas no entrypoint
-- para simularmos um ambiente real de produção com milhões de eventos.

-- ==============================================================================
-- 3. TABELA ACUMULADA — UPSERT (Cumulative Table Pattern)
-- Comprime N linhas de fato em 1 array por usuário/snapshot
-- Ideal para: contagem de dias ativos, retention, cohort analysis
-- ==============================================================================

DROP TABLE IF EXISTS usuarios_atividade_acumulada;
CREATE TABLE IF NOT EXISTS usuarios_atividade_acumulada (
    usuario_id        INTEGER  NOT NULL,
    data_snapshot     DATE     NOT NULL,
    datas_atividade   DATE[]   NOT NULL DEFAULT '{}',
    -- Campos derivados para evitar recalculo em queries (O(1) Access)
    total_dias_ativos INTEGER  GENERATED ALWAYS AS (CARDINALITY(datas_atividade)) STORED,
    PRIMARY KEY (usuario_id, data_snapshot)
);


-- ------------------------------------------------------------------------------
-- PASSO 1: Carga do "estado de ontem"
-- Agrega todos os eventos históricos até :data_ontem
-- ------------------------------------------------------------------------------

INSERT INTO usuarios_atividade_acumulada (usuario_id, data_snapshot, datas_atividade)
SELECT
    f.usuario_id,
    :'data_ontem'::DATE                                   AS data_snapshot,
    ARRAY_AGG(f.data_evento ORDER BY f.data_evento)       AS datas_atividade
FROM usuarios_atividade_fato f
WHERE f.data_evento <= :'data_ontem'::DATE
GROUP BY f.usuario_id
ON CONFLICT (usuario_id, data_snapshot) DO UPDATE
    SET datas_atividade = EXCLUDED.datas_atividade;


-- ------------------------------------------------------------------------------
-- PASSO 2: Pipeline incremental "Yesterday + Today" 
-- Este é o coração do padrão: lê apenas o delta do dia, faz merge no array
-- ------------------------------------------------------------------------------

WITH yesterday AS (
    -- Estado comprimido do dia anterior (leitura de 1 linha por usuário)
    SELECT usuario_id, datas_atividade
    FROM usuarios_atividade_acumulada
    WHERE data_snapshot = :'data_ontem'::DATE
),
today AS (
    -- Delta: apenas eventos novos do dia corrente (APPEND da fato)
    SELECT usuario_id, data_evento
    FROM usuarios_atividade_fato
    WHERE data_evento = :'data_hoje'::DATE
),
merged AS (
    SELECT
        COALESCE(y.usuario_id, t.usuario_id)              AS usuario_id,
        :'data_hoje'::DATE                                AS data_snapshot,
        COALESCE(y.datas_atividade, ARRAY[]::DATE[])
            || CASE
                 WHEN t.usuario_id IS NOT NULL
                 THEN ARRAY[t.data_evento]
                 ELSE ARRAY[]::DATE[]
               END                                        AS datas_atividade
    FROM yesterday y
    FULL OUTER JOIN today t 
    ON y.usuario_id = t.usuario_id
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
    'usuarios_atividade_acumulada (snapshot ' || :'data_hoje' || ')',
    COUNT(*),
    ROUND(AVG(total_dias_ativos), 2)
FROM usuarios_atividade_acumulada
WHERE data_snapshot = :'data_hoje'::DATE;


-- ------------------------------------------------------------------------------
-- V2. Consulta O(1): dias ativos de um usuário (sem histórico scan)
-- ------------------------------------------------------------------------------
SELECT
    a.usuario_id,
    d.segmento,
    a.total_dias_ativos,
    a.datas_atividade[1]                                  AS primeira_atividade,
    a.datas_atividade[CARDINALITY(a.datas_atividade)]     AS ultima_atividade
FROM usuarios_atividade_acumulada a
JOIN usuarios_dim d ON d.usuario_id = a.usuario_id
WHERE a.data_snapshot = :'data_hoje'::DATE
  AND a.usuario_id IN (1, 2, 3);


-- ------------------------------------------------------------------------------
-- V3. Distribuição de engajamento (retention bucket)
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
    COUNT(*)                                              AS usuarios
FROM usuarios_atividade_acumulada a
JOIN usuarios_dim d ON d.usuario_id = a.usuario_id
WHERE a.data_snapshot = :'data_hoje'::DATE
GROUP BY d.segmento, bucket
ORDER BY d.segmento, usuarios DESC;
