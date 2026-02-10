-- ==============================================================================
-- Aula 5: Tabelas Fato vs. Tabelas Cumulativas (Big Data)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- ABORDAGEM CLÁSSICA (KIMBALL) - FACT TABLES
-- ------------------------------------------------------------------------------
-- Padrão de Ouro para BI e Relatórios
-- Cada evento é uma linha, agregamos no Read-Time.

-- 1. Fato Transacional (Transaction Fact)
CREATE TABLE IF NOT EXISTS fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    tempo_id INTEGER NOT NULL,
    cliente_id INTEGER NOT NULL,
    valor_total DECIMAL(10,2)
);

-- Problema em Escala: "Quantos dias o usuário X comprou no último ano?"
-- Exige SCAN na tabela inteira de fatos (potencialmente bilhões de linhas)
/*
SELECT count(distinct data_venda) 
FROM fato_vendas 
WHERE cliente_id = 123 AND data_venda > '2023-01-01';
*/

-- ------------------------------------------------------------------------------
-- ABORDAGEM BIG DATA - CUMULATIVE TABLE
-- ------------------------------------------------------------------------------
-- Padrão "Yesterday + Today = Tomorrow"
-- Usado para evitar scans históricos. O estado é carregado dia a dia.

-- 1. Tabela Acumulada (State Table)
CREATE TABLE IF NOT EXISTS usuarios_atividade_acumulada (
    usuario_id INTEGER,
    data_snapshot DATE,
    -- Array com TODAS as datas de atividade (State)
    datas_atividade DATE[],
    PRIMARY KEY (usuario_id, data_snapshot)
);

-- 2. Pipeline de Carga (Eficiência Máxima)
/*
WITH yesterday AS (
    SELECT * FROM usuarios_atividade_acumulada WHERE data_snapshot = '2024-06-01'
),
today AS (
    SELECT usuario_id, data_evento FROM staging_events WHERE data_evento = '2024-06-02'
)
INSERT INTO usuarios_atividade_acumulada
SELECT 
    COALESCE(y.usuario_id, t.usuario_id),
    '2024-06-02',
    -- Merge do Array Antigo com o Novo Dado
    COALESCE(y.datas_atividade, ARRAY[]) || 
    CASE WHEN t.usuario_id IS NOT NULL THEN ARRAY[t.data_evento] ELSE ARRAY[] END
FROM yesterday y 
FULL OUTER JOIN today t ON y.usuario_id = t.usuario_id;
*/

-- 3. Query Big Data (Zero Scan em Histórico)
-- "Quantos dias o usuário X comprou?" -> Lê apenas a LATEST partition
/*
SELECT CARDINALITY(datas_atividade) 
FROM usuarios_atividade_acumulada
WHERE data_snapshot = '2024-06-02' AND usuario_id = 123;
*/

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
