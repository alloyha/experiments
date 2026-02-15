-- ==============================================================================
-- Aula 9: SCD Type 2 - Implementação Prática e Completa
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- ABORDAGEM CLÁSSICA (KIMBALL) - MULTI-ROW SCD TYPE 2
-- ------------------------------------------------------------------------------
-- Padrão de Ouro para DW Relacionais (Snowflake, Redshift, PostgreSQL)

-- 1. Estrutura Recomendada
-- 1. Estrutura Recomendada
DROP TABLE IF EXISTS dim_cliente_scd2 CASCADE;
CREATE TABLE dim_cliente_scd2 (
    -- Chaves
    cliente_sk SERIAL PRIMARY KEY,           -- Surrogate (nunca muda)
    cliente_id INTEGER NOT NULL,             -- Natural key (business)

    -- Atributos do negócio
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    cidade VARCHAR(100),
    segmento VARCHAR(50),                    -- Pode mudar (SCD)

    -- Campos de controle SCD
    data_inicio DATE NOT NULL,
    data_fim DATE NOT NULL DEFAULT '9999-12-31',
    versao INTEGER NOT NULL DEFAULT 1,
    registro_ativo BOOLEAN NOT NULL DEFAULT TRUE,

    UNIQUE (cliente_id, versao)
);

-- Índices essenciais para performance
CREATE INDEX idx_cliente_natural_ativo_scd2 ON dim_cliente_scd2 (
    cliente_id, registro_ativo
) WHERE registro_ativo
= TRUE;

-- ------------------------------------------------------------------------------
-- ABORDAGEM BIG DATA - NESTED HISTORY (STRUCT ARRAY)
-- ------------------------------------------------------------------------------
-- Otimização para Data Lakes (Spark/Trino) para evitar Shuffle de chaves
-- Histórico compactado na MESMA LINHA do cliente

-- 1. Definir o Tipo Complexo (Struct)
DROP TABLE IF EXISTS dim_cliente_historico_array_scd2;
DROP TYPE IF EXISTS SEGMENTO_HISTORICO_STRUCT_SCD2;

CREATE TYPE SEGMENTO_HISTORICO_STRUCT_SCD2 AS (
    segmento VARCHAR (50),
    data_inicio DATE,
    data_fim DATE
);

-- 2. Tabela de Histórico Aninhado
CREATE TABLE dim_cliente_historico_array_scd2 (
    cliente_id INTEGER PRIMARY KEY,
    nome VARCHAR(100),

    -- Estado atual (Cache para acesso rápido)
    segmento_atual VARCHAR(50),

    -- Todo o histórico vive aqui dentro!
    historico_segmentos SEGMENTO_HISTORICO_STRUCT_SCD2 []
);

-- Comparativo de Queries: "Como era o cliente em 2023-08-15?"

/*
-- Query Kimball (Clássica)
SELECT segmento FROM dim_cliente
WHERE cliente_id = 101 AND '2023-08-15' BETWEEN data_inicio AND data_fim;
*/

/*
-- Query Big Data (Unnest sem Join)
SELECT (h).segmento
FROM dim_cliente_historico_array c,
     UNNEST(c.historico_segmentos) as h
WHERE c.cliente_id = 101
  AND (h).data_inicio <= '2023-08-15'
  AND ((h).data_fim >= '2023-08-15' OR (h).data_fim IS NULL);
*/

-- ------------------------------------------------------------------------------
-- IMPLEMENTAÇÃO TÉCNICA (SCD Type 2 Clássico)
-- ------------------------------------------------------------------------------
-- O restante do script foca na implementação do UPDATE clássico, que é o
-- objetivo principal deste curso de PostgreSQL.

/*
-- Processo de Update Type 2 (Simplificado):
WITH clientes_alterados AS (
    SELECT c.id, c.segmento
    FROM staging c JOIN dim_cliente d ON c.id = d.cliente_id
    WHERE d.ativo = TRUE AND c.segmento != d.segmento
)
UPDATE dim_cliente SET ativo = FALSE, data_fim = NOW() ...
INSERT INTO dim_cliente ...
*/
