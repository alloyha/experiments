-- ==============================================================================
-- Aula 6: Tabelas Dimensão
-- ==============================================================================

-- 1. Anatomia Básica de uma Dimensão
-- 1. Anatomia Básica de uma Dimensão
CREATE TABLE IF NOT EXISTS dim_produto (
    produto_id SERIAL PRIMARY KEY,  -- Surrogate Key (Interne)
    produto_sk VARCHAR(20) UNIQUE,  -- Natural Key (Sistema Origem)
    nome_produto VARCHAR(200),
    descricao TEXT,
    categoria VARCHAR(50),
    subcategoria VARCHAR(50),
    marca VARCHAR(50),
    preco_sugerido DECIMAL(10, 2),
    data_cadastro DATE,
    ativo BOOLEAN DEFAULT TRUE
);

-- 2. Hierarquias
-- Abordagem Desnormalizada (Star Schema - Recomendado)
CREATE TABLE IF NOT EXISTS dim_localização (
    localizacao_id SERIAL PRIMARY KEY,
    cidade VARCHAR(100),
    estado VARCHAR(50),
    regiao VARCHAR(50),
    pais VARCHAR(50),
    continente VARCHAR(50)
);

-- Abordagem Normalizada (Snowflake Schema - Menos Comum)
CREATE TABLE IF NOT EXISTS dim_cidade (
    cidade_id SERIAL PRIMARY KEY,
    nome_cidade VARCHAR(100),
    estado_id INTEGER
);

CREATE TABLE IF NOT EXISTS dim_estado (
    estado_id SERIAL PRIMARY KEY,
    nome_estado VARCHAR(50),
    regiao_id INTEGER
);

-- 3. Dimensão Tempo (Date Dimension)
CREATE TABLE IF NOT EXISTS dim_tempo (
    tempo_id SERIAL PRIMARY KEY,
    data_completa DATE UNIQUE NOT NULL,
    ano INTEGER,
    trimestre INTEGER,
    mes INTEGER,
    mes_nome VARCHAR(20),
    semana_ano INTEGER,
    dia_mes INTEGER,
    dia_semana INTEGER,
    dia_semana_nome VARCHAR(20),
    dia_ano INTEGER,
    fim_de_semana BOOLEAN,
    feriado BOOLEAN,
    nome_feriado VARCHAR(50)
);

-- Script para popular (gerar dias)
/*
INSERT INTO dim_tempo (data_completa, ano, mes, dia_mes, dia_semana_nome)
SELECT
    d::date,
    EXTRACT(YEAR FROM d),
    EXTRACT(MONTH FROM d),
    EXTRACT(DAY FROM d),
    TO_CHAR(d, 'Day')
FROM generate_series('2024-01-01'::date, '2025-12-31'::date, '1 day') d;
*/

-- 4. Role-Playing Dimension (Exemplo de uso)
CREATE TABLE IF NOT EXISTS fato_pedido (
    pedido_id SERIAL PRIMARY KEY,
    data_pedido_id INTEGER REFERENCES dim_tempo (tempo_id), -- Role 1
    data_envio_id INTEGER REFERENCES dim_tempo (tempo_id),  -- Role 2
    data_entrega_id INTEGER REFERENCES dim_tempo (tempo_id),-- Role 3
    cliente_id INTEGER,
    valor DECIMAL(10, 2)
);

-- 5. Junk Dimension (Indicadores de baixa cardinalidade)
CREATE TABLE IF NOT EXISTS dim_indicadores_venda (
    indicador_id SERIAL PRIMARY KEY,
    forma_pagamento VARCHAR(20),    -- cartao, boleto, pix
    tipo_frete VARCHAR(20),         -- normal, expresso
    tem_cupom BOOLEAN,
    eh_primeira_compra BOOLEAN
);

-- 6. Registros Especiais (Handling NULLs)
/*
INSERT INTO dim_cliente (cliente_id, nome, tipo) VALUES
(-1, 'Desconhecido', 'UNKNOWN'),
(0, 'Não se Aplica', 'N/A');
*/

-- 7. Atributos de Auditoria
/*
ALTER TABLE dim_produto ADD COLUMN data_carga TIMESTAMP DEFAULT NOW();
ALTER TABLE dim_produto ADD COLUMN usuario_carga VARCHAR(50);
*/
