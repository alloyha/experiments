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
CREATE TABLE IF NOT EXISTS dim_localizacao (
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
-- NOTA DIDÁTICA: Embora clássica no modelo Kimball, em Big Data moderno
-- prefere-se usar colunas de DATE e funções nativas (EXTRACT, TO_CHAR)
-- para evitar JOINS massivos e facilitar o particionamento.

/*
-- Exemplo de Tabela de Tempo legada:
CREATE TABLE IF NOT EXISTS dim_tempo (
    tempo_id SERIAL PRIMARY KEY,
    data_completa DATE UNIQUE NOT NULL,
    ano INTEGER,
    mes_nome VARCHAR(20)
);
*/

-- 4. Quando usar data nativa (Abordagem Recomendada)
CREATE TABLE IF NOT EXISTS fato_pedido (
    pedido_id SERIAL PRIMARY KEY,
    data_pedido DATE NOT NULL,     -- Role 1 (Fácil de particionar)
    data_envio DATE,               -- Role 2
    data_entrega DATE,             -- Role 3
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
