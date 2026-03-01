-- ==============================================================================
-- Aula 6: Tabelas Dimensão (Anatomia, Hierarquia e SCD)
-- ==============================================================================

-- 0. SETUP: LIMPEZA
-- Reutilizaremos o contexto de usuários da Aula 5 para facilitar o aprendizado
DROP TABLE IF EXISTS dim_usuarios_scd;
DROP TABLE IF EXISTS dim_produto;
DROP TABLE IF EXISTS dim_localizacao_snowflake;
DROP TABLE IF EXISTS dim_localizacao_star;
DROP TABLE IF EXISTS dim_junk_venda;

-- ==============================================================================
-- 1. ANATOMIA BÁSICA E SURROGATE KEYS
-- ==============================================================================

-- Dimensão Produto (Clássica)
CREATE TABLE IF NOT EXISTS dim_produto (
    produto_sk     SERIAL PRIMARY KEY,  -- Surrogate Key (Interne/Serial)
    produto_nk     VARCHAR(20) UNIQUE,  -- Natural Key (Sistema Origem)
    nome           VARCHAR(200),
    descricao      TEXT,
    categoria      VARCHAR(50),
    preco_unitario DECIMAL(10, 2),
    data_carga     TIMESTAMP DEFAULT NOW() -- Atributo de Auditoria
);

-- Registros Especiais: Para tratar nulos no fato sem quebrar integridade referencial
INSERT INTO dim_produto (produto_sk, nome, categoria) 
VALUES (-1, 'NÃO INFORMADO', 'N/A'), (0, 'NÃO SE APLICA', 'N/A');


-- ==============================================================================
-- 2. HIERARQUIAS: STAR SCHEMA vs. SNOWFLAKE
-- ==============================================================================

-- A. Abordagem Normalizada (SNOWFLAKE): Muitas tabelas, economia de espaço
CREATE TABLE dim_localizacao_snowflake (
    cidade_id SERIAL PRIMARY KEY,
    nome_cidade VARCHAR(100),
    estado_id   INTEGER -- Requer JOIN para saber o Estado
);

-- B. Abordagem Desnormalizada (STAR SCHEMA): Uma única tabela, Otimizada para Leitura
-- RECOMENDADO EM BIG DATA / DATA WAREHOUSE
CREATE TABLE dim_localizacao_star (
    localizacao_sk SERIAL PRIMARY KEY,
    cidade         VARCHAR(100),
    estado         VARCHAR(50),
    regiao         VARCHAR(50) -- Todas as hierarquias estão na mesma linha
);


-- ==============================================================================
-- 3. JUNK DIMENSIONS & DEGENERATE DIMENSIONS
-- ==============================================================================

-- Junk Dimension: Agrupa flags para limpar a tabela fato
CREATE TABLE dim_junk_venda (
    junk_sk SERIAL PRIMARY KEY,
    forma_pagamento VARCHAR(20), -- pix, cartao, boleto
    tem_cupom       BOOLEAN,
    tipo_frete      VARCHAR(20)  -- normal, expresso
);

-- Inserindo combinações possíveis (Cross Join simplificado)
INSERT INTO dim_junk_venda (forma_pagamento, tem_cupom, tipo_frete)
VALUES ('pix', true, 'normal'), ('cartao', false, 'expresso');

-- Degenerate Dimension (VIVE NO FATO):
-- Exemplo: Número da NF ou ID do Cupom Fiscal. 
-- Não faz sentido criar uma tabela para isso.
-- CREATE TABLE fato_vendas ( ... numero_cupom VARCHAR(20) ... );
