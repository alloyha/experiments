-- ==============================================================================
-- Aula 7: Tabelas Ponte (Bridge Tables) vs Array Metrics (Big Data)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- ABORDAGEM CLÁSSICA (KIMBALL) - BRIDGE TABLE
-- ------------------------------------------------------------------------------
-- Cenário: Produtos com Múltiplas Categorias (Many-to-Many)
-- Solução: Tabela intermediária de relacionamento

-- 1. Dimensões Normalizadas
-- 1. Dimensões Normalizadas
DROP TABLE IF EXISTS bridge_produto_categoria_demo;
DROP TABLE IF EXISTS dim_produto_simples CASCADE;
DROP TABLE IF EXISTS dim_categoria_demo CASCADE;

CREATE TABLE dim_produto_simples (
    produto_id SERIAL PRIMARY KEY,
    nome_produto VARCHAR(200),
    marca VARCHAR(50)
);

CREATE TABLE dim_categoria_demo (
    categoria_id SERIAL PRIMARY KEY,
    nome_categoria VARCHAR(100)
);

-- 2. A Bridge Table (Gargalo de Join em Big Data)
CREATE TABLE bridge_produto_categoria_demo (
    produto_id INTEGER,
    categoria_id INTEGER,
    peso_alocacao DECIMAL(5,4),  -- soma = 1.0 por produto
    PRIMARY KEY (produto_id, categoria_id),
    FOREIGN KEY (produto_id) REFERENCES dim_produto_simples(produto_id),
    FOREIGN KEY (categoria_id) REFERENCES dim_categoria_demo(categoria_id)
);

-- 3. Query Clássica (3 JOINS)
/*
SELECT 
    dc.nome_categoria,
    SUM(fv.valor * bpc.peso_alocacao) as valor_alocado
FROM fato_vendas fv
JOIN bridge_produto_categoria_demo bpc ON fv.produto_id = bpc.produto_id -- JOIN 1
JOIN dim_categoria_demo dc ON bpc.categoria_id = dc.categoria_id         -- JOIN 2
GROUP BY dc.nome_categoria;
*/

-- ------------------------------------------------------------------------------
-- ABORDAGEM BIG DATA - ARRAY METRICS
-- ------------------------------------------------------------------------------
-- Cenário: Mesmo problema de N:N
-- Solução: Arrays desnormalizados na própria dimensão (No-Shuffle)

-- 1. Tabela Desnormalizada
DROP TABLE IF EXISTS dim_produto_bigdata;
CREATE TABLE dim_produto_bigdata (
    produto_id SERIAL PRIMARY KEY,
    nome_produto VARCHAR(200),
    -- Array de categorias (Elimina a Bridge)
    categorias TEXT[], 
    -- Array de relevância/peso (Elimina coluna de peso da Bridge)
    pesos_relevancia DECIMAL[] 
);

-- 2. Inserindo dados (Exemplo: Notebook é Informática E Eletrônicos)
INSERT INTO dim_produto_bigdata (nome_produto, categorias, pesos_relevancia)
VALUES 
    ('Notebook Dell i5', ARRAY['Informática', 'Eletrônicos'], ARRAY[0.7, 0.3]),
    ('Mouse Logitech', ARRAY['Informática'], ARRAY[1.0]);

-- 3. Query Big Data (EXPLODE/UNNEST - Zero Joins na Dimensão)
/*
SELECT 
    categoria,
    SUM(peso) as relevancia_total
FROM dim_produto_bigdata,
     UNNEST(categorias, pesos_relevancia) AS t(categoria, peso)
GROUP BY categoria;
*/

-- ------------------------------------------------------------------------------
-- OUTRO EXEMPLO: CONTA BANCÁRIA
-- ------------------------------------------------------------------------------

-- Clássico (Bridge)
DROP TABLE IF EXISTS bridge_conta_titular;
CREATE TABLE bridge_conta_titular (
    conta_id INTEGER,
    cliente_id INTEGER,
    peso_alocacao DECIMAL(5,4),
    PRIMARY KEY (conta_id, cliente_id)
);

-- Big Data (Array de Structs)
DROP TABLE IF EXISTS conta_bigdata;
DROP TYPE IF EXISTS titular_struct;

CREATE TYPE titular_struct AS (
    cliente_id INTEGER,
    tipo VARCHAR,
    peso DECIMAL
);

CREATE TABLE conta_bigdata (
    conta_id INTEGER PRIMARY KEY,
    titulares titular_struct[] -- Array complexo aninhado
);
