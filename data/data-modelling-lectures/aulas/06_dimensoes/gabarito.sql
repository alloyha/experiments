-- ==============================================
-- GABARITO AULA 06: MODELAGEM DIMENSIONAL & BIG DATA
-- ==============================================

-- ==============================================
-- RESPOSTA 1: Estrutura
-- ==============================================

-- a) Clássica (Snowflake)
DROP TABLE IF EXISTS dim_produto_snowflake;
DROP TABLE IF EXISTS dim_categoria_snowflake;

CREATE TABLE dim_categoria_snowflake (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100)
);

CREATE TABLE dim_produto_snowflake (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100),
    categoria_id INTEGER REFERENCES dim_categoria_snowflake(id) -- Normalizado
);

-- b) Moderna (Big Data)
-- No PostgreSQL, usamos um TYPE composto para simular STRUCT
DROP TABLE IF EXISTS dim_produto_moderna;
DROP TYPE IF EXISTS categoria_struct;

CREATE TYPE categoria_struct AS (
    id INTEGER,
    nome VARCHAR(100)
);

CREATE TABLE dim_produto_moderna (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100),
    -- Categoria: campo composto (1 struct por produto)
    categoria categoria_struct
);

-- Inserindo dados na moderna:
INSERT INTO dim_produto_moderna (nome, categoria)
VALUES ('Notebook', (1, 'Informática')::categoria_struct)
ON CONFLICT (id) DO UPDATE 
SET nome = EXCLUDED.nome; -- SCD Type 1: Sobrescreve se mudar o nome

-- c) Comparação:
-- Moderna é ideal para leituras (sem JOIN). Snowflake economiza espaço se categoria for muito descritiva.

-- ==============================================
-- RESPOSTA 2: Consultas
-- ==============================================

-- a) Query Big Data (Zero-Join)
SELECT * 
FROM dim_produto_moderna
WHERE (categoria).nome = 'Informática'; 
-- Custo: Zero Join. Acessa estrutura interna diretamente.

-- b) Query Clássica
SELECT p.* 
FROM dim_produto_snowflake p
JOIN dim_categoria_snowflake c ON p.categoria_id = c.id
WHERE c.nome = 'Informática';
-- Custo: Requer buscar ID na tabela de categorias e fazer JOIN.

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Contagem de Produtos na estrutura moderna
   IF (SELECT COUNT(*) FROM dim_produto_moderna) != 1 THEN
      RAISE EXCEPTION 'Erro: Esperado 1 produto em dim_produto_moderna, encontrado %', (SELECT COUNT(*) FROM dim_produto_moderna);
   END IF;

   -- Validação 2: Filtro por estrutura aninhada
   IF (SELECT COUNT(*) FROM dim_produto_moderna WHERE (categoria).nome = 'Informática') != 1 THEN
      RAISE EXCEPTION 'Erro: Falha no filtro por estrutura aninhada (categoria).nome';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 06: SUCESSO! ✅';
END $$;
