-- ==============================================================================
-- Aula 10: Modelagem de Grafos (Graph Data Modeling)
-- ==============================================================================
-- Tópico Avançado: Quando o Relacional não é suficiente.
-- Cenário: Rede Social de Leitores - Quem segue quem? Quem recomenda o quê?
-- Problema do Relacional: Queries de múltiplos níveis (amigos de amigos, influenciadores)
--                         explodem em complexidade com JOINs recursivos.
-- Solução: Modelar como Vértices (Pessoas, Livros) e Arestas (Relacionamentos).

-- Setup do Ambiente (será controlado via search_path pelo validador)
-- SET search_path TO rede_social, public;

-- ==============================================================================
-- 1. CONCEITOS FUNDAMENTAIS
-- ==============================================================================
-- VÉRTICE (VERTEX): A entidade (Ex: Leitor, Livro, Autor).
-- ARESTA (EDGE): O relacionamento e suas propriedades (Ex: "Segue", "Recomendou",
--                "Leu").
-- PROPRIEDADES: Atributos tanto nos vértices quanto nas arestas (peso, timestamp).

-- Exemplo Prático: Rede Social de Leitores (BookClub Online)

-- ==============================================================================
-- 2. EXPLORANDO O GRAFO: Estrutura de Tabelas
-- ==============================================================================

-- Criando a infraestrutura de tabelas para o Modelo de Grafo
DROP TABLE IF EXISTS grafo_arestas;
DROP TABLE IF EXISTS grafo_vertices;

CREATE TABLE grafo_vertices (
    vertice_id SERIAL PRIMARY KEY,
    tipo VARCHAR(50), -- 'Leitor', 'Livro'
    propriedades JSONB -- Flexibilidade para atributos variados
);

CREATE TABLE grafo_arestas (
    aresta_id SERIAL PRIMARY KEY,
    origem_id INTEGER REFERENCES grafo_vertices (vertice_id),
    destino_id INTEGER REFERENCES grafo_vertices (vertice_id),
    tipo_relacao VARCHAR(50), -- 'SEGUE', 'RECOMENDOU', 'LEU'
    peso DECIMAL(5, 2), -- Força da relação (0-1)
    propriedades JSONB,
    data_criacao DATE DEFAULT CURRENT_DATE
);

-- Índices para navegação rápida (Traversal)
CREATE INDEX idx_grafo_origem ON grafo_arestas (origem_id);
CREATE INDEX idx_grafo_destino ON grafo_arestas (destino_id);
CREATE INDEX idx_grafo_tipo ON grafo_arestas (tipo_relacao);


-- ==============================================================================
-- ETL: TRANSFORMANDO DADOS OLTP EM MODELO DE GRAFO
-- ==============================================================================
-- O seed criou tabelas relacionais tradicionais (pessoa, livro, conexao_social, leitura).
-- Agora vamos transformá-las em grafo (vértices e arestas).
-- IMPORTANTE: Veja como relações podem ser DERIVADAS (ex: RECOMENDOU vem de nota >= 4.0)

-- Limpar tabelas de grafo antes de popular (idempotência)
-- Usamos RESTART IDENTITY para começar os IDs do zero para fins didáticos
TRUNCATE TABLE grafo_arestas, grafo_vertices RESTART IDENTITY CASCADE;

-- 1. Transformar PESSOAS em VÉRTICES
INSERT INTO grafo_vertices (vertice_id, tipo, propriedades)
SELECT
    pessoa_id,
    'Leitor' AS tipo,
    JSONB_BUILD_OBJECT(
        'nome', nome,
        'idade', idade,
        'genero_favorito', genero_favorito
    ) AS propriedades
FROM rede_social.pessoa;

-- 2. Transformar LIVROS em VÉRTICES (offset +1000 para evitar conflito de IDs)
INSERT INTO grafo_vertices (vertice_id, tipo, propriedades)
SELECT
    1000 + livro_id AS vertice_id,
    'Livro' AS tipo,
    JSONB_BUILD_OBJECT(
        'titulo', titulo,
        'autor', autor,
        'ano', ano_publicacao
    ) AS propriedades
FROM rede_social.livro;

-- 3. Transformar CONEXAO_SOCIAL em ARESTAS (pessoa → pessoa)
INSERT INTO grafo_arestas (origem_id, destino_id, tipo_relacao, peso, propriedades)
SELECT
    seguidor_id,
    seguido_id,
    'SEGUE' AS tipo_relacao,
    forca_conexao AS peso,
    JSONB_BUILD_OBJECT('data_conexao', data_conexao) AS propriedades
FROM rede_social.conexao_social;

-- 4. Derivar RECOMENDOU: Leituras com nota >= 4.0 viram recomendações
--    (Relação que NÃO existe explicitamente no OLTP, mas é derivada!)
INSERT INTO grafo_arestas (
    origem_id, destino_id, tipo_relacao, peso, propriedades
)
SELECT
    pessoa_id AS origem_id,
    1000 + livro_id AS destino_id,  -- Offset para IDs de livros
    'RECOMENDOU' AS tipo_relacao,
    nota / 5.0 AS peso,  -- Normaliza nota para peso 0-1
    JSONB_BUILD_OBJECT(
        'nota', nota,
        'data_leitura', data_leitura
    ) AS propriedades
FROM rede_social.leitura
WHERE nota >= 4.0;  -- Critério: só recomenda se gostou (nota alta)

-- 5. Transformar TODAS as leituras em ARESTAS (inclusive as não recomendadas)
INSERT INTO grafo_arestas (
    origem_id, destino_id, tipo_relacao, peso, propriedades
)
SELECT
    pessoa_id AS origem_id,
    1000 + livro_id AS destino_id,
    'LEU' AS tipo_relacao,
    nota / 5.0 AS peso,
    JSONB_BUILD_OBJECT(
        'nota', nota,
        'data_leitura', data_leitura
    ) AS propriedades
FROM rede_social.leitura;

-- ==============================================================================
-- 3. EXPLORANDO OS DADOS: O que temos?
-- ==============================================================================

-- Dados OLTP Originais (antes da transformação)
-- - rede_social.pessoa: 4 leitores
-- - rede_social.livro: 3 livros  
-- - rede_social.conexao_social: quem segue quem (6 conexões)
-- - rede_social.leitura: quem leu o quê e qual nota deu (7 leituras)

-- Dados no Grafo (após ETL)
-- - grafo_vertices: 7 vértices (4 Leitores + 3 Livros)
-- - grafo_arestas: 3 tipos de relações
--   * SEGUE: conexões sociais (6 arestas)
--   * RECOMENDOU: leituras com nota >= 4.0 (5 arestas) ← DERIVADA!
--   * LEU: todas as leituras (7 arestas)

-- Visualizar todos os leitores
SELECT
    vertice_id,
    propriedades ->> 'nome' AS nome,
    propriedades ->> 'idade' AS idade,
    propriedades ->> 'genero_favorito' AS genero_favorito
FROM grafo_vertices
WHERE tipo = 'Leitor'
ORDER BY vertice_id;

-- Visualizar todos os livros
SELECT
    vertice_id,
    (propriedades ->> 'ano')::INTEGER AS ano,
    propriedades ->> 'titulo' AS titulo,
    propriedades ->> 'autor' AS autor
FROM grafo_vertices
WHERE tipo = 'Livro'
ORDER BY vertice_id;

-- ==============================================================================
-- 4. QUERIES DE GRAFO (Traversals)
-- ==============================================================================

-- Q1: "Quem segue quem?" (Rede Social Básica)
SELECT
    a.peso AS forca_relacao,
    v_origem.propriedades ->> 'nome' AS seguidor,
    v_destino.propriedades ->> 'nome' AS seguido
FROM grafo_arestas AS a
INNER JOIN grafo_vertices AS v_origem ON a.origem_id = v_origem.vertice_id
INNER JOIN grafo_vertices AS v_destino ON a.destino_id = v_destino.vertice_id
WHERE a.tipo_relacao = 'SEGUE'
ORDER BY a.peso DESC;

-- Q2: "Quais livros foram recomendados por leitores que eu sigo?" (1 Grau)
-- Exemplo: Recomendações para Ana (vertice_id = 1)
SELECT DISTINCT
    v_livro.propriedades ->> 'titulo' AS livro_recomendado,
    v_livro.propriedades ->> 'autor' AS autor,
    v_recomendador.propriedades ->> 'nome' AS recomendado_por
FROM grafo_arestas AS a_segue
INNER JOIN grafo_vertices AS v_seguido
    ON a_segue.destino_id = v_seguido.vertice_id
INNER JOIN grafo_arestas AS a_recomenda
    ON v_seguido.vertice_id = a_recomenda.origem_id
INNER JOIN grafo_vertices AS v_livro
    ON a_recomenda.destino_id = v_livro.vertice_id
INNER JOIN grafo_vertices AS v_recomendador
    ON a_recomenda.origem_id = v_recomendador.vertice_id
WHERE
    a_segue.origem_id = 1  -- Ana
    AND a_segue.tipo_relacao = 'SEGUE'
    AND a_recomenda.tipo_relacao = 'RECOMENDOU'
    AND v_livro.tipo = 'Livro';

-- Q3: "Influenciadores" - Quem tem mais seguidores? (Agregação)
SELECT
    v.propriedades ->> 'nome' AS nome,
    COUNT(*) AS num_seguidores
FROM grafo_arestas AS a
INNER JOIN grafo_vertices AS v ON a.destino_id = v.vertice_id
WHERE a.tipo_relacao = 'SEGUE'
GROUP BY v.vertice_id, v.propriedades
ORDER BY num_seguidores DESC;

-- Q4: Busca Recursiva - "Rede Estendida" (Amigos de Amigos até 3 graus)
WITH RECURSIVE rede_estendida AS (
    -- Caso Base: Quem Ana segue diretamente (1º grau)
    SELECT
        destino_id AS pessoa_id,
        1 AS grau_separacao,
        ARRAY[origem_id, destino_id] AS caminho
    FROM grafo_arestas
    WHERE origem_id = 1 AND tipo_relacao = 'SEGUE'

    UNION ALL

    -- Recursão: Amigos dos amigos (2º, 3º grau...)
    SELECT
        a.destino_id AS pessoa_id,
        r.grau_separacao + 1 AS grau_separacao,
        r.caminho || a.destino_id AS caminho
    FROM grafo_arestas AS a
    INNER JOIN rede_estendida AS r ON a.origem_id = r.pessoa_id
    WHERE
        a.tipo_relacao = 'SEGUE'
        AND NOT (a.destino_id = ANY(r.caminho)) -- Evita ciclos
        AND r.grau_separacao < 3 -- Limite de profundidade
)

SELECT
    r.grau_separacao,
    v.propriedades ->> 'nome' AS nome,
    v.propriedades ->> 'genero_favorito' AS genero_favorito
FROM rede_estendida AS r
INNER JOIN grafo_vertices AS v ON r.pessoa_id = v.vertice_id
ORDER BY r.grau_separacao, nome;

-- ==============================================================================
-- 5. QUANDO USAR GRAFOS?
-- ==============================================================================
-- Use Grafos quando:
-- 1. O relacionamento é tão importante quanto os dados (Rede Social).
-- 2. Queries envolvem múltiplos níveis de conexão (amigos de amigos).
-- 3. Recomendações baseadas em conexões sociais.
-- 4. Detecção de comunidades/clusters (quem interage mais com quem).

-- Ferramentas Especializadas: Neo4j, Amazon Neptune, TigerGraph.
-- No Big Data: Spark GraphX para processar bilhões de conexões.

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Contagem de Vértices (4 leitores + 3 livros)
   IF (SELECT COUNT(*) FROM grafo_vertices) != 7 THEN
      RAISE EXCEPTION 'Erro: Esperado 7 vértices, encontrado %', (SELECT COUNT(*) FROM grafo_vertices);
   END IF;

   -- Validação 2: Contagem de Arestas (6 SEGUE + 5 RECOMENDOU + 7 LEU = 18)
   IF (SELECT COUNT(*) FROM grafo_arestas) != 18 THEN
      RAISE EXCEPTION 'Erro: Esperado 18 arestas, encontrado %', (SELECT COUNT(*) FROM grafo_arestas);
   END IF;

   -- Validação 3: Teste de Traversal (Ana segue Bruno)
   IF NOT EXISTS (
       SELECT 1 FROM grafo_arestas
       WHERE origem_id = 1 AND destino_id = 2 AND tipo_relacao = 'SEGUE'
   ) THEN
      RAISE EXCEPTION 'Erro: Relacionamento Ana→Bruno não encontrado';
   END IF;
   
   -- Validação 4: Verificar relação derivada RECOMENDOU
   IF NOT EXISTS (
       SELECT 1 FROM grafo_arestas
       WHERE tipo_relacao = 'RECOMENDOU'
   ) THEN
      RAISE EXCEPTION 'Erro: Relações RECOMENDOU não foram derivadas corretamente';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 10: SUCESSO! ✅';
END $$;
