-- ==============================================================================
-- Aula 10: Modelagem de Grafos (Graph Data Modeling)
-- ==============================================================================
-- Tópico Avançado: Quando o Relacional não é suficiente.
-- Cenário: Redes Sociais, Detecção de Fraude, Recomendação, Rotas Logísticas.
-- Problema do Relacional: Joins de N:N recursivos explodem em complexidade e custo.
-- Solução: Modelar como Vértices (Nodes) e Arestas (Edges).

-- Setup do Ambiente
SET search_path TO curso_modelagem;

-- ==============================================================================
-- 1. CONCEITOS FUNDAMENTAIS
-- ==============================================================================
-- VÉRTICE (VERTEX): A entidade (Ex: Pessoa, Local, Filme).
-- ARESTA (EDGE): O relacionamento e suas propriedades (Ex: "Amigo de", "Assistiu",
--               "Localizado em").
-- PROPRIEDADES: Atributos tanto nos vértices quanto nas arestas.

-- Exemplo Prático: Rede de Co-autoria (Quem escreveu com quem?)

-- ==============================================================================
-- 2. MODELAGEM RELACIONAL DE UM GRAFO (Estratégia Híbrida)
-- ==============================================================================
-- Podemos simular um grafo eficiente no PostgreSQL usando tabelas de Edges dedicadas.

-- Tabela de Vértices (Todas as entidades do grafo)
DROP TABLE IF EXISTS grafo_arestas_demo;
DROP TABLE IF EXISTS grafo_vertices_demo;

CREATE TABLE grafo_vertices_demo (
    vertice_id SERIAL PRIMARY KEY,
    tipo VARCHAR(50), -- 'Autor', 'Livro', 'Editora'
    propriedades JSONB -- Flexibilidade para atributos variados
);

-- Tabela de Arestas (Relacionamentos)
CREATE TABLE grafo_arestas_demo (
    aresta_id SERIAL PRIMARY KEY,
    origem_id INTEGER REFERENCES grafo_vertices_demo (vertice_id),
    destino_id INTEGER REFERENCES grafo_vertices_demo (vertice_id),
    tipo_relacao VARCHAR(50), -- 'ESCREVEU', 'PUBLICADO_POR', 'AMIGO_DE'
    peso DECIMAL(5, 2), -- Ex: Força da conexão
    propriedades JSONB,
    data_criacao DATE DEFAULT CURRENT_DATE
);

-- Índices para navegação rápida (Traversal)
CREATE INDEX idx_grafo_origem ON grafo_arestas_demo (origem_id);
CREATE INDEX idx_grafo_destino ON grafo_arestas_demo (destino_id);
CREATE INDEX idx_grafo_tipo ON grafo_arestas_demo (tipo_relacao);

-- ==============================================================================
-- 3. POPULANDO O GRAFO (Exemplo)
-- ==============================================================================

-- Inserindo Vértices
INSERT INTO grafo_vertices_demo (tipo, propriedades) VALUES
('Autor', '{"nome": "Machado de Assis", "ano": 1839}'),
('Autor', '{"nome": "Clarice Lispector", "ano": 1920}'),
('Livro', '{"titulo": "Contos Brasileiros"}');

-- Inserindo Arestas (Relacionamentos)
-- Machado -> Escreveu -> Contos
-- Clarice -> Escreveu -> Contos
-- Machado -> Influenciou -> Clarice (Hipocético)

INSERT INTO grafo_arestas_demo (origem_id, destino_id, tipo_relacao, peso) VALUES
(1, 3, 'ESCREVEU', 1.0),
(2, 3, 'ESCREVEU', 1.0),
(1, 2, 'INFLUENCIOU', 0.8);

-- ==============================================================================
-- 4. QUERIES DE GRAFO (Traversals)
-- ==============================================================================

-- P1: "Quem são os co-autores de um livro?" (1 Grau de Separação)
-- Autor A -> Livro <- Autor B
SELECT
    v_autor_a.propriedades ->> 'nome' AS autor_a,
    v_livro.propriedades ->> 'titulo' AS livro,
    v_autor_b.propriedades ->> 'nome' AS autor_b
FROM grafo_arestas_demo AS a1
INNER JOIN grafo_vertices_demo AS v_autor_a ON a1.origem_id = v_autor_a.vertice_id
INNER JOIN grafo_vertices_demo AS v_livro ON a1.destino_id = v_livro.vertice_id
INNER JOIN grafo_arestas_demo AS a2 ON v_livro.vertice_id = a2.destino_id
INNER JOIN grafo_vertices_demo AS v_autor_b ON a2.origem_id = v_autor_b.vertice_id
WHERE
    a1.tipo_relacao = 'ESCREVEU'
    AND a2.tipo_relacao = 'ESCREVEU'
    AND a1.origem_id != a2.origem_id;

-- P2: Busca Recursiva (Recursive CTE) - "Amigos de Amigos" ou "Influência"
-- Quem foi influenciado direta ou indiretamente por Machado?
WITH RECURSIVE influencia_net AS (
    -- Caso Base: Influência direta
    SELECT
        destino_id,
        1 AS nivel,
        ARRAY[origem_id, destino_id] AS caminho
    FROM grafo_arestas_demo
    WHERE origem_id = 1 AND tipo_relacao = 'INFLUENCIOU'

    UNION ALL

    -- Parte Recursiva: Quem esses influenciaram?
    SELECT
        a.destino_id,
        i.nivel + 1 AS nivel,
        i.caminho || a.destino_id AS caminho
    FROM grafo_arestas_demo AS a
    INNER JOIN influencia_net AS i ON a.origem_id = i.destino_id
    WHERE
        a.tipo_relacao = 'INFLUENCIOU'
        AND NOT (a.destino_id = ANY(i.caminho)) -- Evita ciclos
)

SELECT
    i.nivel,
    i.caminho,
    v.propriedades ->> 'nome' AS influenciado
FROM influencia_net AS i
INNER JOIN grafo_vertices_demo AS v ON i.destino_id = v.vertice_id;

-- ==============================================================================
-- 5. QUANDO USAR?
-- ==============================================================================
-- Use Grafos quando:
-- 1. O relacionamento é tão importante quanto o dado (Redes Sociais).
-- 2. Queries envolvem padrões de caminho (Path Finding, Shortest Path).
-- 3. A estrutura é altamente flexível e muda rápido (Schema-less).

-- Ferramentas Especializadas: Neo4j, Amazon Neptune, TigerGraph.
-- No Big Data: Spark GraphX para processamento em massa de arestas.

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Contagem de Vértices
   IF (SELECT COUNT(*) FROM grafo_vertices_demo) != 3 THEN
      RAISE EXCEPTION 'Erro: Esperado 3 vértices, encontrado %', (SELECT COUNT(*) FROM grafo_vertices_demo);
   END IF;

   -- Validação 2: Contagem de Arestas
   IF (SELECT COUNT(*) FROM grafo_arestas_demo) != 3 THEN
      RAISE EXCEPTION 'Erro: Esperado 3 arestas, encontrado %', (SELECT COUNT(*) FROM grafo_arestas_demo);
   END IF;

   -- Validação 3: Teste de Traversal (Influência)
   -- Machado (1) influenciou Clarice (2)
   IF NOT EXISTS (
       SELECT 1 FROM grafo_arestas_demo 
       WHERE origem_id = 1 AND destino_id = 2 AND tipo_relacao = 'INFLUENCIOU'
   ) THEN
      RAISE EXCEPTION 'Erro: Relacionamento de influência não encontrado';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 10: SUCESSO! ✅';
END $$;
