-- ==============================================
-- GABARITO AULA 10: MODELAGEM DE GRAFOS
-- ==============================================

-- ==============================================
-- EXERCÍCIO 1: Grafo de Rede Social
-- ==============================================

DROP TABLE IF EXISTS rede_arestas;
DROP TABLE IF EXISTS rede_vertices;

CREATE TABLE rede_vertices (
    vertice_id SERIAL PRIMARY KEY,
    nome VARCHAR(50) NOT NULL
);

CREATE TABLE rede_arestas (
    aresta_id SERIAL PRIMARY KEY,
    origem_id INTEGER REFERENCES rede_vertices (vertice_id),
    destino_id INTEGER REFERENCES rede_vertices (vertice_id),
    tipo_relacao VARCHAR(20) DEFAULT 'AMIGO',
    UNIQUE (origem_id, destino_id)
);

-- Inserir Pessoas
INSERT INTO rede_vertices (vertice_id, nome) VALUES
(1, 'Alice'), (2, 'Bob'), (3, 'Carol'), (4, 'Dave'), (5, 'Eve');
-- Reset sequence if needed: SELECT setval('rede_vertices_vertice_id_seq', 5);

-- Inserir Conexões (Arestas)
INSERT INTO rede_arestas (origem_id, destino_id) VALUES
(1, 2), -- Alice -> Bob
(2, 3), -- Bob -> Carol
(3, 4), -- Carol -> Dave
(4, 5), -- Dave -> Eve
(5, 1); -- Eve -> Alice

-- ==============================================
-- EXERCÍCIO 2: Consultas
-- ==============================================

-- a) Amigos de Bob (ID 2)
SELECT v.nome
FROM rede_arestas AS a
INNER JOIN rede_vertices AS v ON a.destino_id = v.vertice_id
WHERE a.origem_id = 2;
-- Resultado esperado: Carol

-- b) Amigos de Amigos de Alice (ID 1)
SELECT DISTINCT v_amigo_do_amigo.nome
FROM rede_arestas AS a1
INNER JOIN rede_arestas AS a2 ON a1.destino_id = a2.origem_id
INNER JOIN rede_vertices AS v_amigo_do_amigo ON a2.destino_id = v_amigo_do_amigo.vertice_id
WHERE a1.origem_id = 1;
-- Alice -> Bob -> Carol (Resultado: Carol)

-- c) Caminho entre Alice (1) e Eve (5) - Usando CTE Recursiva
WITH RECURSIVE caminho (origem, destino, path_string) AS (
    -- Caso base: amigos diretos de Alice
    SELECT
        origem_id,
        destino_id,
        CAST(origem_id || '->' || destino_id AS TEXT)
    FROM rede_arestas
    WHERE origem_id = 1

    UNION ALL

    -- Recursão: amigos dos amigos
    SELECT
        c.origem,
        a.destino_id,
        c.path_string || '->' || a.destino_id
    FROM rede_arestas AS a
    INNER JOIN caminho AS c ON a.origem_id = c.destino
    WHERE NOT POSITION(CAST(a.destino_id AS TEXT) IN c.path_string) > 0 -- Evita ciclos infinitos
)

SELECT path_string
FROM caminho
WHERE destino = 5;
-- Resultado esperado: 1->2->3->4->5

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- 1. Validação de Vértices
   IF (SELECT COUNT(*) FROM rede_vertices) != 5 THEN
      RAISE EXCEPTION 'Erro: Esperado 5 vértices, encontrado %', (SELECT COUNT(*) FROM rede_vertices);
   END IF;

   -- 2. Validação de Conexão Indireta (Alice -> ... -> Eve)
   -- Vamos checar se o caminho existe via CTE simples
   IF NOT EXISTS (
       WITH RECURSIVE search_graph(id, path) AS (
           SELECT vertice_id, ARRAY[vertice_id] FROM rede_vertices WHERE vertice_id = 1
           UNION ALL
            SELECT
                a.destino_id,
                s.path || a.destino_id
            FROM rede_arestas AS a
            INNER JOIN search_graph AS s ON a.origem_id = s.id
            WHERE NOT a.destino_id = ANY (s.path)
        )
        SELECT 1 FROM search_graph WHERE id = 5
   ) THEN
      RAISE EXCEPTION 'Erro: Caminho entre Alice e Eve não encontrado!';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 10: SUCESSO! ✅';
END $$;
