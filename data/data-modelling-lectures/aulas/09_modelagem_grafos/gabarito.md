# GABARITO AULA 9: MODELAGEM DE GRAFOS

## EXERCÍCIO 1: Grafo de Rede Social

```sql
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
```

## EXERCÍCIO 2: Consultas

**a) Amigos de Bob (ID 2)**

```sql
SELECT v.nome
FROM rede_arestas AS a
INNER JOIN rede_vertices AS v ON a.destino_id = v.vertice_id
WHERE a.origem_id = 2;
-- Resultado esperado: Carol
```

**b) Amigos de Amigos de Alice (ID 1)**

```sql
SELECT DISTINCT v_amigo_do_amigo.nome
FROM rede_arestas AS a1
INNER JOIN rede_arestas AS a2 ON a1.destino_id = a2.origem_id
INNER JOIN rede_vertices AS v_amigo_do_amigo ON a2.destino_id = v_amigo_do_amigo.vertice_id
WHERE a1.origem_id = 1;
-- Alice -> Bob -> Carol (Resultado: Carol)
```

**c) Caminho entre Alice (1) e Eve (5) - Usando CTE Recursiva**

```sql
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
```

**d) Conexão Ponte (Interesse em Comum)**

Assumindo que adicionamos vértices de "Interesse" e relações:

```sql
SELECT 
    v1.nome AS pessoa_1, 
    v_ponte.nome AS interesse_comum, 
    v2.nome AS pessoa_2
FROM rede_vertices v1
JOIN rede_arestas a1 ON v1.vertice_id = a1.origem_id AND a1.tipo_relacao = 'INTERESSE'
JOIN rede_vertices v_ponte ON a1.destino_id = v_ponte.vertice_id
JOIN rede_arestas a2 ON v_ponte.vertice_id = a2.destino_id AND a2.tipo_relacao = 'INTERESSE'
JOIN rede_vertices v2 ON a2.origem_id = v2.vertice_id
WHERE v1.vertice_id < v2.vertice_id -- Evita duplicatas (A-B e B-A)
  AND NOT EXISTS ( -- Garante que não se seguem
      SELECT 1 FROM rede_arestas a3 
      WHERE a3.origem_id = v1.vertice_id AND a3.destino_id = v2.vertice_id
  );
```

### ASSERTIONS (VALIDAÇÃO DE RESULTADOS)

```sql
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
```
