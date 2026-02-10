-- ==============================================
-- GABARITO AULA 09 (SCD ANINHADO & BIG DATA)
-- ==============================================

-- ==============================================
-- RESPOSTA 1: Conceitual
-- ==============================================

-- a) 5 Linhas (uma para cada segment_id/versão).
-- b) 1 Linha (todos os segmentos dentro de um array).
-- c) **Benefício:** Eliminação de Shuffle (reparticionamento) de chaves duplicadas.
--    O cliente é processado em um único executor (task Spark) com histórico completo.

-- ==============================================
-- RESPOSTA 2: Implementação
-- ==============================================

DROP TABLE IF EXISTS dim_cliente_historico_array_demo_bd;
DROP TYPE IF EXISTS segmento_historico_demo_bd;

CREATE TYPE segmento_historico_demo_bd AS (
    segmento VARCHAR(50),
    data_inicio DATE,
    data_fim DATE
);

CREATE TABLE dim_cliente_historico_array_demo_bd (
    cliente_id INTEGER PRIMARY KEY,
    nome VARCHAR(100),
    historico segmento_historico_demo_bd[]
);

-- Inserindo cliente com histórico (Bronze -> Prata)
INSERT INTO dim_cliente_historico_array_demo_bd VALUES
(
    1, 
    'João Silva',
    ARRAY[
        ('Bronze', '2022-01-01', '2022-12-31')::segmento_historico_demo_bd,
        ('Prata', '2023-01-01', NULL)::segmento_historico_demo_bd
    ]
)
ON CONFLICT (cliente_id) 
DO UPDATE SET 
    -- Merge Inteligente: Concatena o novo histórico ao existente
    historico = dim_cliente_historico_array_demo_bd.historico || EXCLUDED.historico,
    nome = EXCLUDED.nome;

-- ==============================================
-- RESPOSTA 3: Point-in-Time Query
-- ==============================================

-- Procurando segmento ativo em 15/06/2022
SELECT nome, (h).segmento AS segmento_ativo
FROM dim_cliente_historico_array_demo_bd,
     UNNEST(historico) AS h
WHERE cliente_id = 1
  AND (h).data_inicio <= '2022-06-15'::DATE
  AND ((h).data_fim >= '2022-06-15'::DATE OR (h).data_fim IS NULL);

-- Resultado: 'Bronze'

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Point-in-Time Histórico (Bronze)
   IF (SELECT (h).segmento FROM dim_cliente_historico_array_demo_bd, UNNEST(historico) h 
       WHERE cliente_id = 1 AND '2022-06-15'::DATE BETWEEN (h).data_inicio AND COALESCE((h).data_fim, '9999-12-31')) != 'Bronze' THEN
      RAISE EXCEPTION 'Erro: Esperado segmento Bronze para 2022-06-15';
   END IF;

   -- Validação 2: Point-in-Time Atual (Prata)
   IF (SELECT (h).segmento FROM dim_cliente_historico_array_demo_bd, UNNEST(historico) h 
       WHERE cliente_id = 1 AND '2023-01-15'::DATE BETWEEN (h).data_inicio AND COALESCE((h).data_fim, '9999-12-31')) != 'Prata' THEN
      RAISE EXCEPTION 'Erro: Esperado segmento Prata para 2023-01-15';
   END IF;

   -- Validação 3: Estrutura Aninhada (Deve haver apenas 1 linha para o cliente 1)
   IF (SELECT COUNT(*) FROM dim_cliente_historico_array_demo_bd WHERE cliente_id = 1) != 1 THEN
      RAISE EXCEPTION 'Erro: A modelagem Big Data deveria ter apenas 1 linha por cliente';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 09 (BIG DATA): SUCESSO! ✅';
END $$;
