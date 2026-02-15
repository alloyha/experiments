-- ==============================================
-- GABARITO AULA 05: TABELAS FATO E PARADIGMAS DE DADOS
-- ==============================================

-- ==============================================
-- RESPOSTA 1: Conceitual
-- ==============================================

/*
a) Abordagem Clássica:
   SELECT count(distinct user_id)
   FROM logs
   WHERE data IN ('2024-06-01', '2024-06-02');
   -- Custo: Scan na tabela de logs para esses dois dias (se não particionada, scan total).
   -- Problema: Alta complexidade de Leitura em tempo de query.

b) Tabela Acumulada:
   SELECT count(*)
   FROM usuarios_atividade_acumulada
   WHERE data_snapshot = '2024-06-02'
     AND ARRAY['2024-06-01', '2024-06-02'] <@ datas_atividade; -- Operador array contains
   -- Custo: Scan apenas na partição do dia atual (State Table).
   -- Vantagem: State acumulado elimina necessidade de ler partição antiga.
*/


-- ==============================================
-- RESPOSTA 2: Prática
-- ==============================================

-- a) Tabela Acumulada
DROP TABLE IF EXISTS usuarios_atividade_acumulada;
CREATE TABLE usuarios_atividade_acumulada (
    usuario_id INTEGER PRIMARY KEY, -- Simplificado sem snapshot date para exemplo
    datas_atividade DATE []
);

-- b) Inserção Inicial
INSERT INTO usuarios_atividade_acumulada VALUES (1, ARRAY['2024-01-01', '2024-01-10']::DATE [])
ON CONFLICT (usuario_id) DO NOTHING;

-- c) Simulação de Novo Dia ('2024-06-02')
-- Usando UPSERT (INSERT ON CONFLICT) para simplificar merge
INSERT INTO usuarios_atividade_acumulada (usuario_id, datas_atividade)
VALUES (1, ARRAY['2024-06-02']::DATE [])
ON CONFLICT (usuario_id)
DO UPDATE
    SET datas_atividade = usuarios_atividade_acumulada.datas_atividade || excluded.datas_atividade;

-- d) Verificação
SELECT
    usuario_id,
    datas_atividade
FROM usuarios_atividade_acumulada
WHERE usuario_id = 1;
-- Retorno esperado: {2024-01-01, 2024-01-10, 2024-06-02}

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Contagem de Registros
   IF (SELECT COUNT(*) FROM usuarios_atividade_acumulada) != 1 THEN
      RAISE EXCEPTION 'Erro: Esperado 1 registro, encontrado %', (SELECT COUNT(*) FROM usuarios_atividade_acumulada);
   END IF;

   -- Validação 2: Tamanho do Array (State)
   IF (SELECT CARDINALITY(datas_atividade) FROM usuarios_atividade_acumulada WHERE usuario_id = 1) != 3 THEN
      RAISE EXCEPTION 'Erro: Esperado 3 datas no histórico, encontrado %', (SELECT CARDINALITY(datas_atividade) FROM usuarios_atividade_acumulada WHERE usuario_id = 1);
   END IF;

   -- Validação 3: Presença de data específica
   IF NOT ('2024-06-02'::DATE = ANY(SELECT UNNEST(datas_atividade) FROM usuarios_atividade_acumulada WHERE usuario_id = 1)) THEN
      RAISE EXCEPTION 'Erro: Data 2024-06-02 não encontrada no histórico acumulado';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 05: SUCESSO! ✅';
END $$;
