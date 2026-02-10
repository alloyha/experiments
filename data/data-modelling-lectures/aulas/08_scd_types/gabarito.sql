-- ==============================================
-- GABARITO AULA 08: SLOWLY CHANGING DIMENSIONS
-- ==============================================

-- ==============================================
-- RESPOSTA 1: Conceitual
-- ==============================================

-- a) Telefone muda frequentemente e não é crítico.
--    **Recomendado:** SCD Type 1 (Sobrescreve).
--    **Motivo:** Evitar crescimento exponencial da dimensão sem ganho analítico.

-- b) Categoria VIP/Regular é crítica para histórico.
--    **Recomendado:** SCD Type 2 (Histórico).
--    **Motivo:** Saber se o cliente era VIP na compra de 2022 é fundamental.

-- ==============================================
-- RESPOSTA 2: Performance
-- ==============================================

-- a) SCD Type 2 Clássico exige UPDATE na linha antiga para fechar a data de fim.
--    Estratégia Append-Only (apenas INSERT) assume que validade_fim = data do próximo registro.
--
-- b) Performance Massiva (Big Data)
--    **Vencedor:** Append-Only.
--    **Motivo:** UPDATES são muito custosos em Data Lakes (Parquet/Delta).
--              É melhor apenas INSERIR novas versões e resolver a validade na leitura (LEAD/LAG window function).

-- ==============================================
-- RESPOSTA 3: Implementação
-- ==============================================

-- a) Estrutura Type 2 Simplificada
DROP TABLE IF EXISTS dim_cliente_scd2;
CREATE TABLE dim_cliente_scd2 (
    cliente_sk SERIAL PRIMARY KEY,
    cliente_id INTEGER,
    nome VARCHAR(100),
    cidade VARCHAR(100),
    data_validade_inicio DATE,
    data_validade_fim DATE DEFAULT '9999-12-31',
    ativo BOOLEAN DEFAULT TRUE
);

-- b) Simulação Type 2 Clássica
-- Passo 1: Fechar registro antigo
UPDATE dim_cliente_scd2 
SET data_validade_fim = CURRENT_DATE, ativo = FALSE 
WHERE cliente_id = 101 AND ativo = TRUE;

-- Passo 2: Inserir novo
INSERT INTO dim_cliente_scd2 (cliente_id, nome, cidade, data_validade_inicio) 
VALUES (101, 'João Silva', 'Nova Cidade', CURRENT_DATE)
ON CONFLICT DO NOTHING;

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Contagem de Registros
   IF (SELECT COUNT(*) FROM dim_cliente_scd2) != 1 THEN
      RAISE EXCEPTION 'Erro: Esperado 1 registro em dim_cliente_scd2, encontrado %', (SELECT COUNT(*) FROM dim_cliente_scd2);
   END IF;

   -- Validação 2: Status do registro atual
   IF (SELECT ativo FROM dim_cliente_scd2 WHERE cliente_id = 101) != TRUE THEN
      RAISE EXCEPTION 'Erro: Registro deveria estar ativo';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 08: SUCESSO! ✅';
END $$;
