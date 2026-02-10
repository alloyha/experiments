-- ==============================================
-- GABARITO AULA 09: SCD TYPE 2 AVANÇADO
-- ==============================================

-- ==============================================
-- EXERCÍCIO 5: SCD Type 2 Avançado
-- ==============================================

-- a) Criar procedure para automatizar mudança de segmento
CREATE OR REPLACE PROCEDURE sp_atualiza_segmento_cliente(
    p_cliente_id INTEGER, 
    p_novo_segmento VARCHAR
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_sk_antigo INTEGER;
    v_versao_antiga INTEGER;
BEGIN
    -- Pegar dados atuais
    SELECT cliente_sk, versao INTO v_sk_antigo, v_versao_antiga
    FROM dim_cliente 
    WHERE cliente_id = p_cliente_id AND registro_ativo = TRUE;

    IF v_sk_antigo IS NOT NULL THEN
        -- 1. Fechar registro anterior
        UPDATE dim_cliente 
        SET data_fim = CURRENT_DATE - 1, registro_ativo = FALSE
        WHERE cliente_sk = v_sk_antigo;

        -- 2. Inserir novo
        INSERT INTO dim_cliente (cliente_id, nome, email, telefone, cidade, estado, segmento, data_inicio, versao, registro_ativo)
        SELECT cliente_id, nome, email, telefone, cidade, estado, p_novo_segmento, CURRENT_DATE, v_versao_antiga + 1, TRUE
        FROM dim_cliente WHERE cliente_sk = v_sk_antigo;
    END IF;
END;
$$;

-- b) Consultar histórico completo de mudanças de um cliente
SELECT * FROM dim_cliente 
WHERE cliente_id = 101 
ORDER BY versao;

-- c) Análise: vendas antes vs depois da mudança de segmento
SELECT 
    dc.segmento,
    COUNT(fv.venda_id) as num_vendas,
    SUM(fv.valor_total) as total_gasto
FROM fato_vendas fv
JOIN dim_cliente dc ON fv.cliente_sk = dc.cliente_sk
WHERE dc.cliente_id = 101
GROUP BY dc.segmento;

-- Testando a procedure
CALL sp_atualiza_segmento_cliente(103, 'VIP');

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Registro histórico (2 versões para o 103)
   IF (SELECT COUNT(*) FROM dim_cliente WHERE cliente_id = 103) != 2 THEN
      RAISE EXCEPTION 'Erro: Esperado 2 versões para o cliente 103, encontrado %', (SELECT COUNT(*) FROM dim_cliente WHERE cliente_id = 103);
   END IF;

   -- Validação 2: Status do VIP
   IF (SELECT segmento FROM dim_cliente WHERE cliente_id = 103 AND registro_ativo = TRUE) != 'VIP' THEN
      RAISE EXCEPTION 'Erro: Segmento ativo deveria ser VIP';
   END IF;

   -- Validação 3: Fechamento do anterior
   IF (SELECT COUNT(*) FROM dim_cliente WHERE cliente_id = 103 AND registro_ativo = FALSE) != 1 THEN
      RAISE EXCEPTION 'Erro: Deveria haver exatamente 1 registro inativo para o cliente 103';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 09: SUCESSO! ✅';
END $$;
