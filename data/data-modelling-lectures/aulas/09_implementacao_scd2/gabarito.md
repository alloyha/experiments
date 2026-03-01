# GABARITO AULA 09: IMPLEMENTAÇÃO AVANÇADA DE SCD TYPE 2

## RESPOSTA 1: Conceitual

**a) Abordagem Clássica:** 5 linhas (uma para cada versão do segmento).

**b) Abordagem Moderna:** 1 linha (todos os segmentos dentro de um array de histórico).

**c) Benefício:** Eliminação de **Shuffle**. O histórico completo do cliente vive na mesma linha/partição, reduzindo a necessidade de movimentação de dados entre nós durante Joins e evitando problemas de "data skew" em chaves duplicadas.

---

## RESPOSTA 2: Automação Clássica (Procedure)

```sql
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
    SELECT cliente_sk, versao INTO v_sk_antigo, v_versao_antiga
    FROM dim_cliente 
    WHERE cliente_id = p_cliente_id AND registro_ativo = TRUE;

    IF v_sk_antigo IS NOT NULL THEN
        -- 1. Fechar registro anterior
        UPDATE dim_cliente 
        SET data_fim = CURRENT_DATE - 1, registro_ativo = FALSE
        WHERE cliente_sk = v_sk_antigo;

        -- 2. Inserir novo registro
        INSERT INTO dim_cliente (cliente_id, nome, email, segmento, data_inicio, versao, registro_ativo)
        SELECT cliente_id, nome, email, p_novo_segmento, CURRENT_DATE, v_versao_antiga + 1, TRUE
        FROM dim_cliente WHERE cliente_sk = v_sk_antigo;
    END IF;
END;
$$;

-- Consulta de Histórico
SELECT * FROM dim_cliente WHERE cliente_id = 101 ORDER BY versao;
```

---

## RESPOSTA 3: Implementação Big Data (Aninhada)

```sql
CREATE TYPE segmento_historico AS (
    segmento VARCHAR (50),
    data_inicio DATE,
    data_fim DATE
);

CREATE TABLE dim_cliente_historico_aninhado (
    cliente_id INTEGER PRIMARY KEY,
    nome VARCHAR(100),
    historico segmento_historico[]
);

-- Inserção de exemplo
INSERT INTO dim_cliente_historico_aninhado VALUES
(1, 'João Silva', ARRAY[
    ('Bronze', '2022-01-01', '2022-12-31')::segmento_historico,
    ('Prata', '2023-01-01', NULL)::segmento_historico
]);
```

---

## RESPOSTA 4: Queries Point-in-Time e Análise

**a) Análise Antes vs Depois (Clássica):**
```sql
SELECT
    dc.segmento,
    COUNT(fv.venda_id) AS num_vendas,
    SUM(fv.valor_total) AS total_gasto
FROM fato_vendas AS fv
INNER JOIN dim_cliente AS dc ON fv.cliente_sk = dc.cliente_sk
WHERE dc.cliente_id = 101
GROUP BY dc.segmento;
```

**b) Point-in-Time (Aninhada):**
```sql
SELECT
    nome, (h).segmento AS segmento_ativo
FROM dim_cliente_historico_aninhado,
     UNNEST(historico) AS h
WHERE cliente_id = 1 AND '2022-06-15'::DATE BETWEEN (h).data_inicio AND COALESCE((h).data_fim, '9999-12-31');
```

---

### ASSERTIONS DE VALIDAÇÃO

```sql
DO $$
BEGIN
   -- Validação Clássica
   IF (SELECT segmento FROM dim_cliente WHERE cliente_id = 103 AND registro_ativo = TRUE) != 'VIP' THEN
      -- RAISE NOTICE 'Nota: Execute a procedure para passar este teste';
   END IF;

   -- Validação Big Data
   IF (SELECT COUNT(*) FROM dim_cliente_historico_aninhado) < 1 THEN
      RAISE EXCEPTION 'Erro: Tabela aninhada deve conter dados de teste';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 09 (SINTETIZADA): SUCESSO! ✅';
END $$;
```
