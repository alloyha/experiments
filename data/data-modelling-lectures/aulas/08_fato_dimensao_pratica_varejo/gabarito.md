# GABARITO AULA 08: FATO E DIMENSÃO NA PRÁTICA (VAREJO + SCD)

## RESPOSTA 1: Estratégias Analíticas

**a) Estado (SCD Type 1):**
**Recomendado:** Type 1 (Sobrescrever). Produz relatórios **As-Is** (estado atual ignora que ele teve outro endereço no passado). Útil para o dia a dia logístico de onde a loja enviará as próximas correspondências.

**b) Segmento (SCD Type 2):**
**Recomendado:** Type 2 (Histórico Completo). Produz relatórios **As-Was** (preserva quem ele era na época). Métrica essencial no varejo: ticket médio em função de quando ele "era Bronze" cruzado contra quando se graduou para "Prata".

---

## RESPOSTA 2: Procedures Incrementais e Detecção de Deltas

**a) Ineficiência do full-scan:**
Sem filtro, a procedure envia **todos os 10.000 registros** para o mecanismo de UPSERT. O banco precisa tentar o INSERT, detectar o conflito, e executar o UPDATE para cada linha — mesmo que os dados sejam idênticos. Isso gera I/O desnecessário, locks de tabela e desperdício de WAL (Write-Ahead Log). Se apenas 1 cliente mudou, estamos fazendo 9.999 operações inúteis.

**b) Procedure Incremental (Delta-Only):**
```sql
CREATE OR REPLACE PROCEDURE varejo.proc_upsert_clientes_scd1()
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO varejo.dim_cliente_type1 AS tgt (cliente_id, nome, estado, segmento)
    SELECT src.cliente_id, src.nome, src.estado, src.segmento
    FROM varejo.dim_cliente src
    LEFT JOIN varejo.dim_cliente_type1 cur ON cur.cliente_id = src.cliente_id
    WHERE cur.cliente_id IS NULL                                    -- Novo
       OR ROW(cur.nome, cur.estado, cur.segmento)
          IS DISTINCT FROM ROW(src.nome, src.estado, src.segmento)  -- Mudou
    ON CONFLICT (cliente_id) 
    DO UPDATE SET 
        nome = EXCLUDED.nome,
        estado = EXCLUDED.estado,
        segmento = EXCLUDED.segmento;
END;
$$;
```
O `LEFT JOIN + IS DISTINCT FROM` garante que apenas os deltas reais sejam selecionados. Se nada mudou, o SELECT retorna 0 linhas e o INSERT não executa.

---

## RESPOSTA 3: SCD Type 2 com JSONB e Diff Automático

**a) DDL e CTE de Deltas:**
```sql
CREATE TABLE varejo.dim_cliente_type2 (
    cliente_sk SERIAL PRIMARY KEY,
    cliente_id INTEGER,
    nome VARCHAR(100),
    properties JSONB,
    properties_diff JSONB,
    data_inicio DATE,
    data_fim DATE DEFAULT NULL,
    versao INTEGER,
    ativo BOOLEAN DEFAULT TRUE,
    UNIQUE (cliente_id, versao)
);

-- CTE que identifica APENAS registros novos ou divergentes
WITH deltas AS (
    SELECT 
        src.cliente_id,
        src.nome,
        jsonb_build_object('estado', src.estado, 'segmento', src.segmento) AS new_props,
        tgt.properties AS old_props,
        tgt.cliente_sk AS old_sk
    FROM varejo.dim_cliente src
    LEFT JOIN varejo.dim_cliente_type2 tgt 
        ON tgt.cliente_id = src.cliente_id AND tgt.ativo = TRUE
    WHERE tgt.cliente_sk IS NULL
       OR tgt.properties IS DISTINCT FROM jsonb_build_object('estado', src.estado, 'segmento', src.segmento)
)
SELECT * FROM deltas; -- Apenas para visualização
```

**b) Transação Atômica (UPDATE + INSERT em uma CTE):**
```sql
WITH deltas AS (
    SELECT 
        src.cliente_id, src.nome,
        jsonb_build_object('estado', src.estado, 'segmento', src.segmento) AS new_props,
        tgt.properties AS old_props, tgt.cliente_sk AS old_sk
    FROM varejo.dim_cliente src
    LEFT JOIN varejo.dim_cliente_type2 tgt 
        ON tgt.cliente_id = src.cliente_id AND tgt.ativo = TRUE
    WHERE tgt.cliente_sk IS NULL
       OR tgt.properties IS DISTINCT FROM jsonb_build_object('estado', src.estado, 'segmento', src.segmento)
),
fechados AS (
    UPDATE varejo.dim_cliente_type2 AS tgt
    SET data_fim = CURRENT_DATE - 1, ativo = FALSE
    FROM deltas d
    WHERE tgt.cliente_id = d.cliente_id AND tgt.ativo = TRUE AND d.old_sk IS NOT NULL
    RETURNING tgt.cliente_id
)
INSERT INTO varejo.dim_cliente_type2 (cliente_id, nome, properties, properties_diff, data_inicio, data_fim, versao, ativo)
SELECT 
    d.cliente_id, d.nome, d.new_props,
    varejo.get_jsonb_diff(d.old_props, d.new_props),
    CURRENT_DATE, NULL::DATE,
    COALESCE((SELECT MAX(versao) FROM varejo.dim_cliente_type2 WHERE cliente_id = d.cliente_id), 0) + 1,
    TRUE
FROM deltas d
ON CONFLICT (cliente_id, versao)
DO UPDATE SET nome = EXCLUDED.nome, properties = EXCLUDED.properties, properties_diff = EXCLUDED.properties_diff;
```

**c) Consulta Auditora (Buscando Promoções de Segmento):**
```sql
SELECT cliente_id, data_inicio AS data_que_virou_prata
FROM varejo.dim_cliente_type2
WHERE properties_diff @> '{"segmento": {"from": "Bronze", "to": "Prata"}}'::JSONB;
```

---

## RESPOSTA 4: Backfill como Fast-Forward Incremental

**a) Loop de Fast-Forward:**
```sql
DO $$
DECLARE
    dt DATE;
BEGIN
    FOR dt IN SELECT generate_series('2023-01-01'::DATE, '2023-12-31'::DATE, '1 day'::INTERVAL)::DATE LOOP
        CALL varejo.proc_upsert_clientes_scd2(dt);
        CALL varejo.proc_upsert_produtos_scd3(dt);
        CALL varejo.proc_ingestao_fato_vendas(dt);
        CALL varejo.proc_acumular_atividade(dt - 1, dt);  -- Yesterday + Today
    END LOOP;
END $$;
```

**b) Por que Fast-Forward é melhor que uma procedure de backfill separada:**
1. **Código Único:** Manter uma procedure para incremental e outra para backfill é duplicação de lógica. Se um bug é corrigido num, precisa ser corrigido no outro.
2. **Joins Mínimos:** Cada iteração processa apenas o delta daquele dia. Uma procedure monolítica de backfill tentaria processar milhões de registros em uma única transação, causando locks prolongados e uso massivo de memória.
3. **Idempotência Natural:** Se o loop falhar no dia 183, basta reiniciar do dia 183. Cada dia é atômico e independente.
4. **Paralelizável:** Em orquestradores como Airflow/Dagster, cada dia pode virar uma task independente executada em paralelo (respeitando dependência dimensão → fato).

---

## RESPOSTA 5: Ingestão Idempotente de Tabela Fato

**a) Por que Delete-Insert:**
Tabelas Fato podem ter bilhões de linhas sem chave natural óbvia (uma venda pode ter o mesmo cliente, produto e data). Usar `ON CONFLICT` exigiria uma UNIQUE constraint artificial que raramente existe no mundo real. O padrão **Delete-Insert por partição temporal** resolve elegantemente: limpa todos os fatos daquele dia e reinsere. Se rodar duas vezes, o resultado é idêntico.

**b) Procedure Transacional com resolução de Surrogate Key:**
```sql
CREATE OR REPLACE PROCEDURE varejo.proc_ingestao_fato_vendas(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM varejo.fato_vendas WHERE data_venda = p_data_processamento;

    INSERT INTO varejo.fato_vendas (data_venda, produto_sk, cliente_sk, quantidade, valor_total)
    SELECT 
        src.data_venda,
        p.produto_sk,
        c.cliente_sk,
        src.quantidade,
        src.valor_total
    FROM varejo.fato_vendas_staging src
    JOIN varejo.dim_produto p
        ON p.produto_id = src.produto_id
    JOIN varejo.dim_cliente_type2 c
        ON c.cliente_id = src.cliente_id
       AND c.data_inicio <= src.data_venda
       AND (c.data_fim >= src.data_venda OR c.data_fim IS NULL)
    WHERE src.data_venda = p_data_processamento;
END;
$$;
```

**c) Cumulative Table (Revisão Aula 05 encapsulada):**
```sql
CREATE OR REPLACE PROCEDURE varejo.proc_acumular_atividade(p_data_ontem DATE, p_data_hoje DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    WITH yesterday AS (
        SELECT usuario_id, datas_atividade
        FROM varejo.usuarios_atividade_acumulada
        WHERE data_snapshot = p_data_ontem
    ),
    today AS (
        SELECT usuario_id, data_evento
        FROM varejo.usuarios_atividade_fato
        WHERE data_evento = p_data_hoje
    ),
    merged AS (
        SELECT
            COALESCE(y.usuario_id, t.usuario_id)              AS usuario_id,
            p_data_hoje                                       AS data_snapshot,
            COALESCE(y.datas_atividade, ARRAY[]::DATE[])
                || CASE
                     WHEN t.usuario_id IS NOT NULL
                     THEN ARRAY[t.data_evento]
                     ELSE ARRAY[]::DATE[]
                   END                                        AS datas_atividade
        FROM yesterday y
        FULL OUTER JOIN today t ON y.usuario_id = t.usuario_id
    )
    INSERT INTO varejo.usuarios_atividade_acumulada (usuario_id, data_snapshot, datas_atividade)
    SELECT usuario_id, data_snapshot, datas_atividade
    FROM merged
    ON CONFLICT (usuario_id, data_snapshot) DO UPDATE
        SET datas_atividade = EXCLUDED.datas_atividade;
END;
$$;
```

---

### ASSERTIONS DE VALIDAÇÃO

```sql
DO $$
BEGIN
   -- Validação SCD1: Procedure deve ter sincronizado os dados
   IF (SELECT count(*) FROM varejo.dim_cliente_type1) = 0 THEN
      RAISE EXCEPTION 'Atenção: Tabela SCD1 está vazia. Execute proc_upsert_clientes_scd1().';
   END IF;

   -- Validação SCD2: Deve haver ao menos um registro ativo
   IF NOT EXISTS (SELECT 1 FROM varejo.dim_cliente_type2 WHERE ativo = TRUE) THEN
      RAISE EXCEPTION 'Atenção: Nenhum registro ativo em dim_cliente_type2.';
   END IF;

   -- Validação Fato: Deve haver vendas carregadas
   IF (SELECT count(*) FROM varejo.fato_vendas) = 0 THEN
      RAISE EXCEPTION 'Atenção: Tabela fato_vendas está vazia.';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 08 (PRÁTICA VAREJO): SUCESSO! ✅';
END $$;
```
