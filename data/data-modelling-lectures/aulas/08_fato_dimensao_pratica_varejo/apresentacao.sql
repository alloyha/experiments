-- ==============================================================================
-- Aula 8: Fato e Dimensão na Prática (Caso Varejo + SCDs Clássicos)
-- ==============================================================================
-- IMPORTANTE: Para executar este script, você deve rodar antes o script de 
-- setup do Varejo (scripts/setup_varejo.sql), que irá popular as tabelas 
-- dim_cliente, dim_produto, usuarios_atividade_fato, etc.
-- ==============================================================================
-- Princípio de Design: Toda Procedure é INCREMENTAL por natureza.
-- Backfill = Fast-Forward de chamadas incrementais em sequência.
-- ==============================================================================
CREATE SCHEMA IF NOT EXISTS varejo;

-- ==============================================================================
-- SCD TYPE 1: SOBRESCREVE (SEM HISTÓRICO)
-- ==============================================================================
-- A Procedure apenas toca nas linhas que divergem entre Origem e Destino.
-- Linhas iguais = zero I/O.

DROP TABLE IF EXISTS varejo.dim_cliente_type1;
CREATE TABLE varejo.dim_cliente_type1 (
    cliente_id INTEGER PRIMARY KEY,
    nome VARCHAR(100),
    estado VARCHAR(2),
    segmento VARCHAR(50)
);

CREATE OR REPLACE PROCEDURE varejo.proc_upsert_clientes_scd1()
LANGUAGE plpgsql
AS $$
BEGIN
    -- Apenas insere/atualiza registros que divergem ou são novos.
    -- O WHERE filtra ANTES do UPSERT, evitando tocar em linhas iguais.
    INSERT INTO varejo.dim_cliente_type1 AS tgt (cliente_id, nome, estado, segmento)
    SELECT src.cliente_id, src.nome, src.estado, src.segmento
    FROM varejo.dim_cliente src
    LEFT JOIN varejo.dim_cliente_type1 cur ON cur.cliente_id = src.cliente_id
    WHERE cur.cliente_id IS NULL                                 -- Novo
       OR ROW(cur.nome, cur.estado, cur.segmento)
          IS DISTINCT FROM ROW(src.nome, src.estado, src.segmento) -- Mudou
    ON CONFLICT (cliente_id) 
    DO UPDATE SET 
        nome = EXCLUDED.nome,
        estado = EXCLUDED.estado,
        segmento = EXCLUDED.segmento;
END;
$$;

-- ==============================================================================
-- SCD TYPE 2: HISTÓRICO COMPLETO COM JSONB
-- ==============================================================================

DROP TABLE IF EXISTS varejo.dim_cliente_type2;
CREATE TABLE varejo.dim_cliente_type2 (
    cliente_sk SERIAL PRIMARY KEY,        -- Surrogate key
    cliente_id INTEGER,                   -- Natural key
    nome VARCHAR(100),
    properties JSONB,                     -- Atributos variáveis (ex: estado, segmento)
    properties_diff JSONB,                -- Registro de deltas: {"estado": {"from":"SP","to":"RJ"}}
    data_inicio DATE,                     -- Início validade
    data_fim DATE,                        -- NULL = registro atual
    versao INTEGER,                       -- Número da versão
    ativo BOOLEAN,                        -- Flag atual
    UNIQUE (cliente_id, versao)
);

-- Função Auxiliar para calcular a Diferença de Propriedades (JSONB Diff)
CREATE OR REPLACE FUNCTION varejo.get_jsonb_diff(old_json JSONB, new_json JSONB)
RETURNS JSONB AS $$
DECLARE
    result JSONB := '{}'::JSONB;
    k TEXT;
    v JSONB;
BEGIN
    IF old_json IS NULL THEN old_json := '{}'::JSONB; END IF;
    IF new_json IS NULL THEN new_json := '{}'::JSONB; END IF;

    FOR k, v IN SELECT * FROM jsonb_each(new_json) LOOP
        IF old_json->k IS DISTINCT FROM v THEN
            result := jsonb_set(result, ARRAY[k], jsonb_build_object('from', old_json->k, 'to', v));
        END IF;
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Procedure Incremental SCD2
-- Toca APENAS nos clientes que mudaram ou são novos.
CREATE OR REPLACE PROCEDURE varejo.proc_upsert_clientes_scd2(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    -- CTE: Identifica APENAS os deltas (linhas novas ou com properties diferentes)
    -- Essa CTE é o coração do incrementalismo: ela faz o menor JOIN possível.
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
        -- Filtra: só prossegue se é novo (old_sk IS NULL) ou mudou (properties divergem)
        WHERE tgt.cliente_sk IS NULL
           OR tgt.properties IS DISTINCT FROM jsonb_build_object('estado', src.estado, 'segmento', src.segmento)
    ),
    -- SUB-PASSO 1: Fechar registros antigos que sofreram alterações
    fechados AS (
        UPDATE varejo.dim_cliente_type2 AS tgt
        SET data_fim = p_data_processamento - 1,
            ativo = FALSE
        FROM deltas d
        WHERE tgt.cliente_id = d.cliente_id
          AND tgt.ativo = TRUE
          AND d.old_sk IS NOT NULL -- Só fecha se existia versão anterior
        RETURNING tgt.cliente_id
    )
    -- SUB-PASSO 2: Inserir as novas versões (somente dos deltas identificados)
    INSERT INTO varejo.dim_cliente_type2 (cliente_id, nome, properties, properties_diff, data_inicio, data_fim, versao, ativo)
    SELECT 
        d.cliente_id,
        d.nome,
        d.new_props,
        varejo.get_jsonb_diff(d.old_props, d.new_props),
        p_data_processamento,
        NULL::DATE,
        COALESCE((SELECT MAX(versao) FROM varejo.dim_cliente_type2 WHERE cliente_id = d.cliente_id), 0) + 1,
        TRUE
    FROM deltas d
    ON CONFLICT (cliente_id, versao) 
    DO UPDATE SET 
        nome = EXCLUDED.nome,
        properties = EXCLUDED.properties,
        properties_diff = EXCLUDED.properties_diff;
END;
$$;

-- ==============================================================================
-- SCD TYPE 3: HISTÓRICO LIMITADO (COLUNA ANTERIOR - PRODUTO)
-- ==============================================================================

DROP TABLE IF EXISTS varejo.dim_produto_type3;
CREATE TABLE varejo.dim_produto_type3 (
    produto_id VARCHAR(20) PRIMARY KEY,
    nome_produto VARCHAR(200),
    categoria_atual VARCHAR(50),
    categoria_anterior VARCHAR(50),
    data_mudanca_categoria DATE
);

-- Procedure Incremental SCD3
-- Toca APENAS nos produtos novos ou com categoria divergente.
CREATE OR REPLACE PROCEDURE varejo.proc_upsert_produtos_scd3(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO varejo.dim_produto_type3 (
        produto_id, nome_produto, categoria_atual, categoria_anterior, data_mudanca_categoria
    )
    SELECT 
        src.produto_id, 
        src.nome_produto, 
        src.categoria, 
        NULL,
        p_data_processamento
    FROM varejo.dim_produto src
    LEFT JOIN varejo.dim_produto_type3 cur ON cur.produto_id = src.produto_id
    WHERE cur.produto_id IS NULL                             -- Novo
       OR cur.categoria_atual IS DISTINCT FROM src.categoria -- Mudou
    ON CONFLICT (produto_id) 
    DO UPDATE SET 
        categoria_anterior = CASE 
            WHEN varejo.dim_produto_type3.categoria_atual <> EXCLUDED.categoria_atual 
            THEN varejo.dim_produto_type3.categoria_atual 
            ELSE varejo.dim_produto_type3.categoria_anterior 
        END,
        data_mudanca_categoria = CASE 
            WHEN varejo.dim_produto_type3.categoria_atual <> EXCLUDED.categoria_atual 
            THEN p_data_processamento 
            ELSE varejo.dim_produto_type3.data_mudanca_categoria 
        END,
        categoria_atual = EXCLUDED.categoria_atual,
        nome_produto = EXCLUDED.nome_produto;
END;
$$;

-- ==============================================================================
-- TABELA FATO: PROCEDURES IDEMPOTENTES DE INGESTÃO
-- ==============================================================================
-- Revisitamos aqui os dois padrões de ingestão de Fato aprendidos na Aula 05,
-- agora encapsulados em Procedures parametrizadas para Backfill/Incremental.

-- ------------------------------------------------------------------------------
-- FATO TRANSACIONAL (Delete-Insert por Partição de Data)
-- ------------------------------------------------------------------------------
-- Padrão clássico para fatos de evento (vendas, cliques, transações):
-- Cada dia é uma "partição lógica". Reprocessar = deletar o dia e reinserir.

CREATE TABLE IF NOT EXISTS varejo.fato_vendas_staging (
    data_venda DATE,
    produto_id VARCHAR(20),
    cliente_id INTEGER,
    quantidade INTEGER,
    valor_total DECIMAL(10, 2)
);

CREATE OR REPLACE PROCEDURE varejo.proc_ingestao_fato_vendas(p_data_processamento DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    -- 1. Idempotência: Limpa a janela do dia
    DELETE FROM varejo.fato_vendas WHERE data_venda = p_data_processamento;

    -- 2. Insere os fatos do dia resolvendo a Surrogate Key da Dimensão SCD2 e da SCD3/Type 0
    INSERT INTO varejo.fato_vendas (data_venda, produto_sk, cliente_sk, quantidade, valor_total)
    SELECT 
        src.data_venda,
        p.produto_sk,
        c.cliente_sk,          -- Surrogate Key resolvida!
        src.quantidade,
        src.valor_total
    FROM varejo.fato_vendas_staging src
    JOIN varejo.dim_produto p
        ON p.produto_id = src.produto_id -- Resolve Produto
    JOIN varejo.dim_cliente_type2 c
        ON c.cliente_id = src.cliente_id -- Resolve NK -> SK via Point-In-Time
       AND c.data_inicio <= src.data_venda
       AND (c.data_fim >= src.data_venda OR c.data_fim IS NULL)
    WHERE src.data_venda = p_data_processamento;
END;
$$;

-- ------------------------------------------------------------------------------
-- FATO ACUMULADA (Cumulative Table - Yesterday + Today Pattern)
-- Revisão do padrão da Aula 05, agora encapsulado em Procedure parametrizada.
-- ------------------------------------------------------------------------------
-- Na Aula 05 aprendemos que a Cumulative Table comprime N linhas de evento
-- em 1 array por usuario/snapshot. Aqui encapsulamos o pipeline Yesterday+Today
-- numa Procedure que aceita (ontem, hoje) para funcionar tanto incremental 
-- quanto em backfill (fast-forward).

DROP TABLE IF EXISTS varejo.usuarios_atividade_acumulada;
CREATE TABLE IF NOT EXISTS varejo.usuarios_atividade_acumulada (
    usuario_id        INTEGER  NOT NULL,
    data_snapshot     DATE     NOT NULL,
    datas_atividade   DATE[]   NOT NULL DEFAULT '{}',
    -- Campos derivados para evitar recalculo em queries (O(1) Access)
    total_dias_ativos INTEGER  GENERATED ALWAYS AS (CARDINALITY(datas_atividade)) STORED,
    PRIMARY KEY (usuario_id, data_snapshot)
);

CREATE OR REPLACE PROCEDURE varejo.proc_acumular_atividade(p_data_ontem DATE, p_data_hoje DATE)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Padrão Yesterday + Today (idêntico à Aula 05, parametrizado)
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

-- ==============================================================================
-- ORQUESTRAÇÃO: FAST-FORWARD INCREMENTAL (SIMULAÇÃO PRÁTICA)
-- ==============================================================================
-- 0. Reset do Estado Inicial (para garantir reproducibilidade ao rodar múltiplas vezes)
UPDATE varejo.dim_cliente SET estado = 'SP' WHERE cliente_id = 101;
UPDATE varejo.dim_produto SET categoria = 'Informática' WHERE produto_id = 'PROD001';

-- 1. Setup para o Dia 1: '2024-05-04'
INSERT INTO varejo.fato_vendas_staging (data_venda, produto_id, cliente_id, quantidade, valor_total) 
VALUES ('2024-05-04', 'PROD001', 101, 1, 3500.00);

CALL varejo.proc_upsert_clientes_scd1();
CALL varejo.proc_upsert_clientes_scd2('2024-05-04');
CALL varejo.proc_upsert_produtos_scd3('2024-05-04');
CALL varejo.proc_ingestao_fato_vendas('2024-05-04');
CALL varejo.proc_acumular_atividade(NULL, '2024-05-04');  

-- 2. Mudança nas Origens: O João (101) muda de Estado e o Notebook (PROD001) vira Gamer
UPDATE varejo.dim_cliente SET estado = 'PR' WHERE cliente_id = 101;
UPDATE varejo.dim_produto SET categoria = 'Gamer' WHERE produto_id = 'PROD001';

-- Setup para o Dia 2: '2024-05-05'
INSERT INTO varejo.fato_vendas_staging (data_venda, produto_id, cliente_id, quantidade, valor_total) 
VALUES ('2024-05-05', 'PROD001', 101, 2, 7000.00);

-- 3. Carga Incremental do Dia 2: as procedures processarão APENAS João e Notebook
CALL varejo.proc_upsert_clientes_scd1();
CALL varejo.proc_upsert_clientes_scd2('2024-05-05');
CALL varejo.proc_upsert_produtos_scd3('2024-05-05');
CALL varejo.proc_ingestao_fato_vendas('2024-05-05');
CALL varejo.proc_acumular_atividade('2024-05-04', '2024-05-05');

-- 4. Fast-Forward (Backfill) do resto do mês
-- O setup injetou vendas e atividades reais a partir de 2024-05-05 em diante
DO $$
DECLARE
    dt DATE;
BEGIN
    FOR dt IN SELECT generate_series('2024-05-06'::DATE, '2024-05-15'::DATE, '1 day'::INTERVAL)::DATE LOOP
        CALL varejo.proc_upsert_clientes_scd2(dt);
        CALL varejo.proc_upsert_produtos_scd3(dt);
        CALL varejo.proc_ingestao_fato_vendas(dt);
        CALL varejo.proc_acumular_atividade(dt - 1, dt);
    END LOOP;
END $$;


SELECT * FROM varejo.dim_cliente_type1 WHERE cliente_id = 101;
SELECT * FROM varejo.dim_cliente_type2 WHERE cliente_id = 101 ORDER BY versao;
SELECT * FROM varejo.dim_produto_type3 WHERE produto_id = 'PROD001' ORDER BY versao;
SELECT * FROM varejo.fato_vendas WHERE cliente_sk = 101;
SELECT * FROM varejo.usuarios_atividade_acumulada WHERE usuario_id = 101;
