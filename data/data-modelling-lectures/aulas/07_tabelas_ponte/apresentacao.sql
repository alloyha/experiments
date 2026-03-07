-- ==============================================================================
-- Aula 7: Tabelas Ponte (Bridge Tables) vs Array Metrics (Big Data)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- MODELO 1: DIMENSÃO MULTI-VALORADA (FATO ↔ BRIDGE ↔ DIMENSÃO)
-- ------------------------------------------------------------------------------
-- Cenário: Uma Consulta Médica (Fato) pode envolver múltiplos Diagnósticos (Dim).
-- Solução: Tabela Fato ganha uma 'Chave de Grupo'. A Bridge mapeia essa chave.

DROP TABLE IF EXISTS bridge_grupo_diagnostico CASCADE;
DROP TABLE IF EXISTS fato_consulta CASCADE;
DROP TABLE IF EXISTS dim_diagnostico CASCADE;

-- 1. Dimensão
CREATE TABLE dim_diagnostico (
    diagnostico_id SERIAL PRIMARY KEY,
    codigo_cid VARCHAR(10),
    descricao VARCHAR(200)
);

INSERT INTO dim_diagnostico (codigo_cid, descricao) VALUES
('J00', 'Resfriado comum'),
('J01', 'Sinusite aguda'),
('J02', 'Faringite aguda');

-- 2. Tabela Bridge (Mapeia a Group Key para as FKs da Dimensão)
CREATE TABLE bridge_grupo_diagnostico (
    grupo_diagnostico_key INTEGER,
    diagnostico_id INTEGER REFERENCES dim_diagnostico(diagnostico_id),
    peso_alocacao DECIMAL(5, 4),  -- Essencial para não inflar as métricas!
    PRIMARY KEY (grupo_diagnostico_key, diagnostico_id)
);

-- Criar um grupo (Ex: Grupo 100 refere-se a Resfriado + Faringite, 50% cada)
INSERT INTO bridge_grupo_diagnostico (grupo_diagnostico_key, diagnostico_id, peso_alocacao) VALUES
(100, 1, 0.5000), 
(100, 3, 0.5000);

-- 3. Tabela Fato (Contém a Group Key no lugar da chave de dimensão direta)
CREATE TABLE fato_consulta (
    consulta_id SERIAL PRIMARY KEY,
    data_consulta DATE,
    medico_id INTEGER,
    paciente_id INTEGER,
    grupo_diagnostico_key INTEGER, -- Ligação com o Grupo (Bridge)
    valor_consulta DECIMAL(10, 2)
);

INSERT INTO fato_consulta (data_consulta, medico_id, paciente_id, grupo_diagnostico_key, valor_consulta)
VALUES ('2024-03-01', 1, 10, 100, 200.00);

-- QUERY CORRETA (Sem o bridge com pesos, o valor passaria para R$ 400 em análises por diagnóstico)
/*
SELECT 
    d.descricao, 
    SUM(f.valor_consulta * b.peso_alocacao) as valor_alocado
FROM fato_consulta f
JOIN bridge_grupo_diagnostico b ON f.grupo_diagnostico_key = b.grupo_diagnostico_key
JOIN dim_diagnostico d ON b.diagnostico_id = d.diagnostico_id
GROUP BY d.descricao;
*/

-- ------------------------------------------------------------------------------
-- ABORDAGEM BIG DATA - ARRAY METRICS (No-Shuffle)
-- ------------------------------------------------------------------------------
-- Cenário: Mesmo problema Multi-valorado (Diagnósticos), mas sem Join e com Estruturas Aninhadas.

DROP TABLE IF EXISTS fato_consulta_bigdata CASCADE;

CREATE TABLE fato_consulta_bigdata (
    consulta_id SERIAL PRIMARY KEY,
    valor_consulta DECIMAL(10, 2),
    diagnosticos_cid TEXT[],  -- Array de códigos na própria Fato
    pesos_relevancia DECIMAL[] -- Array de pesos equivalentes
);

INSERT INTO fato_consulta_bigdata (valor_consulta, diagnosticos_cid, pesos_relevancia)
VALUES (200.00, ARRAY['J00', 'J02'], ARRAY[0.5, 0.5]);

-- QUERY BIG DATA (EXPLODE/UNNEST)
/*
SELECT 
    cid_diagnostico,
    SUM(valor_consulta * peso) as valor_alocado
FROM fato_consulta_bigdata, 
     UNNEST(diagnosticos_cid, pesos_relevancia) AS t(cid_diagnostico, peso)
GROUP BY cid_diagnostico;
*/

-- ------------------------------------------------------------------------------
-- MODELO 2: BRIDGE ENTRE DIMENSÕES (FATO ↔ DIM PRINCIPAL ↔ BRIDGE ↔ DIM SECUNDÁRIA)
-- ------------------------------------------------------------------------------
-- Cenário: Conta Bancária com múltiplos Titulares.
-- Solução: Fato aponta para Conta. Bridge conecta FK_Conta a FK_Cliente.

DROP TABLE IF EXISTS bridge_conta_titular CASCADE;
DROP TABLE IF EXISTS fato_transacao CASCADE;
DROP TABLE IF EXISTS dim_conta CASCADE;
DROP TABLE IF EXISTS dim_cliente CASCADE;

CREATE TABLE dim_cliente (
    cliente_id SERIAL PRIMARY KEY,
    nome_cliente VARCHAR(100)
);

CREATE TABLE dim_conta (
    conta_id SERIAL PRIMARY KEY,
    agencia VARCHAR(10),
    numero_conta VARCHAR(20)
);

CREATE TABLE bridge_conta_titular (
    conta_id INTEGER REFERENCES dim_conta(conta_id),
    cliente_id INTEGER REFERENCES dim_cliente(cliente_id),
    peso_alocacao DECIMAL(5, 4),
    PRIMARY KEY (conta_id, cliente_id)
);

CREATE TABLE fato_transacao (
    transacao_id SERIAL PRIMARY KEY,
    conta_id INTEGER REFERENCES dim_conta(conta_id),
    valor_transacao DECIMAL(10, 2),
    data_transacao TIMESTAMP
);

