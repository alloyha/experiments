-- ==============================================
-- SETUP: VAREJO (DDL + DML) - BIG DATA VOLUME
-- ==============================================

-- 1. ESTRUTURA BÁSICA (INTRODUÇÃO)
CREATE TABLE IF NOT EXISTS varejo.dim_cliente (
    cliente_sk SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    nome VARCHAR(100) NOT NULL,
    estado VARCHAR(2),
    segmento VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS varejo.dim_produto (
    produto_sk SERIAL PRIMARY KEY,
    produto_id VARCHAR(20) UNIQUE,
    nome_produto VARCHAR(200),
    categoria VARCHAR(50),
    preco_sugerido DECIMAL(10, 2)
);

CREATE TABLE IF NOT EXISTS varejo.fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    data_venda DATE NOT NULL,
    produto_sk INTEGER REFERENCES varejo.dim_produto (produto_sk),
    cliente_sk INTEGER REFERENCES varejo.dim_cliente (cliente_sk),
    quantidade INTEGER NOT NULL,
    valor_total DECIMAL(10, 2) NOT NULL
);

-- 2. ESTRUTURA PARA EXERCÍCIOS DE BIG DATA (DE PARA AULAS 05 e 06)
CREATE TABLE IF NOT EXISTS varejo.usuarios_dim (
    usuario_id   INTEGER      PRIMARY KEY,
    nome         TEXT         NOT NULL,
    segmento     TEXT         NOT NULL,   -- 'premium' | 'standard' | 'trial'
    data_cadastro DATE        NOT NULL
);

CREATE TABLE IF NOT EXISTS varejo.usuarios_atividade_fato (
    usuario_id   INTEGER  NOT NULL,
    data_evento  DATE     NOT NULL,
    PRIMARY KEY (usuario_id, data_evento)
);

-- 3. DADOS DIMENSIONAIS (POPULAÇÃO INICIAL)
INSERT INTO varejo.dim_cliente (cliente_id, nome, estado, segmento) VALUES
(101, 'João Silva', 'SP', 'Ouro'),
(102, 'Maria Santos', 'RJ', 'Bronze'),
(103, 'Pedro Costa', 'MG', 'Prata')
ON CONFLICT DO NOTHING;

INSERT INTO varejo.dim_produto (produto_id, nome_produto, categoria, preco_sugerido) VALUES
('PROD001', 'Notebook Dell i5', 'Informática', 3500.00),
('PROD002', 'Mouse Logitech MX', 'Informática', 250.00),
('PROD003', 'Teclado Mecânico RGB', 'Informática', 350.00)
ON CONFLICT (produto_id) DO NOTHING;

-- 4. SÍNTESE MASSIVA DE USUÁRIOS (10.000 usuários)
SELECT setseed(0.42);
INSERT INTO varejo.usuarios_dim (usuario_id, nome, segmento, data_cadastro)
SELECT
    gs                                                          AS usuario_id,
    'Usuario_' || LPAD(gs::TEXT, 6, '0')                       AS nome,
    CASE
        WHEN random() < 0.15 THEN 'premium'
        WHEN random() < 0.55 THEN 'standard'
        ELSE 'trial'
    END                                                         AS segmento,
    CURRENT_DATE - (random() * 730)::INT                        AS data_cadastro
FROM generate_series(1, 10000) AS gs
ON CONFLICT DO NOTHING;

-- 5. SÍNTESE MASSIVA DE ATIVIDADE (120.000 eventos)
INSERT INTO varejo.usuarios_atividade_fato (usuario_id, data_evento)
SELECT DISTINCT
    (random() * 9999 + 1)::INT   AS usuario_id,
    (DATE '2024-05-05' + (gs % 30))::DATE  AS data_evento
FROM generate_series(1, 120000) AS gs
ON CONFLICT DO NOTHING;

-- Injeções Determinísticas para Testes de Churn/Retention
INSERT INTO varejo.usuarios_atividade_fato (usuario_id, data_evento)
SELECT 1, DATE '2024-05-05' + i FROM generate_series(0, 29) AS i ON CONFLICT DO NOTHING; -- Power User
INSERT INTO varejo.usuarios_atividade_fato (usuario_id, data_evento)
SELECT 2, DATE '2024-05-05' + i FROM generate_series(0, 22) AS i ON CONFLICT DO NOTHING; -- Churn
INSERT INTO varejo.usuarios_atividade_fato (usuario_id, data_evento)
VALUES (3, '2024-06-03') ON CONFLICT DO NOTHING; -- New User

-- 6. SÍNTESE MASSIVA DE VENDAS (100.000 records)
INSERT INTO varejo.fato_vendas (data_venda, produto_sk, cliente_sk, quantidade, valor_total)
SELECT 
    (CURRENT_DATE - (random() * 30)::INT * INTERVAL '1 day')::DATE as data_venda,
    (random() * 2 + 1)::INT as produto_sk,
    (random() * 2 + 1)::INT as cliente_sk,
    (random() * 5 + 1)::INT as quantidade,
    0 as valor_total
FROM generate_series(1, 100000) AS id;

UPDATE varejo.fato_vendas SET valor_total = quantidade * 100 WHERE valor_total = 0;
