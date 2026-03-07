-- ==============================================
-- SETUP: VAREJO (MOCK DO SISTEMA OLTP ORIGEM)
-- ==============================================
-- Este script simula um banco transacional (OLTP) que serve como
-- fonte de dados para o pipeline analítico (DW) da Aula 08.

DROP SCHEMA IF EXISTS varejo CASCADE;
CREATE SCHEMA varejo;

-- ==============================================================================
-- 1. ESTRUTURA OLTP SOURCE (Tabelas que simulam o sistema transacional)
-- ==============================================================================
CREATE TABLE varejo.origem_cliente (
    cliente_id    INTEGER PRIMARY KEY,
    nome          VARCHAR(100) NOT NULL,
    estado        VARCHAR(2),
    segmento      VARCHAR(50),
    data_cadastro DATE NOT NULL
);

CREATE TABLE varejo.origem_produto (
    produto_id     VARCHAR(20) PRIMARY KEY,
    nome_produto   VARCHAR(200),
    categoria      VARCHAR(50),
    preco_sugerido DECIMAL(10, 2)
);

CREATE TABLE varejo.origem_venda (
    venda_id   SERIAL PRIMARY KEY,
    data_venda DATE NOT NULL,
    produto_id VARCHAR(20) NOT NULL REFERENCES varejo.origem_produto (produto_id),
    cliente_id INTEGER NOT NULL REFERENCES varejo.origem_cliente (cliente_id),
    quantidade INTEGER NOT NULL,
    valor_total DECIMAL(10, 2) NOT NULL
);

-- ==============================================================================
-- 2. DADOS INICIAIS (Casos de Estudo Claros: João, Maria, Pedro)
-- ==============================================================================
INSERT INTO varejo.origem_cliente (cliente_id, nome, estado, segmento, data_cadastro) VALUES
(101, 'João Silva',    'SP', 'Ouro',   '2023-01-15'),
(102, 'Maria Santos',  'RJ', 'Bronze', '2023-05-20'),
(103, 'Pedro Costa',   'MG', 'Prata',  '2024-02-10');

INSERT INTO varejo.origem_produto (produto_id, nome_produto, categoria, preco_sugerido) VALUES
('PROD001', 'Notebook Dell i5',      'Informática', 3500.00),
('PROD002', 'Mouse Logitech MX',     'Informática',  250.00),
('PROD003', 'Teclado Mecânico RGB',  'Informática',  350.00);

-- Vendas determinísticas do João para demonstrar Point-In-Time na aula:
INSERT INTO varejo.origem_venda (data_venda, produto_id, cliente_id, quantidade, valor_total) VALUES
('2024-05-04', 'PROD001', 101, 1, 3500.00),
('2024-05-05', 'PROD001', 101, 2, 7000.00),
('2024-06-15', 'PROD003', 101, 1,  350.00),
('2024-06-28', 'PROD002', 101, 1,  250.00),
('2024-07-01', 'PROD002', 101, 1,  250.00),
('2024-07-03', 'PROD001', 101, 1, 3500.00);

-- ==============================================================================
-- 3. SÍNTESE DE CLIENTES (2.000 clientes simulando volume suficiente para análises)
-- ==============================================================================
SELECT setseed(0.42);
INSERT INTO varejo.origem_cliente (cliente_id, nome, estado, segmento, data_cadastro)
SELECT
    gs AS cliente_id,
    'Cliente_' || LPAD(gs::TEXT, 6, '0') AS nome,
    (ARRAY['SP','RJ','MG','PR','SC','RS','BA','PE'])[floor(random()*8+1)] AS estado,
    CASE
        WHEN random() < 0.15 THEN 'Ouro'
        WHEN random() < 0.55 THEN 'Prata'
        ELSE 'Bronze'
    END AS segmento,
    CURRENT_DATE - (random() * 730)::INT AS data_cadastro
FROM generate_series(200, 2000) AS gs;

-- ==============================================================================
-- 4. SÍNTESE DE VENDAS (30.000 transações mapeando o período de 2 meses)
-- ==============================================================================
-- Vendas espalhadas num intervalo de 60 dias para clientes sintéticos (>= 200)
INSERT INTO varejo.origem_venda (data_venda, produto_id, cliente_id, quantidade, valor_total)
SELECT
    (DATE '2024-05-04' + FLOOR(random() * 62)::INT),
    'PROD00' || (floor(random() * 3 + 1))::TEXT,
    (random() * 1800 + 200)::INT,
    (floor(random() * 5 + 1))::INT,
    0
FROM generate_series(1, 30000) AS id;

UPDATE varejo.origem_venda SET valor_total = quantidade * 100 WHERE valor_total = 0;
