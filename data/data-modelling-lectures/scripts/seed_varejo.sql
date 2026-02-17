-- ==============================================
-- SEED: VAREJO
-- ==============================================
-- Dataset completo para modelagem dimensional

CREATE TABLE IF NOT EXISTS varejo.dim_cliente (
    cliente_sk SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    nome VARCHAR(100) NOT NULL,
    estado VARCHAR(2),
    segmento VARCHAR(50)
);

INSERT INTO varejo.dim_cliente (cliente_id, nome, estado, segmento) VALUES
(101, 'João Silva', 'SP', 'Ouro'),
(102, 'Maria Santos', 'RJ', 'Bronze'),
(103, 'Pedro Costa', 'MG', 'Prata')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS varejo.dim_produto (
    produto_sk SERIAL PRIMARY KEY,
    produto_id VARCHAR(20) UNIQUE,
    nome_produto VARCHAR(200),
    categoria VARCHAR(50),
    preco_sugerido DECIMAL(10, 2)
);

INSERT INTO varejo.dim_produto (produto_id, nome_produto, categoria, preco_sugerido) VALUES
('PROD001', 'Notebook Dell i5', 'Informática', 3500.00),
('PROD002', 'Mouse Logitech MX', 'Informática', 250.00),
('PROD003', 'Teclado Mecânico RGB', 'Informática', 350.00)
ON CONFLICT (produto_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS varejo.fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    data_venda DATE NOT NULL,
    produto_sk INTEGER REFERENCES varejo.dim_produto (produto_sk),
    cliente_sk INTEGER REFERENCES varejo.dim_cliente (cliente_sk),
    quantidade INTEGER NOT NULL,
    valor_total DECIMAL(10, 2) NOT NULL
);

-- Inserir vendas apenas se a tabela estiver vazia (evita duplicação)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM varejo.fato_vendas LIMIT 1) THEN
        INSERT INTO varejo.fato_vendas (data_venda, produto_sk, cliente_sk, quantidade, valor_total)
        SELECT
            '2024-06-01'::DATE,
            p.produto_sk,
            c.cliente_sk,
            (RANDOM() * 5 + 1)::INTEGER,
            0
        FROM varejo.dim_produto p, varejo.dim_cliente c
        LIMIT 10;
        
        UPDATE varejo.fato_vendas SET valor_total = quantidade * 100;
    END IF;
END $$;

CREATE OR REPLACE VIEW varejo.v_vendas_completo AS
SELECT
    fv.venda_id,
    fv.data_venda,
    EXTRACT(YEAR FROM fv.data_venda) AS ano,
    TO_CHAR(fv.data_venda, 'Month') AS mes_nome,
    dp.nome_produto,
    dc.nome AS cliente_nome,
    fv.valor_total
FROM varejo.fato_vendas fv
JOIN varejo.dim_produto dp ON fv.produto_sk = dp.produto_sk
JOIN varejo.dim_cliente dc ON fv.cliente_sk = dc.cliente_sk;
