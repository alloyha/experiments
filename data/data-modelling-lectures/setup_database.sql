-- ==============================================
-- SCRIPT COMPLETO: CURSO MODELAGEM DE DADOS
-- ==============================================

-- Drop tudo se existir (Reset)
DROP TABLE IF EXISTS autor, livro, livro_autor, usuario, emprestimo, multa CASCADE;
DROP TABLE IF EXISTS dim_tempo, dim_cliente, dim_produto, dim_loja, dim_categoria, bridge_produto_categoria, dim_indicadores_venda, fato_vendas, fato_estoque_diario CASCADE;
DROP VIEW IF EXISTS v_clientes_ativos, v_vendas_completo, v_vendas_por_categoria CASCADE;
DROP SCHEMA IF EXISTS curso_modelagem CASCADE;

-- ==============================================
-- PARTE 1: ERD - SISTEMA BIBLIOTECA
-- ==============================================

CREATE TABLE autor (
    autor_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    nacionalidade VARCHAR(50),
    data_nascimento DATE
);

CREATE TABLE livro (
    livro_id SERIAL PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    isbn VARCHAR(13) UNIQUE,
    ano_publicacao INTEGER,
    quantidade_disponivel INTEGER DEFAULT 0,
    CHECK (quantidade_disponivel >= 0)
);

CREATE TABLE livro_autor (
    livro_id INTEGER,
    autor_id INTEGER,
    PRIMARY KEY (livro_id, autor_id),
    FOREIGN KEY (livro_id) REFERENCES livro(livro_id) ON DELETE CASCADE,
    FOREIGN KEY (autor_id) REFERENCES autor(autor_id) ON DELETE CASCADE
);

CREATE TABLE usuario (
    usuario_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    tipo VARCHAR(20) CHECK (tipo IN ('aluno', 'professor')),
    email VARCHAR(100) UNIQUE,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

CREATE TABLE emprestimo (
    emprestimo_id SERIAL PRIMARY KEY,
    usuario_id INTEGER NOT NULL,
    livro_id INTEGER NOT NULL,
    data_emprestimo DATE DEFAULT CURRENT_DATE,
    data_devolucao_prevista DATE NOT NULL,
    data_devolucao_real DATE,
    FOREIGN KEY (usuario_id) REFERENCES usuario(usuario_id),
    FOREIGN KEY (livro_id) REFERENCES livro(livro_id),
    CHECK (data_devolucao_prevista >= data_emprestimo)
);

CREATE TABLE multa (
    multa_id SERIAL PRIMARY KEY,
    emprestimo_id INTEGER UNIQUE NOT NULL,
    valor_multa DECIMAL(10,2) CHECK (valor_multa >= 0),
    pago BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (emprestimo_id) REFERENCES emprestimo(emprestimo_id)
);

-- ==============================================
-- PARTE 2: MODELAGEM DIMENSIONAL - VENDAS
-- ==============================================

-- Dimensão Tempo
CREATE TABLE dim_tempo (
    tempo_id SERIAL PRIMARY KEY,
    data_completa DATE UNIQUE NOT NULL,
    ano INTEGER,
    trimestre INTEGER,
    mes INTEGER,
    mes_nome VARCHAR(20),
    semana_ano INTEGER,
    dia_mes INTEGER,
    dia_semana INTEGER,
    dia_semana_nome VARCHAR(20),
    dia_ano INTEGER,
    fim_de_semana BOOLEAN,
    feriado BOOLEAN,
    nome_feriado VARCHAR(50)
);

-- Popular dimensão tempo
INSERT INTO dim_tempo (
    data_completa, ano, trimestre, mes, mes_nome, 
    semana_ano, dia_mes, dia_semana, dia_semana_nome, 
    dia_ano, fim_de_semana
)
SELECT 
    d::date,
    EXTRACT(YEAR FROM d)::INTEGER,
    EXTRACT(QUARTER FROM d)::INTEGER,
    EXTRACT(MONTH FROM d)::INTEGER,
    TO_CHAR(d, 'Month'),
    EXTRACT(WEEK FROM d)::INTEGER,
    EXTRACT(DAY FROM d)::INTEGER,
    EXTRACT(DOW FROM d)::INTEGER,
    TO_CHAR(d, 'Day'),
    EXTRACT(DOY FROM d)::INTEGER,
    CASE WHEN EXTRACT(DOW FROM d) IN (0,6) THEN TRUE ELSE FALSE END
FROM generate_series('2024-01-01'::date, '2025-12-31'::date, '1 day') d;

-- Dimensão Cliente (com SCD Type 2)
CREATE TABLE dim_cliente (
    cliente_sk SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    telefone VARCHAR(20),
    cidade VARCHAR(100),
    estado VARCHAR(2),
    segmento VARCHAR(50),
    data_inicio DATE NOT NULL,
    data_fim DATE NOT NULL DEFAULT '9999-12-31',
    versao INTEGER NOT NULL DEFAULT 1,
    registro_ativo BOOLEAN NOT NULL DEFAULT TRUE,
    data_carga TIMESTAMP DEFAULT NOW(),
    UNIQUE(cliente_id, versao),
    CHECK (data_fim >= data_inicio)
);

CREATE INDEX idx_cliente_natural_ativo 
ON dim_cliente(cliente_id, registro_ativo) 
WHERE registro_ativo = TRUE;

-- Inserir clientes especiais
INSERT INTO dim_cliente (cliente_sk, cliente_id, nome, segmento, data_inicio, versao) VALUES
(-1, -1, 'Desconhecido', 'UNKNOWN', '1900-01-01', 1),
(0, 0, 'Não se Aplica', 'N/A', '1900-01-01', 1);

-- Dimensão Produto
CREATE TABLE dim_produto (
    produto_id SERIAL PRIMARY KEY,
    produto_sk VARCHAR(20) UNIQUE,
    nome_produto VARCHAR(200),
    descricao TEXT,
    categoria VARCHAR(50),
    subcategoria VARCHAR(50),
    marca VARCHAR(50),
    preco_sugerido DECIMAL(10,2),
    ativo BOOLEAN DEFAULT TRUE
);

-- Dimensão Loja
CREATE TABLE dim_loja (
    loja_id SERIAL PRIMARY KEY,
    nome_loja VARCHAR(100),
    cidade VARCHAR(100),
    estado VARCHAR(50),
    regiao VARCHAR(50),
    gerente VARCHAR(100)
);

-- Dimensão Categoria (para bridge)
CREATE TABLE dim_categoria (
    categoria_id SERIAL PRIMARY KEY,
    nome_categoria VARCHAR(100),
    departamento VARCHAR(50)
);

-- Bridge Table: Produto-Categoria
CREATE TABLE bridge_produto_categoria (
    produto_id INTEGER,
    categoria_id INTEGER,
    peso_alocacao DECIMAL(5,4),
    PRIMARY KEY (produto_id, categoria_id),
    FOREIGN KEY (produto_id) REFERENCES dim_produto(produto_id),
    FOREIGN KEY (categoria_id) REFERENCES dim_categoria(categoria_id),
    CHECK (peso_alocacao > 0 AND peso_alocacao <= 1)
);

-- Dimensão Junk (indicadores de venda)
CREATE TABLE dim_indicadores_venda (
    indicador_id SERIAL PRIMARY KEY,
    forma_pagamento VARCHAR(20),
    tipo_frete VARCHAR(20),
    tem_cupom BOOLEAN,
    eh_primeira_compra BOOLEAN
);

-- Popular combinações comuns de indicadores
INSERT INTO dim_indicadores_venda (forma_pagamento, tipo_frete, tem_cupom, eh_primeira_compra) VALUES
('cartao', 'normal', FALSE, FALSE),
('cartao', 'expresso', FALSE, FALSE),
('pix', 'normal', TRUE, FALSE),
('boleto', 'normal', FALSE, TRUE);

-- Fato Vendas (Transacional)
CREATE TABLE fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    tempo_id INTEGER NOT NULL,
    produto_id INTEGER NOT NULL,
    cliente_sk INTEGER NOT NULL,
    loja_id INTEGER NOT NULL,
    indicador_id INTEGER NOT NULL,
    -- Métricas aditivas
    quantidade INTEGER NOT NULL CHECK (quantidade > 0),
    valor_unitario DECIMAL(10,2) NOT NULL,
    valor_desconto DECIMAL(10,2) DEFAULT 0,
    valor_total DECIMAL(10,2) NOT NULL,
    custo_produto DECIMAL(10,2),
    -- Dimensão degenerada
    numero_nota_fiscal VARCHAR(20),
    -- FKs
    FOREIGN KEY (tempo_id) REFERENCES dim_tempo(tempo_id),
    FOREIGN KEY (produto_id) REFERENCES dim_produto(produto_id),
    FOREIGN KEY (cliente_sk) REFERENCES dim_cliente(cliente_sk),
    FOREIGN KEY (loja_id) REFERENCES dim_loja(loja_id),
    FOREIGN KEY (indicador_id) REFERENCES dim_indicadores_venda(indicador_id)
);

-- Índices para performance
CREATE INDEX idx_fato_vendas_tempo ON fato_vendas(tempo_id);
CREATE INDEX idx_fato_vendas_produto ON fato_vendas(produto_id);
CREATE INDEX idx_fato_vendas_cliente ON fato_vendas(cliente_sk);
CREATE INDEX idx_fato_vendas_loja ON fato_vendas(loja_id);

-- Fato Estoque Diário (Snapshot Periódico)
CREATE TABLE fato_estoque_diario (
    snapshot_id SERIAL PRIMARY KEY,
    data_snapshot_id INTEGER NOT NULL,
    produto_id INTEGER NOT NULL,
    loja_id INTEGER NOT NULL,
    -- Métricas semi-aditivas
    quantidade_estoque INTEGER NOT NULL,
    valor_estoque DECIMAL(12,2),
    -- FKs
    FOREIGN KEY (data_snapshot_id) REFERENCES dim_tempo(tempo_id),
    FOREIGN KEY (produto_id) REFERENCES dim_produto(produto_id),
    FOREIGN KEY (loja_id) REFERENCES dim_loja(loja_id),
    UNIQUE(data_snapshot_id, produto_id, loja_id)
);

-- ==============================================
-- DADOS DE EXEMPLO
-- ==============================================

-- Inserir lojas
INSERT INTO dim_loja (nome_loja, cidade, estado, regiao, gerente) VALUES
('Loja SP Centro', 'São Paulo', 'SP', 'Sudeste', 'Ana Silva'),
('Loja RJ Botafogo', 'Rio de Janeiro', 'RJ', 'Sudeste', 'Carlos Souza'),
('Loja BH Savassi', 'Belo Horizonte', 'MG', 'Sudeste', 'Maria Santos');

-- Inserir categorias
INSERT INTO dim_categoria (nome_categoria, departamento) VALUES
('Informática', 'Tecnologia'),
('Eletrônicos', 'Tecnologia'),
('Livros Técnicos', 'Educação'),
('Games', 'Entretenimento');

-- Inserir produtos
INSERT INTO dim_produto (produto_sk, nome_produto, categoria, subcategoria, marca, preco_sugerido) VALUES
('PROD001', 'Notebook Dell i5', 'Informática', 'Notebooks', 'Dell', 3500.00),
('PROD002', 'Mouse Logitech MX', 'Informática', 'Periféricos', 'Logitech', 250.00),
('PROD003', 'Teclado Mecânico RGB', 'Informática', 'Periféricos', 'Redragon', 350.00);

-- Bridge: produtos com múltiplas categorias
INSERT INTO bridge_produto_categoria (produto_id, categoria_id, peso_alocacao) VALUES
(1, 1, 0.7),  -- Notebook: 70% Informática
(1, 2, 0.3),  -- Notebook: 30% Eletrônicos
(2, 1, 1.0),  -- Mouse: 100% Informática
(3, 1, 0.8),  -- Teclado: 80% Informática
(3, 4, 0.2);  -- Teclado: 20% Games

-- Inserir clientes
INSERT INTO dim_cliente (cliente_id, nome, email, cidade, estado, segmento, data_inicio) VALUES
(101, 'João Silva', 'joao@email.com', 'São Paulo', 'SP', 'Bronze', '2024-01-15'),
(102, 'Maria Santos', 'maria@email.com', 'Rio de Janeiro', 'RJ', 'Ouro', '2024-02-01'),
(103, 'Pedro Costa', 'pedro@email.com', 'Belo Horizonte', 'MG', 'Prata', '2024-03-10');

-- Exemplo de mudança SCD Type 2 para cliente 101
UPDATE dim_cliente 
SET data_fim = '2024-06-30', registro_ativo = FALSE
WHERE cliente_id = 101 AND registro_ativo = TRUE;

INSERT INTO dim_cliente (cliente_id, nome, email, cidade, estado, segmento, data_inicio, versao)
VALUES (101, 'João Silva', 'joao@email.com', 'São Paulo', 'SP', 'Ouro', '2024-07-01', 2);

-- Inserir vendas de exemplo
INSERT INTO fato_vendas (
    tempo_id, produto_id, cliente_sk, loja_id, indicador_id,
    quantidade, valor_unitario, valor_desconto, valor_total, numero_nota_fiscal
)
SELECT 
    t.tempo_id,
    p.produto_id,
    c.cliente_sk,
    l.loja_id,
    1, -- indicador padrão
    (RANDOM() * 3 + 1)::INTEGER,
    p.preco_sugerido,
    (RANDOM() * 100)::DECIMAL(10,2),
    p.preco_sugerido * (RANDOM() * 3 + 1),
    'NF-' || LPAD((ROW_NUMBER() OVER ())::TEXT, 6, '0')
FROM dim_tempo t
CROSS JOIN dim_produto p
CROSS JOIN dim_cliente c
CROSS JOIN dim_loja l
WHERE t.data_completa BETWEEN '2024-06-01' AND '2024-06-07'
  AND c.registro_ativo = TRUE
  AND RANDOM() < 0.1  -- 10% de chance para não gerar muitos dados
LIMIT 100;

-- ==============================================
-- VIEWS ÚTEIS
-- ==============================================

-- View: Clientes ativos
CREATE VIEW v_clientes_ativos AS
SELECT 
    cliente_sk,
    cliente_id,
    nome,
    email,
    cidade,
    estado,
    segmento
FROM dim_cliente
WHERE registro_ativo = TRUE;

-- View: Vendas com todas as dimensões
CREATE VIEW v_vendas_completo AS
SELECT 
    fv.venda_id,
    fv.numero_nota_fiscal,
    dt.data_completa,
    dt.ano,
    dt.mes_nome,
    dp.nome_produto,
    dp.categoria,
    dc.nome as cliente_nome,
    dc.segmento as cliente_segmento,
    dl.nome_loja,
    dl.cidade as loja_cidade,
    fv.quantidade,
    fv.valor_total,
    fv.valor_total - fv.valor_desconto as valor_liquido
FROM fato_vendas fv
JOIN dim_tempo dt ON fv.tempo_id = dt.tempo_id
JOIN dim_produto dp ON fv.produto_id = dp.produto_id
JOIN dim_cliente dc ON fv.cliente_sk = dc.cliente_sk
JOIN dim_loja dl ON fv.loja_id = dl.loja_id;

-- View: Vendas por categoria (com bridge)
CREATE VIEW v_vendas_por_categoria AS
SELECT 
    dc.nome_categoria,
    dt.ano,
    dt.mes,
    COUNT(*) as qtd_vendas,
    SUM(fv.valor_total * bpc.peso_alocacao) as valor_alocado
FROM fato_vendas fv
JOIN dim_tempo dt ON fv.tempo_id = dt.tempo_id
JOIN bridge_produto_categoria bpc ON fv.produto_id = bpc.produto_id
JOIN dim_categoria dc ON bpc.categoria_id = dc.categoria_id
GROUP BY dc.nome_categoria, dt.ano, dt.mes;
