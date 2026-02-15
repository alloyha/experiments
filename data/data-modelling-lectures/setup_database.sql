-- ==============================================
-- SCRIPT DE SETUP: CURSO MODELAGEM DE DADOS
-- ==============================================

-- 1. Resetar o Ambiente
DROP TABLE IF EXISTS autor, livro, livro_autor, usuario, emprestimo, multa CASCADE;
DROP TABLE IF EXISTS dim_tempo,
dim_cliente,
dim_produto,
dim_loja,
dim_categoria,
bridge_produto_categoria,
dim_indicadores_venda,
fato_vendas,
fato_estoque_diario CASCADE;
DROP VIEW IF EXISTS v_clientes_ativos, v_vendas_completo, v_vendas_por_categoria CASCADE;

-- ==============================================
-- PARTE 1: MODELAGEM RELACIONAL (ERD) - FOCO AULAS 01-03
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
    FOREIGN KEY (livro_id) REFERENCES livro (livro_id) ON DELETE CASCADE,
    FOREIGN KEY (autor_id) REFERENCES autor (autor_id) ON DELETE CASCADE
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
    FOREIGN KEY (usuario_id) REFERENCES usuario (usuario_id),
    FOREIGN KEY (livro_id) REFERENCES livro (livro_id),
    CHECK (data_devolucao_prevista >= data_emprestimo)
);

CREATE TABLE multa (
    multa_id SERIAL PRIMARY KEY,
    emprestimo_id INTEGER UNIQUE NOT NULL,
    valor_multa DECIMAL(10, 2) CHECK (valor_multa >= 0),
    pago BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (emprestimo_id) REFERENCES emprestimo (emprestimo_id)
);

-- ==============================================
-- PARTE 2: MODELAGEM DIMENSIONAL (BI/BIG DATA) - FOCO AULAS 04-10
-- ==============================================

-- 2.1 Dimensões Puras (SCD Ready)

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
    fim_de_semana BOOLEAN
);

INSERT INTO dim_tempo (
    data_completa,
    ano,
    trimestre,
    mes,
    mes_nome,
    semana_ano,
    dia_mes,
    dia_semana,
    dia_semana_nome,
    fim_de_semana
)
SELECT
    d::DATE AS data_completa,
    EXTRACT(YEAR FROM d)::INTEGER AS ano,
    EXTRACT(QUARTER FROM d)::INTEGER AS trimestre,
    EXTRACT(MONTH FROM d)::INTEGER AS mes,
    TO_CHAR(d, 'Month') AS mes_nome,
    EXTRACT(WEEK FROM d)::INTEGER AS semana_ano,
    EXTRACT(DAY FROM d)::INTEGER AS dia_mes,
    EXTRACT(DOW FROM d)::INTEGER AS dia_semana,
    TO_CHAR(d, 'Day') AS dia_semana_nome,
    COALESCE(EXTRACT(DOW FROM d) IN (0, 6), FALSE) AS fim_de_semana
FROM GENERATE_SERIES('2024-01-01'::DATE, '2025-12-31'::DATE, '1 day') AS d;

-- dim_cliente: Padronizado com SK e Natural Key
CREATE TABLE dim_cliente (
    cliente_sk SERIAL PRIMARY KEY,           -- Surrogate Key
    cliente_id INTEGER NOT NULL,             -- Natural Key (Idempotente)
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    cidade VARCHAR(100),
    estado VARCHAR(2),
    segmento VARCHAR(50),
    data_inicio DATE DEFAULT CURRENT_DATE,   -- Validade DE
    data_fim DATE DEFAULT '9999-12-31',      -- Validade ATÉ
    versao INTEGER DEFAULT 1,
    registro_ativo BOOLEAN DEFAULT TRUE
);

INSERT INTO dim_cliente (cliente_id, nome, email, cidade, estado, segmento) VALUES
(101, 'João Silva', 'joao@email.com', 'São Paulo', 'SP', 'Ouro'),
(102, 'Maria Santos', 'maria@email.com', 'Rio de Janeiro', 'RJ', 'Bronze'),
(103, 'Pedro Costa', 'pedro@email.com', 'Belo Horizonte', 'MG', 'Prata');

-- dim_produto: Padronizado com SK e Natural Key
CREATE TABLE dim_produto (
    produto_sk SERIAL PRIMARY KEY,           -- Surrogate Key
    produto_id VARCHAR(20) UNIQUE,           -- Natural Key (Idempotente)
    nome_produto VARCHAR(200),
    categoria VARCHAR(50),
    marca VARCHAR(50),
    preco_sugerido DECIMAL(10, 2)
);

INSERT INTO dim_produto (produto_id, nome_produto, categoria, marca, preco_sugerido) VALUES
('PROD001', 'Notebook Dell i5', 'Informática', 'Dell', 3500.00),
('PROD002', 'Mouse Logitech MX', 'Informática', 'Logitech', 250.00),
('PROD003', 'Teclado Mecânico RGB', 'Informática', 'Redragon', 350.00);

CREATE TABLE dim_categoria (
    categoria_id SERIAL PRIMARY KEY,
    nome_categoria VARCHAR(100),
    departamento VARCHAR(50)
);

INSERT INTO dim_categoria (nome_categoria, departamento) VALUES
('Informática', 'Tecnologia'),
('Eletrônicos', 'Tecnologia'),
('Livros Técnicos', 'Educação'),
('Games', 'Entretenimento');

CREATE TABLE dim_loja (
    loja_id SERIAL PRIMARY KEY,
    nome_loja VARCHAR(100),
    cidade VARCHAR(100),
    estado VARCHAR(2),
    regiao VARCHAR(50)
);

INSERT INTO dim_loja (nome_loja, cidade, estado, regiao) VALUES
('Loja SP Centro', 'São Paulo', 'SP', 'Sudeste'),
('Loja RJ Botafogo', 'Rio de Janeiro', 'RJ', 'Sudeste'),
('Loja BH Savassi', 'Belo Horizonte', 'MG', 'Sudeste');

-- 2.2 Relacionamentos
CREATE TABLE bridge_produto_categoria (
    produto_sk INTEGER REFERENCES dim_produto (produto_sk),
    categoria_id INTEGER REFERENCES dim_categoria (categoria_id),
    peso_alocacao DECIMAL(5, 4) DEFAULT 1.0,
    PRIMARY KEY (produto_sk, categoria_id)
);

-- 2.3 Fatos
CREATE TABLE fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    tempo_id INTEGER REFERENCES dim_tempo (tempo_id),
    produto_sk INTEGER REFERENCES dim_produto (produto_sk),
    cliente_sk INTEGER REFERENCES dim_cliente (cliente_sk),
    loja_id INTEGER REFERENCES dim_loja (loja_id),
    quantidade INTEGER NOT NULL,
    valor_unitario DECIMAL(10, 2) NOT NULL,
    valor_total DECIMAL(10, 2) NOT NULL
);

-- Popular Fato Vendas
INSERT INTO fato_vendas (
    tempo_id, produto_sk, cliente_sk, loja_id, quantidade, valor_unitario, valor_total
)
SELECT
    t.tempo_id,
    p.produto_sk,
    c.cliente_sk,
    l.loja_id,
    (RANDOM() * 2 + 1)::INTEGER AS quantidade,
    p.preco_sugerido AS valor_unitario,
    0 AS valor_total
FROM dim_tempo AS t
CROSS JOIN dim_produto AS p
CROSS JOIN dim_cliente AS c
CROSS JOIN dim_loja AS l
WHERE t.data_completa BETWEEN '2024-06-01' AND '2024-06-01';

UPDATE fato_vendas SET valor_total = quantidade * valor_unitario;

-- 2.4 Views
CREATE VIEW v_vendas_completo AS
SELECT
    fv.venda_id,
    dt.data_completa,
    dp.nome_produto,
    dc.nome AS cliente_nome,
    dl.nome_loja,
    fv.quantidade,
    fv.valor_total
FROM fato_vendas AS fv
INNER JOIN dim_tempo AS dt ON fv.tempo_id = dt.tempo_id
INNER JOIN dim_produto AS dp ON fv.produto_sk = dp.produto_sk
INNER JOIN dim_cliente AS dc ON fv.cliente_sk = dc.cliente_sk
INNER JOIN dim_loja AS dl ON fv.loja_id = dl.loja_id;
