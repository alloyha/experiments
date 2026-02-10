-- ==============================================================================
-- Aula 2: Entity Relationship Diagrams - ERD Parte 1
-- ==============================================================================

-- Exemplo: Tabela Simples (Entidade)
CREATE TABLE IF NOT EXISTS cliente (
    cliente_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    data_cadastro DATE DEFAULT CURRENT_DATE
);

-- Exemplo: Chaves Estrangeiras (Relacionamento 1:N)
CREATE TABLE IF NOT EXISTS pedido (
    pedido_id SERIAL PRIMARY KEY,
    cliente_id INTEGER NOT NULL,
    data_pedido TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (cliente_id) REFERENCES cliente(cliente_id)
);

-- Exemplo: Tabela Associativa (Relacionamento N:N)
-- Produto <-> Categoria
-- Tabelas de apoio para o exemplo N:N
CREATE TABLE IF NOT EXISTS produto (produto_id SERIAL PRIMARY KEY, nome VARCHAR(100));
CREATE TABLE IF NOT EXISTS categoria (categoria_id SERIAL PRIMARY KEY, nome VARCHAR(100));

CREATE TABLE IF NOT EXISTS produto_categoria (
    produto_id INTEGER,
    categoria_id INTEGER,
    PRIMARY KEY (produto_id, categoria_id),
    FOREIGN KEY (produto_id) REFERENCES produto(produto_id),
    FOREIGN KEY (categoria_id) REFERENCES categoria(categoria_id)
);
