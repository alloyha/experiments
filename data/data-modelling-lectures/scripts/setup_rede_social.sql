-- ==============================================
-- SETUP: REDE SOCIAL (DDL + DML)
-- ==============================================

-- 1. ESTRUTURA (DDL)
CREATE TABLE IF NOT EXISTS rede_social.pessoa (
    pessoa_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    idade INTEGER
);

CREATE TABLE IF NOT EXISTS rede_social.livro (
    livro_id SERIAL PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    autor VARCHAR(100),
    ano_publicacao INTEGER
);

CREATE TABLE IF NOT EXISTS rede_social.conexao_social (
    seguidor_id INTEGER REFERENCES rede_social.pessoa (pessoa_id),
    seguido_id INTEGER REFERENCES rede_social.pessoa (pessoa_id),
    forca_conexao DECIMAL(5, 2),
    data_conexao DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (seguidor_id, seguido_id),
    CHECK (seguidor_id != seguido_id)
);

CREATE TABLE IF NOT EXISTS rede_social.leitura (
    pessoa_id INTEGER REFERENCES rede_social.pessoa (pessoa_id),
    livro_id INTEGER REFERENCES rede_social.livro (livro_id),
    nota DECIMAL(3, 1),
    data_leitura DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (pessoa_id, livro_id),
    CHECK (nota >= 0 AND nota <= 5)
);

CREATE TABLE IF NOT EXISTS rede_social.genero (
    genero_id SERIAL PRIMARY KEY,
    nome VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS rede_social.livro_genero (
    livro_id INTEGER REFERENCES rede_social.livro (livro_id),
    genero_id INTEGER REFERENCES rede_social.genero (genero_id),
    PRIMARY KEY (livro_id, genero_id)
);

CREATE TABLE IF NOT EXISTS rede_social.pessoa_preferencia (
    pessoa_id INTEGER REFERENCES rede_social.pessoa (pessoa_id),
    genero_id INTEGER REFERENCES rede_social.genero (genero_id),
    PRIMARY KEY (pessoa_id, genero_id)
);

-- 2. DADOS (DML)
INSERT INTO rede_social.pessoa (nome, idade) VALUES
('Ana Silva', 28),
('Bruno Costa', 35),
('Carla Mendes', 42),
('Daniel Santos', 31),
('Evelyn Rocha', 27),
('Fabio Lima', 40),
('Gisele Pires', 33),
('Helder Souza', 29),
('Igor Dias', 36),
('Julia Barros', 24)
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.livro (titulo, autor, ano_publicacao) VALUES
('1984', 'George Orwell', 1949),
('Sapiens', 'Yuval Harari', 2011),
('Cem Anos de Solidão', 'Gabriel García Márquez', 1967),
('Admirável Mundo Novo', 'Aldous Huxley', 1932),
('Meditações', 'Marco Aurélio', 180),
('Duna', 'Frank Herbert', 1965),
('O Alquimista', 'Paulo Coelho', 1988),
('Fundação', 'Isaac Asimov', 1951)
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.genero (nome) VALUES 
('Ficção Científica'), ('Distopia'), ('História'), ('Filosofia'), ('Fantasia'), ('Drama'), ('Romance')
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.livro_genero (livro_id, genero_id) VALUES
(1, 1), (1, 2), -- 1984: Ficção Científica, Distopia
(2, 3),        -- Sapiens: História
(3, 7),        -- Cem Anos: Romance
(4, 1), (4, 2), -- Admirável Mundo: Ficção, Distopia
(5, 4),        -- Meditações: Filosofia
(6, 1),        -- Duna: Ficção
(7, 4), (7, 5), -- Alquimista: Filosofia, Fantasia
(8, 1)         -- Fundação: Ficção
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.pessoa_preferencia (pessoa_id, genero_id) VALUES
(1, 1), (1, 2), -- Ana: Ficção Científica, Distopia
(2, 3), (2, 4), -- Bruno: História, Filosofia
(3, 7),         -- Carla: Romance
(4, 1),         -- Daniel: Ficção Científica
(5, 6),         -- Evelyn: Drama
(6, 3),         -- Fabio: História
(7, 1),         -- Gisele: Ficção Científica
(8, 3), (8, 4), -- Helder: História, Filosofia
(9, 7),         -- Igor: Romance
(10, 1)         -- Julia: Ficção Científica
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.conexao_social (seguidor_id, seguido_id, forca_conexao) VALUES
(1, 2, 0.8), (2, 3, 0.9), (3, 1, 0.7), (4, 1, 0.6), (4, 2, 0.6), (4, 3, 0.5),
(3, 5, 0.85), (5, 6, 0.75), (6, 7, 0.95), (7, 8, 0.8), (8, 9, 0.6), (9, 10, 0.9),
(1, 4, 0.4), (10, 1, 0.2), (5, 1, 0.5), (2, 7, 0.6)
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.leitura (pessoa_id, livro_id, nota, data_leitura) VALUES
(1, 1, 5.0, '2024-01-15'), (1, 3, 4.5, '2024-02-20'), (2, 2, 5.0, '2024-01-10'),
(2, 1, 3.0, '2024-03-05'), (3, 3, 4.8, '2024-02-28'), (4, 1, 5.0, '2024-01-20'),
(4, 2, 2.5, '2024-03-10'), (5, 4, 5.0, '2024-04-01'), (6, 5, 4.9, '2024-04-10'),
(7, 6, 4.7, '2024-05-01'), (8, 7, 4.8, '2024-05-15'), (9, 8, 5.0, '2024-06-01'),
(10, 4, 4.2, '2024-06-10'), (7, 1, 4.5, '2024-01-05'), (3, 2, 4.0, '2024-02-01')
ON CONFLICT DO NOTHING;
