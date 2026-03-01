-- ==============================================
-- SETUP: REDE SOCIAL (DDL + DML)
-- ==============================================

-- 1. ESTRUTURA (DDL)
CREATE TABLE IF NOT EXISTS rede_social.pessoa (
    pessoa_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    idade INTEGER,
    genero_favorito VARCHAR(50)
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

-- 2. DADOS (DML)
INSERT INTO rede_social.pessoa (nome, idade, genero_favorito) VALUES
('Ana Silva', 28, 'Ficção'),
('Bruno Costa', 35, 'Não-ficção'),
('Carla Mendes', 42, 'Romance'),
('Daniel Santos', 31, 'Ficção')
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.livro (titulo, autor, ano_publicacao) VALUES
('1984', 'George Orwell', 1949),
('Sapiens', 'Yuval Harari', 2011),
('Cem Anos de Solidão', 'Gabriel García Márquez', 1967)
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.conexao_social (seguidor_id, seguido_id, forca_conexao) VALUES
(1, 2, 0.8), (2, 3, 0.9), (3, 1, 0.7), (4, 1, 0.6), (4, 2, 0.6), (4, 3, 0.5)
ON CONFLICT DO NOTHING;

INSERT INTO rede_social.leitura (pessoa_id, livro_id, nota, data_leitura) VALUES
(1, 1, 5.0, '2024-01-15'), (1, 3, 4.5, '2024-02-20'), (2, 2, 5.0, '2024-01-10'),
(2, 1, 3.0, '2024-03-05'), (3, 3, 4.8, '2024-02-28'), (4, 1, 5.0, '2024-01-20'),
(4, 2, 2.5, '2024-03-10')
ON CONFLICT DO NOTHING;
