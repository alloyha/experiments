-- ==============================================
-- SEED: REDE SOCIAL (Demonstração de Transformação OLTP → Grafo)
-- ==============================================
-- Propósito Didático: Mostrar como dados relacionais são transformados em grafo
-- Etapa 1: Criar tabelas OLTP tradicionais
-- Etapa 2: Tabelas de grafo (vazias, serão populadas na aula)

-- ==============================================
-- ETAPA 1: TABELAS OLTP (Modelo Relacional Tradicional)
-- ==============================================

-- Tabela de Pessoas (Leitores da plataforma)
CREATE TABLE IF NOT EXISTS rede_social.pessoa (
    pessoa_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    idade INTEGER,
    genero_favorito VARCHAR(50)
);

-- Tabela de Livros (Catálogo da plataforma)
CREATE TABLE IF NOT EXISTS rede_social.livro (
    livro_id SERIAL PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    autor VARCHAR(100),
    ano_publicacao INTEGER
);

-- Tabela de Conexões Sociais (Quem segue quem - relação direcional)
CREATE TABLE IF NOT EXISTS rede_social.conexao_social (
    seguidor_id INTEGER REFERENCES rede_social.pessoa (pessoa_id),
    seguido_id INTEGER REFERENCES rede_social.pessoa (pessoa_id),
    forca_conexao DECIMAL(5, 2), -- 0-1: força da conexão
    data_conexao DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (seguidor_id, seguido_id),
    CHECK (seguidor_id != seguido_id) -- Não pode seguir a si mesmo
);

-- Tabela de Leituras (Quem leu qual livro e que nota deu)
CREATE TABLE IF NOT EXISTS rede_social.leitura (
    pessoa_id INTEGER REFERENCES rede_social.pessoa (pessoa_id),
    livro_id INTEGER REFERENCES rede_social.livro (livro_id),
    nota DECIMAL(3, 1), -- 0-5: avaliação do livro
    data_leitura DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (pessoa_id, livro_id),
    CHECK (nota >= 0 AND nota <= 5)
);

-- Populando tabelas OLTP
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
(1, 2, 0.8),  -- Ana → Bruno
(2, 3, 0.9),  -- Bruno → Carla
(3, 1, 0.7),  -- Carla → Ana (reciprocidade)
(4, 1, 0.6),  -- Daniel → Ana
(4, 2, 0.6),  -- Daniel → Bruno
(4, 3, 0.5)  -- Daniel → Carla
ON CONFLICT DO NOTHING;

-- Leituras: Quem leu o quê e qual nota deu
INSERT INTO rede_social.leitura (pessoa_id, livro_id, nota, data_leitura) VALUES
(1, 1, 5.0, '2024-01-15'),  -- Ana leu "1984" - nota máxima
(1, 3, 4.5, '2024-02-20'),  -- Ana leu "Cem Anos" - nota alta
(2, 2, 5.0, '2024-01-10'),  -- Bruno leu "Sapiens" - nota máxima
(2, 1, 3.0, '2024-03-05'),  -- Bruno leu "1984" - nota média (não recomenda)
(3, 3, 4.8, '2024-02-28'),  -- Carla leu "Cem Anos" - nota alta
(4, 1, 5.0, '2024-01-20'),  -- Daniel leu "1984" - nota máxima
(4, 2, 2.5, '2024-03-10')   -- Daniel leu "Sapiens" - nota baixa (não recomenda)
ON CONFLICT DO NOTHING;


