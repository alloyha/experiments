-- ==============================================
-- SEED: BIBLIOTECA
-- ==============================================
-- Tabelas básicas que servem de fundação para as aulas

CREATE TABLE IF NOT EXISTS biblioteca.autor (
    autor_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    nacionalidade VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS biblioteca.livro (
    livro_id SERIAL PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    isbn VARCHAR(13) UNIQUE,
    ano_publicacao INTEGER
);

CREATE TABLE IF NOT EXISTS biblioteca.usuario (
    usuario_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE
);

-- Dados de exemplo para construir o grafo (rede_social)
INSERT INTO biblioteca.autor (nome, nacionalidade) VALUES
('Machado de Assis', 'Brasileira'),
('Clarice Lispector', 'Brasileira')
ON CONFLICT DO NOTHING;

INSERT INTO biblioteca.livro (titulo, isbn, ano_publicacao) VALUES
('Dom Casmurro', '9788535911664', 1899),
('A Hora da Estrela', '9788520925683', 1977)
ON CONFLICT DO NOTHING;
