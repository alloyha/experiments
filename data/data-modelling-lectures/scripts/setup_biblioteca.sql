-- ==============================================
-- SETUP: BIBLIOTECA (DDL + DML)
-- ==============================================

-- 1. ESTRUTURA (DDL)
CREATE TABLE IF NOT EXISTS biblioteca.autor (
    autor_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    nacionalidade VARCHAR(50),
    data_nascimento DATE
);

CREATE TABLE IF NOT EXISTS biblioteca.livro (
    livro_id SERIAL PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    isbn VARCHAR(13) UNIQUE,
    ano_publicacao INTEGER,
    quantidade_disponivel INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS biblioteca.usuario (
    usuario_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    tipo VARCHAR(20) CHECK (tipo IN ('aluno', 'professor')) DEFAULT 'aluno',
    email VARCHAR(100) UNIQUE
);

CREATE TABLE IF NOT EXISTS biblioteca.livro_autor (
    livro_id INTEGER REFERENCES biblioteca.livro(livro_id),
    autor_id INTEGER REFERENCES biblioteca.autor(autor_id),
    PRIMARY KEY (livro_id, autor_id)
);

CREATE TABLE IF NOT EXISTS biblioteca.emprestimo (
    emprestimo_id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES biblioteca.usuario(usuario_id),
    livro_id INTEGER REFERENCES biblioteca.livro(livro_id),
    data_emprestimo DATE DEFAULT CURRENT_DATE,
    data_devolucao_prevista DATE NOT NULL,
    data_devolucao_real DATE
);

CREATE TABLE IF NOT EXISTS biblioteca.multa (
    multa_id SERIAL PRIMARY KEY,
    emprestimo_id INTEGER UNIQUE REFERENCES biblioteca.emprestimo(emprestimo_id),
    valor_multa DECIMAL(10, 2),
    pago BOOLEAN DEFAULT FALSE
);

-- 2. DADOS (DML)
INSERT INTO biblioteca.autor (nome, nacionalidade) VALUES
('Machado de Assis', 'Brasileira'),
('Clarice Lispector', 'Brasileira'),
('Jorge Amado', 'Brasileira')
ON CONFLICT DO NOTHING;

INSERT INTO biblioteca.livro (titulo, isbn, ano_publicacao) VALUES
('Dom Casmurro', '9788525044648', 1899),
('A Hora da Estrela', '9788532508126', 1977),
('Capitães da Areia', '9788535914064', 1937)
ON CONFLICT DO NOTHING;

INSERT INTO biblioteca.usuario (nome, email) VALUES
('Ana Silva', 'ana@email.com'),
('Bruno Costa', 'bruno@email.com')
ON CONFLICT DO NOTHING;