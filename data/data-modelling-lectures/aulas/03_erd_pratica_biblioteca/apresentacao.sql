-- ==============================================================================
-- Aula 3: ERD Parte 2 - Prática (Sistema de Biblioteca)
-- ==============================================================================

-- 1. Tabelas Principais (Entidades Fortes)
CREATE TABLE IF NOT EXISTS autor (
    autor_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    nacionalidade VARCHAR(50),
    data_nascimento DATE
);

-- Garantir que a coluna existe (caso a tabela já existisse sem ela)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='autor' AND column_name='data_nascimento') THEN
        ALTER TABLE autor ADD COLUMN data_nascimento DATE;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS livro (
    livro_id SERIAL PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    isbn VARCHAR(13) UNIQUE,
    ano_publicacao INTEGER,
    quantidade_disponivel INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usuario (
    usuario_id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    tipo VARCHAR(20) CHECK (tipo IN ('aluno', 'professor')),
    email VARCHAR(100) UNIQUE
);

-- Garantir compatibilidade com Aula 01 (adicionar colunas se faltar)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='usuario' AND column_name='tipo') THEN
        ALTER TABLE usuario ADD COLUMN tipo VARCHAR(20) CHECK (tipo IN ('aluno', 'professor'));
    END IF;
END $$;

-- 2. Tabela Associativa (Livro N:N Autor)
CREATE TABLE IF NOT EXISTS livro_autor (
    livro_id INTEGER,
    autor_id INTEGER,
    PRIMARY KEY (livro_id, autor_id),
    FOREIGN KEY (livro_id) REFERENCES livro (livro_id),
    FOREIGN KEY (autor_id) REFERENCES autor (autor_id)
);

-- 3. Tabela Transacional (Empréstimo 1:N)
CREATE TABLE IF NOT EXISTS emprestimo (
    emprestimo_id SERIAL PRIMARY KEY,
    usuario_id INTEGER NOT NULL,
    livro_id INTEGER NOT NULL,
    data_emprestimo DATE DEFAULT CURRENT_DATE,
    data_devolucao_prevista DATE NOT NULL,
    data_devolucao_real DATE,
    FOREIGN KEY (usuario_id) REFERENCES usuario (usuario_id),
    FOREIGN KEY (livro_id) REFERENCES livro (livro_id)
);

-- 4. Tabela Dependente (Multa 1:1 Opcional)
CREATE TABLE IF NOT EXISTS multa (
    multa_id SERIAL PRIMARY KEY,
    emprestimo_id INTEGER UNIQUE NOT NULL,
    valor_multa DECIMAL(10, 2),
    pago BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (emprestimo_id) REFERENCES emprestimo (emprestimo_id)
);
