-- ==============================================
-- GABARITO AULA 03: ERD E SQL BÁSICO
-- ==============================================

-- ==============================================
-- EXERCÍCIO 1: ERD Biblioteca
-- ==============================================

-- a) Inserir 3 autores brasileiros
INSERT INTO autor (nome, nacionalidade, data_nascimento) VALUES 
('Machado de Assis', 'Brasileira', '1839-06-21'),
('Clarice Lispector', 'Brasileira', '1920-12-10'),
('Jorge Amado', 'Brasileira', '1912-08-10')
ON CONFLICT DO NOTHING;

-- b) Inserir 5 livros de diferentes gêneros
INSERT INTO livro (titulo, isbn, ano_publicacao, quantidade_disponivel) VALUES 
('Dom Casmurro', '9788525044648', 1899, 3),
('A Hora da Estrela', '9788532508126', 1977, 5),
('Capitães da Areia', '9788535914064', 1937, 4),
('Clean Code', '9780132350884', 2008, 2),
('Design Patterns', '9780201633610', 1994, 2)
ON CONFLICT (isbn) DO NOTHING;

-- c) Criar associação livro-autor (alguns livros com múltiplos autores)
-- Assumindo IDs sequenciais a partir do insert anterior
INSERT INTO livro_autor (livro_id, autor_id) VALUES 
(1, 1), -- Dom Casmurro - Machado
(2, 2), -- Hora da Estrela - Clarice
(3, 3) -- Capitães - Jorge Amado
ON CONFLICT DO NOTHING;

-- d) Inserir 10 usuários (mix de alunos e professores)
INSERT INTO usuario (nome, tipo, email) VALUES 
('Ana Souza', 'aluno', 'ana@uni.edu'),
('Bruno Lima', 'aluno', 'bruno@uni.edu'),
('Carlos Rocha', 'professor', 'carlos@uni.edu'),
('Daniela Silva', 'aluno', 'daniela@uni.edu'),
('Eduardo Santos', 'aluno', 'eduardo@uni.edu'),
('Fernanda Costa', 'professor', 'fernanda@uni.edu'),
('Gabriel Alves', 'aluno', 'gabriel@uni.edu'),
('Helena Dias', 'aluno', 'helena@uni.edu'),
('Igor Martins', 'professor', 'igor@uni.edu'),
('Julia Pereira', 'aluno', 'julia@uni.edu')
ON CONFLICT (email) DO NOTHING;

-- e) Criar 15 empréstimos com datas variadas
INSERT INTO emprestimo (usuario_id, livro_id, data_emprestimo, data_devolucao_prevista, data_devolucao_real) VALUES 
(1, 1, '2024-05-01', '2024-05-15', '2024-05-10'),
(2, 2, '2024-05-02', '2024-05-16', '2024-05-16'),
(3, 3, '2024-05-03', '2024-05-20', '2024-05-18'), -- Professor tem mais prazo
(4, 4, '2024-06-01', '2024-06-15', NULL),    -- Atrasado
(5, 5, '2024-06-05', '2024-06-19', NULL),
(6, 1, '2024-06-10', '2024-06-25', '2024-06-20'),
(7, 2, '2024-06-12', '2024-06-26', NULL),    -- Atrasado
(8, 3, '2024-06-15', '2024-06-29', NULL),
(9, 1, '2024-07-01', '2024-07-15', NULL),
(10, 2, '2024-07-02', '2024-07-16', '2024-07-10'),
(1, 3, '2024-07-03', '2024-07-17', NULL),
(2, 4, '2024-07-04', '2024-07-18', '2024-07-18'),
(3, 5, '2024-07-05', '2024-07-20', NULL),
(4, 1, '2024-07-06', '2024-07-20', '2024-07-15'),
(5, 2, '2024-07-07', '2024-07-21', NULL)
ON CONFLICT (emprestimo_id) DO NOTHING; -- Assuming standard serial behavior, might conflict if IDs manually inserted or implicit

-- f) Criar multas para empréstimos atrasados
-- Assumindo IDs 4 e 7 como atrasados
INSERT INTO multa (emprestimo_id, valor_multa, pago) VALUES 
(4, 5.50, FALSE),
(7, 12.00, TRUE)
ON CONFLICT (emprestimo_id) DO NOTHING;

-- ==============================================
-- EXERCÍCIO 2: Consultas ERD
-- ==============================================

-- a) Listar todos os livros com seus autores
SELECT l.titulo, a.nome as autor 
FROM livro l
JOIN livro_autor la ON l.livro_id = la.livro_id
JOIN autor a ON la.autor_id = a.autor_id;

-- b) Encontrar livros mais emprestados
SELECT l.titulo, COUNT(e.emprestimo_id) as qtd_emprestimos
FROM livro l
JOIN emprestimo e ON l.livro_id = e.livro_id
GROUP BY l.titulo
ORDER BY qtd_emprestimos DESC;

-- c) Listar usuários com empréstimos em atraso
SELECT u.nome, l.titulo, e.data_devolucao_prevista
FROM emprestimo e
JOIN usuario u ON e.usuario_id = u.usuario_id
JOIN livro l ON e.livro_id = l.livro_id
WHERE e.data_devolucao_real IS NULL 
  AND e.data_devolucao_prevista < CURRENT_DATE;

-- d) Calcular total de multas não pagas
SELECT SUM(valor_multa) as total_pendente 
FROM multa 
WHERE pago = FALSE;

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Contagem de Autores
   IF (SELECT COUNT(*) FROM autor) != 3 THEN
      RAISE EXCEPTION 'Erro: Esperado 3 autores, encontrado %', (SELECT COUNT(*) FROM autor);
   END IF;

   -- Validação 2: Contagem de Livros
   IF (SELECT COUNT(*) FROM livro) != 5 THEN
      RAISE EXCEPTION 'Erro: Esperado 5 livros, encontrado %', (SELECT COUNT(*) FROM livro);
   END IF;

   -- Validação 3: Contagem de Usuários
   IF (SELECT COUNT(*) FROM usuario) != 10 THEN
      RAISE EXCEPTION 'Erro: Esperado 10 usuários, encontrado %', (SELECT COUNT(*) FROM usuario);
   END IF;

   -- Validação 4: Contagem de Empréstimos
   IF (SELECT COUNT(*) FROM emprestimo) != 15 THEN
      RAISE EXCEPTION 'Erro: Esperado 15 empréstimos, encontrado %', (SELECT COUNT(*) FROM emprestimo);
   END IF;

   -- Validação 5: Valor de Multas Pendentes
   IF (SELECT SUM(valor_multa) FROM multa WHERE pago = FALSE) != 5.50 THEN
      RAISE EXCEPTION 'Erro: Esperado 5.50 em multas pendentes, encontrado %', (SELECT SUM(valor_multa) FROM multa WHERE pago = FALSE);
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 03: SUCESSO! ✅';
END $$;
