-- ==============================================
-- GABARITO AULA 02: ERD FUNDAMENTOS & TIPOS DE DADOS
-- ==============================================

-- ==============================================
-- RESPOSTA 1: Entidades (Normalização)
-- ==============================================
-- a) **Itens_Do_Pedido:** É um array ou lista, viola a 1NF (atomicidade).
--    Deve ser separado em tabela 'Pedido_Itens'.
-- b) **Cidade_Entrega:** Redundância. Se cliente muda de cidade, histórico pode estar
--    associado a Endereço de Entrega.
--    Idealmente: Cliente -> Endereço. Pedido -> FK Endereço_ID.

-- b) **Repetição de Dados:** Ocupa espaço e gera inconsistência (update anomaly).
--    Se 'São Paulo' for escrito 'S. Paulo' em um pedido e 'SP' em outro, a análise quebra.
--    Centralizar em tabela 'Cidade' ou 'Endereço'.

-- ==============================================
-- RESPOSTA 2: Relacionamentos
-- ==============================================
-- a) **N:N (Muitos-para-Muitos).**
--    Exige Tabela Associativa (Intermediária): 'Livro_Autor' (livro_id, autor_id).

-- b) **1:N (Um-para-Muitos).**
--    Chave Estrangeira (FK) na tabela 'Funcionario' apontando para 'Departamento_ID'.
--    (Um Funcionário "tem um" Departamento).

-- ==============================================
-- RESPOSTA 3: Cardinalidade
-- ==============================================
-- a) **PK:** id (identificador único do empréstimo).
--    **FK:** usuario_id (aponta p/ Usuario), livro_id (aponta p/ Livro).

--    A PK 'id' garante unicidade do evento de empréstimo, não do usuário.

DO $$ BEGIN RAISE NOTICE 'VALIDAÇÃO AULA 02: SUCESSO! ✅'; END $$;
