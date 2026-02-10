-- ==============================================
-- GABARITO AULA 04: INTRODUÇÃO À MODELAGEM DIMENSIONAL
-- ==============================================

-- ==============================================
-- RESPOSTA 1: Paradigmas (Performance)
-- ==============================================
-- a) **Modelo B: Star Schema.**
--    Fatos isolados no centro e dimensões em um único nível de JOIN.
--    No OLAP, menos JOINS = Mais Performance. Dimensões pré-calculadas evitam recálculo a cada query.
--    O Modelo A (OLTP Normalizado) exigiria joins em cadeia (Pedido -> Cliente -> Cidade -> Estado) custosos para leitura.

-- ==============================================
-- RESPOSTA 2: Fatos vs. Dimensões
-- ==============================================
-- a) **Fato** (Métrica, numérico, aditivo).
-- b) **Dimensão** (Contexto Temporal, quem, quando).
-- c) **Dimensão** (Contexto Descritivo, quem).
-- d) **Fato** (Métrica, numérico, aditivo).
-- e) **Dimensão** (Contexto Descritivo, o que).
-- f) **Dimensão** (Contexto Descritivo, onde).

-- ==============================================
-- RESPOSTA 3: Star vs. Snowflake
-- ==============================================
-- a) **Snowflake:** Não. A tabela Categoria estaria ligada à Tabela Produto (Sub-dimensão), exigindo dois saltos (JOINS) até a Fato.
--    (Fato -> Produto -> Categoria)
--
-- b) **Star Schema (Simplicidade).**
--    Ferramentas de BI funcionam melhor com modelos de esquema estrela direto (One-Hop Join).
--    Snowflake exige que a ferramenta de BI gerencie hierarquias complexas de joins, o que pode reduzir performance e intuitividade para o usuário final.

DO $$ BEGIN RAISE NOTICE 'VALIDAÇÃO AULA 04: SUCESSO! ✅'; END $$;
