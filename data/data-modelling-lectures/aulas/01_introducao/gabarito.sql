-- ==============================================
-- GABARITO AULA 01: INTRODUÇÃO À MODELAGEM
-- ==============================================

-- ==============================================
-- RESPOSTA 1: OLTP vs OLAP
-- ==============================================
-- a) **Cenário A é OLTP.**
--    Requer ALTA performance de escrita/leitura para uma única transação (atômica, consistente).
--    Dados normalizados para evitar erro de duplicidade (não cobrar 2x).

--    **Cenário B é OLAP.**
--    Requer leitura de GRANDES volumes de dados históricos.
--    Dados desnormalizados (Star Schema) facilitam joins e queries de agregação (SUM, AVG).

-- b) **Normalizada (OLTP):** Sistema de Pagamento.
--    **Desnormalizada (OLAP):** Dashboard.

-- ==============================================
-- RESPOSTA 2: Tipos de Modelagem (Rede Social)
-- ==============================================
-- a) Conceitual:
--    [Usuário] --- <Curte> --- [Post]
--    (Relação Muitos-para-Muitos: 1 usuário curte N posts, 1 post tem N curtidas).

-- b) Físico:
--    Tabela Associativa: 'curtida' (usuario_id, post_id, data_curtida).
--    CHAVE PRIMÁRIA COMPOSTA (PK): (usuario_id, post_id) -> impede curtir 2x.

-- ==============================================
-- RESPOSTA 3: DDL Básico
-- ==============================================

-- a) SQL:
CREATE TABLE IF NOT EXISTS usuario (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE
);

CREATE TABLE IF NOT EXISTS post (
    id SERIAL PRIMARY KEY,
    texto TEXT,
    usuario_id INTEGER REFERENCES usuario(id)
);

CREATE TABLE IF NOT EXISTS curtida (
    usuario_id INTEGER REFERENCES usuario(id),
    post_id INTEGER REFERENCES post(id),
    data_curtida TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (usuario_id, post_id)
);

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Verificando existência das tabelas criadas no exercício 3
   IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'curtida') THEN
      RAISE EXCEPTION 'Erro: Tabela curtida não foi criada';
   END IF;
   
   RAISE NOTICE 'VALIDAÇÃO AULA 01: SUCESSO! ✅';
END $$;
