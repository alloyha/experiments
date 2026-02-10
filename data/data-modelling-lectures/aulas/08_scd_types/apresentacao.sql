-- ==============================================================================
-- Aula 8: Slowly Changing Dimensions (SCD) - Tipos
-- ==============================================================================

-- SCD Type 0: Nunca Muda
-- Exemplo: CPF e Data de Nascimento não mudam
DROP TABLE IF EXISTS dim_cliente_type0;
CREATE TABLE dim_cliente_type0 (
    cliente_id SERIAL PRIMARY KEY,
    cpf VARCHAR(11) UNIQUE,  -- Type 0
    data_nascimento DATE,    -- Type 0
    nome VARCHAR(100)
);

-- SCD Type 1: Sobrescreve (Sem histórico)
/*
UPDATE dim_cliente 
SET cidade = 'Rio de Janeiro',
    estado = 'RJ'
WHERE cliente_id = 123;
*/

-- SCD Type 2: Histórico Completo (Adiciona Linhas)
DROP TABLE IF EXISTS dim_cliente_type2;
CREATE TABLE dim_cliente_type2 (
    cliente_sk SERIAL PRIMARY KEY,        -- Surrogate key
    cliente_id INTEGER,                   -- Natural key
    nome VARCHAR(100),
    cidade VARCHAR(100),
    categoria VARCHAR(20),                -- Ex: Bronze, Prata, Ouro
    data_inicio DATE,                     -- Início validade
    data_fim DATE,                        -- Fim validade
    versao INTEGER,                       -- Número da versão
    ativo BOOLEAN,                        -- Flag atual
    UNIQUE(cliente_id, versao)
);

/*
-- Processo de Update Type 2:
1. UPDATE registro antigo (setar data_fim e ativo=FALSE)
2. INSERT novo registro (data_inicio=hoje, versao++, ativo=TRUE)
*/

-- SCD Type 3: Histórico Limitado (Coluna Anterior)
DROP TABLE IF EXISTS dim_produto_type3;
CREATE TABLE dim_produto_type3 (
    produto_id SERIAL PRIMARY KEY,
    nome_produto VARCHAR(200),
    categoria_atual VARCHAR(50),
    categoria_anterior VARCHAR(50), -- Guarda apenas o último valor
    data_mudanca_categoria DATE
);

/*
-- Update Type 3:
UPDATE dim_produto 
SET categoria_anterior = categoria_atual,
    categoria_atual = 'Eletrônicos Premium',
    data_mudanca_categoria = CURRENT_DATE
WHERE produto_id = 456;
*/
