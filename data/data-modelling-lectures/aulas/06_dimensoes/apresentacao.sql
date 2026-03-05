-- ==============================================================================
-- Aula 6: Tabelas Dimensão (Taxonomia Completa e SCD)
-- ==============================================================================

-- NOTA PEDAGÓGICA:
-- Dimensões dão o contexto (Quem, Onde, Quando, O Quê) aos fatos numéricos.
-- Diferente das fatos, são tabelas largas, ricas em texto e desnormalizadas.

-- 0. SETUP: LIMPEZA
DROP TABLE IF EXISTS fato_vendas;
DROP TABLE IF EXISTS dim_produto;
DROP TABLE IF EXISTS dim_localizacao_snowflake;
DROP TABLE IF EXISTS dim_localizacao_star;
DROP TABLE IF EXISTS dim_usuarios_scd1;
DROP TABLE IF EXISTS dim_clientes_scd2;
DROP TABLE IF EXISTS dim_usuarios_scd3;
DROP TABLE IF EXISTS dim_usuarios_atual_scd4;
DROP TABLE IF EXISTS dim_usuarios_historico_scd4;
DROP TABLE IF EXISTS dim_perfil_mini_scd5;
DROP TABLE IF EXISTS dim_cliente_scd6;
DROP TABLE IF EXISTS dim_vendedor;
DROP TABLE IF EXISTS dim_escritorio_outrigger;
DROP TABLE IF EXISTS dim_tempo;
DROP TABLE IF EXISTS dim_mes_shrunken;
DROP TABLE IF EXISTS dim_status_estatico;
DROP TABLE IF EXISTS dim_junk_venda;

-- ==============================================================================
-- 1. ANATOMIA BÁSICA E SURROGATE KEYS
-- ==============================================================================

-- Dimensão Produto (Clássica)
CREATE TABLE dim_produto (
    produto_sk     SERIAL PRIMARY KEY,  -- Surrogate Key (Interna do DW)
    produto_nk     VARCHAR(20) UNIQUE,  -- Natural Key (ID do Sistema Origem)
    nome           VARCHAR(200),
    categoria      VARCHAR(50),
    data_carga     TIMESTAMP DEFAULT NOW()
);

-- Registros Especiais: Para evitar FKs órfãs e manter integridade
INSERT INTO dim_produto (produto_sk, nome, categoria) 
VALUES (-1, 'NÃO INFORMADO', 'N/A'), (0, 'NÃO SE APLICA', 'N/A');


-- ==============================================================================
-- 2. HIERARQUIAS: STAR SCHEMA vs. SNOWFLAKE
-- ==============================================================================

-- A. SNOWFLAKE: Normalizado (Economia de espaço, Lento para consultas)
CREATE TABLE dim_localizacao_snowflake (
    cidade_id SERIAL PRIMARY KEY,
    nome_cidade VARCHAR(100),
    estado_id   INTEGER -- Requer JOIN para saber o Estado
);

-- B. STAR SCHEMA: Desnormalizado (Otimizado para Performance/Leitura)
-- PADRÃO PARA DIMENSIONAL MODELLING
CREATE TABLE dim_localizacao_star (
    localizacao_sk SERIAL PRIMARY KEY,
    cidade         VARCHAR(100),
    estado         VARCHAR(50),
    regiao         VARCHAR(50) -- Todas as hierarquias estão na mesma linha
);


-- ==============================================================================
-- 3. SCD (SLOWLY CHANGING DIMENSIONS) - GERENCIANDO MUDANÇAS
-- ==============================================================================

-- SCD TIPO 1: Overwrite (Sem histórico; corrige erros)
CREATE TABLE dim_usuarios_scd1 (
    usuario_id      INTEGER PRIMARY KEY,
    nome            VARCHAR(100),
    cpf             VARCHAR(14) -- Se o CPF mudar, apenas sobrescrevemos
);

-- SCD TIPO 2: Add Row (Histórico Completo - O Padrão Ouro)
CREATE TABLE dim_clientes_scd2 (
    cliente_sk      SERIAL PRIMARY KEY,
    cliente_nk      INTEGER,
    nome            VARCHAR(100),
    endereco        VARCHAR(200),
    versao          INTEGER DEFAULT 1,
    data_inicio     TIMESTAMP,
    data_fim        TIMESTAMP,
    eh_atual        BOOLEAN DEFAULT TRUE
);

-- SCD TIPO 3: Add Column (Valor Atual vs Anterior Lado a Lado)
CREATE TABLE dim_usuarios_scd3 (
    usuario_id      INTEGER PRIMARY KEY,
    nome            VARCHAR(100),
    segmento_atual   VARCHAR(20),
    segmento_anterior VARCHAR(20),
    data_mudanca    DATE
);

-- SCD TIPO 4: History Table (Tabela de Estado Atual + Tabela de Histórico)
CREATE TABLE dim_usuarios_atual_scd4 (
    usuario_id      INTEGER PRIMARY KEY,
    nome            VARCHAR(100),
    segmento        VARCHAR(20)
);

CREATE TABLE dim_usuarios_historico_scd4 (
    historico_sk    SERIAL PRIMARY KEY,
    usuario_id      INTEGER,
    segmento        VARCHAR(20),
    data_inicio     TIMESTAMP,
    data_fim        TIMESTAMP
);

-- SCD TIPO 5 (Tipo 4 + Mini-Dimension): Atributos voláteis (Idade, Score)
-- NOTA: Usar tipos numéricos/faixas (typed) é melhor que VARCHAR para performance
CREATE TABLE dim_perfil_mini_scd5 (
    perfil_sk       SERIAL PRIMARY KEY,
    idade_min       SMALLINT,
    idade_max       SMALLINT,
    renda_min       DECIMAL(10,2),
    renda_max       DECIMAL(10,2),
    score_credito_classe CHAR(1) -- A, B, C, D
);

-- SCD TIPO 6: Híbrido (1 + 2 + 3)
-- Nova linha (T2) com coluna de "Valor Atual" (T3) que é propagada (T1)
CREATE TABLE dim_cliente_scd6 (
    cliente_sk      SERIAL PRIMARY KEY,
    cliente_nk      INTEGER,
    nome            VARCHAR(100),
    segmento_historico VARCHAR(20), -- Valor da época (Tipo 2)
    segmento_atual   VARCHAR(20),     -- Valor atualizado para todos (Tipo 1)
    data_inicio     DATE,
    data_fim        DATE
);


-- ==============================================================================
-- 4. TIPOS DE ARQUITETURA
-- ==============================================================================

-- OUTRIGGER: Dimensão que aponta para outra (Escrita -> Endereço)
CREATE TABLE dim_escritorio_outrigger (
    escritorio_sk SERIAL PRIMARY KEY,
    nome_escritorio VARCHAR(100),
    endereco_completo TEXT
);

CREATE TABLE dim_vendedor (
    vendedor_sk SERIAL PRIMARY KEY,
    nome_vendedor VARCHAR(100),
    escritorio_sk INTEGER REFERENCES dim_escritorio_outrigger(escritorio_sk)
);

-- SHRUNKEN DIMENSION: Subconjunto de uma maior para fatos agregados
CREATE TABLE dim_tempo (
    tempo_sk        INTEGER PRIMARY KEY, -- YYYYMMDD
    data            DATE,
    mes             SMALLINT,
    ano             SMALLINT,
    dia_semana      SMALLINT -- 1 a 7 (Typed)
);

CREATE TABLE dim_mes_shrunken (
    mes_sk          INTEGER PRIMARY KEY, -- YYYYMM
    mes_nome        VARCHAR(20),
    ano             SMALLINT
);

-- CONFORMADA (Conformed): Mesma dim_produto usada por Vales Vendas e Fato Estoque
-- Fato Vendas (Refers dim_produto)
-- Fato Estoque (Refers dim_produto)


-- ==============================================================================
-- 5. TIPOS DE USO E MISCELÂNEA
-- ==============================================================================

-- ROLE-PLAYING: Uma tabela, múltiplos papéis no JOIN
-- Ex: dim_tempo servindo como dt_pedido e dt_entrega (resolvido com ALIAS no SQL)

-- JUNK DIMENSION: Agrupa Flags e Atributos de Baixa Cardinalidade
-- NOTA: Usar BOOLEAN e SMALLINT é mais aconselhado que VARCHAR para estas propriedades
CREATE TABLE dim_junk_venda (
    junk_sk         SERIAL PRIMARY KEY,
    forma_pag_cod   SMALLINT, -- 1: Pix, 2: Cartão, 3: Boleto (Typed)
    tem_cupom       BOOLEAN,
    eh_entrega_full BOOLEAN,
    tipo_cliente    CHAR(1)   -- P: PF, J: PJ
);

-- DEGENERATE DIMENSION: Número da NF ou ID do Pedido (Vive na Tabela Fato)
-- SELECT valor, numero_nf FROM fato_vendas; 

-- STATIC DIMENSION: Listas fixas criadas no DW
CREATE TABLE dim_status_estatico (
    status_id INTEGER PRIMARY KEY,
    descricao VARCHAR(50)
);
INSERT INTO dim_status_estatico VALUES (1, 'Ativo'), (2, 'Inativo'), (3, 'Cancelado');

-- SCD TIPO 7: Dual PK (Surrogate Key para histórico, Natural Key para atual)
-- A fato armazena DUAS chaves para o mesmo cliente.
CREATE TABLE fato_vendas (
    venda_id        SERIAL PRIMARY KEY,
    cliente_sk      INTEGER, -- Histórico (As Was)
    cliente_nk      INTEGER, -- Atual (As Is)
    produto_sk      INTEGER,
    valor           DECIMAL(10,2)
);
