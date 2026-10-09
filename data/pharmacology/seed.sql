-- ============================================================
-- Seed script: patologias, sintomas, remédios, apresentações
-- e princípios ativos
--
-- Banco: DuckDB
--
-- Objetivo:
--
--   Representar o domínio como um grafo clínico quantitativo
--   capaz de alimentar um otimizador de regimes terapêuticos.
--
--
-- Modelo conceitual:
--
-- Patologia ──apresenta───────────────> Sintoma
--
-- Apresentação ──alivia──────────────> Sintoma
-- Apresentação ──evento_adverso──────> Sintoma
-- Apresentação ──indicada_para───────> Patologia
-- Apresentação ──interage_com────────> Apresentação
--
-- Remédio ──possui───────────────────> Apresentação
--
-- Apresentação ──contém──────────────> Princípio Ativo
--
--
-- ============================================================
-- FORMALISMO QUANTITATIVO
-- ============================================================
--
-- Para patologia p, sintoma s e regime M:
--
-- d_p(s)
--     carga gerada pela patologia.
--
-- a_M(s)
--     carga adversa agregada introduzida pelo regime.
--
-- b_p,M(s)
--     carga gerada total:
--
--         b_p,M(s) = d_p(s) + a_M(s)
--
-- q_M(s)
--     fator de atenuação terapêutica:
--
--         q_M(s) = Π_m (1 - e_m(s))
--
-- r_p,M(s)
--     carga residual:
--
--         r_p,M(s) = b_p,M(s) * q_M(s)
--
-- rho_p,M(s)
--     redução relativa:
--
--         rho = 1 - r / b
--
--
-- ============================================================
-- PARTIÇÕES DE SINTOMAS
-- ============================================================
--
-- D_p
--     sintomas originados pela patologia.
--
-- A_M
--     sintomas adversos originados pelo tratamento.
--
-- G_p(M)
--     sintomas gerados:
--
--         G = D ∪ A
--
-- O_p(M)
--     sobreposição adversa:
--
--         O = D ∩ A
--
-- N_p(M)
--     sintomas adversos novos:
--
--         N = A \ D
--
--
-- O conjunto G é particionado em:
--
-- C_p(M)
--     sintomas controlados.
--
-- L_p(M)
--     sintomas tolerados.
--
-- U_p(M)
--     sintomas não resolvidos.
--
-- de forma que:
--
--     G = C ⊔ L ⊔ U
--
--
-- ============================================================
-- POLÍTICA
-- ============================================================
--
-- limiar_controle:
--
--     eta_{p,s}
--
--     redução relativa mínima necessária para considerar
--     o sintoma controlado.
--
--
-- limiar_tolerancia:
--
--     tau_{p,s}
--
--     carga residual máxima considerada tolerável.
--
--
-- risco_maximo_admissivel:
--
--     limite máximo admissível para o risco agregado de um
--     evento adverso.
--
--
-- IMPORTANTE:
--
-- Todos os valores deste seed são DIDÁTICOS.
-- Eles não constituem recomendação clínica nem dados
-- epidemiológicos validados.
-- ============================================================


-- ============================================================
-- LIMPEZA
-- ============================================================

DROP VIEW IF EXISTS vw_matriz_efeitos;
DROP VIEW IF EXISTS vw_carga_patologia;
DROP VIEW IF EXISTS vw_risco_evento_adverso;
DROP VIEW IF EXISTS vw_beneficio_terapeutico;

DROP TABLE IF EXISTS apresentacao_interacao;
DROP TABLE IF EXISTS apresentacao_evento_adverso;
DROP TABLE IF EXISTS apresentacao_alivia;
DROP TABLE IF EXISTS apresentacao_indicacao;
DROP TABLE IF EXISTS apresentacao_principio_ativo;
DROP TABLE IF EXISTS patologia_sintoma;

DROP TABLE IF EXISTS apresentacao;
DROP TABLE IF EXISTS remedio;
DROP TABLE IF EXISTS principio_ativo;
DROP TABLE IF EXISTS sintoma;
DROP TABLE IF EXISTS patologia;


-- ============================================================
-- ENTIDADES
-- ============================================================


-- ------------------------------------------------------------
-- Patologia
-- ------------------------------------------------------------

CREATE TABLE patologia (
    id          INTEGER PRIMARY KEY,
    nome        VARCHAR NOT NULL UNIQUE,
    descricao   VARCHAR
);


-- ------------------------------------------------------------
-- Sintoma
--
-- Uma única entidade representa:
--
--   - manifestação de uma patologia;
--   - alvo terapêutico;
--   - evento adverso.
--
-- O papel é definido pela relação.
-- ------------------------------------------------------------

CREATE TABLE sintoma (
    id          INTEGER PRIMARY KEY,
    nome        VARCHAR NOT NULL UNIQUE,
    descricao   VARCHAR,

    categoria   VARCHAR CHECK (
        categoria IS NULL OR categoria IN (
            'sistemico',
            'neurologico',
            'respiratorio',
            'gastrointestinal',
            'dermatologico',
            'cardiovascular',
            'musculoesqueletico',
            'psiquiatrico',
            'outro'
        )
    )
);


-- ------------------------------------------------------------
-- Princípio ativo
-- ------------------------------------------------------------

CREATE TABLE principio_ativo (
    id                  INTEGER PRIMARY KEY,
    nome                VARCHAR NOT NULL UNIQUE,
    classe_terapeutica  VARCHAR
);


-- ------------------------------------------------------------
-- Remédio
--
-- Produto comercial.
--
-- A composição pertence à apresentação, não diretamente
-- ao remédio.
-- ------------------------------------------------------------

CREATE TABLE remedio (
    id              INTEGER PRIMARY KEY,
    nome_comercial  VARCHAR NOT NULL,
    fabricante      VARCHAR,

    UNIQUE (
        nome_comercial,
        fabricante
    )
);


-- ------------------------------------------------------------
-- Apresentação
--
-- Unidade real utilizada pelo otimizador.
-- ------------------------------------------------------------

CREATE TABLE apresentacao (
    id                   INTEGER PRIMARY KEY,

    remedio_id           INTEGER NOT NULL
        REFERENCES remedio(id),

    forma_farmaceutica   VARCHAR NOT NULL,

    via_administracao    VARCHAR NOT NULL,

    descricao            VARCHAR,

    custo_complexidade   DOUBLE NOT NULL DEFAULT 0.1
        CHECK (
            custo_complexidade >= 0
            AND custo_complexidade <= 1
        ),

    UNIQUE (
        remedio_id,
        forma_farmaceutica,
        via_administracao,
        descricao
    )
);


-- ============================================================
-- RELACIONAMENTOS
-- ============================================================


-- ------------------------------------------------------------
-- Patologia ──apresenta──> Sintoma
--
-- prevalencia:
--
--     frequência relativa aproximada do sintoma no contexto
--     da patologia.
--
-- peso_clinico:
--
--     importância relativa da carga desse sintoma.
--
-- limiar_tolerancia:
--
--     tau_{p,s}
--
-- limiar_controle:
--
--     eta_{p,s}
--
-- A política é contextual à relação patologia-sintoma.
-- ------------------------------------------------------------

CREATE TABLE patologia_sintoma (
    patologia_id    INTEGER NOT NULL
        REFERENCES patologia(id),

    sintoma_id      INTEGER NOT NULL
        REFERENCES sintoma(id),

    frequencia      VARCHAR CHECK (
        frequencia IS NULL OR frequencia IN (
            'muito_comum',
            'comum',
            'ocasional',
            'incomum',
            'raro',
            'muito_raro'
        )
    ),

    prevalencia     DOUBLE NOT NULL
        CHECK (
            prevalencia >= 0
            AND prevalencia <= 1
        ),

    peso_clinico    DOUBLE NOT NULL
        CHECK (
            peso_clinico >= 0
            AND peso_clinico <= 1
        ),

    limiar_tolerancia DOUBLE NOT NULL
        CHECK (
            limiar_tolerancia >= 0
            AND limiar_tolerancia <= 1
        ),

    limiar_controle DOUBLE NOT NULL
        CHECK (
            limiar_controle >= 0
            AND limiar_controle <= 1
        ),

    PRIMARY KEY (
        patologia_id,
        sintoma_id
    )
);


-- ------------------------------------------------------------
-- Remédio ──possui──> Apresentação
--
-- Representada por:
--
--     apresentacao.remedio_id -> remedio.id
-- ------------------------------------------------------------


-- ------------------------------------------------------------
-- Apresentação ──contém──> Princípio Ativo
-- ------------------------------------------------------------

CREATE TABLE apresentacao_principio_ativo (
    apresentacao_id     INTEGER NOT NULL
        REFERENCES apresentacao(id),

    principio_ativo_id  INTEGER NOT NULL
        REFERENCES principio_ativo(id),

    quantidade          DECIMAL(12, 4),

    unidade             VARCHAR,

    volume_referencia   DECIMAL(12, 4),

    unidade_referencia  VARCHAR,

    PRIMARY KEY (
        apresentacao_id,
        principio_ativo_id
    ),

    CHECK (
        quantidade IS NULL
        OR quantidade > 0
    ),

    CHECK (
        volume_referencia IS NULL
        OR volume_referencia > 0
    )
);


-- ------------------------------------------------------------
-- Apresentação ──indicada_para──> Patologia
--
-- Independente de "alivia".
--
-- Isso evita assumir:
--
--     alivia sintoma
--         => indicada para patologia
-- ------------------------------------------------------------

CREATE TABLE apresentacao_indicacao (
    apresentacao_id  INTEGER NOT NULL
        REFERENCES apresentacao(id),

    patologia_id     INTEGER NOT NULL
        REFERENCES patologia(id),

    evidencia        VARCHAR,

    PRIMARY KEY (
        apresentacao_id,
        patologia_id
    )
);


-- ------------------------------------------------------------
-- Apresentação ──alivia──> Sintoma
--
-- e_m(s):
--
--     eficacia * probabilidade_resposta
-- ------------------------------------------------------------

CREATE TABLE apresentacao_alivia (
    apresentacao_id         INTEGER NOT NULL
        REFERENCES apresentacao(id),

    sintoma_id              INTEGER NOT NULL
        REFERENCES sintoma(id),

    eficacia                DOUBLE NOT NULL
        CHECK (
            eficacia >= 0
            AND eficacia <= 1
        ),

    probabilidade_resposta  DOUBLE NOT NULL
        CHECK (
            probabilidade_resposta >= 0
            AND probabilidade_resposta <= 1
        ),

    evidencia               VARCHAR,

    PRIMARY KEY (
        apresentacao_id,
        sintoma_id
    )
);


-- ------------------------------------------------------------
-- Apresentação ──evento_adverso──> Sintoma
--
-- probabilidade:
--
--     P(evento | apresentação)
--
-- gravidade:
--
--     classe semântica explícita.
--
-- gravidade_peso:
--
--     peso quantitativo associado à gravidade.
--
-- peso_clinico:
--
--     importância relativa do evento.
--
-- risco_maximo_admissivel:
--
--     limite usado pela camada de viabilidade.
-- ------------------------------------------------------------

CREATE TABLE apresentacao_evento_adverso (
    apresentacao_id  INTEGER NOT NULL
        REFERENCES apresentacao(id),

    sintoma_id       INTEGER NOT NULL
        REFERENCES sintoma(id),

    probabilidade    DOUBLE NOT NULL
        CHECK (
            probabilidade >= 0
            AND probabilidade <= 1
        ),

    gravidade        VARCHAR NOT NULL
        CHECK (
            gravidade IN (
                'leve',
                'moderada',
                'grave',
                'critica'
            )
        ),

    gravidade_peso   DOUBLE NOT NULL
        CHECK (
            gravidade_peso >= 0
            AND gravidade_peso <= 1
        ),

    peso_clinico     DOUBLE NOT NULL
        CHECK (
            peso_clinico >= 0
            AND peso_clinico <= 1
        ),

    risco_maximo_admissivel DOUBLE NOT NULL
        CHECK (
            risco_maximo_admissivel >= 0
            AND risco_maximo_admissivel <= 1
        ),

    evidencia        VARCHAR,

    PRIMARY KEY (
        apresentacao_id,
        sintoma_id
    )
);


-- ------------------------------------------------------------
-- Apresentação ──interage_com──> Apresentação
--
-- Relação simétrica armazenada uma única vez.
--
-- Exigimos:
--
--     apresentacao_a_id < apresentacao_b_id
-- ------------------------------------------------------------

CREATE TABLE apresentacao_interacao (
    apresentacao_a_id  INTEGER NOT NULL
        REFERENCES apresentacao(id),

    apresentacao_b_id  INTEGER NOT NULL
        REFERENCES apresentacao(id),

    severidade         DOUBLE NOT NULL
        CHECK (
            severidade >= 0
            AND severidade <= 1
        ),

    descricao          VARCHAR,

    PRIMARY KEY (
        apresentacao_a_id,
        apresentacao_b_id
    ),

    CHECK (
        apresentacao_a_id < apresentacao_b_id
    )
);


-- ============================================================
-- DADOS DE EXEMPLO
-- ============================================================


-- ============================================================
-- Patologias
-- ============================================================

INSERT INTO patologia (
    id,
    nome,
    descricao
)
VALUES
    (
        1,
        'Gripe',
        'Infecção viral respiratória aguda'
    ),
    (
        2,
        'Enxaqueca',
        'Distúrbio neurológico caracterizado por episódios recorrentes de cefaleia'
    ),
    (
        3,
        'Rinite alérgica',
        'Inflamação da mucosa nasal desencadeada por exposição a alérgenos'
    );


-- ============================================================
-- Sintomas
-- ============================================================

INSERT INTO sintoma (
    id,
    nome,
    descricao,
    categoria
)
VALUES
    (
        1,
        'Febre',
        'Elevação da temperatura corporal',
        'sistemico'
    ),
    (
        2,
        'Dor de cabeça',
        'Dor localizada na região craniana',
        'neurologico'
    ),
    (
        3,
        'Congestão nasal',
        'Obstrução parcial ou completa das vias nasais',
        'respiratorio'
    ),
    (
        4,
        'Coriza',
        'Secreção nasal excessiva',
        'respiratorio'
    ),
    (
        5,
        'Tosse',
        'Reflexo de expulsão de ar das vias respiratórias',
        'respiratorio'
    ),
    (
        6,
        'Espirros',
        'Expulsão reflexa de ar pelas vias nasais',
        'respiratorio'
    ),
    (
        7,
        'Sensibilidade à luz',
        'Desconforto ou intolerância à exposição luminosa',
        'neurologico'
    ),
    (
        8,
        'Náusea',
        'Sensação de enjoo acompanhada ou não de vontade de vomitar',
        'gastrointestinal'
    ),
    (
        9,
        'Dor de estômago',
        'Dor ou desconforto localizado na região gástrica',
        'gastrointestinal'
    ),
    (
        10,
        'Sonolência',
        'Tendência aumentada ao sono',
        'neurologico'
    ),
    (
        11,
        'Reação alérgica',
        'Resposta imunológica exacerbada após exposição a uma substância',
        'sistemico'
    );


-- ============================================================
-- Princípios ativos
-- ============================================================

INSERT INTO principio_ativo (
    id,
    nome,
    classe_terapeutica
)
VALUES
    (
        1,
        'Paracetamol',
        'Analgésico e antitérmico'
    ),
    (
        2,
        'Ibuprofeno',
        'Anti-inflamatório não esteroidal'
    ),
    (
        3,
        'Loratadina',
        'Anti-histamínico'
    ),
    ( 4, 'Ativo Antitussivo A', 'Antitussivo sintético didático' ), ( 5, 'Ativo Antiemético A', 'Antiemético sintético didático' ), ( 6, 'Ativo Antiemético B', 'Antiemético sintético didático' ), ( 7, 'Ativo Fotofobia A', 'Modulador neurológico sintético didático' ), ( 8, 'Ativo Descongestionante A', 'Descongestionante sintético didático' ), ( 9, 'Ativo Descongestionante B', 'Descongestionante sintético didático' ), ( 10, 'Ativo Gástrico A', 'Agente gastrointestinal sintético didático' );


-- ============================================================
-- Remédios
-- ============================================================

INSERT INTO remedio (
    id,
    nome_comercial,
    fabricante
)
VALUES
    (
        1,
        'Tylenol',
        'Johnson & Johnson'
    ),
    (
        2,
        'Advil',
        'Pfizer'
    ),
    (
        3,
        'Claritin',
        'Bayer'
    ),
    ( 4, 'DemoAntitussive', 'Synthetic Labs' ), ( 5, 'DemoAntiemetic A', 'Synthetic Labs' ), ( 6, 'DemoAntiemetic B', 'Synthetic Labs' ), ( 7, 'DemoPhotophobia', 'Synthetic Labs' ), ( 8, 'DemoDecongestant A', 'Synthetic Labs' ), ( 9, 'DemoDecongestant B', 'Synthetic Labs' ), ( 10, 'DemoGastric', 'Synthetic Labs' ), ( 11, 'DemoFlu Multi', 'Synthetic Labs' );


-- ============================================================
-- Apresentações
-- ============================================================

INSERT INTO apresentacao (
    id,
    remedio_id,
    forma_farmaceutica,
    via_administracao,
    descricao,
    custo_complexidade
)
VALUES
    (
        1,
        1,
        'comprimido',
        'oral',
        '500 mg',
        0.10
    ),
    (
        2,
        1,
        'xarope',
        'oral',
        '160 mg / 5 mL',
        0.15
    ),
    (
        3,
        2,
        'comprimido',
        'oral',
        '400 mg',
        0.12
    ),
    (
        4,
        3,
        'comprimido',
        'oral',
        '10 mg',
        0.08
    ),
    ( 5, 4, 'comprimido', 'oral', '20 mg', 0.10 ), ( 6, 5, 'comprimido', 'oral', '10 mg', 0.10 ), ( 7, 6, 'comprimido', 'oral', '10 mg', 0.07 ), ( 8, 7, 'comprimido', 'oral', '5 mg', 0.11 ), ( 9, 8, 'comprimido', 'oral', '10 mg', 0.10 ), ( 10, 9, 'comprimido', 'oral', '10 mg', 0.08 ), ( 11, 10, 'comprimido', 'oral', '20 mg', 0.09 ), ( 12, 11, 'comprimido', 'oral', 'combinação 20 mg + 10 mg', 0.16 );


-- ============================================================
-- Patologia ──apresenta──> Sintoma
--
-- Valores didáticos.
--
-- Campos:
--
-- prevalencia
-- peso_clinico
-- limiar_tolerancia = tau_{p,s}
-- limiar_controle   = eta_{p,s}
-- ============================================================

INSERT INTO patologia_sintoma (
    patologia_id,
    sintoma_id,
    frequencia,
    prevalencia,
    peso_clinico,
    limiar_tolerancia,
    limiar_controle
)
VALUES
    -- Gripe
    (
        1, 1,
        'comum',
        0.70,
        0.70,
        0.10,
        0.70
    ),
    (
        1, 3,
        'comum',
        0.70,
        0.40,
        0.10,
        0.65
    ),
    (
        1, 4,
        'comum',
        0.75,
        0.25,
        0.05,
        0.70
    ),
    (
        1, 5,
        'comum',
        0.65,
        0.45,
        0.10,
        0.60
    ),

    -- Enxaqueca
    (
        2, 2,
        'comum',
        0.95,
        0.95,
        0.15,
        0.70
    ),
    (
        2, 7,
        'ocasional',
        0.60,
        0.55,
        0.10,
        0.60
    ),
    (
        2, 8,
        'ocasional',
        0.50,
        0.60,
        0.10,
        0.60
    ),

    -- Rinite alérgica
    (
        3, 3,
        'comum',
        0.80,
        0.45,
        0.10,
        0.60
    ),
    (
        3, 4,
        'comum',
        0.85,
        0.35,
        0.05,
        0.70
    ),
    (
        3, 6,
        'comum',
        0.80,
        0.30,
        0.05,
        0.70
    );


-- ============================================================
-- Apresentação ──contém──> Princípio Ativo
-- ============================================================

INSERT INTO apresentacao_principio_ativo (
    apresentacao_id,
    principio_ativo_id,
    quantidade,
    unidade,
    volume_referencia,
    unidade_referencia
)
VALUES
    (
        1,
        1,
        500,
        'mg',
        NULL,
        NULL
    ),
    (
        2,
        1,
        160,
        'mg',
        5,
        'mL'
    ),
    (
        3,
        2,
        400,
        'mg',
        NULL,
        NULL
    ),
    (
        4,
        3,
        10,
        'mg',
        NULL,
        NULL
    ),
    (5, 4, 20, 'mg', NULL, NULL), (6, 5, 10, 'mg', NULL, NULL), (7, 6, 10, 'mg', NULL, NULL), (8, 7, 5, 'mg', NULL, NULL), (9, 8, 10, 'mg', NULL, NULL), (10, 9, 10, 'mg', NULL, NULL), (11, 10, 20, 'mg', NULL, NULL), (12, 4, 20, 'mg', NULL, NULL), (12, 8, 10, 'mg', NULL, NULL);


-- ============================================================
-- Apresentação ──indicada_para──> Patologia
-- ============================================================

INSERT INTO apresentacao_indicacao (
    apresentacao_id,
    patologia_id,
    evidencia
)
VALUES
    (
        1,
        1,
        'Uso sintomático para febre e dor associadas a quadros gripais'
    ),
    (
        2,
        1,
        'Uso sintomático para febre associada a quadros gripais'
    ),
    (
        3,
        2,
        'Uso analgésico e anti-inflamatório para episódios de cefaleia'
    ),
    (
        4,
        3,
        'Uso anti-histamínico para sintomas de rinite alérgica'
    ),
    ( 5, 1, 'Indicação sintética didática para tosse associada à gripe' ), 
    ( 6, 2, 'Indicação sintética didática para náusea associada à enxaqueca' ), 
    ( 7, 2, 'Indicação sintética didática para náusea associada à enxaqueca' ), 
    ( 8, 2, 'Indicação sintética didática para sensibilidade à luz' ), 
    ( 9, 1, 'Indicação sintética didática para congestão nasal' ), 
    ( 9, 3, 'Indicação sintética didática para congestão nasal' ), 
    ( 10, 1, 'Indicação sintética didática para sintomas nasais' ), 
    ( 10, 3, 'Indicação sintética didática para sintomas nasais' ), 
    ( 12, 1, 'Combinação sintética didática multi-sintoma para gripe' );


-- ============================================================
-- Apresentação ──alivia──> Sintoma
--
-- e_m(s) =
--
--     eficacia * probabilidade_resposta
-- ============================================================

INSERT INTO apresentacao_alivia (
    apresentacao_id,
    sintoma_id,
    eficacia,
    probabilidade_resposta,
    evidencia
)
VALUES
    -- Tylenol comprimido
    (
        1,
        1,
        0.85,
        0.90,
        'Valor didático'
    ),
    (
        1,
        2,
        0.65,
        0.80,
        'Valor didático'
    ),

    -- Tylenol xarope
    (
        2,
        1,
        0.80,
        0.88,
        'Valor didático'
    ),

    -- Advil
    (
        3,
        1,
        0.80,
        0.88,
        'Valor didático'
    ),
    (
        3,
        2,
        0.90,
        0.90,
        'Valor didático'
    ),

    -- Claritin
    (
        4,
        3,
        0.65,
        0.80,
        'Valor didático'
    ),
    (
        4,
        4,
        0.85,
        0.90,
        'Valor didático'
    ),
    (
        4,
        6,
        0.90,
        0.92,
        'Valor didático'
    ),
    ( 5, 5, 0.80, 0.85, 'Valor sintético didático' ),
    ( 6, 8, 0.80, 0.85, 'Valor sintético didático' ),
    ( 7, 8, 0.90, 0.92, 'Valor sintético didático' ),
    ( 8, 7, 0.75, 0.80, 'Valor sintético didático' ),
    ( 9, 3, 0.80, 0.85, 'Valor sintético didático' ),
    ( 10, 3, 0.70, 0.85, 'Valor sintético didático' ), 
    ( 10, 4, 0.65, 0.80, 'Valor sintético didático' ),
    ( 11, 9, 0.85, 0.90, 'Valor sintético didático' ),
    ( 12, 5, 0.70, 0.80, 'Valor sintético didático' ), 
    ( 12, 3, 0.75, 0.80, 'Valor sintético didático' ), 
    ( 12, 4, 0.70, 0.78, 'Valor sintético didático' );


-- ============================================================
-- Apresentação ──evento_adverso──> Sintoma
--
-- Valores didáticos.
--
-- risco_maximo_admissivel será usado posteriormente para
-- definir a região de viabilidade F_p.
-- ============================================================

INSERT INTO apresentacao_evento_adverso (
    apresentacao_id,
    sintoma_id,
    probabilidade,
    gravidade,
    gravidade_peso,
    peso_clinico,
    risco_maximo_admissivel,
    evidencia
)
VALUES
    -- Tylenol -> Náusea
    (
        1,
        8,
        0.03,
        'leve',
        0.20,
        0.40,
        0.10,
        'Valor didático'
    ),

    -- Advil -> Dor de estômago
    (
        3,
        9,
        0.25,
        'moderada',
        0.50,
        0.60,
        0.30,
        'Valor didático'
    ),

    -- Advil -> Náusea
    (
        3,
        8,
        0.10,
        'leve',
        0.25,
        0.40,
        0.15,
        'Valor didático'
    ),

    -- Claritin -> Sonolência
    (
        4,
        10,
        0.20,
        'leve',
        0.25,
        0.35,
        0.25,
        'Valor didático'
    ),

    -- Claritin -> Reação alérgica
    --
    -- Deliberadamente configurado para tornar essa situação
    -- potencialmente inviável:
    --
    --     0.01 > 0.005
    --
    (
        4,
        11,
        0.01,
        'grave',
        1.00,
        1.00,
        0.005,
        'Valor didático'
    ),
    
    ( 5, 10, 0.12, 'leve', 0.30, 0.35, 0.18, 'Valor sintético didático' ),
    ( 6, 10, 0.10, 'leve', 0.25, 0.30, 0.18, 'Valor sintético didático' ),
    ( 8, 8, 0.08, 'leve', 0.25, 0.40, 0.15, 'Valor sintético didático' ),
    ( 9, 2, 0.08, 'moderada', 0.40, 0.50, 0.12, 'Valor sintético didático'),
    ( 11, 10, 0.07, 'leve', 0.20, 0.30, 0.18, 'Valor sintético didático' ),
    ( 12, 10, 0.08, 'leve', 0.25, 0.35, 0.18, 'Valor sintético didático' ),
    ( 12, 2, 0.06, 'leve', 0.25, 0.40, 0.12, 'Valor sintético didático' );


-- ============================================================
-- Apresentação ──interage_com──> Apresentação
--
-- Valores exclusivamente didáticos.
-- ============================================================

INSERT INTO apresentacao_interacao (
    apresentacao_a_id,
    apresentacao_b_id,
    severidade,
    descricao
)
VALUES
    (
        1,
        3,
        0.15,
        'Interação didática para testar penalidade combinatória'
    ),
    (
        3,
        4,
        0.10,
        'Interação didática para testar penalidade combinatória'
    ),
    ( 5, 6, 0.08, 'Interação sintética pequena; risco agregado de sonolência é o efeito experimental principal' ), ( 6, 8, 0.12, 'Interação didática entre tratamento da fotofobia e antiemético' ), ( 7, 8, 0.05, 'Interação didática de baixa intensidade' ), ( 3, 11, 0.05, 'Interação didática entre Advil e tratamento do ADR gástrico' ), ( 9, 12, 0.70, 'Interação didática forte entre descongestionante isolado e combinação' ), ( 10, 12, 0.20, 'Interação didática entre tratamentos multi-target concorrentes' );



-- ============================================================
-- VIEWS ANALÍTICAS
-- ============================================================


-- ------------------------------------------------------------
-- Benefício terapêutico esperado
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW vw_beneficio_terapeutico AS
SELECT
    aa.apresentacao_id,
    aa.sintoma_id,

    aa.eficacia,
    aa.probabilidade_resposta,

    aa.eficacia
        * aa.probabilidade_resposta
        AS beneficio_esperado

FROM apresentacao_alivia AS aa;


-- ------------------------------------------------------------
-- Risco esperado de evento adverso
--
-- Observação:
--
-- risco_esperado é uma medida de burden.
--
-- A decisão de viabilidade usa também:
--
--     probabilidade
--     gravidade
--     risco_maximo_admissivel
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW vw_risco_evento_adverso AS
SELECT
    ae.apresentacao_id,
    ae.sintoma_id,

    ae.probabilidade,
    ae.gravidade,
    ae.gravidade_peso,
    ae.peso_clinico,
    ae.risco_maximo_admissivel,

    ae.probabilidade
        * ae.gravidade_peso
        * ae.peso_clinico
        AS risco_esperado

FROM apresentacao_evento_adverso AS ae;


-- ------------------------------------------------------------
-- Carga inicial por patologia e sintoma
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW vw_carga_patologia AS
SELECT
    ps.patologia_id,
    ps.sintoma_id,

    ps.prevalencia,
    ps.peso_clinico,

    ps.limiar_tolerancia,
    ps.limiar_controle,

    ps.prevalencia
        * ps.peso_clinico
        AS carga_inicial

FROM patologia_sintoma AS ps;


-- ------------------------------------------------------------
-- Matriz positiva/negativa apresentação x sintoma
--
-- Facilita inspeção do espaço utilizado pelo solver.
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW vw_matriz_efeitos AS

WITH positivos AS (
    SELECT
        apresentacao_id,
        sintoma_id,
        beneficio_esperado
    FROM vw_beneficio_terapeutico
),

negativos AS (
    SELECT
        apresentacao_id,
        sintoma_id,
        risco_esperado
    FROM vw_risco_evento_adverso
),

pares AS (
    SELECT
        apresentacao_id,
        sintoma_id
    FROM positivos

    UNION

    SELECT
        apresentacao_id,
        sintoma_id
    FROM negativos
)

SELECT
    pares.apresentacao_id,
    pares.sintoma_id,

    COALESCE(
        positivos.beneficio_esperado,
        0
    ) AS beneficio_esperado,

    COALESCE(
        negativos.risco_esperado,
        0
    ) AS risco_esperado,

    COALESCE(
        positivos.beneficio_esperado,
        0
    )
    -
    COALESCE(
        negativos.risco_esperado,
        0
    ) AS efeito_liquido

FROM pares

LEFT JOIN positivos
    ON positivos.apresentacao_id
        = pares.apresentacao_id
   AND positivos.sintoma_id
        = pares.sintoma_id

LEFT JOIN negativos
    ON negativos.apresentacao_id
        = pares.apresentacao_id
   AND negativos.sintoma_id
        = pares.sintoma_id;


-- ============================================================
-- CONSULTAS DE VALIDAÇÃO
-- ============================================================


-- ------------------------------------------------------------
-- 1. Patologia, sintomas e política contextual
-- ------------------------------------------------------------

SELECT
    p.nome AS patologia,

    s.nome AS sintoma,

    ps.frequencia,
    ps.prevalencia,
    ps.peso_clinico,

    ps.limiar_tolerancia,
    ps.limiar_controle,

    ps.prevalencia
        * ps.peso_clinico
        AS carga_inicial

FROM patologia_sintoma AS ps

JOIN patologia AS p
    ON p.id = ps.patologia_id

JOIN sintoma AS s
    ON s.id = ps.sintoma_id

ORDER BY
    p.nome,
    carga_inicial DESC;


-- ------------------------------------------------------------
-- 2. Composição das apresentações
-- ------------------------------------------------------------

SELECT
    r.nome_comercial,

    a.forma_farmaceutica,
    a.descricao AS apresentacao,

    pa.nome AS principio_ativo,

    apa.quantidade,
    apa.unidade,
    apa.volume_referencia,
    apa.unidade_referencia

FROM remedio AS r

JOIN apresentacao AS a
    ON a.remedio_id = r.id

JOIN apresentacao_principio_ativo AS apa
    ON apa.apresentacao_id = a.id

JOIN principio_ativo AS pa
    ON pa.id = apa.principio_ativo_id

ORDER BY
    r.nome_comercial,
    a.id;


-- ------------------------------------------------------------
-- 3. Indicações
-- ------------------------------------------------------------

SELECT
    p.nome AS patologia,

    r.nome_comercial,

    a.forma_farmaceutica,
    a.descricao AS apresentacao,

    ai.evidencia

FROM apresentacao_indicacao AS ai

JOIN patologia AS p
    ON p.id = ai.patologia_id

JOIN apresentacao AS a
    ON a.id = ai.apresentacao_id

JOIN remedio AS r
    ON r.id = a.remedio_id

ORDER BY
    p.nome,
    r.nome_comercial;


-- ------------------------------------------------------------
-- 4. Benefício esperado
-- ------------------------------------------------------------

SELECT
    r.nome_comercial,

    a.descricao AS apresentacao,

    s.nome AS sintoma,

    bt.eficacia,
    bt.probabilidade_resposta,
    bt.beneficio_esperado

FROM vw_beneficio_terapeutico AS bt

JOIN apresentacao AS a
    ON a.id = bt.apresentacao_id

JOIN remedio AS r
    ON r.id = a.remedio_id

JOIN sintoma AS s
    ON s.id = bt.sintoma_id

ORDER BY
    r.nome_comercial,
    bt.beneficio_esperado DESC;


-- ------------------------------------------------------------
-- 5. Eventos adversos e política de admissibilidade
-- ------------------------------------------------------------

SELECT
    r.nome_comercial,

    a.descricao AS apresentacao,

    s.nome AS evento_adverso,

    rea.probabilidade,
    rea.gravidade,
    rea.gravidade_peso,
    rea.peso_clinico,

    rea.risco_esperado,
    rea.risco_maximo_admissivel,

    rea.probabilidade
        <= rea.risco_maximo_admissivel
        AS admissivel_individualmente

FROM vw_risco_evento_adverso AS rea

JOIN apresentacao AS a
    ON a.id = rea.apresentacao_id

JOIN remedio AS r
    ON r.id = a.remedio_id

JOIN sintoma AS s
    ON s.id = rea.sintoma_id

ORDER BY
    rea.gravidade_peso DESC,
    rea.probabilidade DESC;


-- ------------------------------------------------------------
-- 6. Interações
-- ------------------------------------------------------------

SELECT
    ia.nome_comercial AS remedio_a,
    aa.descricao AS apresentacao_a,

    ib.nome_comercial AS remedio_b,
    ab.descricao AS apresentacao_b,

    i.severidade,
    i.descricao

FROM apresentacao_interacao AS i

JOIN apresentacao AS aa
    ON aa.id = i.apresentacao_a_id

JOIN remedio AS ia
    ON ia.id = aa.remedio_id

JOIN apresentacao AS ab
    ON ab.id = i.apresentacao_b_id

JOIN remedio AS ib
    ON ib.id = ab.remedio_id

ORDER BY
    i.severidade DESC;


-- ------------------------------------------------------------
-- 7. Matriz apresentação x sintoma
-- ------------------------------------------------------------

SELECT
    r.nome_comercial,

    a.descricao AS apresentacao,

    s.nome AS sintoma,

    me.beneficio_esperado,
    me.risco_esperado,
    me.efeito_liquido

FROM vw_matriz_efeitos AS me

JOIN apresentacao AS a
    ON a.id = me.apresentacao_id

JOIN remedio AS r
    ON r.id = a.remedio_id

JOIN sintoma AS s
    ON s.id = me.sintoma_id

ORDER BY
    r.nome_comercial,
    s.nome;


-- ------------------------------------------------------------
-- 8. Sanity check:
--    eventos adversos individualmente acima do limite
-- ------------------------------------------------------------

SELECT
    r.nome_comercial,
    a.descricao AS apresentacao,
    s.nome AS evento_adverso,

    ae.probabilidade,
    ae.gravidade,
    ae.risco_maximo_admissivel

FROM apresentacao_evento_adverso AS ae

JOIN apresentacao AS a
    ON a.id = ae.apresentacao_id

JOIN remedio AS r
    ON r.id = a.remedio_id

JOIN sintoma AS s
    ON s.id = ae.sintoma_id

WHERE
    ae.probabilidade
    > ae.risco_maximo_admissivel

ORDER BY
    ae.probabilidade DESC;
