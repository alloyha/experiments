# GABARITO AULA 06: MODELAGEM DIMENSIONAL & CONTEXTO

## RESPOSTA 1: Star vs. Snowflake

**a) Snowflake (Normalizado):**
```sql
CREATE TABLE dim_estado (
    id SERIAL PRIMARY KEY,
    nome_estado VARCHAR(50)
);

CREATE TABLE dim_localizacao_snowflake (
    cidade_id SERIAL PRIMARY KEY,
    nome_cidade VARCHAR(100),
    estado_id   INTEGER REFERENCES dim_estado (id)
);
```

**b) Star Schema (Desnormalizado):**
```sql
CREATE TABLE dim_localizacao_star (
    localizacao_sk SERIAL PRIMARY KEY,
    cidade         VARCHAR(100),
    estado         VARCHAR(50),
    regiao         VARCHAR(50)
);
```

**c) Comparação:**
Star Schema é preferido em Big Data (Analítico) porque reduz o custo de JOINs durante a análise. Snowflake economiza pouco espaço frente ao alto custo de JOIN em tabelas massivas.

---

## RESPOSTA 2: Junk & Degenerate Dimensions

**a) Junk Dimension:** `status_entrega`, `eh_primeira_tentativa`, `tipo_veiculo`.

**b) Degenerate Dimension:** `codigo_rastreamento`. Por ser de alta cardinalidade (quase um ID único para cada fato) e não possuir atributos descritivos próprios.

**c) Benefício Técnico:** Reduzir o número de colunas (e chaves estrangeiras) na tabela fato, facilitando o gerenciamento e otimizando o armazenamento ao agrupar flags comuns em uma única "lixeira" (junk).

---

## RESPOSTA 3: SCD Tipo 3 vs. Tipo 2

**a) Justificativa:** O SCD Tipo 3 é mais simples para este caso porque permite a comparação direta entre o estado atual e o anterior através de colunas na mesma linha (`segmento_atual` vs `segmento_anterior`). No SCD Tipo 2, seria necessário realizar um `JOIN` da tabela com ela mesma ou usar funções de janela (`LAG`) para buscar o valor da linha anterior.

**b) DDL SCD Tipo 3:**
```sql
CREATE TABLE dim_usuario_marketing (
    usuario_id        INTEGER PRIMARY KEY,
    nome              VARCHAR(100),
    categoria_atual   VARCHAR(20),
    categoria_anterior VARCHAR(20),
    data_ultima_mudanca DATE
);
```

---

## RESPOSTA 4: Role-Playing & Outrigger

**a) Role-Playing:** O conceito é **Role-Playing Dimension**. Resolvemos isso no SQL utilizando **Aliases (Apelidos)** para as tabelas no momento do JOIN.
Exemplo:
```sql
SELECT f.valor, de.data as emissao, dt.data as entrega
FROM fato_vendas f
JOIN dim_tempo de ON f.data_emissao_sk = de.tempo_sk
JOIN dim_tempo dt ON f.data_entrega_sk = dt.tempo_sk;
```

**b) Outrigger:** O Escritório se torna uma **Outrigger Dimension**. É uma dimensão que serve de "extensão" para outra dimensão, criando uma pequena cadeia que lembra o Snowflake, mas é permitida em casos específicos para evitar o crescimento exagerado de colunas em uma dimensão principal muito pesada.
