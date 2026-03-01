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
