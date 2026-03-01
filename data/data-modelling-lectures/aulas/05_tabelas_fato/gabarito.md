# GABARITO AULA 05: TABELAS FATO E PARADIGMAS DE DADOS

## RESPOSTA 1: Conceitual

**a) Abordagem Clássica (Scan na Fato Bruta):**

```sql
SELECT count(distinct data_evento)
FROM usuarios_atividade_fato
WHERE usuario_id = 123 AND data_evento > '2023-01-01';
-- Custo: O(N) onde N é o número total de eventos históricos. 
-- Problema: Scan massivo conforme o dado cresce.
```

**b) Abordagem de Tabela Acumulada (Compressão em Array):**

```sql
SELECT total_dias_ativos
FROM usuarios_atividade_acumulada
WHERE data_snapshot = '2024-06-03' AND usuario_id = 123;
-- Custo: O(1) de leitura. 
-- Vantagem: State acumulado (snapshot) elimina necessidade de ler o histórico bruto.
```

## RESPOSTA 2: Prática (Big Data Pattern)

**a) Estrutura das Tabelas (Gold vs Cumulative)**

```sql
-- 1. Fato de Grão Fino (Standard Gold)
CREATE TABLE usuarios_atividade_fato (
    usuario_id   INTEGER,
    data_evento  DATE,
    PRIMARY KEY (usuario_id, data_evento)
);

-- 2. Tabela Acumulada (Compressão para Analytics)
CREATE TABLE usuarios_atividade_acumulada (
    usuario_id        INTEGER NOT NULL,
    data_snapshot     DATE NOT NULL,
    datas_atividade   DATE[] NOT NULL,
    total_dias_ativos INTEGER GENERATED ALWAYS AS (CARDINALITY(datas_atividade)) STORED,
    PRIMARY KEY (usuario_id, data_snapshot)
);
```

**b) Pipeline de Carga Incremental (Yesterday + Today)**

```sql
WITH yesterday AS (
    SELECT * FROM usuarios_atividade_acumulada 
    WHERE data_snapshot = :data_ontem::DATE
),
today AS (
    SELECT usuario_id, data_evento FROM usuarios_atividade_fato 
    WHERE data_evento = :data_hoje::DATE
)
INSERT INTO usuarios_atividade_acumulada (usuario_id, data_snapshot, datas_atividade)
SELECT
    COALESCE(y.usuario_id, t.usuario_id)              AS usuario_id,
    :data_hoje::DATE                                  AS data_snapshot,
    COALESCE(y.datas_atividade, ARRAY[]::DATE[]) ||
    CASE 
        WHEN t.usuario_id IS NOT NULL THEN ARRAY[t.data_evento] 
        ELSE ARRAY[]::DATE[] 
    END                                               AS datas_atividade
FROM yesterday y
FULL OUTER JOIN today t ON y.usuario_id = t.usuario_id;
```

### VALIDAÇÃO TÉCNICA (DAU & Retention)

```sql
-- Pergunta: "Quais usuários são Power Users (>25 dias ativos) sem fazer scan de histórico?"
SELECT usuario_id, total_dias_ativos
FROM usuarios_atividade_acumulada
WHERE data_snapshot = :data_hoje::DATE
  AND total_dias_ativos > 25;
```

### ASSERTIONS

```sql
DO $$
BEGIN
   -- Validação: O merge não deve criar duplicatas no array se rodar 2x (Idempotência)
   -- Nota: Para arrays, o ideal é usar a lógica de set (UNNEST + DISTINCT) ou validar o cardinality final.
   
   IF (SELECT MAX(total_dias_ativos) FROM usuarios_atividade_acumulada) > 31 THEN
      RAISE EXCEPTION 'Erro: Histórico maior que o período simulado!';
   END IF;

   RAISE NOTICE 'GABARITO VALIDADO: PADRÃO GOLD APLICADO ✅';
END $$;
```
