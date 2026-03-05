# GABARITO AULA 07: TABELAS PONTE (BRIDGE TABLES)

## EXERCÍCIO 1: Dimensão Multi-Valorada (Consultas e Diagnósticos)

**a) Insira 2 novos diagnósticos na `dim_diagnostico`**

```sql
INSERT INTO dim_diagnostico (codigo_cid, descricao) VALUES
('G43', 'Enxaqueca Crônica'),
('I10', 'Hipertensão');
```

**b) Mapeie um novo `grupo_diagnostico_key` na Bridge**

```sql
-- Assumindo que os IDs gerados sequencialmente foram 4 e 5 para os novos diagnósticos
INSERT INTO bridge_grupo_diagnostico (grupo_diagnostico_key, diagnostico_id, peso_alocacao) VALUES
(200, 4, 0.5000),
(200, 5, 0.5000);
```

**c) Insira 1 nova consulta na Fato associada ao Grupo**

```sql
INSERT INTO fato_consulta (data_consulta, medico_id, paciente_id, grupo_diagnostico_key, valor_consulta)
VALUES ('2024-03-05', 2, 11, 200, 500.00);
```

**d) Query de Análise Alocada (Evitando Dupla Contagem)**

```sql
SELECT 
    d.descricao, 
    SUM(f.valor_consulta * b.peso_alocacao) as valor_alocado
FROM fato_consulta f
JOIN bridge_grupo_diagnostico b ON f.grupo_diagnostico_key = b.grupo_diagnostico_key
JOIN dim_diagnostico d ON b.diagnostico_id = d.diagnostico_id
GROUP BY d.descricao;
```

---

## EXERCÍCIO 2: Bridge Entre Dimensões (Contas e Titulares)

**a) Crie 2 novos clientes e 1 nova conta**

```sql
INSERT INTO dim_cliente (nome_cliente) VALUES
('João Silva'),
('Maria Oliveira');

INSERT INTO dim_conta (agencia, numero_conta) VALUES
('0001', '99999');
```

**b) Crie o vínculo bridge com 50% de peso para cada titular**

```sql
-- Assumindo cliente_id 1 e 2, conta_id 1 baseados nas novas inserções (ou uso de sequence/CURRVAL logado)
INSERT INTO bridge_conta_titular (conta_id, cliente_id, peso_alocacao) VALUES
(1, 1, 0.5000),
(1, 2, 0.5000);
```

**c) Registre 1 transação (Fato) associada à conta conjunta**

```sql
INSERT INTO fato_transacao (conta_id, valor_transacao, data_transacao)
VALUES (1, 1200.00, '2024-03-05 10:00:00');
```

**d) Query calculando o volume financeiro por cliente**

```sql
SELECT 
    c.nome_cliente, 
    SUM(ft.valor_transacao * bct.peso_alocacao) AS volume_financeiro_alocado
FROM fato_transacao ft
JOIN bridge_conta_titular bct ON ft.conta_id = bct.conta_id
JOIN dim_cliente c ON bct.cliente_id = c.cliente_id
GROUP BY c.nome_cliente;
```

### ASSERTIONS (VALIDAÇÃO DE RESULTADOS)

```sql
DO $$
BEGIN
   -- Validação 1: Verificar integridade do peso da bridge de diagnóstico grupo 200
   IF (SELECT SUM(peso_alocacao) FROM bridge_grupo_diagnostico WHERE grupo_diagnostico_key = 200) <> 1.0000 THEN
      RAISE EXCEPTION 'Erro: A soma dos pesos do grupo 200 na bridge de diagnósticos não é 1.0';
   END IF;

   -- Validação 2: Verificar se a soma das transações na bridge é igual a 100% da conta (id genérico 1)
   IF (SELECT SUM(peso_alocacao) FROM bridge_conta_titular WHERE conta_id = 1) <> 1.0000 THEN
      RAISE EXCEPTION 'Erro: A soma dos pesos para a conta 1 não é 1.0 na bridge de titulares';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 07: SUCESSO! ✅';
END $$;
```
