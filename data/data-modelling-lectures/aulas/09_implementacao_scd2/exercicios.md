# EXERCÍCIOS AULA 09: IMPLEMENTAÇÃO AVANÇADA DE SCD TYPE 2

## EXERCÍCIO 1: Arquiteturas de Histórico (Clássica vs. Moderna)

**Cenário:** Um cliente mudou de segmento 5 vezes ao longo de 2 anos.

**a)** Na abordagem clássica de SCD Type 2 (uma linha por versão), quantas linhas o cliente terá na tabela de dimensão?

**b)** Na abordagem moderna de SCD aninhado (usando Arrays de Structs), quantas linhas o cliente terá na tabela?

**c)** Qual o principal benefício da abordagem aninhada (Modern/Big Data) em termos de performance de processamento distribuído (Spark/Presto)?

---

## EXERCÍCIO 2: Automação SCD Type 2 (Abordagem Clássica)

**Tarefa:**
**a)** Desenvolva uma `PROCEDURE` que receba o `cliente_id` e o `novo_segmento`. A procedure deve:
   1. Fechar o registro atual (marcar `registro_ativo = FALSE` e definir `data_fim`).
   2. Inserir a nova versão com `versao = versao_anterior + 1`.

**b)** Escreva uma query para listar o histórico completo de um cliente específico, ordenado pela versão.

---

## EXERCÍCIO 3: Implementação Big Data (Abordagem Aninhada)

**Tarefa:**
**a)** Crie uma tabela `dim_cliente_historico_aninhado` com um campo `historico` do tipo ARRAY de STRUCTS (ou um TYPE composto no PostgreSQL).

**b)** Insira um cliente manualmente simulando sua evolução de segmento (ex: 'Bronze' em 2022 -> 'Prata' em 2023).

---

## EXERCÍCIO 4: Queries Analíticas e Point-in-Time

**Tarefa:**
**a)** Utilizando a abordagem clássica, compare o volume de vendas de um cliente antes e depois de ele se tornar 'VIP' (mudança de segmento).

**b)** Utilizando a abordagem aninhada e a cláusula `UNNEST`, encontre em qual segmento o cliente estava exatamente na data '2022-06-15'.
