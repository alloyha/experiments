# EXERCÍCIOS AULA 06: MODELAGEM DIMENSIONAL & BIG DATA

## EXERCÍCIO 1: Estrutura da Dimensão Produto (Clássica vs. Big Data)

**a)** Crie uma tabela `dim_produto_snowflake` com a categoria isolada numa tabela separada.

**b)** Crie uma tabela `dim_produto_moderna` usando o tipo STRUCT (TYPE composto em PostgreSQL) e ARRAY para representar a mesma categoria.

**c)** Compare conceitualmente: qual das duas tabelas é mais eficiente para uma query simples?

## EXERCÍCIO 2: Junção e Desnormalização (Zero-Join)

**a)** Tente selecionar todos os produtos da `dim_produto_moderna` que possuem a categoria 'Informática' (simulando filtro em array).

**b)** Escreva a query equivalente na abordagem clássica, que exigiria JOIN.
