# EXERCÍCIOS AULA 05: TABELAS FATO E PARADIGMAS DE DADOS

## EXERCÍCIO 1: Comparação de Performance (Conceitual)

**Cenário:** Você tem uma tabela de logs com 1 bilhão de linhas.
**Tarefa:** Escreva (no papel ou comentário) a query para contar usuários ativos ontem E hoje.

**a)** Usando a abordagem clássica (SCAN na tabela de fatos bruta).

**b)** Usando a abordagem de Tabela Acumulada (FULL OUTER JOIN de arrays).

## EXERCÍCIO 2: Implementando Tabela Acumulada

**a)** Crie a tabela `usuarios_atividade_acumulada` com um campo ARRAY de datas.

**b)** Insira um registro inicial para o usuário 1 com datas antigas.

**c)** Simule um novo dia de atividade e faça o UPDATE/MERGE do array.

**d)** Selecione o usuário e verifique se o array contém o histórico + data nova.
