# EXERCÍCIOS AULA 07: TABELAS PONTE (BRIDGE TABLES)

## EXERCÍCIO 1: Dimensão Multi-Valorada (Consultas e Diagnósticos)

**a)** Insira 2 novos diagnósticos na `dim_diagnostico` (Ex: Enxaqueca Crônica, Hipertensão).

**b)** Mapeie um novo `grupo_diagnostico_key` (ex: 200) na `bridge_grupo_diagnostico`, vinculando esses 2 novos diagnósticos com peso de 50% (`0.5000`) cada.

**c)** Insira 1 nova consulta na `fato_consulta`, com valor de R$ 500,00, associada ao `grupo_diagnostico_key` 200.

**d)** Construa a Query que traga o `valor_alocado` total por `descricao` de diagnóstico, garantindo a multiplicação pelo `peso_alocacao` para não inflarmos a soma das faturas contabilizadas na clínica.

---

## EXERCÍCIO 2: Bridge Entre Dimensões (Contas e Titulares)

**a)** Crie 2 novos clientes na `dim_cliente` (Ex: João e Maria) e 1 nova conta na `dim_conta` (Agência 0001, Conta 99999).

**b)** Crie o vínculo desta conta com ambos os clientes na `bridge_conta_titular`, atribuindo 50% (`0.5000`) de `peso_alocacao` para João e 50% para Maria.

**c)** Registre 1 transação (Ex: R$ 1.200,00) na `fato_transacao` associada à conta conjunta criada.

**d)** Construa uma Query que mostre o volume financeiro total transacionado e alocado por `nome_cliente` (multiplicando pelo peso da bridge para não gerar dupla contagem contábil de movimentações da conta conjunta).
