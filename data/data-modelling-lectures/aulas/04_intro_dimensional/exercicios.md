# EXERCÍCIOS AULA 04: INTRODUÇÃO À MODELAGEM DIMENSIONAL

## EXERCÍCIO 1: Paradigmas OLTP vs. OLAP

**Cenário:** Você precisa desenvolver um relatório de Vendas por Mês e Categoria.

**a)** Qual dos dois modelos abaixo seria mais performático para LEITURA (SELECT)? Por quê?
- **Modelo A**: Tabelas normalizadas (3NF) com muitos Joins.
- **Modelo B**: Star Schema (Fato + Dimensões desnormalizadas).

## EXERCÍCIO 2: Identificando Fatos e Dimensões

Dado o cenário de um E-commerce, classifique os itens abaixo como **Fato** ou **Dimensão**:

**a)** Valor Total da Venda
**b)** Data da Compra
**c)** Nome do Cliente
**d)** Quantidade de Itens
**e)** Categoria do Produto
**f)** Cidade de Entrega

## EXERCÍCIO 3: Star Schema vs. Snowflake

**a)** Em um Snowflake Schema, a tabela de 'Categoria' estaria ligada diretamente à Fato?

**b)** Qual a principal vantagem do Star Schema sobre o Snowflake para ferramentas de BI (PowerBI/Tableau)?
