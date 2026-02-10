-- ==============================================================================
-- Aula 1: Introdução a Data Modelling
-- ==============================================================================

/*
OBJETIVOS:
- Compreender o que é modelagem de dados
- Diferenciar OLTP (Operacional) vs OLAP (Analítico)
- Conhecer os tipos de modelagem: Conceitual, Lógica, Física

CONCEITOS CHAVE:

1. OLTP (Online Transaction Processing)
   - Foco: Operação do dia a dia (Vendas, Cadastros)
   - Característica: Alta velocidade de escrita, normalizado (3NF)
   - Exemplo: Sistema de E-commerce (Tabelas de Pedidos, Clientes)

2. OLAP (Online Analytical Processing)
   - Foco: Análise e Tomada de Decisão
   - Característica: Leitura otimizada, desnormalizado, histórico
   - Exemplo: Data Warehouse (Relatórios de Vendas por Trimestre)

3. TIPOS DE MODELAGEM
   - Conceitual: Visão de alto nível (Entidades e Relações) - "O Que"
   - Lógica: Atributos, Chaves, Tipos de dados (independente de banco) - "Como"
   - Física: SQL, DDL, Índices (PostgreSQL, Oracle, etc) - "Implementação"
*/

-- ==============================================================================
-- PREPARAÇÃO DO AMBIENTE (Física)
-- ==============================================================================

-- Criar Schema para organizar o curso
DROP SCHEMA IF EXISTS curso_modelagem CASCADE;
CREATE SCHEMA curso_modelagem;

-- Definir como schema padrão para a sessão
SET search_path TO curso_modelagem;

-- Próximos passos: Aula 2 (ERD e Tabelas Simples)
