# Aula 1: Introdução a Data Modelling

## 🗺️ Mapa da Jornada
Este curso foi estruturado para levar você do design relacional básico às arquiteturas de Big Data de alta escala.

1.  **Fundamentos (Aulas 1-3):** Conceitos OLTP vs OLAP, Normalização e Tabelas Reais.
2.  **Modelagem Dimensional (Aulas 4-7):** Star Schema, Fatos, Dimensões e otimizações com Arrays/Structs.
3.  **Estado e Histórico (Aulas 8-9):** Mudanças dimensionais (SCD) e Tabelas Cumulativas.
4.  **Grafos (Aula 10):** Relações complexas e caminhos não lineares.

---

## 🎯 Objetivos
- Compreender o que é modelagem de dados e sua importância.
- Diferenciar modelagem operacional (OLTP) vs analítica (OLAP).
- Entender o papel da **Normalização** na integridade dos dados.
- Conhecer os tipos de modelagem: Conceitual, Lógica e Física.

---

## 🏗️ O que é Modelagem de Dados?
Modelagem de dados é o processo de criar uma representação visual ou um esquema que define como os dados são coletados, armazenados e acessados.

> **Analogia:** Pense na modelagem de dados como a **planta de uma casa**. Sem ela, a construção pode ser instável, difícil de manter e impossível de expandir.

### Impactos de uma má modelagem:
- **Performance:** Consultas lentas e travamentos.
- **Manutenção:** Dificuldade em corrigir erros ou adicionar campos.
- **Escalabilidade:** O sistema não aguenta o crescimento do volume de dados.

---

## ⚡ Modelagem Operacional (OLTP)
**OLTP** stands for *Online Transactional Processing*.

- **Objetivo:** Suportar as operações e transações do dia a dia.
- **Características:** 
    - Alta normalização (evitar redundância).
    - Foco na integridade dos dados.
    - Muitas escritas e atualizações rápidas.
- **Exemplo:** Sistema de e-commerce gerenciando pedidos em tempo real.

---

## 🧩 O Pilar da Integridade: Normalização

A **Normalização** é a técnica central da modelagem relacional (OLTP). Seu objetivo principal é organizar os dados para reduzir a redundância e dependência contraditória.

### Por que normalizar?
1.  **Economia de Espaço:** Evita salvar a mesma informação (ex: nome de um estado) em milhares de linhas.
2.  **Integridade (Anomalias de Escrita):** Se o CPF de um cliente mudar, você altera em **um só lugar** (Tabela Cliente), e não em cada linha de pedido.
3.  **Flexibilidade:** Facilita a expansão do banco de dados sem quebrar o que já existe.

> **Regra de Ouro:** No OLTP, cada pedaço de dado deve ter "uma única fonte da verdade".

---

## 📊 Modelagem Analítica (OLAP)
**OLAP** stands for *Online Analytical Processing*.

- **Objetivo:** Facilitar análises complexas, relatórios e tomada de decisão.
- **Características:**
    - Desnormalização (facilitar leitura).
    - Foco em grandes volumes de dados de leitura.
    - Armazenamento de histórico (snapshots).
- **Exemplo:** Data Warehouse para analisar tendências de vendas nos últimos 5 anos.

---

## 📐 Tipos de Modelagem
1. **Conceitual:** Nível mais alto (Linguagem ubíqua). Foca no negócio. (Entidades e Relacionamentos).
2. **Lógica:** Nível intermediário. Define tabelas e colunas, mas é independente de tecnologia.
3. **Física:** Implementação real no banco de dados (ex: PostgreSQL), definindo tipos de dados, índices e constraints.

---

## 🛠️ O Vocabulário do SQL
Para o modelador, o SQL se divide em dois grandes papéis:

1. **DDL (Data Definition Language):** É a **"Planta"**. Define a estrutura e as regras.
   - *Ex:* `CREATE`, `ALTER`, `DROP`.
   - Foco da Modelagem Física.

2. **DML (Data Manipulation Language):** É o **"Fluxo"**. Move e transforma os dados.
   - *Ex:* `INSERT`, `SELECT`, `UPDATE`, `DELETE`.
   - Foco da Engenharia/Uso no dia a dia.

---

## 🏁 Fechamento
- Modelagem é a fundação de qualquer sistema de dados.
- Escolher entre OLTP e OLAP depende do seu caso de uso.
- **Preview:** Na próxima aula, vamos aprender a desenhar diagramas ERD!
