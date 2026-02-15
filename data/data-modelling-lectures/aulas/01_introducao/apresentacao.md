# Aula 1: Introdução a Data Modelling

## 🎯 Objetivos
- Compreender o que é modelagem de dados e sua importância.
- Diferenciar modelagem operacional (OLTP) vs analítica (OLAP).
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
1. **Conceitual:** Nível mais alto. Foca no negócio. (Entidades e Relacionamentos).
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
