# Curso de Modelagem de Dados com PostgreSQL

Este repositório contém o material completo do curso prático de Modelagem de Dados, indo desde os fundamentos relacionais até técnicas avançadas de Big Data Engineering e Grafos.

## 🎯 Objetivo

O objetivo deste curso é ensinar a modelar dados de forma eficiente, pragmática e escalável. Diferente de cursos tradicionais, aqui combinamos a teoria clássica (Kimball) com as práticas modernas das Big Techs (Netflix, Airbnb, Facebook), inspiradas nas melhores práticas de Engenharia de Dados.

## 👥 Público Alvo

Este material é destinado a:
*   **Engenheiros de Dados** que buscam consolidar conceitos de Data Warehousing e aprender padrões modernos de Big Data.
*   **Desenvolvedores Backend** interessados em otimização de banco de dados e design de schemas robustos.
*   **Analistas de Dados/BI** que desejam entender a estrutura por trás de relatórios performáticos.
*   **Estudantes** de computação que querem ir além do básico "SELECT * FROM".

## 📊 Nível do Conteúdo

*   **Nível:** Intermediário a Avançado.
*   **Pré-requisitos:** Conhecimento básico de SQL (SELECT, INSERT, JOINs).
*   **Foco:** A transição do Modelo Relacional (3NF) para Modelagem Dimensional (Star Schema) e técnicas de **Engenharia de Dados em Escala** (Structs, Arrays, Cumulative Tables).

## 📚 Estrutura do Curso

O curso é dividido em 10 aulas práticas, cada uma contendo:
*   `apresentacao.sql`: Conceitos e exemplos explicados.
*   `exercicios.sql`: Desafios para fixação.
*   `gabarito.sql`: Solução comentada.

### Módulos

*   **Aula 01: Introdução** - OLTP vs OLAP, Modelagem Conceitual/Lógica/Física.
*   **Aula 02: ERD (Entity Relationship Diagrams)** - Entidades, Atributos, Normalização (1NF, 2NF, 3NF).
*   **Aula 03: Prática de ERD** - Modelagem completa de um sistema de Biblioteca.
*   **Aula 04: Introdução Dimensional** - Star Schema vs Snowflake, Fatos e Dimensões.
*   **Aula 05: Tabelas Fato & Big Data** - Fatos Transacionais, Snapshots e **Cumulative Tables** (State Management).
*   **Aula 06: Tabelas Dimensão** - Dimensões, Hierarquias e **Structs/Arrays** para eliminar Joins.
*   **Aula 07: Bridge Tables & Array Metrics** - Resolvendo relacionamentos N:N com tabelas ponte e Arrays.
*   **Aula 08: SCD (Slowly Changing Dimensions)** - Tipos 0, 1, 2, 3 e estratégias de performance.
*   **Aula 09: Implementação de SCD Type 2** - O padrão clássico vs **Nested History** (Histórico numa única linha).
*   **Aula 10: Modelagem de Grafos** - Quando o relacional falha: modelando redes complexas.

## 🚀 Como Executar

Este projeto utiliza **Docker** para subir um ambiente PostgreSQL pronto para uso.

### Pré-requisitos
*   Docker & Docker Compose instalados.
*   Um cliente SQL (DBeaver, VSCode SQLTools, Datagrip) ou terminal (`psql`).

### Passo a Passo

1.  **Subir o Banco de Dados:**
    ```bash
    docker compose up -d
    ```
    Isso iniciará um container PostgreSQL e executará automaticamente o script `setup_database.sql`, criando as tabelas base e inserindo dados de exemplo.

2.  **Conectar ao Banco:**
    *   **Host:** `localhost`
    *   **Port:** `5432`
    *   **Database:** `curso_modelagem`
    *   **User:** `aluno`
    *   **Password:** `modelagem_password`

3.  **Explorar as Aulas:**
    Navegue pelas pastas `aula_XX` e execute os scripts SQL no seu cliente favorito.

## 🌟 Destaques "Big Data"

Além do currículo tradicional de Data Warehousing, este curso inclui adaptações para **Engenharia de Dados em Escala (Big Data)**:

*   **Cumulative Table Design:** "Yesterday + Today = Tomorrow". Como gerenciar estado de usuários sem scans históricos massivos.
*   **Array Metrics:** Substituição de Bridge Tables custosas por Arrays desnormalizados.
*   **Nested Data (Structs):** Como compactar histórico de SCD Type 2 em uma única linha para evitar Shuffle em processamento distribuído (Spark/Trino).
*   **Idempotência:** Uso de `INSERT ON CONFLICT` para garantir pipelines robustos.

---
*Material desenvolvido para estudo e prática de engenharia de dados.*
