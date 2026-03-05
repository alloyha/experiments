# Aula 6: Tabelas Dimensão - Além do Básico

## 🎯 Objetivos
- Entender o propósito das **Dimensões** (Contexto) no Data Warehouse.
- Dominar a taxonomia completa de dimensões (SCD, Junk, Degenerate, Outrigger, etc).
- Implementar **Surrogate Keys** e manutenção de histórico em múltiplos níveis.
- Comparar modelagem clássica (Snowflake) vs. Big Data (Star Schema / Denormalização).

---

## 📦 1. Anatomia de uma Dimensão
Diferente das tabelas fato, dimensões são "largas", ricas em texto e desnormalizadas.

- **Surrogate Key (SK):** Chave primária gerada internamente (ex: SERIAL). Protege o DW de mudanças no sistema origem.
- **Natural Key (NK):** O ID original do sistema operacional (ex: `usuario_id` do banco de produção).
- **Atributos:** Campos descritivos (Nome, Segmento, Região) usados para filtros e agrupamentos.

---

## 🎭 2. Tipos de Dimensão

### Dimensões no Tempo (SCD)

#### Básicas

- **SCD TIPO 0 (Fixed):** O valor nunca muda (ex: Data de Nascimento).
- **SCD TIPO 1 (Overwrite):** Sobrescreve o antigo. Sem histórico (ex: correção de CPF).
- **SCD TIPO 2 (Add Row):** Nova linha para cada mudança. Histórico completo (O Padrão Ouro).
- **SCD TIPO 3 (Add Column):** Mantém o valor atual e o "anterior" em colunas lado a lado na mesma linha.

#### Avançadas

- **SCD TIPO 4 (History Table):** Usa uma tabela para o estado atual e uma tabela separada para o histórico completo. Mantém a dimensão "quente" pequena.
- **SCD TIPO 5 (Mini-Dimension + Type 1):** Usado para atributos que mudam muito rápido (ex: idade, score). Os atributos voláteis vão para uma "mini-dimensão" ligada à fato.
- **SCD TIPO 6 (Híbrido 1+2+3):** Combina linhas de histórico (Tipo 2) com colunas de "valor atual" (Tipo 3) que são sobrescritas (Tipo 1) em todos os registros daquela entidade.
- **SCD TIPO 7 (Dual SK/NK):** A tabela fato armazena duas chaves: uma Surrogate Key (para histórico "as was") e uma Natural Key (para estado atual "as is").

### Dimensões de Arquitetura

- **Conformada:** Dimensão idêntica compartilhada por múltiplos fatos (ex: Mesma `dim_produto` para Vendas e estoque).
- **Junk Dimension:** Uma "lixeira organizada" que agrupa flags SIM/NÃO e indicadores de baixa cardinalidade em uma única tabela.
- **Degenerada:** Atributo que vive no fato (ex: Número da Nota) pois não tem outros atributos próprios para justificar uma tabela.
- **Outrigger:** Uma dimensão que aponta para outra dimensão (ex: `dim_vendedor` -> `dim_escritorio`). Cuidado: isso vira um Snowflake se exagerar.
- **Shrunken:** Um subconjunto de uma dimensão maior, usada para fatos agregados (ex: uma `dim_mes` derivada da `dim_tempo`).

### Dimensões de Uso

- **Role-Playing:** Uma única tabela física desempenhando múltiplos papéis no mesmo fato (ex: `dim_tempo` servindo como Data do Pedido, Data do Pagamento e Data da Entrega).
- **Static:** Tabelas que não vem da fonte original, mas são criadas no DW (ex: Tabela de Status, Categorias Fixas).

---

## 📐 3. Estruturação: Snowflake vs. Star Schema

- **Snowflake (Normalizado):** Muitas tabelas pequenas, difícil de consultar e lento (JOINs).
- **Star Schema (Desnormalizado):** Poucas tabelas largas, rápido e intuitivo. É o requisito para Big Data.

---

## 🏁 Fechamento

- Dimensões dão o significado aos números.
- Diferentes tipos resolvem diferentes problemas de custo computacional e precisão histórica.
- **Dica:** Comece sempre pelo Star Schema, só use Outriggers em casos extremos.
