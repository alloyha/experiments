# Aula 6: Tabelas Dimensão

## 🎯 Objetivos
- Entender o propósito das **Dimensões** (Contexto) no Data Warehouse.
- Diferenciar os principais **Tipos de Dimensão** (SCD1, SCD2, Junk, Degenerate).
- Implementar **Surrogate Keys** e manutenção de histórico.
- Comparar modelagem clássica (Snowflake) vs. Big Data (Star Schema / Denormalização).

---

## 📦 1. Anatomia de uma Dimensão
Diferente das tabelas fato, dimensões são "largas", ricas em texto e desnormalizadas.

- **Surrogate Key (SK):** Chave primária gerada internamente (ex: SERIAL). Protege o DW de mudanças no sistema origem.
- **Natural Key (NK):** O ID original do sistema operacional (ex: `usuario_id` do banco de produção).
- **Atributos:** Campos descritivos (Nome, Segmento, Região) usados para filtros e agrupamentos.

---

## 🎭 2. Tipos de Dimensão

### SCD (Slowly Changing Dimensions)
Como lidar com mudanças nos atributos (ex: um usuário mudou de categoria):
- **SCD TIPO 1 (Overwrite):** Sobrescreve o valor antigo. Não guarda histórico. (Uso: correção de erros).
- **SCD TIPO 2 (Add Row):** Cria uma nova linha para a nova versão. Mantém histórico completo. (Uso: rastreio de carreira/segmento).

### Outras Dimensões Especiais
- **Degenerada:** Vive na tabela fato (ex: Número da Nota Fiscal).
- **Junk Dimension:** Agrupa múltiplos flags e indicadores pequenos (SIM/NÃO) em uma única tabela para reduzir o número de colunas no fato.
- **Conformada:** Dimensão idêntica usada por múltiplos fatos (ex: `dim_tempo`).

---

## 📐 3. Estruturação: Snowflake vs. Star Schema
- **Snowflake (Normalizado):** Economiza espaço, mas requer muitos JOINs.
- **Star Schema (Desnormalizado):** Padrão Gold para performance. Traz atributos de tabelas relacionadas (ex: Cidade/Estado) diretamente para a dimensão principal.

---

## 🏁 Fechamento
- Dimensões dão o significado aos números.
- Escolha SCD2 quando o histórico for vital para o negócio.
- Surrogate keys são fundamentais para a estabilidade do DW.
