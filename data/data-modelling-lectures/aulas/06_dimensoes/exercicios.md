# EXERCÍCIOS AULA 06: MODELAGEM DIMENSIONAL & CONTEXTO

## EXERCÍCIO 1: Estrutura da Dimensão (Star Schema vs. Snowflake)

**Cenário:** Precisamos mapear todos os atributos de uma `Localização` (Cidade, Estado, Região, País) para enriquecer o contexto de cada usuário.

**Tarefa:**
a) Crie a tabela `dim_localizacao_snowflake` dividindo Cidade e Estado em tabelas diferentes (Normalizado).
b) Crie a tabela `dim_localizacao_star` consolidando todos os atributos em uma única tabela (Desnormalizado).
c) No contexto de Big Data e Engenharia Analítica, por que a abordagem **Star Schema** (Desnormalizada) é preferida em relação ao Snowflake (Normalizada)?

---

## EXERCÍCIO 2: Junk Dimension vs. Degenerate Dimension

**Cenário:** Uma Tabela Fato de Logística contém atributos como `status_entrega` (SIM/NÃO), `eh_primeira_tentativa` (SIM/NÃO), `codigo_rastreamento` (ex: TRACK12345) e `tipo_veiculo` (Caminhão, Van, Moto).

**Tarefa:**
a) Quais desses atributos você colocaria em uma **Junk Dimension**? 
b) Qual desses atributos seria uma **Degenerate Dimension** e por quê?
c) Qual o benefício técnico de agrupar Flags (SIM/NÃO) em uma Junk Dimension em vez de deixá-los na tabela fato?
