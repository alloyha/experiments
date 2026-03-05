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

---

## EXERCÍCIO 3: SCD Tipo 3 vs. Tipo 2

**Cenário:** O departamento de Marketing quer analisar a mudança de categoria dos usuários. Eles precisam comparar *apenas* a categoria atual com a categoria anterior para ver quem subiu de nível no último mês.

**Tarefa:**
a) Justifique por que o **SCD Tipo 3** (Add Column) seria mais simples para essa query específica do que o SCD Tipo 2 (Add Row).

b) Escreva o DDL de uma tabela `dim_usuario_marketing` que siga o padrão SCD Tipo 3.

---

## EXERCÍCIO 4: Role-Playing & Outrigger

**Cenário:** Em um sistema de vendas, temos um `Pedido` que possui uma `Data de Emissão` e uma `Data de Entrega`. Ambas apontam para a mesma tabela `dim_tempo`. Além disso, o `Vendedor` do pedido pertence a um `Escritório`, que tem muitos detalhes de endereço.

**Tarefa:**
a) Como chamamos o conceito de usar a mesma `dim_tempo` para duas datas diferentes no mesmo fato? Como resolvemos isso no SQL?

b) Se decidirmos não desnormalizar o endereço do escritório dentro da tabela do vendedor para economizar espaço em uma base muito grande, qual tipo de dimensão o `Escritório` se torna em relação ao `Vendedor`?
