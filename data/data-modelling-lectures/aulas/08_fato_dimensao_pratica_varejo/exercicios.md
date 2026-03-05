# EXERCÍCIOS AULA 08: FATO E DIMENSÃO NA PRÁTICA (VAREJO + SCD)

## EXERCÍCIO 1: Estratégias Analíticas (As-Is vs As-Was)

**Cenário:** Você tem a tabela `varejo.dim_cliente`.

**a)** O campo 'estado' (SP, RJ, etc) pode mudar com relocações, mas a loja prefere não crescer a dimensão para cada pequena mudança de endereço dos clientes comuns. Qual tipo de SCD usar aqui? E que tipo de análise ele gera?

**b)** O campo 'segmento' (Ouro, Prata, Bronze) é crítico para o board diretivo. Um relatório precisa mostrar o comportamento do cliente *na época* (Ex: ticket médio que ele gerava no varejo quando ainda era Bronze). Qual tipo de SCD usar?

---

## EXERCÍCIO 2: Procedures Incrementais e Detecção de Deltas

**Cenário:** Você tem 10.000 clientes em `varejo.dim_cliente` e precisa manter uma cópia SCD Type 1 em `varejo.dim_cliente_type1`.

**a)** Por que é ineficiente escrever uma procedure que faz `INSERT ... SELECT * FROM varejo.dim_cliente ON CONFLICT DO UPDATE` sem nenhum filtro? O que acontece se apenas 1 cliente mudou?

**b)** Escreva uma procedure `proc_upsert_clientes_scd1()` que utilize `LEFT JOIN` e `IS DISTINCT FROM` para identificar **apenas** os registros novos ou alterados antes de executar o UPSERT. Garanta que, se nenhum dado mudou, o I/O seja zero.

---

## EXERCÍCIO 3: SCD Type 2 com JSONB e Diff Automático

**a)** Crie a tabela `varejo.dim_cliente_type2` com colunas `properties` e `properties_diff` (ambas JSONB), e escreva a CTE `deltas` que identifica em uma única passagem quais clientes são novos ou têm properties divergentes do registro ativo.

**b)** Usando o resultado da CTE `deltas`, demonstre como o UPDATE (fechar versão antiga) e o INSERT (abrir versão nova) podem ocorrer **atomicamente** numa única transação, usando `WITH ... UPDATE ... INSERT`.

**c)** Escreva uma Query para buscar em todos os históricos dos clientes quais deles **mudaram de Segmento** ("Bronze" para "Prata") em algum momento de sua vida útil, utilizando nossa coluna auditora `properties_diff`.

---

## EXERCÍCIO 4: Backfill como Fast-Forward Incremental

**Cenário:** Você precisa carregar 1 ano de histórico de clientes (2023-01-01 a 2023-12-31).

**a)** Sabendo que a procedure `proc_upsert_clientes_scd2(DATE)` é incremental por natureza (só toca nos deltas), escreva o bloco PL/pgSQL que executa o Backfill como um loop de chamadas incrementais.

**b)** Explique por que essa abordagem ("Fast-Forward") é preferível a uma procedure de backfill separada que tenta processar o ano inteiro de uma vez.

---

## EXERCÍCIO 5: Ingestão Idempotente de Tabela Fato

**a)** Explique por que tabelas Fato Transacionais (`fato_vendas`) utilizam o padrão **Delete-Insert por partição temporal** ao invés de `ON CONFLICT DO UPDATE`, e como isso garante idempotência.

**b)** Escreva a procedure `proc_ingestao_fato_vendas(DATE)` que resolve a **Surrogate Key** correta da dimensão SCD Type 2 usando o filtro Point-In-Time (tratando `data_fim IS NULL` para o registro atual).

**c)** Na Aula 05 aprendemos o padrão **Cumulative Table (Yesterday + Today)** para comprimir a `usuarios_atividade_fato` em arrays. Encapsule esse pipeline numa Procedure `proc_acumular_atividade(p_data_ontem DATE, p_data_hoje DATE)` que funcione tanto para carga incremental quanto para backfill via fast-forward.
