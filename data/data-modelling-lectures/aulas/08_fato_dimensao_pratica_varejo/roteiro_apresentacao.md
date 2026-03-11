# Roteiro de Apresentação (20 Minutos)
**Tema:** Fato e Dimensão na Prática (Caso Varejo Pipeline Completo)

> **Nota ao Instrutor:** 20 minutos é um tempo extremamente enxuto para a densidade deste material. O foco deve ser na **"Mecânica do Fluxo"** e no **"Por que"** (Arquitetura) em vez de ler cada linha de sintaxe SQL. Deixe o código falar através das execuções.

---

## ⏱️ Bloco 1: O "Big Picture" e o Grafo Medallion (00:00 - 03:00 = 3 min)
**Objetivo:** Mostrar de onde viemos, para onde vamos, e como isso se encaixa no mundo real.

- **[1 min] O Desafio:** "Até agora vimos a teoria. Hoje vamos responder: *como um engenheiro de dados atualiza o Data Warehouse de ponta a ponta todos os dias sem quebrar os dados?*"
- **[2 min] O Grafo (Mostrar o `apresentacao.md`):**
  - Exiba o diagrama Mermaid no Markdown.
  - Explique a **Medallion Architecture**: "Deixamos a origem limpa (Bronze), conformamos regras e mantemos a história (Silver) e criamos as tabelas para os Dashboards (Gold)".
  - Destaque o sentido das setas (Dependências DAG): "A Fato não pode rodar antes da Dimensão. A Gold não roda antes da Fato".

---

## ⏱️ Bloco 2: A Camada Silver - Dimensões as-was vs as-is (03:00 - 08:00 = 5 min)
**Objetivo:** Demonstrar visualmente a diferença brutal entre sobrescrever e guardar o histórico.

- **[2 min] SCD Type 1 vs Type 3:**
  - Mostre rapidamente a `proc_upsert_clientes_scd1` (Type 1): "Sobrescreve, perde a história, útil para logística atual".
  - Mostre a `proc_upsert_produtos_scd3` (Type 3): "Guarda a coluna `anterior`. Simples, mas limitado a apenas 1 salto no passado".
- **[3 min] O Padrão Ouro e Reconstrução - SCD Type 2:**
  - Mostre a `dim_cliente_type2` no `apresentacao.sql`.
  - Explique os elementos essenciais: "A `Surrogate Key` (cliente_sk), a janela de tempo (`data_inicio` / `data_fim`) e a nossa inovação geométrica: o `JSONB Diff` que audita o que mudou na hora".
  - *Dica:* Não explique o código do JSONB função a função, foque no conceito mercadológico — "Saber *quando* ele mudou e *para o que* ele mudou".
  - **Bridge para Gap and Island:** Imediatamente após, mostre a tabela `cliente_snapshot_diario` e a procedure `proc_snapshot_clientes`. Explique: "Antes de rodar o SCD2, tiramos uma 'foto' do OLTP todo dia. Com semanas de fotos acumuladas, a view `view_reconstrucao_scd2` usa LAG e SUM cumulativo para reconstruir o histórico Type 2 do zero — sem depender da `dim_cliente_type2`. É assim que se audita ou reconstrói um SCD2 a partir de um log CDC real".

---

## ⏱️ Bloco 3: A Camada Silver - Tabelas Fato & Point-in-Time (08:00 - 12:00 = 4 min)
**Objetivo:** Provar que Idempotência garante noites de sono tranquilas.

- **[2 min] Idempotência (`DELETE + INSERT`):**
  - Mostre a `proc_ingestao_fato_vendas`.
  - Pergunte/Provoque: "Se o pipeline quebrar e eu rodar duas vezes o mesmo dia, o que acontece? A primeira linha da procedure é um `DELETE WHERE data = hoje`. Isso chama-se Idempotência".
- **[2 min] A Mágica do Point-In-Time:**
  - Mostre o `JOIN` na fato com a dimensão Type 2.
  - Explique a condição de contorno: `AND (c.data_fim >= src.data_venda OR c.data_fim IS NULL)`. 
  - "Nós amarramos a venda não à ID natural do cliente, mas ali onde ele era na exata foto daquele dia. Isso protege o Dashboard Financeiro (as-was)".

---

## ⏱️ Bloco 4: A Engenhosidade Analítica (12:00 - 16:00 = 4 min)
**Objetivo:** Mostrar por que modelagem comportamental vai além do Star Schema tradicional.

- **[2 min] Tabelas Acumuladas & Arrays Posicionais:**
  - Mostre a tabela `cliente_vendas_array_mensal`.
  - "Ao invés de processar 365 linhas por usuário com JOINs diários, empacotamos o mês num Array de 30 posições O(1). O índice do array *é* o dia do mês."
- **[2 min] Datint / Retenção Bitwise (O Show-Off):**
  - Explique o conceito por alto: "A 'atividade' do cliente é derivada diretamente das vendas (fato_vendas). Se o cliente comprou, liga o bit '1'. Dia sem compra = '0'. Guardamos 32 dias em 32-bits dentro de um `Integer`".
  - Mostre o bitmap de um cliente: "Olhem os gaps — nem todo cliente compra todos os dias. Isso gera padrões reais de retenção, não máscaras artificiais".
  - Mostre o `mart_retencao_d7` (WHERE bitmask & 1 << 7).
  - "Essa álgebra binária calcula retenção de milhões de usuários na margem de nanosegundos com zero I/O."

---

## ⏱️ Bloco 5: A Execução Prática e o Profiling Final (16:00 - 20:00 = 4 min)
**Objetivo:** Provar que o barco flutua, finalizando forte.

- **[2 min] O Fast-Forward (Orquestração):**
  - Vá para a seção de Orquestração do SQL.
  - Selecione o bloco inteiro e dê **Run**.
  - Aponte os **"RAISE NOTICE"** aparecendo no Console: "Estamos rodando o DAG. Dia 1 entra. Dia 2 o João muda de UF. Depois, simulamos 60 dias de backfill contínuo em massa — e olha como os índices B-Tree e comitações fracionadas seguram milhares de sub-processamentos iterativos de forma estável. Em um ambiente local simulado, processar os ~60 dias leva em torno de 19 minutos, com acompanhamento de telemetria para cada dia".
- **[2 min] Auditoria e Duração (Encerramento):**
  - Mostre o resultado da execução. Olhe o output final do Profiling (Final Duration Report).
  - Abra o resultado da query da Fato mostrando João amarrado a `SP` na primeira compra e a `PR` na segunda.
  - **Mensagem Final:** "Modelagem Dimensional não é teoria acadêmica. É o esqueleto de como os dados interagem, mantendo a história e escalando no armazenamento."

---
### 💡 Dicas Táticas de Velocidade:
1. Deixe o arquivo SQL já aberto e as queries pré-selecionadas no seu IDE.
2. Evite a tentação de se perder lendo o código de CTEs complexas (como a construção do JSON). Foque nos comentários acima da procedure.
3. Foque o impacto visual nos outputs tabulares (exibir a J-Curve do Bitwise e o Histórico Type 2).
