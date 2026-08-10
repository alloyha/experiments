# CDC Stack — Arquitetura e Design

## 🏛️ Visão Geral

```
┌──────────────────────────────────────────────────────────────────────┐
│  PostgreSQL — cdc-postgres (1 container, 3 databases isolados)       │
│                                                                       │
│  cdc_db            cdc_source.customers / .orders / .products       │
│                     wal_level=logical, REPLICA IDENTITY FULL          │
│                     publication dbz_publication FOR ALL TABLES        │
│                                                                       │
│  iceberg_catalog    metastore JDBC do Iceberg (Flink + dbt-duckdb)   │
│  analytics_db       Gold + Diamond (destino final do dbt)            │
└───────────────────────────────┬────────────────────────────────────┘
                                 │ logical replication (pgoutput)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Debezium (Kafka Connect) — connector postgres-cdc-connector         │
│  snapshot.mode=initial · table.include.list=cdc_source.*             │
│  RegexRouter: topic = nome da tabela (customers/orders/products)     │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Kafka — tópicos customers / orders / products (JSON, sem schema)    │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Flink (SQL Gateway + cluster)                                       │
│                                                                       │
│  BRONZE (streaming, roda continuamente, 1 job por entidade):          │
│    cdc_source_customers/_orders/_products (Kafka source, scan.startup │
│    .mode=earliest-offset) → INSERT INTO brz_*_cdc (Iceberg, append)  │
│                                                                       │
│  SILVER (batch, re-executado a cada 60s):                            │
│    brz_*_cdc → ROW_NUMBER() dedup por PK, WHERE operation<>'delete'  │
│    → slv_customers/_orders/_products (Iceberg, "estado atual")       │
└───────────────────────────────┬────────────────────────────────────┘
                                 │ Iceberg REST via catálogo JDBC (Postgres)
                                 │ dados físicos em MinIO (S3)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  dbt-duckdb (target duckdb) — lê Iceberg via plugin pyiceberg,        │
│  materializa em Postgres via `attach` (alias "pg")                    │
│                                                                       │
│  dim_customers_attributes_history / dim_products_history:            │
│    lê brz_*_cdc (histórico COMPLETO, não slv_*) → LAG() detecta cada │
│    transição real de valor → uma linha por versão                    │
│                                                                       │
│  customer_metrics_current: slv_customers + slv_orders → métricas      │
│    derivadas de pedido (estado atual, NÃO versionado)                │
│                                                                       │
│  fct_orders: slv_orders + slv_customers                              │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  dbt-postgres (target postgres) — lê/escreve analytics_db             │
│                                                                       │
│  GOLD (SCD2 final):                                                   │
│    dim_customers / dim_products: LAG()/LEAD() sobre o *_history →     │
│    properties (JSONB) + properties_diff (JSONB) + valid_from/         │
│    valid_to/is_current — rebuild completo a cada ciclo                │
│                                                                       │
│  DIAMOND (semântico, OBT):                                            │
│    obt_customer_orders: fct_orders ⋈ dim_customers (ponto-no-tempo)   │
│                         ⋈ customer_metrics_current (estado atual)     │
│    agg_daily_sales_by_segment                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## 🗄️ Por que 3 bancos Postgres em 1 container

| Banco | Papel | Por quê separado |
|-------|-------|-------------------|
| `cdc_db` | OLTP transacional, fonte do Debezium | Isolado do resto para não misturar dados operacionais com metadados de catálogo ou tabelas analíticas |
| `iceberg_catalog` | Metastore JDBC do Iceberg (`org.apache.iceberg.jdbc.JdbcCatalog`) | Compartilhado por Flink (grava) e pelo plugin `iceberg` do dbt-duckdb (lê) — precisa ser o mesmo catálogo para os dois enxergarem as mesmas tabelas |
| `analytics_db` | Gold + Diamond | Onde o dbt materializa o resultado final; separado do OLTP para não competir por recursos nem misturar schemas de camadas diferentes |

Postgres não permite queries cross-database (`Cross-db references not allowed`), então
cada dbt target aponta para exatamente um banco (ver `dbt/profiles.yml`). Um model que
precisa ser lido tanto do target `duckdb` (onde `database='pg'` é o alias do `attach`)
quanto do target `postgres` (onde `database` deve ser `none`, resolvido implicitamente
pelo `schema`) usa o padrão:

```jinja
{{ config(database=('pg' if target.type == 'duckdb' else none)) }}
```

## 🎯 Modelagem SCD Type 2 (dim_customers / dim_products)

**Chave idempotente:** `id` (= `customer_id`/`product_id` da origem).
**Payload:** `properties` (JSONB, snapshot completo da versão) e `properties_diff`
(JSONB, só as chaves que mudaram vs. a versão imediatamente anterior).

### O problema que essa modelagem resolve

O gerador de dados roda a cada 10s; o ciclo dbt roda a cada 60s. Se o SCD2 fosse
construído a partir de uma leitura periódica do "estado atual" (Silver, que só guarda a
última versão de cada linha), duas mudanças de atributo que acontecessem entre dois
ciclos apareceriam como **uma única** transição — a intermediária seria perdida.

**Solução:** reconstruir o histórico inteiro a partir do log bruto e append-only do
Bronze (`brz_customers_cdc`/`brz_products_cdc`), não do Silver. Isso é zero-perda
independente de quão espaçado (ou frequente) for o refresh:

1. **`dim_customers_attributes_history` / `dim_products_history`** (staging, target
   `duckdb`): para cada `id`, ordena os eventos CDC por `event_timestamp` e usa
   `LAG()` para comparar cada evento com o anterior — só sobrevive uma linha quando
   pelo menos um atributo rastreado realmente mudou (ou é a primeira versão). Isso
   colapsa updates "ruído" (ex: só `updated_at` mudou) em nada, e preserva toda
   transição real, não importa quantas aconteceram no mesmo ciclo de 60s.

2. **`dim_customers` / `dim_products`** (final, target `postgres`): sobre essa lista de
   versões, `LAG(properties)` computa `properties_diff` (comparando chave a chave via
   `jsonb_each`) e `LEAD(changed_at)` computa `valid_to`. **Rebuild completo a cada
   ciclo** (`materialized='table'`) — não há incremental nem `pre_hook`: como a fonte já
   é o histórico completo (não um "estado atual" pesquisado periodicamente),
   recalcular do zero sempre produz a mesma timeline correta, e nunca perde uma
   mudança que aconteceu entre dois refreshes.

### Duas armadilhas reais encontradas (e corrigidas)

- **Primeira versão com `valid_from` = quando o Bronze observou a linha pela primeira
  vez.** Isso não é o início real de validade — é só o instante em que o
  Debezium/snapshot capturou o estado inicial. Fatos genuinamente anteriores a esse
  instante (ex: pedidos de um seed inicial que rodou antes do snapshot CDC) ficavam
  órfãos no join ponto-no-tempo (`customer_name` nulo). Corrigido forçando
  `valid_from = '-infinity'::timestamptz` só na primeira versão de cada `id`:
  ```sql
  CASE WHEN prev_properties IS NULL THEN '-infinity'::timestamptz ELSE changed_at END AS valid_from
  ```

- **Eventos CDC duplicados com o mesmo `event_timestamp`.** Depois de um restart do
  Flink sem checkpoint (ver seção de operação abaixo), o job de streaming Bronze pode
  reprocessar mensagens Kafka já vistas, gravando no Bronze duas linhas idênticas
  (mesmo `event_timestamp`, `ingested_at` diferente). Como não há uma segunda chave de
  desempate, `ROW_NUMBER()`/`LAG()` computados com o mesmo `ORDER BY event_timestamp`
  podem resolver o empate de forma inconsistente entre si, fabricando uma versão SCD2
  de largura zero (`valid_from = valid_to`) e uma chave (`scd_key`) duplicada.
  Corrigido com um `DISTINCT ON (id, event_timestamp)` antes de detectar transições —
  colapsa qualquer réplica exata antes que ela chegue nas window functions.

### Separação identidade × métricas derivadas

`dim_customers.properties` é **só** os atributos de identidade que mudam devagar
(nome/email/telefone/país) — SCD2 é para isso. Métricas derivadas de pedido
(`total_orders`, `customer_lifetime_value`, `customer_segment`, `customer_status`, ...)
mudam a cada novo pedido, o que não é o que SCD2 modela; elas vivem em
`customer_metrics_current` (current-state, sem versionamento). O Diamond
(`obt_customer_orders`) junta os dois: `dim_customers` via join ponto-no-tempo (pelo
`order_date`) para atributos como estavam na hora do pedido, e
`customer_metrics_current` via join simples por `customer_id` para as métricas como
estão *agora*.

```sql
-- ponto-no-tempo (dim_customers, SCD2)
LEFT JOIN dim_customers dc
  ON fo.customer_id = dc.id
  AND fo.order_date >= dc.valid_from
  AND (dc.valid_to IS NULL OR fo.order_date < dc.valid_to)

-- estado atual (customer_metrics_current)
LEFT JOIN customer_metrics_current cm
  ON fo.customer_id = cm.customer_id
```

## 🩺 Healthchecks

Cada container tem um healthcheck adequado às ferramentas realmente disponíveis na
imagem (vários dos padrões "óbvios" não funcionam):

| Container | Teste | Por quê |
|-----------|-------|---------|
| `postgres` | `pg_isready` | padrão |
| `zookeeper` | `echo srvr \| nc -w2 localhost 2181 \| grep 'Zookeeper version'` | o comando `ruok`/`imok` (4-letter word) vem desabilitado por padrão nessa imagem — só `srvr` está na whitelist |
| `kafka` | `kafka-broker-api-versions --bootstrap-server kafka:29092` | pode falhar por timeout (5s) sob carga da JVM mesmo saudável — flakiness conhecida, sem impacto real |
| `debezium` | `curl -f http://localhost:8083/connectors` | padrão |
| `minio` | `curl -f http://localhost:9000/minio/health/live` | padrão |
| `minio-sidecar` | compara `$(date +%s)` gravado no *conteúdo* de `/tmp/healthy` contra o relógio atual | a imagem `minio/mc` não tem `find`/`grep -m`/`stat`/`which` — qualquer teste baseado em mtime de arquivo falha com exit 127 |
| `flink-cluster` | `curl -f http://localhost:8081/` | padrão |
| `flink-taskmanager` | `curl -sf http://flink-cluster:8081/taskmanagers \| grep '"id"'` | confirma que já se registrou no jobmanager, não só que o processo subiu |
| `flink-sql-gateway` | `curl -f http://localhost:8084/v1/info` | padrão |
| `data-generator` | `find /app/logs/heartbeat -mmin -1 \| grep heartbeat` | o script grava um heartbeat a cada ciclo (10s) em vez de depender do processo continuar vivo |
| `dbt-flink-ingestao` / `dbt-duckdb-medallhao` | `scripts/healthcheck-dbt.sh` | idem, adaptado ao ciclo de 60s |

## ⚙️ Operação: armadilhas conhecidas e recuperação

### Estado do Flink não sobrevive a um restart do jobmanager

Diferente do Debezium (cujo estado de replicação fica no slot do Postgres + tópicos de
config do Kafka Connect, ambos persistentes), os **jobs em execução do Flink vivem em
memória do jobmanager**. Um restart do Docker Desktop, da máquina, ou uma recriação do
`flink-cluster` derruba os 3 jobs de streaming Bronze — `curl
http://localhost:8081/jobs/overview` volta `{"jobs":[]}` mesmo com Kafka/Debezium
intactos e produzindo. É preciso resubmeter Bronze manualmente (recriar/reiniciar
`dbt-flink-ingestao`, cujo startup já faz `dbt run -t flink -s bronze`).

### Containers de vida longa após restart do Docker Desktop

Se o daemon do Docker Desktop reinicia (crash, ou `Restart Docker Desktop`) mas um
container **não** é recriado (só o processo dentro dele continua rodando), o resolver
DNS embutido do container (`127.0.0.11`) pode ficar apontando para o estado antigo do
daemon — chamadas para outro container pelo nome (`flink-sql-gateway`, por exemplo)
passam a falhar com `Name or service not known` mesmo que ambos os containers estejam
saudáveis e na mesma network. Sintoma: `docker inspect` mostra os dois containers com o
mesmo `NetworkID`/subnet, mas a resolução de nome falha mesmo assim. Corrigido
recriando o container afetado (`docker restart <container>` já é suficiente — força
uma nova configuração de rede).

### Réplica de eventos CDC após resubmissão do Flink sem checkpoint

Ver a seção "Duas armadilhas reais encontradas" acima — resubmeter o job de streaming
Bronze sem um checkpoint salvo pode reprocessar mensagens Kafka já consumidas
(`scan.startup.mode = earliest-offset`), duplicando linhas no Bronze. Isso é esperado
nesse cenário de dev local (sem Hadoop/checkpoint storage persistente configurado); os
models de reconstrução de histórico já absorvem isso via `DISTINCT ON`.

## 📊 Características por Camada

| Camada | Tecnologia | Função | Persistência |
|--------|-----------|--------|---------------|
| Origem | PostgreSQL (`cdc_db`) | Dados operacionais | Volume Docker |
| Captura | Debezium | Extrair mudanças (WAL lógico) | Kafka Connect offsets (Kafka) |
| Stream | Kafka | Fila distribuída | Tópicos, retenção padrão |
| Bronze/Silver | Flink + Iceberg | Streaming append / batch dedup | Iceberg (MinIO) + catálogo JDBC (`iceberg_catalog`) |
| Gold | dbt-duckdb → Postgres | SCD2 + métricas + fatos | `analytics_db` |
| Diamond | dbt-postgres | Semântico / OBT | `analytics_db` |

## 🚀 Escalabilidade (stack local)

- **Kafka**: aumentar partições por tópico (hoje 1, compatível com paralelismo 1 do
  consumo Flink por entidade)
- **Flink**: `taskmanager.numberOfTaskSlots` em `docker-compose.yml`
- **DuckDB**: `threads: 1` no target `duckdb` é proposital — múltiplos models
  concorrentes tocando a mesma fonte Iceberg via plugin causam
  `Catalog write-write conflict on create`; não aumentar sem resolver isso primeiro

## 🔍 Monitoramento

- Flink Dashboard: `http://localhost:8081` (jobs de streaming Bronze devem aparecer
  como `RUNNING` permanentemente; jobs de batch Silver aparecem e terminam a cada
  ciclo)
- Status do connector: `curl http://localhost:8083/connectors/postgres-cdc-connector/status`
- MinIO Console: `http://localhost:9001`
- Logs: `docker compose logs -f <serviço>`

---

Para o passo a passo de subir a stack e comandos do dia a dia, ver [README.md](README.md).
