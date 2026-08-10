# CDC Local Stack — Postgres → Debezium → Kafka → Flink → Iceberg → dbt → Postgres

Pipeline de **Change Data Capture (CDC)** local, ponta a ponta, seguindo a arquitetura
medalhão (Bronze/Silver/Gold/Diamond). Todo o stack roda em Docker Compose; nenhuma
dependência precisa ser instalada no host.

## 🏗️ Arquitetura (visão geral)

```
PostgreSQL (cdc_db, OLTP)
    │  (Debezium — logical replication / pgoutput)
    ▼
Kafka (tópicos customers/orders/products)
    │  (Flink SQL — streaming)
    ▼
Iceberg Bronze  (brz_*_cdc — log bruto, append-only, 1 job de streaming por entidade)
    │  (Flink SQL — batch a cada 60s, dedup por PK)
    ▼
Iceberg Silver  (slv_* — "estado atual" deduplicado)
    │  (dbt-duckdb lê Iceberg via plugin nativo, materializa em Postgres)
    ▼
Postgres Gold (analytics_db)
    ├─ dim_customers / dim_products  → SCD2, reconstruído do histórico COMPLETO do Bronze
    ├─ customer_metrics_current      → métricas derivadas de pedidos, estado atual (não SCD2)
    └─ fct_orders
    │  (dbt-postgres — joins ponto-no-tempo)
    ▼
Postgres Diamond (analytics_db)  → obt_customer_orders, agg_daily_sales_by_segment
```

Iceberg usa um **catálogo JDBC** (metastore em Postgres, banco `iceberg_catalog`) com os
dados físicos em MinIO (S3-compatível) — sem Hadoop/Hive metastore.

Veja [ARCHITECTURE.md](ARCHITECTURE.md) para o design detalhado (por que 3 bancos
Postgres, como o SCD2 garante zero perda de dados, e as armadilhas operacionais já
mapeadas).

## 📋 Pré-requisitos

- Docker & Docker Compose (v3.8+)
- 8GB+ RAM disponível para os containers

## 🚀 Quick Start

```bash
cd postgres-debezium-kafka-flink-iceberg

# Sobe toda a stack (healthchecks garantem a ordem de dependência correta)
docker compose up -d

# Acompanhe até todos ficarem "healthy"
watch docker compose ps
```

Depois que `postgres`, `kafka` e `debezium` estiverem saudáveis, registre o connector:

```bash
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d @config/debezium/postgres-source.json

curl http://localhost:8083/connectors/postgres-cdc-connector/status
```

A partir daí, dois containers rodam em loop **interno** (não dependem de
`restart: always` — o loop é um `while true` dentro do próprio script) e mantêm o
pipeline vivo sozinhos:

- **`data-generator`**: a cada `DATA_GENERATOR_INTERVAL` (10s) insere/atualiza clientes,
  produtos e pedidos direto no Postgres OLTP.
- **`dbt-flink-ingestao`**: submete os 3 jobs de streaming Bronze uma vez (rodam
  continuamente no Flink) e, a cada `DBT_RUN_INTERVAL` (60s), roda o batch Silver.
- **`dbt-duckdb-medallhao`**: a cada 60s reconstrói o histórico SCD2 a partir do Bronze,
  recalcula `customer_metrics_current`/`fct_orders`, e materializa Gold + Diamond em
  Postgres.

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| **PostgreSQL** | `localhost:5432` | `postgres` / `postgres` |
| **Kafka** | `localhost:9092` | - |
| **Debezium (Kafka Connect REST)** | `http://localhost:8083` | - |
| **MinIO Console** | `http://localhost:9001` | `minioadmin` / `minioadmin` |
| **Flink Dashboard** | `http://localhost:8081` | - |
| **Flink SQL Gateway** | `http://localhost:8084` | - |

## 🗄️ Bancos de dados

Um único container Postgres (`cdc-postgres`) hospeda 3 bancos isolados
(criados em `scripts/init-postgres.sql`):

| Banco | Uso |
|-------|-----|
| `cdc_db` | OLTP transacional (`cdc_source.customers/orders/products`) — fonte do Debezium |
| `iceberg_catalog` | Metastore JDBC do Iceberg (usado por Flink e pelo plugin `iceberg` do dbt-duckdb) |
| `analytics_db` | Gold + Diamond (dbt materializa aqui) |

Postgres não permite queries cross-database — cada dbt target (`flink`, `duckdb`,
`postgres`) aponta para o banco certo; ver `dbt/profiles.yml`.

## 🔄 Fluxo de CDC

### 1. Dados chegam no PostgreSQL

```bash
docker exec -it cdc-postgres psql -U postgres -d cdc_db -c \
  "INSERT INTO cdc_source.customers (name, email, phone, country) VALUES ('Alice', 'alice@example.com', '+1000000000', 'Brazil');"
```

### 2. Debezium captura e publica no Kafka

Publication `FOR ALL TABLES` (`puballtables: true`) — qualquer tabela nova em
`cdc_source` é capturada automaticamente, sem reiniciar o connector.

```bash
docker exec cdc-kafka kafka-topics --list --bootstrap-server kafka:29092

docker exec cdc-kafka kafka-console-consumer \
  --bootstrap-server kafka:29092 --topic customers --from-beginning
```

### 3. Flink consome e escreve Bronze/Silver em Iceberg

```bash
# Ver os jobs de streaming Bronze rodando
curl -s http://localhost:8081/jobs/overview

# Rodar manualmente um ciclo (normalmente automático via loop interno)
docker exec cdc-dbt-flink-ingestao sh -c "cd /dbt && dbt run -t flink -s silver"
```

### 4. Gold/Diamond materializados em Postgres

```bash
docker exec cdc-dbt-duckdb-medallhao sh -c \
  "cd /app/dbt && dbt run -t duckdb -s dim_customers_attributes_history dim_products_history customer_metrics_current fct_orders"

docker exec cdc-dbt-duckdb-medallhao sh -c \
  "cd /app/dbt && dbt run -t postgres -s dim_customers dim_products diamond"
```

```bash
docker exec cdc-postgres psql -U postgres -d analytics_db -c \
  "SELECT * FROM diamond.obt_customer_orders LIMIT 10;"
```

## 📁 Estrutura de Diretórios

```
postgres-debezium-kafka-flink-iceberg/
├── docker-compose.yml            # Orquestrador central (healthchecks em todos os serviços)
├── scripts/
│   ├── init-postgres.sql         # Cria os 3 bancos + tabelas cdc_source + seed
│   ├── entrypoint-data-generator.sh  # Loop interno do data-generator
│   ├── data_generator_once.py    # Um ciclo: insert/update customer/order/product
│   └── healthcheck-dbt.sh        # Healthcheck compartilhado pelos containers dbt
├── config/
│   ├── debezium/postgres-source.json   # Config do connector (publication FOR ALL TABLES)
│   ├── postgresql.conf
│   └── dbt/profiles.yml          # Cópia bind-mounted (mantida em sync com dbt/profiles.yml)
├── docker/
│   ├── Dockerfile.flink          # Imagem Flink (cluster/taskmanager/sql-gateway)
│   ├── Dockerfile.dbt-flink      # dbt-flink-ingestao (Bronze streaming + Silver batch)
│   ├── Dockerfile.dbt-duckdb     # dbt-duckdb-medallhao (Gold histórico + SCD2 + Diamond)
│   └── Dockerfile.data-generator
└── dbt/
    ├── dbt_project.yml
    ├── profiles.yml               # targets: flink / duckdb / postgres
    ├── macros/macros.sql          # generate_schema_name, SCD2 properties, Iceberg connector, hooks
    └── models/
        ├── _sources.yml           # cdc_source (Postgres), kafka, iceberg_bronze, iceberg_silver
        ├── bronze/                # brz_customers_cdc, brz_orders_cdc, brz_products_cdc (streaming)
        ├── silver/                # slv_customers, slv_orders, slv_products (batch, dedup)
        ├── gold/
        │   ├── dim_customers_attributes_history.sql  # staging: toda versão real, do Bronze
        │   ├── dim_products_history.sql
        │   ├── dim_customers.sql / dim_products.sql  # SCD2 final (properties/properties_diff)
        │   ├── customer_metrics_current.sql          # métricas de pedido, estado atual
        │   └── fct_orders.sql
        └── diamond/
            ├── obt_customer_orders.sql
            └── agg_daily_sales_by_segment.sql
```

## 🔧 Operações Comuns

### Parar / reiniciar a stack

```bash
docker compose down          # mantém os volumes (dados persistem)
docker compose down -v       # reset completo (apaga tudo)
docker compose up -d
```

### Ver logs de um serviço

```bash
docker compose logs -f postgres
docker compose logs -f kafka
docker compose logs -f debezium
docker compose logs -f dbt-flink-ingestao
docker compose logs -f dbt-duckdb-medallhao
docker compose logs -f data-generator
```

### Conectar ao PostgreSQL (por banco)

```bash
docker exec -it cdc-postgres psql -U postgres -d cdc_db          # OLTP
docker exec -it cdc-postgres psql -U postgres -d iceberg_catalog # metastore Iceberg
docker exec -it cdc-postgres psql -U postgres -d analytics_db    # Gold/Diamond
```

### Rodar dbt manualmente

```bash
# Bronze (streaming, roda uma vez — os jobs ficam vivos no Flink)
docker exec cdc-dbt-flink-ingestao sh -c "cd /dbt && dbt run -t flink -s bronze"

# Silver (batch)
docker exec cdc-dbt-flink-ingestao sh -c "cd /dbt && dbt run -t flink -s silver"

# Gold: histórico + métricas + fatos (lê Iceberg via DuckDB)
docker exec cdc-dbt-duckdb-medallhao sh -c \
  "cd /app/dbt && dbt run -t duckdb -s dim_customers_attributes_history dim_products_history customer_metrics_current fct_orders"

# Gold: SCD2 final + Diamond (lê/escreve Postgres)
docker exec cdc-dbt-duckdb-medallhao sh -c \
  "cd /app/dbt && dbt run -t postgres -s dim_customers dim_products diamond"

docker exec cdc-dbt-duckdb-medallhao sh -c "cd /app/dbt && dbt test -t postgres -s gold diamond"
```

## 🐛 Troubleshooting

### Debezium não conecta ao PostgreSQL

```bash
docker compose logs debezium
docker exec cdc-postgres psql -U postgres -c "SHOW wal_level;"   # deve ser 'logical'
```

### Kafka "unhealthy" mesmo respondendo

O healthcheck (`kafka-broker-api-versions`) ocasionalmente estoura o timeout de 5s sob
carga da JVM mesmo com o broker saudável. Se `docker compose up` travar esperando Kafka,
suba o container diretamente (contorna o gate do `depends_on`):

```bash
docker start cdc-kafka
```

### Depois de reiniciar o Docker Desktop / a máquina

O estado do Flink jobmanager **não sobrevive** a um restart — os 3 jobs de streaming
Bronze somem (`curl http://localhost:8081/jobs/overview` retorna `{"jobs":[]}`) mesmo
com o Debezium/Kafka intactos. Resubmeta-os recriando o `dbt-flink-ingestao` (ele
resubmete Bronze automaticamente no start):

```bash
docker compose up -d postgres minio zookeeper
docker compose up -d kafka
docker compose up -d debezium flink-cluster flink-taskmanager
docker compose up -d flink-sql-gateway
docker restart cdc-dbt-flink-ingestao   # recria também a resolução DNS, que pode
                                          # ficar "presa" no daemon antigo em containers
                                          # de vida longa
```

Se o Flink foi resubmetido sem checkpoint, ele pode reprocessar mensagens Kafka já
vistas, criando eventos Bronze duplicados com o mesmo `event_timestamp`. Os modelos
`dim_customers_attributes_history`/`dim_products_history` já fazem
`DISTINCT ON (id, event_timestamp)` antes de reconstruir o histórico, então isso não
gera versões SCD2 espúrias — mas vale saber que a duplicação em Bronze é esperada
nesse cenário.

### "Cross-db references not allowed in postgres (pg vs analytics_db)"

Algum model do target `duckdb` está com `database='pg'` fixo (em vez de
`database=('pg' if target.type == 'duckdb' else none)`) e está sendo referenciado
via `ref()` a partir do target `postgres`. Ver `dbt/models/gold/customer_metrics_current.sql`
para o padrão correto.

### MinIO buckets não criados

```bash
docker exec cdc-minio-sidecar mc alias set local http://minio:9000 minioadmin minioadmin
docker exec cdc-minio-sidecar mc mb --ignore-existing local/iceberg-warehouse
```

## 📚 Recursos Adicionais

- [Debezium Docs](https://debezium.io/)
- [dbt-flink (PyPI)](https://pypi.org/project/dbt-flink/)
- [dbt-duckdb (Iceberg plugin)](https://github.com/duckdb/dbt-duckdb)
- [Apache Iceberg](https://iceberg.apache.org/)
- [Apache Kafka](https://kafka.apache.org/)
- [Apache Flink](https://flink.apache.org/)

## 📝 Próximos Passos

1. Adicionar mais tabelas fonte: criar em `cdc_source` (capturada automaticamente
   pela publication `FOR ALL TABLES`) e replicar o padrão bronze/silver/gold.
2. Conectar ferramentas de BI a `analytics_db` (schemas `gold`/`diamond`).
3. Ajustar paralelismo do Flink (`taskmanager.numberOfTaskSlots` em `docker-compose.yml`).

## 📄 Licença

Open source - use livremente.
