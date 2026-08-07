# CDC Stack - Arquitetura e Design

## 🏛️ Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                       ORIGEM DE DADOS                        │
│                      PostgreSQL (CDC)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ cdc_source.customers                                │   │
│  │ cdc_source.orders                                   │   │
│  │ (WAL - Write-Ahead Logs com replicação lógica)     │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ↓
        ┌────────────────────────────────────────┐
        │    DEBEZIUM CONNECTOR (CDC)            │
        │  ┌────────────────────────────────────┤
        │  │ Mode: Logical Decoding (pgoutput)  │
        │  │ Plugin: pgoutput                   │
        │  │ Snapshot: initial                  │
        │  │ Key: id (primary key)              │
        │  │ Value: JSON com valores antigos    │
        │  └────────────────────────────────────┤
        │                                        │
        │  Captura:                              │
        │  - INSERT (op: c)                      │
        │  - UPDATE (op: u)                      │
        │  - DELETE (op: d)                      │
        │  - TRUNCATE (op: t)                    │
        └────────────────────────────────────────┘
                             │
                             ↓
        ┌──────────────────────────────────────────┐
        │   APACHE KAFKA (Message Streaming)       │
        │  ┌──────────────────────────────────────┤
        │  │ Broker: kafka:29092                 │
        │  │ Replication Factor: 1               │
        │  │ Partitions: 1 (por tabela)          │
        │  │ Retention: Indefinido               │
        │  │                                     │
        │  │ Topics:                             │
        │  │ - customers (CDC events)            │
        │  │ - orders (CDC events)               │
        │  │                                     │
        │  │ Format: JSON (chave/valor)          │
        │  └──────────────────────────────────────┤
        └──────────────────────────────────────────┘
                             │
                             ↓
        ┌──────────────────────────────────────────┐
        │   DBT-FLINK (Transformação Streaming)    │
        │  ┌──────────────────────────────────────┤
        │  │ Consumer Group: dbt-flink-consumer  │
        │  │ Parallelism: 4                      │
        │  │ Backend: RocksDB                    │
        │  │                                     │
        │  │ Models:                             │
        │  │ - stg_customers (staging layer)    │
        │  │ - mart_customer_orders (marts)     │
        │  │                                     │
        │  │ Processamento:                      │
        │  │ - Limpeza de dados deletados      │
        │  │ - Join de múltiplas tabelas       │
        │  │ - Agregações em tempo real        │
        │  └──────────────────────────────────────┤
        └──────────────────────────────────────────┘
                             │
                             ↓
        ┌──────────────────────────────────────────┐
        │   APACHE ICEBERG (Data Lake Format)      │
        │  ┌──────────────────────────────────────┤
        │  │ Warehouse: s3://iceberg-warehouse  │
        │  │ Backend: MinIO (S3 local)           │
        │  │                                     │
        │  │ Estrutura:                          │
        │  │ /staging/                           │
        │  │   - stg_customers                   │
        │  │   - stg_orders                      │
        │  │                                     │
        │  │ /marts/                             │
        │  │   - mart_customer_orders            │
        │  │   - (mais modelos)                  │
        │  │                                     │
        │  │ Features:                           │
        │  │ - ACID transactions                 │
        │  │ - Time travel                       │
        │  │ - Schema evolution                  │
        │  │ - Data locality                     │
        │  └──────────────────────────────────────┤
        └──────────────────────────────────────────┘
                             │
                             ↓
        ┌──────────────────────────────────────────┐
        │   MinIO (S3-compatible Object Store)     │
        │  ┌──────────────────────────────────────┤
        │  │ Endpoint: http://minio:9000         │
        │  │ Console: http://minio:9001          │
        │  │ Root: minioadmin/minioadmin         │
        │  │                                     │
        │  │ Buckets:                            │
        │  │ - iceberg-warehouse (data)          │
        │  │ - iceberg-metadata (manifest)       │
        │  │ - kafka-data (raw Kafka events)     │
        │  └──────────────────────────────────────┤
        └──────────────────────────────────────────┘
```

## 🔄 Fluxo de Dados Detalhado

### Fase 1: Captura (PostgreSQL → Kafka)

1. **Inserção de dados** no PostgreSQL
   ```sql
   INSERT INTO cdc_source.customers (name, email) 
   VALUES ('Alice', 'alice@example.com');
   ```

2. **PostgreSQL WAL** registra a mudança
   - Modo replicação lógica ativado
   - Dados armazenados em `/var/lib/postgresql/data/pg_wal/`

3. **Debezium Connector** lê WAL via `pgoutput`
   ```json
   {
     "before": null,
     "after": {
       "id": 1,
       "name": "Alice",
       "email": "alice@example.com",
       "country": "Brazil",
       "created_at": "2024-01-15T10:30:00Z"
     },
     "op": "c",
     "ts": 1705318200000
   }
   ```

4. **Kafka** recebe a mensagem
   - Tópico: `customers`
   - Partição: 0
   - Offset: incrementa a cada evento

### Fase 2: Transformação (Kafka → Iceberg)

1. **dbt-Flink Job** consome tópico Kafka
   ```sql
   SELECT * FROM kafka_topic_customers
   WHERE __deleted = false
   ```

2. **Aplicar modelo dbt**
   ```sql
   -- stg_customers filtra dados deletados
   SELECT id, name, email, country, created_at
   FROM raw.customers
   WHERE __deleted = false
   ```

3. **Computar marts**
   ```sql
   -- mart_customer_orders agrega dados
   SELECT 
     customer_id,
     COUNT(*) as total_orders,
     SUM(total_amount) as total_spent
   FROM stg_customers c
   LEFT JOIN raw.orders o ON c.id = o.customer_id
   GROUP BY customer_id
   ```

4. **Escrever em Iceberg**
   - Formato: Apache Iceberg (ORC/Parquet)
   - Localização: `s3://iceberg-warehouse/marts/mart_customer_orders/`
   - Snapshot ID: incrementado a cada write

### Fase 3: Armazenamento (Iceberg → MinIO)

1. **Iceberg metadata**
   ```
   s3://iceberg-warehouse/
   ├── staging/
   │   └── stg_customers/
   │       ├── metadata/
   │       │   ├── v1.metadata.json
   │       │   └── snap-123.manifest.avro
   │       └── data/
   │           └── 00000-1-hash.parquet
   └── marts/
       └── mart_customer_orders/
           └── (similar)
   ```

2. **MinIO armazena** os arquivos
   - Backend: disco local (`/data`)
   - Acesso: S3-compatible API

## 🎯 Padrões de Design

### 1. **Medallion Architecture** (Bronze → Silver → Gold)

```
Bronze (Raw):
  - Dados brutos do Kafka/Debezium
  - Sem transformação
  - Pasta: /raw/

Silver (Staging):
  - Limpeza e enriquecimento
  - Modelos: stg_*
  - Pasta: /staging/

Gold (Marts):
  - Agregações finais para BI
  - Modelos: mart_*
  - Pasta: /marts/
```

### 2. **Tratamento de Deletes** (Soft Deletes)

Debezium marca deletes com flag `__deleted = true`:

```sql
-- Na ingestão
INSERT INTO raw.customers VALUES (1, 'Alice', ..., __deleted=false)

-- Após DELETE
UPDATE raw.customers SET __deleted=true WHERE id=1

-- Nos models staging
WHERE __deleted = false  -- Filtra deletes
```

### 3. **Idempotência** (Re-executar sem duplicatas)

- **Primary Key**: garantido em Iceberg ACID
- **Particionamento**: por data (`created_at`)
- **Upsert**: Flink garante exatamente-uma-vez

### 4. **Late-Arriving Data**

Flink aguarda dados atrasados por default:

```yaml
# Configuração de time windows
watermark_delay: 10min
```

## 📊 Características por Camada

| Camada | Tecnologia | Função | Durabilidade |
|--------|-----------|--------|-------------|
| **Origem** | PostgreSQL | Dados operacionais | Persistente |
| **Captura** | Debezium | Extrair mudanças | Logs WAL |
| **Stream** | Kafka | Fila distribuída | Topic Retention |
| **Processamento** | Flink | Transform/Aggregate | State Checkpoints |
| **Armazenamento** | Iceberg | Data lake ACID | S3-compatible |
| **Objeto** | MinIO | Object storage | Disco local |

## 🔐 Segurança e Confiabilidade

### Consistência End-to-End

1. **PostgreSQL**: ACID transactions
2. **Kafka**: Exactly-once semantics
3. **Flink**: Exactly-once with checkpointing
4. **Iceberg**: ACID transactions com snapshots

### Recuperação e Rollback

```bash
# Ver histórico de snapshots
SELECT * FROM iceberg_snapshots;

# Voltar para snapshot anterior
SELECT * FROM table AT VERSION AS OF 123;

# Restaurar tudo
docker-compose down -v
docker-compose up -d
```

## 🚀 Escalabilidade

### Horizontal Scaling

- **Kafka**: aumentar partições (compatível com Flink parallelism)
- **Flink**: aumentar `taskmanager.numberOfTaskSlots`
- **Iceberg**: particionamento automático

### Limites Locais

- **RAM**: 8GB recomendado (ajustável em docker-compose.yml)
- **Disco**: 50GB+ para volumes docker
- **CPU**: 4 cores (Flink parallelism)

## 🔍 Monitoramento

### Métricas Importantes

- **Lag do Debezium**: offset Kafka vs Postgres WAL
- **Lag do Flink**: offset consumido vs produzido
- **Data Freshness**: timestamp última transformação
- **Tamanho Iceberg**: bytes por snapshot

### Dashboards

- Flink UI: `http://localhost:8081`
- MinIO Console: `http://localhost:9001`
- PostgreSQL Logs: `docker-compose logs postgres`

---

**Última atualização:** 2024

Para mais detalhes, consulte a documentação oficial de cada ferramenta.
