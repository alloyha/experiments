# CDC Local Stack - PostgreSQL → Debezium → Kafka → dbt-Flink → Iceberg

Um projeto completo de **Change Data Capture (CDC)** local usando PostgreSQL, Debezium, Kafka, dbt-Flink e Iceberg.

## 🏗️ Arquitetura

```
PostgreSQL (CDC Source)
    ↓ (Debezium Connector)
Kafka (Event Streaming)
    ↓ (dbt-Flink Consumer)
Iceberg (Data Lake)
    └─ MinIO (S3 Storage)
```

## 📋 Pré-requisitos

- Docker & Docker Compose (v3.8+)
- Bash shell (ou similar para executar scripts)
- 8GB+ RAM disponível

## 🚀 Quick Start

### 1. Iniciar a Stack

```bash
cd postgres-debezium-kafka-flink-iceberg
chmod +x scripts/*.sh
./scripts/start-all.sh
```

### 2. Verificar Saúde

```bash
./scripts/health-check.sh
```

### 3. Acessar Serviços

| Serviço | URL | Credenciais |
|---------|-----|-----------|
| **PostgreSQL** | `localhost:5432` | `postgres` / `postgres` |
| **Kafka** | `localhost:9092` | - |
| **Debezium** | `http://localhost:8083` | - |
| **MinIO** | `http://localhost:9001` | `minioadmin` / `minioadmin` |
| **Flink** | `http://localhost:8081` | - |

## 📊 Configuração do Debezium

### 1. Criar Publication e Slot (automático com init-postgres.sql)

PostgreSQL já vem configurado com:
- Replicação lógica ativada (`wal_level = logical`)
- Tabelas de teste: `cdc_source.customers` e `cdc_source.orders`
- REPLICA IDENTITY configurada

### 2. Registrar Connector

```bash
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d @config/debezium/postgres-source.json
```

### 3. Verificar Connector

```bash
curl http://localhost:8083/connectors/postgres-cdc-connector/status
```

## 🔄 Fluxo de CDC

### 1. Dados chegam no PostgreSQL
```sql
-- Conectar ao Postgres
psql -h localhost -U postgres -d cdc_db

-- Inserir dados de teste
INSERT INTO cdc_source.customers (name, email) VALUES ('Alice', 'alice@example.com');

-- Ver registros
SELECT * FROM cdc_source.customers;
```

### 2. Debezium captura mudanças
- Mensagens aparecem em tópicos Kafka: `customers` e `orders`
- Cada operação (INSERT, UPDATE, DELETE) gera um evento

```bash
# Ver tópicos Kafka
docker exec cdc-kafka kafka-topics --list --bootstrap-server kafka:29092

# Consumir tópicos
docker exec cdc-kafka kafka-console-consumer \
  --bootstrap-server kafka:29092 \
  --topic customers \
  --from-beginning
```

### 3. dbt-Flink processa e escreve em Iceberg
```bash
# Executar models dbt-flink
docker exec cdc-dbt-flink dbt run

# Ver status
docker exec cdc-dbt-flink dbt docs generate
```

### 4. Consultar dados em Iceberg (via MinIO)
```bash
# Acessar MinIO Console
# http://localhost:9001
# Buckets: iceberg-warehouse, kafka-data
```

## 📁 Estrutura de Diretórios

```
postgres-debezium-kafka-flink-iceberg/
├── docker-compose.yml           # Orquestrador central
├── scripts/
│   ├── start-all.sh            # Subir stack completa
│   ├── stop-all.sh             # Desligar stack
│   ├── health-check.sh         # Verificar saúde dos serviços
│   └── init-postgres.sql       # Inicialização do PostgreSQL
├── config/
│   ├── debezium/
│   │   └── postgres-source.json # Connector config
│   ├── minio/
│   │   └── init-buckets.sh     # Criação de buckets
│   └── dbt-flink/
│       └── (configs adicionais)
├── dbt/
│   ├── dbt_project.yml         # Config do projeto
│   ├── profiles.yml            # Conexão Flink
│   └── models/
│       ├── staging/            # Raw data
│       ├── marts/              # Dados transformados
│       └── schema.yml          # Definição de sources
└── README.md
```

## 🔧 Operações Comuns

### Parar a Stack

```bash
./scripts/stop-all.sh
```

### Remover volumes (reset completo)

```bash
docker-compose down -v
```

### Ver logs de um serviço

```bash
docker-compose logs -f postgres        # PostgreSQL
docker-compose logs -f kafka           # Kafka
docker-compose logs -f debezium        # Debezium
docker-compose logs -f dbt-flink       # Flink
```

### Conectar ao PostgreSQL

```bash
docker exec -it cdc-postgres psql -U postgres -d cdc_db
```

### Executar dbt commands

```bash
docker exec cdc-dbt-flink dbt run
docker exec cdc-dbt-flink dbt test
docker exec cdc-dbt-flink dbt docs generate
```

### Acessar Flink Shell

```bash
docker exec -it cdc-dbt-flink /opt/flink/bin/flink
```

## 🐛 Troubleshooting

### Debezium não conecta ao PostgreSQL
```bash
# Verificar se Postgres está saudável
docker-compose logs postgres

# Verificar credenciais e wal_level
docker exec cdc-postgres psql -U postgres -c "SHOW wal_level;"
```

### Kafka topics vazios
```bash
# Criar publication (se necessário)
docker exec cdc-postgres psql -U postgres -d cdc_db -c \
  "CREATE PUBLICATION dbz_publication FOR TABLE cdc_source.customers, cdc_source.orders;"

# Reiniciar Debezium
docker-compose restart debezium
```

### MinIO buckets não criados
```bash
bash config/minio/init-buckets.sh
```

### Flink job não inicia
```bash
docker-compose logs dbt-flink
# Verificar se Kafka está accessible
docker exec cdc-dbt-flink nc -zv kafka 29092
```

## 📚 Recursos Adicionais

- [Debezium Docs](https://debezium.io/)
- [dbt-flink](https://github.com/getdbt/dbt-flink)
- [Apache Iceberg](https://iceberg.apache.org/)
- [Apache Kafka](https://kafka.apache.org/)
- [Apache Flink](https://flink.apache.org/)

## 📝 Próximos Passos

1. **Customizar models dbt**: Editar `dbt/models/` com suas transformações
2. **Adicionar tabelas**: Criar mais tabelas no PostgreSQL e registrar com Debezium
3. **Escalar**: Ajustar paralelismo do Flink em `docker-compose.yml`
4. **Integrar**: Conectar com ferramentas de BI (Metabase, Superset, etc.)

## 🤝 Contribuindo

Sugestões e melhorias são bem-vindas!

## 📄 Licença

Open source - use livremente.
