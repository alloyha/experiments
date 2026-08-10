# 📜 Script Index - Guia Completo de Scripts

## 📂 Estrutura de Diretórios

```
scripts/
├── 🚀 setup/                 # Inicialização e configuração
├── 🔍 query/                 # Consultas de dados
├── ✅ check/                 # Validação e diagnóstico  
├── 🚀 deploy/                # Deployment e CI/CD
├── 📊 monitor/               # Monitoramento em tempo real
└── 🔧 utility/               # Utilitários e helpers

(Raiz scripts/) - Sidecars principais para Docker
```

---

## 🚀 Setup & Initialization (`scripts/setup/`)

| Script | Propósito | Comando |
|--------|-----------|---------|
| **start-all.sh** | Inicia todos os serviços | `./scripts/setup/start-all.sh` |
| **stop-all.sh** | Para todos os serviços | `./scripts/setup/stop-all.sh` |
| **start-with-sidecars.sh** | Inicia com sidecars | `./scripts/setup/start-with-sidecars.sh` |
| **rebuild-and-start.sh** | Reconstrói e inicia tudo | `./scripts/setup/rebuild-and-start.sh` |
| **start-flink.sh** | Inicia apenas Flink | `./scripts/setup/start-flink.sh` |
| **install_dependencies.sh** | Instala dependências | `./scripts/setup/install_dependencies.sh` |
| **init-postgres.sql** | SQL inicial PostgreSQL | `.sql file` |
| **entrypoints/** | Scripts de entrada dos containers | |
| └ **entrypoint-data-generator.sh** | Entrada data-generator | (Docker entry) |
| └ **entrypoint-dbt-runner.sh** | Entrada dbt-runner | (Docker entry) |
| └ **entrypoint-flink-consumer.sh** | Entrada flink-consumer | (Docker entry) |

### Exemplos de Uso
```bash
# Setup completo
cd scripts/setup
./rebuild-and-start.sh

# Apenas parar
./stop-all.sh

# Usar make
make start
make stop
```

---

## 🔍 Query & Data Access (`scripts/query/`)

| Script | Propósito | Comando |
|--------|-----------|---------|
| **query-iceberg.sh** | Query genérica em Iceberg | `./scripts/query/query-iceberg.sh` |
| **query-iceberg-flink.sh** | Query via Flink API | `./scripts/query/query-iceberg-flink.sh` |
| **query-iceberg-duckdb.py** | Query via DuckDB Python | `python3 ./scripts/query/query-iceberg-duckdb.py` |
| **query-iceberg-pyiceberg.py** | Query via PyIceberg | `python3 ./scripts/query/query-iceberg-pyiceberg.py` |
| **consume_kafka.py** | Consome tópicos Kafka | `python3 ./scripts/query/consume_kafka.py` |
| **flink_stream_consumer.py** | Consumer streaming Flink | `python3 ./scripts/query/flink_stream_consumer.py` |

### Exemplos de Uso
```bash
# Query em Iceberg (Silver layer)
./scripts/query/query-iceberg.sh

# Consumir evento Kafka
TOPIC=customers python3 ./scripts/query/consume_kafka.py

# Query via Python
python3 ./scripts/query/query-iceberg-duckdb.py
```

---

## ✅ Check & Validation (`scripts/check/`)

| Script | Propósito | Comando |
|--------|-----------|---------|
| **health-check.sh** | Verifica saúde da stack | `./scripts/check/health-check.sh` |
| **check-consumer.sh** | Valida Kafka consumer | `./scripts/check/check-consumer.sh` |
| **check-pipeline.sh** | Valida pipeline CDC | `./scripts/check/check-pipeline.sh` |
| **check-status.sh** | Status geral da stack | `./scripts/check/check-status.sh` |
| **check_connector.py** | Status do connector Debezium | `python3 ./scripts/check/check_connector.py` |

### Exemplos de Uso
```bash
# Verificar saúde completa
make health
./scripts/check/health-check.sh

# Status do pipeline
./scripts/check/check-status.sh

# Validar CDC
./scripts/check/check-pipeline.sh
```

---

## 🚀 Deploy & CI/CD (`scripts/deploy/`)

| Script | Propósito | Comando |
|--------|-----------|---------|
| **rebuild.sh** | Reconstrói serviços | `./scripts/deploy/rebuild.sh` |
| **rebuild-dbt.sh** | Reconstrói modelos dbt | `./scripts/deploy/rebuild-dbt.sh` |
| **deploy-medallhao.sh** | Deploy medalhão completo | `./scripts/deploy/deploy-medallhao.sh` |
| **deploy-dbt-flink.sh** | Deploy dbt-flink service | `./scripts/deploy/deploy-dbt-flink.sh` |

### Exemplos de Uso
```bash
# Deploy medalhão
./scripts/deploy/deploy-medallhao.sh

# Reconstrói apenas dbt
./scripts/deploy/rebuild-dbt.sh

# Reconstrói tudo
./scripts/deploy/rebuild.sh
```

---

## 📊 Monitoring (`scripts/monitor/`)

| Script | Propósito | Comando |
|--------|-----------|---------|
| **monitor-stack.sh** | Monitor colorizado em tempo real | `./scripts/monitor/monitor-stack.sh` |
| **monitor.py** | Monitor em Python | `python3 ./scripts/monitor/monitor.py` |

### Exemplos de Uso
```bash
# Monitor com cores e status
./scripts/monitor/monitor-stack.sh

# Monitor Python
make monitor
python3 ./scripts/monitor/monitor.py

# Ver logs em tempo real
docker compose logs -f
make logs-generator
make logs-dbt-runner
```

---

## 🔧 Utility (`scripts/utility/`)

| Script | Propósito | Comando |
|--------|-----------|---------|
| **healthcheck-dbt.sh** | Health check para dbt services | (Docker health check) |
| **QUICK_REFERENCE.sh** | Referência rápida de comandos | `source ./scripts/utility/QUICK_REFERENCE.sh` |
| **validate_migration.py** | Valida migração pyproject.toml | `python3 ./scripts/utility/validate_migration.py` |
| **update_health_checks.py** | Atualiza health checks | `python3 ./scripts/utility/update_health_checks.py` |
| **validate_sidecars.py** | Valida sidecars | `python3 ./scripts/utility/validate_sidecars.py` |

### Exemplos de Uso
```bash
# Validar dependências
python3 ./scripts/utility/validate_migration.py

# Ver referência rápida
./scripts/utility/QUICK_REFERENCE.sh

# Validar sidecars
python3 ./scripts/utility/validate_sidecars.py
```

---

## 🐳 Docker Sidecars (Raiz `scripts/`)

Estes scripts rodam dentro de containers Docker:

| Script | Container | Propósito |
|--------|-----------|-----------|
| **data_generator.py** | data-generator | Gera dados sintéticos (10s cycle) |
| **data_generator_once.py** | data-generator | One-shot data generation |
| **dbt_runner.py** | dbt-runner | Executa dbt models (loop contínuo) |
| **dbt_runner_once.py** | dbt-runner | One-shot dbt execution |
| **debezium_manager.py** | Host | Gerencia connector Debezium |

### Configuração
```bash
# Registrar Debezium connector
make register-connector
python3 scripts/debezium_manager.py register

# Verificar status
make status-connector
python3 scripts/debezium_manager.py status

# Setup cron jobs (alternativa sem Docker)
python3 scripts/setup_cron.py install
```

---

## 🎯 Referência Rápida por Tarefa

### Iniciar Stack
```bash
make start
# ou
./scripts/setup/start-all.sh
```

### Parar Stack
```bash
make stop
# ou
./scripts/setup/stop-all.sh
```

### Monitorar Pipeline
```bash
make monitor
# ou
./scripts/monitor/monitor-stack.sh
```

### Verificar Saúde
```bash
make health
# ou
./scripts/check/health-check.sh
```

### Query Gold (PostgreSQL)
```bash
make psql
psql> SELECT * FROM gold.fct_orders LIMIT 10;
```

### Query Silver (Iceberg)
```bash
./scripts/query/query-iceberg-duckdb.py
```

### Consumir Kafka
```bash
make kafka-topics
TOPIC=customers make kafka-consume
# ou
TOPIC=customers python3 ./scripts/query/consume_kafka.py
```

### Executar dbt
```bash
# Todos os targets
make dbt-run

# Apenas Gold (PostgreSQL)
dbt run -t postgres

# Com freshness check
dbt run -t postgres --freshness
```

### Validar Setup
```bash
python3 ./scripts/utility/validate_migration.py
./scripts/check/check-pipeline.sh
./scripts/check/check-status.sh
```

---

## 📝 Notas

- ✅ Scripts movidos de root para `scripts/*/` para melhor organização
- ✅ Todos os paths atualizados no Makefile
- ✅ Docker entry points estão em `scripts/setup/entrypoints/`
- ✅ Python scripts de sidecars permanecem na raiz `scripts/` (referenciados pelos Dockerfiles)

---

## 🔗 Links Relacionados

- [REORGANIZATION.md](REORGANIZATION.md) - Detalhes da reorganização
- [SIDECARS.md](SIDECARS.md) - Guia de sidecars contínuos
- [Makefile](Makefile) - Targets make disponíveis
- [docker-compose.yml](docker-compose.yml) - Serviços e configuração
