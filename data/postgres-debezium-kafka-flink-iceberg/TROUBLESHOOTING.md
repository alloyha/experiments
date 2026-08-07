# CDC Stack - Troubleshooting Guide

## 🔍 Diagnóstico Rápido

### 1. Verificar Saúde Geral
```bash
make health
# ou
./scripts/health-check.sh
```

### 2. Verificar Logs
```bash
make logs              # Todos os serviços
make logs-postgres     # PostgreSQL
make logs-kafka        # Kafka
make logs-debezium     # Debezium
make logs-flink        # Flink
```

---

## ❌ Problemas Comuns e Soluções

### PostgreSQL

#### "Conexão recusada na porta 5432"
```bash
# Verificar se container está rodando
docker ps | grep cdc-postgres

# Ver logs
docker-compose logs postgres

# Reiniciar
docker-compose restart postgres
```

#### "FATAL: wal_level must be 'logical' for CDC"
```bash
# Já vem configurado no init-postgres.sql
# Se o erro persiste, fazer manualmente:
docker exec cdc-postgres psql -U postgres -c "SHOW wal_level;"

# Deve retornar: wal_level | logical
```

#### Dados não aparecem na tabela
```bash
docker exec -it cdc-postgres psql -U postgres -d cdc_db
SELECT * FROM cdc_source.customers;
```

---

### Kafka

#### "Broker not available"
```bash
# Verificar conexão com Zookeeper
docker exec cdc-kafka zookeeper-shell localhost:2181 ls /brokers/ids

# Reiniciar Kafka
docker-compose restart kafka

# Aguardar ~20 segundos
```

#### Topics vazios
```bash
# Listar tópicos
docker exec cdc-kafka kafka-topics --list --bootstrap-server kafka:29092

# Descrever tópico
docker exec cdc-kafka kafka-topics --describe --bootstrap-server kafka:29092 --topic customers

# Consumir com offset reset
docker exec cdc-kafka kafka-console-consumer \
  --bootstrap-server kafka:29092 \
  --topic customers \
  --from-beginning
```

#### "Topic already exists"
```bash
# Deletar e recriar
docker exec cdc-kafka kafka-topics --delete --bootstrap-server kafka:29092 --topic customers
docker exec cdc-kafka kafka-topics --create --bootstrap-server kafka:29092 --topic customers --partitions 1 --replication-factor 1
```

---

### Debezium

#### "Connector não encontrado ao registrar"
```bash
# Verificar se Debezium está pronto
curl http://localhost:8083/

# Aguardar mais tempo antes de registrar
python3 scripts/debezium_manager.py wait
make register-connector
```

#### "Connector em estado FAILED"
```bash
# Ver status detalhado
make status-connector

# Deletar e recriar
python3 scripts/debezium_manager.py delete
sleep 5
make register-connector
```

#### "Slot 'dbz_slot' já existe"
```bash
# Acessar PostgreSQL e remover
docker exec cdc-postgres psql -U postgres -d cdc_db -c "SELECT * FROM pg_replication_slots;"

# Dropar se existir
docker exec cdc-postgres psql -U postgres -d cdc_db -c "SELECT pg_drop_replication_slot('dbz_slot');"
```

#### "Publication não existe"
```bash
# Verificar
docker exec cdc-postgres psql -U postgres -d cdc_db -c "SELECT * FROM pg_publication;"

# Criar se não existir
docker exec cdc-postgres psql -U postgres -d cdc_db -c \
  "CREATE PUBLICATION dbz_publication FOR TABLE cdc_source.customers, cdc_source.orders;"
```

#### Connector conectando mas não capturando mudanças
```bash
# 1. Verificar se REPLICA IDENTITY está configurado
docker exec cdc-postgres psql -U postgres -d cdc_db -c \
  "SELECT schemaname, tablename, replica_identity FROM pg_tables WHERE schemaname = 'cdc_source';"

# Deve retornar 'f' (full) para as tabelas

# 2. Se não estiver, configurar:
docker exec cdc-postgres psql -U postgres -d cdc_db -c \
  "ALTER TABLE cdc_source.customers REPLICA IDENTITY FULL;"

docker exec cdc-postgres psql -U postgres -d cdc_db -c \
  "ALTER TABLE cdc_source.orders REPLICA IDENTITY FULL;"

# 3. Pausar e resumir connector
python3 scripts/debezium_manager.py pause
sleep 5
python3 scripts/debezium_manager.py resume
```

---

### MinIO

#### "Não consigo acessar console (9001)"
```bash
# Verificar se container está rodando
docker ps | grep cdc-minio

# Ver logs
docker-compose logs minio

# Reiniciar
docker-compose restart minio
```

#### "Buckets não foram criados"
```bash
# Criar manualmente
docker exec cdc-minio mc alias set minio http://localhost:9000 minioadmin minioadmin
docker exec cdc-minio mc mb minio/iceberg-warehouse
docker exec cdc-minio mc mb minio/iceberg-metadata
```

#### Credenciais padrão não funcionam
```bash
# Credenciais padrão:
# User: minioadmin
# Pass: minioadmin

# Se não funcionar, verificar logs
docker-compose logs minio

# Reiniciar com reset
docker-compose down -v
docker-compose up -d minio
```

---

### Flink / dbt-flink

#### "dbt: command not found"
```bash
# dbt-flink ainda está em desenvolvimento
# Use Python local:
pip install -r requirements.txt
cd dbt
dbt run
```

#### Flink não consegue acessar Kafka
```bash
# Verificar conectividade dentro do container
docker exec cdc-dbt-flink nc -zv kafka 29092

# Se falhar, verificar network
docker network ls
docker network inspect postgres-debezium-kafka-flink-iceberg_cdc_network
```

#### Job não inicia
```bash
# Ver logs detalhados
docker-compose logs -f dbt-flink

# Verificar UI
# http://localhost:8081

# Reiniciar
docker-compose restart dbt-flink
```

---

## 🔧 Operações de Manutenção

### Reset Completo (com perda de dados)
```bash
docker-compose down -v
make start
make register-connector
```

### Backup de dados PostgreSQL
```bash
docker exec cdc-postgres pg_dump -U postgres cdc_db > backup.sql
```

### Restore de dados PostgreSQL
```bash
cat backup.sql | docker exec -i cdc-postgres psql -U postgres cdc_db
```

### Limpar tópicos Kafka
```bash
docker exec cdc-kafka kafka-topics --delete --bootstrap-server kafka:29092 --topic customers
docker exec cdc-kafka kafka-topics --delete --bootstrap-server kafka:29092 --topic orders
```

### Listar todos os connectors (REST API)
```bash
curl -s http://localhost:8083/connectors | jq .
```

### Pausar todas as tarefas Flink
```bash
curl -X PATCH http://localhost:8081/v1/jobs/JOBID?state=SUSPENDED
```

---

## 📊 Monitoramento

### Dashboard Flink
```
http://localhost:8081
```

### MinIO Console
```
http://localhost:9001
Username: minioadmin
Password: minioadmin
```

### Verificar offset do Kafka
```bash
docker exec cdc-kafka kafka-consumer-groups \
  --bootstrap-server kafka:29092 \
  --list
```

---

## 🚀 Performance Tuning

### Aumentar Parallelismo do Flink
```yaml
# docker-compose.yml
dbt-flink:
  environment:
    FLINK_PROPERTIES: |
      taskmanager.numberOfTaskSlots: 8  # aumentar de 4 para 8
```

### Aumentar Retenção de Dados Kafka
```yaml
# docker-compose.yml
kafka:
  environment:
    KAFKA_LOG_RETENTION_MS: 604800000  # 7 dias
```

### Ajustar Batch Size do Debezium
```json
// config/debezium/postgres-source.json
{
  "config": {
    "max.batch.size": 2048,
    "poll.interval.ms": 1000
  }
}
```

---

## 📞 Suporte

Se nenhuma solução funcionar:

1. Coletar logs de todos os serviços
2. Salvar output de `docker ps` e `docker-compose logs`
3. Salvar configurações (sem senhas)
4. Abrir issue com essas informações

---

**Última atualização:** 2024
