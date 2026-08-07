# CDC Local Stack - Getting Started Guide

## 🎯 5 Minutos para Começar

### 1. Subir a Stack
```bash
cd postgres-debezium-kafka-flink-iceberg
make start
```

### 2. Verificar Saúde
```bash
make health
```

Você deve ver:
- ✓ PostgreSQL (localhost:5432)
- ✓ Kafka (localhost:9092)
- ✓ Debezium (http://localhost:8083)
- ✓ MinIO (http://localhost:9001)
- ✓ Flink (http://localhost:8081)

### 3. Registrar Connector Debezium
```bash
make register-connector
```

Espere ~10 segundos para o connector inicializar.

### 4. Inserir Dados de Teste
```bash
make insert-test-data
```

Ou conecte ao PostgreSQL:
```bash
make psql
```

```sql
-- Dentro do psql
INSERT INTO cdc_source.customers (name, email, country) 
VALUES ('Alice', 'alice@example.com', 'Brazil');
```

### 5. Ver Mudanças no Kafka
```bash
make kafka-consume TOPIC=customers
```

Você verá uma mensagem JSON com:
- Dados do cliente inserido
- Operação (INSERT)
- Timestamp

## 📖 Próximos Passos

### Ver Tópicos Kafka
```bash
make kafka-topics
```

### Acessar Consoles

| Serviço | URL |
|---------|-----|
| MinIO | http://localhost:9001 |
| Flink | http://localhost:8081 |

### Executar dbt Models
```bash
make dbt-run
```

Isso criará tabelas Iceberg:
- `stg_customers` (staging)
- `mart_customer_orders` (marts)

### Ver Status do Connector
```bash
make status-connector
```

## 🐛 Problemas Comuns

### "Connector não encontrado" ao registrar
```bash
# Aguarde Debezium estar pronto
sleep 10
make register-connector
```

### Kafka topics vazios
```bash
# Verificar se publication foi criada
make psql
```

Dentro do psql:
```sql
SELECT * FROM pg_publication;
CREATE PUBLICATION dbz_publication FOR TABLE cdc_source.customers, cdc_source.orders;
```

Depois reiniciar Debezium:
```bash
docker-compose restart debezium
```

### Parar tudo
```bash
make stop
```

### Reset completo (remove volumes)
```bash
make clean
make start
```

## 💡 Dicas

- Use `make help` para ver todos os comandos
- Verifique logs com `make logs-<serviço>`
- O PostgreSQL já vem com dados de teste
- MinIO é como S3 local - acesse via console

## 🎓 Arquitetura Simplificada

```
Postgres         (seu banco de origem)
    ↓ (CDC)
Kafka           (fila de eventos)
    ↓
dbt-Flink       (processa dados em tempo real)
    ↓
Iceberg (MinIO) (seu data lake)
```

## 📚 Mais Informações

Veja `README.md` para documentação completa.

---

Divirta-se com seu CDC local! 🚀
