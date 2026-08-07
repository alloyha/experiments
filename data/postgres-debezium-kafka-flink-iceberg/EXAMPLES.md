# CDC Stack - Exemplos Práticos

## 🏁 Exemplo 1: Fluxo Completo End-to-End (10 minutos)

### Passo 1: Subir Stack
```bash
cd postgres-debezium-kafka-flink-iceberg
make start
```

Aguarde todos os serviços ficarem healthy (~30-60 segundos).

### Passo 2: Verificar Saúde
```bash
make health
```

Todos devem aparecer com ✓

### Passo 3: Registrar Debezium Connector
```bash
make register-connector
```

Aguarde confirmação de sucesso.

### Passo 4: Inserir Dados de Teste
```bash
make psql
```

Dentro do psql:
```sql
INSERT INTO cdc_source.customers (name, email, phone, country) 
VALUES 
  ('Bob Johnson', 'bob@example.com', '+5511987654321', 'Brazil'),
  ('Carol Davis', 'carol@example.com', '+5511998765432', 'Brazil'),
  ('David Brown', 'david@example.com', '+5511999999999', 'USA');

INSERT INTO cdc_source.orders (customer_id, total_amount, status) 
VALUES 
  (1, 299.99, 'COMPLETED'),
  (2, 150.00, 'PENDING'),
  (1, 75.50, 'SHIPPED'),
  (3, 999.99, 'PROCESSING');

\q  -- sair
```

### Passo 5: Ver Eventos no Kafka
```bash
make kafka-consume TOPIC=customers
```

Você verá mensagens JSON como:
```json
{
  "before": null,
  "after": {
    "id": 4,
    "name": "Bob Johnson",
    "email": "bob@example.com",
    "phone": "+5511987654321",
    "country": "Brazil",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "op": "c",
  "ts": 1705318200000
}
```

### Passo 6: Executar dbt Models
```bash
make dbt-run
```

Isso criará tabelas Iceberg com os dados transformados.

### Passo 7: Consultar Resultados
```bash
# Acessar MinIO Console
# http://localhost:9001
# Credenciais: minioadmin / minioadmin
```

Navegue até o bucket `iceberg-warehouse` para ver os arquivos parquet criados.

---

## 📝 Exemplo 2: Capturar Updates

### Fazer UPDATE no PostgreSQL
```bash
make psql
```

```sql
UPDATE cdc_source.customers 
SET phone = '+5511912345678', country = 'Canada'
WHERE name = 'Bob Johnson';

SELECT * FROM cdc_source.customers WHERE name = 'Bob Johnson';

\q
```

### Ver o evento de UPDATE no Kafka
```bash
make kafka-consume TOPIC=customers
```

Procure pela operação `"op": "u"` (update):
```json
{
  "before": {
    "id": 4,
    "name": "Bob Johnson",
    "phone": "+5511987654321",
    "country": "Brazil",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "after": {
    "id": 4,
    "name": "Bob Johnson",
    "phone": "+5511912345678",  // <-- mudado
    "country": "Canada",         // <-- mudado
    "created_at": "2024-01-15T10:30:00Z"
  },
  "op": "u",
  "ts": 1705318500000
}
```

---

## 🗑️ Exemplo 3: Capturar Deletes

### Deletar registro do PostgreSQL
```bash
make psql
```

```sql
DELETE FROM cdc_source.customers WHERE name = 'David Brown';

SELECT * FROM cdc_source.customers WHERE name = 'David Brown';

\q
```

### Ver o evento de DELETE no Kafka
```bash
make kafka-consume TOPIC=customers
```

Procure por `"op": "d"` (delete):
```json
{
  "before": {
    "id": 6,
    "name": "David Brown",
    "email": "david@example.com",
    "country": "USA",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "after": null,
  "op": "d",
  "ts": 1705318800000
}
```

O modelo dbt `stg_customers` filtrará automaticamente com `WHERE __deleted = false`.

---

## 🔄 Exemplo 4: Modificar um Model dbt

### Customizar modelo de staging

Editar `dbt/models/staging/stg_customers.sql`:

```sql
{{
  config(
    materialized='table',
    schema='staging',
    location='s3://iceberg-warehouse/staging/customers'
  )
}}

SELECT
    id,
    name,
    LOWER(email) as email,  -- normalizar email
    phone,
    UPPER(country) as country,  -- país em maiúscula
    CURRENT_TIMESTAMP() as dbt_loaded_at,
    CAST(created_at AS DATE) as created_date
FROM {{ source('kafka_cdc', 'customers') }}
WHERE __deleted = false
  AND email IS NOT NULL  -- filtro adicional
```

Depois executar:
```bash
make dbt-run
```

---

## 📊 Exemplo 5: Adicionar Nova Tabela ao CDC

### 1. Criar tabela no PostgreSQL
```bash
make psql
```

```sql
CREATE TABLE cdc_source.products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    stock INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE cdc_source.products REPLICA IDENTITY FULL;

INSERT INTO cdc_source.products (name, price, stock) VALUES 
  ('Laptop', 999.99, 10),
  ('Mouse', 29.99, 100),
  ('Monitor', 299.99, 25);

\q
```

### 2. Atualizar Publication do Debezium
```bash
make psql
```

```sql
ALTER PUBLICATION dbz_publication ADD TABLE cdc_source.products;
\q
```

### 3. Parar e reativar o connector
```bash
python3 scripts/debezium_manager.py pause
sleep 5
python3 scripts/debezium_manager.py resume
```

### 4. Ver novo tópico Kafka
```bash
make kafka-topics
```

Você verá `products` na lista.

### 5. Criar modelo dbt para products
```bash
cat > dbt/models/staging/stg_products.sql << 'EOF'
{{
  config(
    materialized='table',
    schema='staging',
    location='s3://iceberg-warehouse/staging/products'
  )
}}

SELECT
    id,
    name,
    price,
    stock,
    created_at,
    updated_at
FROM {{ source('kafka_cdc', 'products') }}
WHERE __deleted = false
EOF
```

### 6. Atualizar schema.yml
Adicionar à seção `kafka_cdc.tables`:

```yaml
      - name: products
        description: "Product catalog from CDC"
        columns:
          - name: id
            description: "Product ID"
            data_type: int
          - name: name
            description: "Product name"
            data_type: string
          - name: price
            description: "Product price"
            data_type: decimal
          - name: stock
            description: "Stock quantity"
            data_type: int
```

### 7. Executar novo modelo
```bash
make dbt-run
```

---

## 🧪 Exemplo 6: Criar Mart Customizado

### Criar novo model em `dbt/models/marts/mart_sales_summary.sql`
```sql
{{
  config(
    materialized='table',
    schema='marts',
    location='s3://iceberg-warehouse/marts/sales_summary'
  )
}}

SELECT
    o.customer_id,
    c.name as customer_name,
    c.country,
    COUNT(o.id) as total_orders,
    SUM(o.total_amount) as total_revenue,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.order_date) as most_recent_order,
    CASE 
        WHEN SUM(o.total_amount) > 1000 THEN 'Premium'
        WHEN SUM(o.total_amount) > 500 THEN 'Gold'
        ELSE 'Silver'
    END as customer_tier,
    CURRENT_TIMESTAMP() as dbt_updated_at
FROM {{ ref('stg_customers') }} c
LEFT JOIN {{ source('kafka_cdc', 'orders') }} o
    ON c.id = o.customer_id
    AND o.__deleted = false
WHERE c.__deleted = false
GROUP BY 
    o.customer_id,
    c.name,
    c.country
ORDER BY total_revenue DESC
```

### Executar
```bash
make dbt-run
```

---

## 🧪 Exemplo 7: Testes dbt

### Criar teste customizado em `dbt/tests/test_customer_unique.sql`
```sql
-- Verificar se não há customer_id duplicado em orders
SELECT customer_id, COUNT(*) as cnt
FROM {{ ref('mart_customer_orders') }}
GROUP BY customer_id
HAVING COUNT(*) > 1
```

### Executar testes
```bash
make dbt-test
```

---

## 📈 Exemplo 8: Monitorar Performance

### Ver lag do Kafka
```bash
docker exec cdc-kafka kafka-consumer-groups \
  --bootstrap-server kafka:29092 \
  --group dbt-flink-consumer \
  --describe
```

Coluna `LAG` mostra quantas mensagens ainda não foram processadas.

### Ver UI do Flink
```
http://localhost:8081
```

- **Jobs**: Mostra status do job
- **Task Managers**: Performance e memory
- **Logs**: Erros e warnings

### Ver MinIO
```
http://localhost:9001
```

- Buckets criados
- Arquivos Iceberg
- Histórico de uploads

---

## 🔧 Exemplo 9: Pausar e Resumir CDC

### Pausar (útil para manutenção)
```bash
python3 scripts/debezium_manager.py pause
```

Dados continuam no PostgreSQL, CDC não capta novos eventos.

### Resumir
```bash
python3 scripts/debezium_manager.py resume
```

Capturará novos eventos a partir de onde parou.

---

## 🧹 Exemplo 10: Reset Completo

Se algo der errado ou quiser começar do zero:

```bash
# Parar tudo
make stop

# Remover volumes (perdida de dados)
docker-compose down -v

# Subir novamente
make start

# Registrar connector
make register-connector

# Pronto! Volta ao estado inicial
```

---

## 💡 Dicas Importantes

1. **Debezium precisa de replica identity**: Se não capturar valores antigos em UPDATEs, executar:
   ```sql
   ALTER TABLE cdc_source.sua_tabela REPLICA IDENTITY FULL;
   ```

2. **Kafka topics são imutáveis**: Delete topics se quiser limpar:
   ```bash
   docker exec cdc-kafka kafka-topics --delete --bootstrap-server kafka:29092 --topic customers
   ```

3. **dbt models são idempotentes**: Seguro executar múltiplas vezes
   ```bash
   make dbt-run  # Executa novamente, sobrescreve tables
   ```

4. **Monitorar via Logs**:
   ```bash
   make logs-debezium      # Ver captura CDC
   make logs-kafka         # Ver fila de eventos
   make logs-flink         # Ver processamento
   ```

5. **Backup de dados Postgres**:
   ```bash
   docker exec cdc-postgres pg_dump -U postgres cdc_db > backup.sql
   ```

---

Divirta-se! 🎉
