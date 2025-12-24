# Quickstart Guide

Get Sage running in 5 minutes.

## Installation

```bash
pip install sage-saga
```

Or with extras:

```bash
pip install sage-saga[postgres,rabbitmq]  # For production
pip install sage-saga[postgres,kafka]     # With Kafka
```

## Basic Usage

### 1. Define a Saga

```python
from sage import Saga, SagaContext

# Define your step functions
async def reserve_inventory(ctx: SagaContext):
    order = ctx.get("order")
    # Reserve items...
    ctx.set("reservation_id", "RES-123")

async def release_inventory(ctx: SagaContext):
    reservation_id = ctx.get("reservation_id")
    # Release reservation...

async def charge_payment(ctx: SagaContext):
    order = ctx.get("order")
    # Charge customer...
    ctx.set("payment_id", "PAY-456")

async def refund_payment(ctx: SagaContext):
    payment_id = ctx.get("payment_id")
    # Refund payment...

# Build the saga
saga = (
    Saga("order-processing")
    .step("reserve_inventory")
        .action(reserve_inventory)
        .compensation(release_inventory)
    .step("charge_payment")
        .action(charge_payment)
        .compensation(refund_payment)
    .build()
)
```

### 2. Execute the Saga

```python
from sage import SagaContext

# Create context with initial data
ctx = SagaContext(
    saga_id="order-001",
    data={"order": {"id": "ORD-123", "amount": 99.99}}
)

# Execute
result = await saga.execute(ctx)

if result.status == "completed":
    print(f"Order processed successfully!")
else:
    print(f"Order failed: {result.error}")
```

---

## With Transactional Outbox

For reliable event delivery:

### 1. Setup Database

```sql
-- Run migration
CREATE TABLE saga_outbox (
    event_id UUID PRIMARY KEY,
    saga_id VARCHAR(255) NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2. Configure Storage & Broker

```python
from sage.outbox import OutboxWorker
from sage.outbox.storage.postgresql import PostgreSQLOutboxStorage
from sage.outbox.brokers.rabbitmq import RabbitMQBroker

# Storage
storage = PostgreSQLOutboxStorage(
    connection_string="postgresql://user:pass@localhost/db"
)

# Broker
broker = RabbitMQBroker(
    url="amqp://guest:guest@localhost/"
)
```

### 3. Run Worker

```python
from sage.outbox import OutboxWorker, OutboxConfig

config = OutboxConfig(
    batch_size=100,
    poll_interval_seconds=1.0,
    max_retries=5,
)

worker = OutboxWorker(storage, broker, config)
await worker.start()  # Runs until stopped
```

---

## Deploy to Kubernetes

```bash
# Create namespace and secrets
kubectl create namespace sage
kubectl apply -f k8s/secrets-local.yaml

# Deploy PostgreSQL
kubectl apply -f k8s/postgresql-local.yaml

# Run migrations
kubectl apply -f k8s/migration-job.yaml

# Deploy RabbitMQ
kubectl apply -f k8s/rabbitmq.yaml

# Deploy workers
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/outbox-worker.yaml

# Verify
kubectl get pods -n sage
```

---

## Next Steps

| Topic | Link |
|-------|------|
| Architecture | [Overview](architecture/overview.md) |
| Full API Reference | [API Docs](reference/api.md) |
| Kubernetes Deployment | [K8s Guide](guides/kubernetes.md) |
| Performance Tuning | [Benchmarking](guides/benchmarking.md) |
