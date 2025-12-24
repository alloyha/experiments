# Architecture Overview

This document describes the high-level architecture of the Sage library.

## System Context

```
┌────────────────────────────────────────────────────────┐
│                           Application Layer            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Service A  │  │   Service B  │  │   Service C  │  │
│  │  (Orders)    │  │  (Payments)  │  │  (Inventory) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │          │
│         └─────────────────┼─────────────────┘          │
│                           │                            │
│                     ┌─────▼─────┐                      │
│                     │   Sage    │                      │
│                     │  Library  │                      │
│                     └─────┬─────┘                      │
└───────────────────────────┼────────────────────────────┘
                            │
          ┌─────────────────┼────────────────┐
          │                 │                │
    ┌─────▼──────┐    ┌─────▼─────┐    ┌─────▼─────┐
    │ PostgreSQL │    │  RabbitMQ │    │   Kafka   │
    │  (Outbox)  │    │  (Broker) │    │  (Broker) │
    └────────────┘    └───────────┘    └───────────┘
```

## Core Components

### 1. Saga Engine

The central orchestrator that manages saga execution and compensation.

| Component | Purpose | Location |
|-----------|---------|----------|
| `Saga` | Declarative saga builder | `sage/core.py` |
| `SagaStep` | Individual step with action + compensation | `sage/core.py` |
| `SagaContext` | Shared state across steps | `sage/context.py` |
| `SagaOrchestrator` | Executes sagas with retry/timeout | `sage/orchestrator.py` |

### 2. Outbox System

Ensures reliable event delivery through the transactional outbox pattern.

| Component | Purpose | Location |
|-----------|---------|----------|
| `OutboxStorage` | Abstract storage interface | `sage/outbox/storage/base.py` |
| `PostgreSQLStorage` | PostgreSQL implementation | `sage/outbox/storage/postgresql.py` |
| `OutboxWorker` | Background event publisher | `sage/outbox/worker.py` |
| `MessageBroker` | Abstract broker interface | `sage/outbox/brokers/base.py` |

### 3. Compensation Graph

Manages complex compensation dependencies for parallel execution.

| Component | Purpose | Location |
|-----------|---------|----------|
| `CompensationGraph` | DAG of compensation dependencies | `sage/compensation_graph.py` |
| `CompensationNode` | Single compensation action | `sage/compensation_graph.py` |

---

## Deployment Architecture

### Kubernetes Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Namespace: sage                         ││
│  │                                                              ││
│  │  ┌──────────────────┐    ┌──────────────────┐               ││
│  │  │   PostgreSQL     │    │    RabbitMQ      │               ││
│  │  │   (StatefulSet)  │    │   (Deployment)   │               ││
│  │  │                  │    │                  │               ││
│  │  │  ┌────────────┐  │    │  Port: 5672      │               ││
│  │  │  │saga_outbox │  │    │  Port: 15672     │               ││
│  │  │  │   table    │  │    │    (mgmt)        │               ││
│  │  │  └────────────┘  │    └────────┬─────────┘               ││
│  │  └────────┬─────────┘             │                         ││
│  │           │                       │                         ││
│  │           │    ┌──────────────────┤                         ││
│  │           │    │                  │                         ││
│  │  ┌────────▼────▼───┐    ┌────────▼────────┐                ││
│  │  │  Outbox Worker  │    │  Outbox Worker  │  ... (N pods)  ││
│  │  │    (Pod 1)      │    │    (Pod 2)      │                ││
│  │  │                 │    │                 │                ││
│  │  │ - Poll outbox   │    │ - Poll outbox   │                ││
│  │  │ - Publish events│    │ - Publish events│                ││
│  │  │ - Mark sent     │    │ - Mark sent     │                ││
│  │  └─────────────────┘    └─────────────────┘                ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
Application                    Sage                           Infrastructure
    │                           │                                   │
    │  saga.execute()           │                                   │
    ├──────────────────────────►│                                   │
    │                           │                                   │
    │                           │  BEGIN TRANSACTION                │
    │                           ├──────────────────────────────────►│ PostgreSQL
    │                           │                                   │
    │                           │  Execute step.action()            │
    │                           │  Write to saga_outbox             │
    │                           ├──────────────────────────────────►│
    │                           │                                   │
    │                           │  COMMIT                           │
    │                           ├──────────────────────────────────►│
    │                           │                                   │
    │  SagaResult               │                                   │
    │◄──────────────────────────┤                                   │
    │                           │                                   │
    │                           │         ┌─────────────────────────┤
    │                           │         │  Outbox Worker          │
    │                           │         │  (async, separate)      │
    │                           │         │                         │
    │                           │         │  Poll pending events    │
    │                           │         ├────────────────────────►│ PostgreSQL
    │                           │         │                         │
    │                           │         │  Publish to broker      │
    │                           │         ├────────────────────────►│ RabbitMQ
    │                           │         │                         │
    │                           │         │  Mark as sent           │
    │                           │         ├────────────────────────►│ PostgreSQL
    │                           │         └─────────────────────────┤
```

---

## Design Principles

### 1. Exactly-Once Semantics

- Events stored in DB within same transaction as business data
- Worker uses `FOR UPDATE SKIP LOCKED` for safe concurrent processing
- Consumer inbox pattern prevents duplicate processing

### 2. Failure Isolation

- Each saga step is independent
- Compensation runs in reverse order on failure
- Dead-letter queue for unrecoverable events

### 3. Horizontal Scalability

- Stateless workers can scale independently
- `SKIP LOCKED` prevents worker contention
- HPA scales based on pending event count

### 4. Observability

- Prometheus metrics for monitoring
- Structured logging (JSON format)
- OpenTelemetry tracing support

---

## Future: CDC (Change Data Capture)

For high-throughput requirements (50K+ events/sec), the polling-based outbox worker can be replaced with CDC:

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────┐     ┌──────────┐
│   PostgreSQL    │ ──► │   Debezium    │ ──► │    Kafka    │ ──► │ Consumers│
│   (WAL stream)  │     │ (CDC capture) │     │   (events)  │     │          │
└─────────────────┘     └───────────────┘     └─────────────┘     └──────────┘
```

| Mode | Throughput | Use Case |
|------|------------|----------|
| **Polling** (current) | 1-5K/sec | Most applications |
| **CDC** (planned) | 50-100K/sec | High-throughput requirements |

See [ADR-011: CDC Support](adr-011-cdc-support.md) for full design.

---

## Next Steps

- [Component Details](components.md) - Deep dive into service artifacts
- [Dataflow](dataflow.md) - Event flow and state machines
- [Architecture Decisions](decisions.md) - Why we made these choices
- [CDC Support (ADR-011)](adr-011-cdc-support.md) - High-throughput upgrade path
- [Roadmap](../ROADMAP.md) - Planned features
