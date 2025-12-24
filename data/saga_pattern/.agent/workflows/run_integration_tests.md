---
description: Run integration tests with Docker containers
---

# Run Integration Tests

This workflow runs the integration tests which require Docker for PostgreSQL, Kafka, and RabbitMQ.

## Prerequisites

- Docker Desktop / Docker Engine running
- `uv` installed
- `python` 3.11+

## Steps

1. Install dependencies with all optional groups:
```bash
uv sync --all-groups
```

2. Run tests with the integration flag:
```bash
RUN_INTEGRATION=1 pytest tests/test_integration_containers.py -v
```

3. To run all tests including integration:
```bash
RUN_INTEGRATION=1 pytest tests/ -v
```

4. To check coverage:
```bash
RUN_INTEGRATION=1 pytest tests/ -v --cov=sage --cov-report=term-missing
```
