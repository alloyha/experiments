#!/bin/bash
set -e

# Script de entrada para dbt-runner
# Executa uma vez (run + test + docs) e sai; docker-compose cuida do loop

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting dbt_runner_once.py..."

# Aguarda PostgreSQL e Kafka
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
  if psql -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-cdc_db}" -c "SELECT 1" 2>/dev/null; then
    echo "✓ PostgreSQL is ready!"
    break
  fi
  echo -n "."
  sleep 1
done

echo "Waiting for Kafka to be ready..."
for i in {1..20}; do
  if kafka-topics --bootstrap-server kafka:9092 --list 2>/dev/null | grep -q .; then
    echo "✓ Kafka is ready!"
    break
  fi
  echo -n "."
  sleep 1
done

# Executa dbt runner
python dbt_runner_once.py
EXIT_CODE=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] dbt_runner_once.py exited with code $EXIT_CODE"
exit $EXIT_CODE
