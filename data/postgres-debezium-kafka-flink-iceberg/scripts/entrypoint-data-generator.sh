#!/bin/bash
set -e

# Script de entrada para data-generator
# Loop interno (mesmo padrao usado por dbt-flink-ingestao / dbt-duckdb-medallhao)
# em vez de depender do `restart: always` do docker-compose para reiniciar o
# container a cada ciclo -- o container sobe uma vez e fica rodando,
# agendando os proprios ciclos via sleep.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando data-generator (loop interno, intervalo: ${DATA_GENERATOR_INTERVAL:-10}s)..."

# Aguarda PostgreSQL ficar pronto (uma vez, no start do container)
# (PGPASSWORD e necessario aqui: pg_hba.conf usa scram-sha-256 para conexoes
# de outros containers, so trust para 127.0.0.1/socket local -- sem a senha
# este check sempre falhava silenciosamente e so avancava apos as 30 tentativas)
export PGPASSWORD="${POSTGRES_PASSWORD:-postgres}"
echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
  if psql -h "${POSTGRES_HOST:-postgres}" -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-cdc_db}" -c "SELECT 1" >/dev/null 2>&1; then
    echo "✓ PostgreSQL is ready!"
    break
  fi
  echo -n "."
  sleep 1
done

while true; do
  python data_generator_once.py
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    # Heartbeat para o healthcheck do container: grava em /app/logs (montado
    # do host, sobrevive caso o container seja recriado).
    date '+%Y-%m-%d %H:%M:%S' > /app/logs/heartbeat
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] data_generator_once.py exited with code $EXIT_CODE"
  fi

  sleep "${DATA_GENERATOR_INTERVAL:-10}"
done
