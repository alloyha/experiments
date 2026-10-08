#!/bin/bash
set -e

# Loop interno (mesmo padrao de entrypoint-data-generator.sh): sobe uma vez e
# fica rodando, agendando os proprios ciclos via sleep, em vez de depender do
# docker-compose reiniciar o container a cada ciclo.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando iceberg-maintenance (loop interno, intervalo: ${ICEBERG_MAINTENANCE_INTERVAL:-3600}s)..."

while true; do
  python /app/scripts/iceberg_maintenance.py
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    mkdir -p /app/logs
    date '+%Y-%m-%d %H:%M:%S' > /app/logs/iceberg_maintenance_heartbeat
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] iceberg_maintenance.py exited with code $EXIT_CODE"
  fi

  sleep "${ICEBERG_MAINTENANCE_INTERVAL:-3600}"
done
