#!/bin/bash
# Entrypoint script for Flink Stream Consumer
# Waits for dependencies before starting

set -e

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Flink Stream Consumer entrypoint..."

# Wait for PostgreSQL
echo "Waiting for PostgreSQL to be ready..."
POSTGRES_WAIT=0
while ! psql -h postgres -U postgres -d cdc_db -c "SELECT 1" > /dev/null 2>&1; do
    POSTGRES_WAIT=$((POSTGRES_WAIT + 1))
    if [ $POSTGRES_WAIT -eq 60 ]; then
        echo "PostgreSQL didn't come up in time!"
        exit 1
    fi
    sleep 1
done
echo "✓ PostgreSQL is ready"

# Wait for Kafka
echo "Waiting for Kafka to be ready..."
KAFKA_WAIT=0
while ! kafka-topics --bootstrap-server kafka:29092 --list > /dev/null 2>&1; do
    KAFKA_WAIT=$((KAFKA_WAIT + 1))
    if [ $KAFKA_WAIT -eq 40 ]; then
        echo "Kafka didn't come up in time!"
        exit 1
    fi
    sleep 1
done
echo "✓ Kafka is ready"

echo ""
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Dependencies ready. Starting consumer..."
echo ""

# Run the consumer
python3 /app/flink_stream_consumer.py
