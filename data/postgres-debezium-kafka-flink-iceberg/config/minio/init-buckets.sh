#!/bin/bash

# Aguardar MinIO estar pronto
echo "Waiting for MinIO to be ready..."
sleep 10

# Configurar cliente MinIO
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin

# Criar buckets necessários para Iceberg
mc alias set myminio http://minio:9000 minioadmin minioadmin

# Criar buckets
mc mb myminio/iceberg-warehouse
mc mb myminio/iceberg-metadata
mc mb myminio/kafka-data

echo "MinIO buckets created successfully"
