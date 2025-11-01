#!/bin/bash
# run_masterclass.sh - Docker wrapper for small cluster benchmark
# 
# Run the Spark Data Layout Masterclass benchmark in Docker
#
# This script:
#   - Starts Docker containers if needed
#   - Runs small_cluster_benchmark.py (500K rows, 5 minutes)
#   - Tests Parquet, Delta Lake, and Iceberg in Docker
#   - Shows performance comparisons
#
# Usage:
#   docker-compose up -d  # Start first
#   ./scripts/run_masterclass.sh
#
# Or directly:
#   docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found"
    echo "Install Docker Desktop or Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if containers are running
if ! docker-compose ps | grep -q spark-master; then
    echo "Starting Docker containers..."
    cd "$PROJECT_DIR"
    docker-compose up -d
    echo "Waiting for services to be ready..."
    sleep 5
fi

cd "$PROJECT_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  SPARK DATA LAYOUT MASTERCLASS - Quick Benchmark       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Configuration:"
echo "   • Rows: 500,000"
echo "   • Formats: Parquet, Delta Lake, Iceberg"
echo "   • Queries: 9 (filters, aggregations)"
echo "   • Storage: MinIO S3"
echo "   • Mode: Docker container"
echo ""
echo "⏱️  Estimated Time: 5-10 minutes"
echo ""
echo "Starting benchmark..."
echo ""

# Run the benchmark in Docker
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py

echo ""
echo "✅ Benchmark complete!"
echo "View results: http://localhost:9001 (MinIO console)"
echo ""
