#!/bin/bash

# Spark Masterclass - Setup Verification Script
# Validates that all services are running correctly

set -e

echo "=========================================="
echo "Spark Masterclass - Setup Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_service() {
    local service_name=$1
    local port=$2
    local container=$3
    
    printf "Checking %-25s " "$service_name..."
    
    if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "^${container}$" 2>/dev/null; then
        # Container is running
        if timeout 2 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} (port $port)"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} (running, port check failed)"
            return 0
        fi
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

echo "=== Service Status ==="
echo ""

failed=0

check_service "PostgreSQL" "5432" "postgres-metastore" || ((failed++))
check_service "MinIO S3" "9000" "minio" || ((failed++))
check_service "Hive Metastore" "9083" "hive-metastore" || ((failed++))
check_service "Spark Master" "7077" "spark-master" || ((failed++))
check_service "Jupyter Lab" "8888" "spark-master" || ((failed++))
check_service "Spark Web UI" "8080" "spark-master" || ((failed++))
check_service "Iceberg REST" "8181" "iceberg-rest" || ((failed++))

echo ""
echo "=== Detailed Service Info ==="
echo ""

echo "Running Containers:"
docker-compose ps 2>/dev/null | grep -E "Up|Exited" || echo "  (no containers)"

echo ""
echo "=== Connectivity Tests ==="
echo ""

# Test PostgreSQL
printf "PostgreSQL connection... "
if docker exec postgres-metastore psql -U hive -d metastore -c "SELECT 1;" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    ((failed++))
fi

# Test MinIO
printf "MinIO S3 connectivity... "
if docker exec minio mc admin info localhost >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠${NC} (may not be fully ready)"
fi

# Test Hive
printf "Hive Metastore health... "
if docker logs hive-metastore 2>&1 | grep -q "Starting Hive Metastore Server"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠${NC} (check logs)"
fi

echo ""
echo "=== Port Status ==="
echo ""

echo "Listening ports:"
(netstat -tuln 2>/dev/null || ss -tuln 2>/dev/null) | grep -E "8080|8888|9083|7077|5432|9000" | \
    sed 's/^/  /' || echo "  (unable to determine)"

echo ""
echo "=== Quick Access Links ==="
echo ""

echo "Service URLs:"
echo "  Spark Master UI:  http://localhost:8080"
echo "  Jupyter Lab:      http://localhost:8888 (token: admin)"
echo "  MinIO Console:    http://localhost:9001 (admin/password)"
echo "  Spark Apps:       http://localhost:4040-4045"

echo ""
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Your environment is ready.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Open Jupyter Lab: http://localhost:8888"
    echo "  2. Create a new Python notebook"
    echo "  3. Run the masterclass: ./scripts/run_masterclass.sh"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the output above.${NC}"
    echo ""
    echo "Debugging tips:"
    echo "  - Check container logs: docker-compose logs <service_name>"
    echo "  - Verify containers are running: docker-compose ps"
    echo "  - Restart services: docker-compose restart"
    exit 1
fi
