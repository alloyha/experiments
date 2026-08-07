#!/bin/bash

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== CDC Stack Health Check ===${NC}\n"

# Função para verificar porta
check_port() {
    local host=$1
    local port=$2
    local service=$3
    
    if timeout 1 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $service ($host:$port)"
        return 0
    else
        echo -e "${RED}✗${NC} $service ($host:$port)"
        return 1
    fi
}

# Função para verificar HTTP endpoint
check_http() {
    local url=$1
    local service=$2
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|404"; then
        echo -e "${GREEN}✓${NC} $service ($url)"
        return 0
    else
        echo -e "${RED}✗${NC} $service ($url)"
        return 1
    fi
}

# Verificações
check_port "localhost" 5432 "PostgreSQL"
check_port "localhost" 9092 "Kafka"
check_http "http://localhost:8083" "Debezium"
check_http "http://localhost:9001" "MinIO Console"
check_http "http://localhost:8081" "Flink UI"

echo ""
echo -e "${YELLOW}=== Detailed Status ===${NC}\n"

# Verificar containers
echo "Running containers:"
docker ps --filter "label!=unrelated" --format "table {{.Names}}\t{{.Status}}" | grep cdc

echo ""
echo -e "${YELLOW}Docker Compose Status:${NC}"
docker-compose ps
