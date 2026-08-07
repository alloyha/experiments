#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Starting CDC Stack ===${NC}"

# Subir containers
docker-compose up -d

echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 5

# Verificar saúde dos serviços
services=("postgres" "kafka" "debezium" "minio" "dbt-flink")

for service in "${services[@]}"; do
    container="cdc-${service}"
    echo -n "Checking ${service}... "
    
    if docker ps | grep -q "$container"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗${NC}"
    fi
done

echo ""
echo -e "${GREEN}=== CDC Stack Started ===${NC}"
echo ""
echo "Services:"
echo "  Postgres:  localhost:5432 (user: postgres, pass: postgres)"
echo "  Kafka:     localhost:9092"
echo "  Debezium:  http://localhost:8083"
echo "  MinIO:     http://localhost:9001 (user: minioadmin, pass: minioadmin)"
echo "  Flink:     http://localhost:8081"
echo ""
echo "Use './stop-all.sh' to stop services"
