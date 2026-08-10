#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Starting CDC Stack with Sidecars ===${NC}"

# Sobe tudo, incluindo data-generator e dbt-runner
docker-compose up -d --build

echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Verificar saúde dos serviços principais
services=("postgres" "kafka" "debezium" "minio" "data-generator" "dbt-runner")

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
echo -e "${YELLOW}Registrando connector Debezium...${NC}"
curl -s -X POST http://localhost:8083/connectors \
    -H "Content-Type: application/json" \
    -d @config/debezium/postgres-source.json > /dev/null && \
    echo -e "${GREEN}✓ Connector registrado${NC}" || \
    echo -e "${YELLOW}⚠ Connector já pode estar registrado${NC}"

echo ""
echo -e "${GREEN}=== CDC Stack com Sidecars Rodando ===${NC}"
echo ""
echo "Sidecars ativos:"
echo "  Data Generator: insere/atualiza/deleta dados a cada 10s"
echo "  dbt Runner:     roda dbt run/test/docs a cada 60s"
echo ""
echo "Acompanhe em terminais separados:"
echo "  make logs-generator"
echo "  make logs-dbt-runner"
echo "  make kafka-topics"
echo ""
echo "Use './stop-all.sh' para parar tudo"
