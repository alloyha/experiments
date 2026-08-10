#!/bin/bash

################################################################################
# CDC Stack Real-Time Monitor
# Monitora o status de todos os serviços, dados e métricas
################################################################################

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Símbolos
CHECK='✓'
CROSS='✗'
CLOCK='⏱'
CIRCLE='●'

clear

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              🚀 CDC STACK REAL-TIME MONITOR                                   ║${NC}"
echo -e "${BLUE}║                                                                                ║${NC}"
echo -e "${BLUE}║  Timestamp: $(date '+%Y-%m-%d %H:%M:%S')                                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${YELLOW}📦 CONTAINER STATUS:${NC}"
echo ""

# Function to check container status
check_container() {
    local service=$1
    local emoji=$2

    status=$(docker compose ps --services --status running 2>/dev/null | grep -c "^${service}$" || echo 0)

    if [ "$status" -eq 1 ]; then
        echo -e "   ${GREEN}${CIRCLE}  ${emoji} ${service}${NC}                    ${GREEN}running${NC}"
    else
        health_status=$(docker compose ps --filter "name=${service}" --format "table {{.Status}}" 2>/dev/null | tail -1)
        if [ -z "$health_status" ]; then
            echo -e "   ${GRAY}${CIRCLE}  ${emoji} ${service}${NC}                    ${GRAY}stopped${NC}"
        else
            if [[ "$health_status" == *"unhealthy"* ]]; then
                echo -e "   ${RED}${CIRCLE}  ${emoji} ${service}${NC}                    ${RED}${health_status}${NC}"
            elif [[ "$health_status" == *"starting"* ]]; then
                echo -e "   ${YELLOW}${CIRCLE}  ${emoji} ${service}${NC}                    ${YELLOW}${health_status}${NC}"
            else
                echo -e "   ${GRAY}${CIRCLE}  ${emoji} ${service}${NC}                    ${GRAY}${health_status}${NC}"
            fi
        fi
    fi
}

check_container "postgres" "🗄️"
check_container "kafka" "📨"
check_container "zookeeper" "🔗"
check_container "debezium" "🔄"
check_container "minio" "💾"
check_container "flink-cluster" "🌊"
check_container "data-generator" "📝"
check_container "dbt-flink-ingestao" "⚙️"
check_container "dbt-duckdb-medallhao" "🎯"

echo ""
echo -e "${YELLOW}📊 DATA FLOW METRICS:${NC}"
echo ""

# Check PostgreSQL customer count
if docker compose ps --filter "name=postgres" --services 2>/dev/null | grep -q postgres; then
    customer_count=$(docker compose exec -T postgres psql -U postgres -d postgres -c "SELECT COUNT(*) FROM customers 2>/dev/null;" 2>/dev/null | grep -E '[0-9]+' | head -1 || echo "?")
    echo -e "   📝 PostgreSQL Customers:     ${BLUE}${customer_count} rows${NC}"
else
    echo -e "   📝 PostgreSQL Customers:     ${GRAY}? rows${NC}"
fi

# Check Kafka topics
if docker compose ps --filter "name=kafka" --services 2>/dev/null | grep -q kafka; then
    topic_count=$(docker compose exec -T kafka kafka-topics --bootstrap-server kafka:29092 --list 2>/dev/null | wc -l || echo "?")
    echo -e "   📨 Kafka Topics:             ${BLUE}${topic_count} topics${NC}"
else
    echo -e "   📨 Kafka Topics:             ${GRAY}? topics${NC}"
fi

echo ""
echo -e "${YELLOW}🤖 SIDECARS STATUS:${NC}"
echo ""

# Data Generator status (tail 1 misses the log line most of the time; look at recent history instead)
if docker compose logs --tail 20 data-generator 2>/dev/null | grep -qi "ciclo\|customer inserido"; then
    last_run=$(docker compose logs --tail 20 data-generator 2>/dev/null | grep -oP '\[\K[^\]]+(?=\])' | tail -1)
    echo -e "   💾 Data Generator:           ${GREEN}Running${NC}                    ${GRAY}[${last_run}]${NC}"
else
    echo -e "   💾 Data Generator:           ${YELLOW}Aguardando...${NC}"
fi

# dbt Runner status (dbt-flink-ingestao + dbt-duckdb-medallhao replaced the old dbt-runner service)
if docker compose logs --tail 20 dbt-flink-ingestao dbt-duckdb-medallhao 2>/dev/null | grep -qi "completed successfully\|done\. PASS"; then
    echo -e "   🔄 dbt Runner:               ${GREEN}Transformando${NC}"
else
    echo -e "   🔄 dbt Runner:               ${YELLOW}Aguardando...${NC}"
fi

# Debezium connector status
if docker compose ps --filter "name=debezium" --services 2>/dev/null | grep -q debezium; then
    connector_status=$(curl -s http://localhost:8083/connectors/cdc-connector/status 2>/dev/null | grep -o '"state":"[^"]*"' | cut -d'"' -f4 || echo "?")
    if [ "$connector_status" = "RUNNING" ]; then
        echo -e "   🔗 Debezium Connector:       ${GREEN}${connector_status}${NC}"
    else
        echo -e "   🔗 Debezium Connector:       ${YELLOW}${connector_status}${NC}"
    fi
else
    echo -e "   🔗 Debezium Connector:       ${GRAY}?${NC}"
fi

echo ""
echo -e "${YELLOW}🔧 QUICK COMMANDS:${NC}"
echo ""
echo -e "   ${BLUE}make logs-generator${NC}       # Ver logs do data generator em tempo real"
echo -e "   ${BLUE}make logs-dbt-runner${NC}      # Ver logs do dbt runner em tempo real"
echo -e "   ${BLUE}make kafka-topics${NC}         # Ver tópicos Kafka"
echo -e "   ${BLUE}make psql${NC}                 # Acessar PostgreSQL"
echo -e "   ${BLUE}make health${NC}               # Verificar saúde completa da stack"
echo -e "   ${BLUE}docker compose logs -f${NC}    # Ver logs de todos os serviços"

echo ""
echo -e "${GRAY}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GRAY}Atualizar: ctrl+l para limpar e executar novamente${NC}"
echo -e "${GRAY}═══════════════════════════════════════════════════════════════════════════════${NC}"
