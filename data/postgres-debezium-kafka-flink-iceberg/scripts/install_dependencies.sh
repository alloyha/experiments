#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Instalando dependências Python (sidecars via cron/host) ===${NC}"

pip3 install psycopg2-binary names python-dotenv dbt-core

echo -e "${GREEN}✓ Dependências instaladas${NC}"
echo "Use 'make setup-cron' para instalar os cron jobs, ou 'make run-data-generator' / 'make run-dbt-runner' para rodar os sidecars diretamente no host."
