#!/bin/bash

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Stopping CDC Stack ===${NC}"

docker-compose down

echo -e "${GREEN}=== CDC Stack Stopped ===${NC}"
echo "To remove volumes as well, run: docker-compose down -v"
