.PHONY: help setup monitor start stop health clean logs register-connector logs-postgres logs-kafka logs-debezium logs-flink setup-cron remove-cron list-cron cron-logs validate-sidecars logs-generator logs-dbt-runner build-sidecars

help:
	@echo "CDC Local Stack - Makefile Commands"
	@echo ""
	@echo "🚀 SETUP RECOMENDADO (Tudo em Docker, zero dependências no host):"
	@echo "  make start              - Inicia stack + sidecars (tudo incluso)"
	@echo "  make register-connector - Registra o connector Debezium"
	@echo "  make monitor            - Monitora stack em tempo real"
	@echo ""
	@echo "🔧 CONTROLE DA STACK:"
	@echo "  make stop               - Para a stack"
	@echo "  make health             - Verifica saúde dos serviços"
	@echo "  make clean              - Para e remove volumes (reset)"
	@echo "  make build-sidecars     - Reconstrói imagens dos sidecars"
	@echo ""
	@echo "💻 ACESSO AOS SERVIÇOS:"
	@echo "  make psql               - Conecta ao PostgreSQL"
	@echo "  make kafka-topics       - Lista tópicos Kafka"
	@echo "  make kafka-consume      - Consome tópico (use TOPIC=customers)"
	@echo ""
	@echo "🔄 TRANSFORMAÇÕES:"
	@echo "  make dbt-run            - Executa models dbt"
	@echo "  make dbt-test           - Executa testes dbt"
	@echo "  make insert-test-data   - Insere dados de teste"
	@echo ""
	@echo "📊 DEBEZIUM & CONNECTOR:"
	@echo "  make status-connector   - Verifica status do connector"
	@echo ""
	@echo "📋 LOGS & MONITORAMENTO:"
	@echo "  make logs               - Mostra logs de todos os serviços"
	@echo "  make logs-postgres      - Logs do PostgreSQL"
	@echo "  make logs-kafka         - Logs do Kafka"
	@echo "  make logs-debezium      - Logs do Debezium"
	@echo "  make logs-flink         - Logs do Flink"
	@echo "  make logs-generator     - Monitora Data Generator (novo dados cada 10s)"
	@echo "  make logs-dbt-runner    - Monitora dbt Runner (transforma cada 60s)"
	@echo ""
	@echo "📌 CRON ALTERNATIVO (Se preferir não usar Docker para sidecars):"
	@echo "  make setup-cron         - Instala cron jobs para data generator + dbt"
	@echo "  make remove-cron        - Remove cron jobs"
	@echo "  make cron-logs          - Monitora logs dos cron jobs"
	@echo ""

start:
	@chmod +x scripts/*.sh
	@./scripts/start-all.sh

stop:
	@./scripts/stop-all.sh

health:
	@chmod +x scripts/health-check.sh
	@./scripts/health-check.sh

clean:
	@docker-compose down -v
	@echo "Stack removida completamente"

register-connector:
	@python3 scripts/debezium_manager.py register
	@echo "Aguardando connector inicializar..."
	@sleep 5
	@python3 scripts/debezium_manager.py status

status-connector:
	@python3 scripts/debezium_manager.py status

logs:
	@docker-compose logs -f

logs-postgres:
	@docker-compose logs -f postgres

logs-kafka:
	@docker-compose logs -f kafka

logs-debezium:
	@docker-compose logs -f debezium

logs-flink:
	@docker-compose logs -f dbt-flink

logs-generator:
	@docker-compose logs -f data-generator

logs-dbt-runner:
	@docker-compose logs -f dbt-runner

psql:
	@docker exec -it cdc-postgres psql -U postgres -d cdc_db

kafka-topics:
	@docker exec cdc-kafka kafka-topics --list --bootstrap-server kafka:29092

kafka-consume:
	@docker exec cdc-kafka kafka-console-consumer \
		--bootstrap-server kafka:29092 \
		--topic $(TOPIC) \
		--from-beginning

flink-shell:
	@docker exec -it cdc-dbt-flink /opt/flink/bin/flink

dbt-run:
	@docker exec cdc-dbt-flink dbt run

dbt-test:
	@docker exec cdc-dbt-flink dbt test

dbt-docs:
	@docker exec cdc-dbt-flink dbt docs generate

insert-test-data:
	@docker exec cdc-postgres psql -U postgres -d cdc_db -c \
		"INSERT INTO cdc_source.customers (name, email, country) VALUES ('Test User', 'test@example.com', 'Brazil');"
	@echo "Dados de teste inseridos"

stop-sidecars:
	@docker-compose stop data-generator dbt-runner
	@echo "Sidecars parados"

build-sidecars:
	@docker-compose build data-generator dbt-runner
	@echo "✓ Sidecars reconstruídos com sucesso"

# 🎯 ONE-SHOT SETUP & MONITOR (Alternativa com dashboard em tempo real)
setup:
	@chmod +x scripts/one_shot_setup.py scripts/monitor.py
	@python3 scripts/one_shot_setup.py

monitor:
	@chmod +x scripts/monitor.py
	@python3 scripts/monitor.py

# 🤖 SIDECARS NO HOST (Alternativa: usar Python scripts diretamente sem Docker)
run-data-generator:
	@chmod +x scripts/data_generator.py
	@python3 scripts/data_generator.py

run-dbt-runner:
	@chmod +x scripts/dbt_runner.py
	@python3 scripts/dbt_runner.py

# 📌 CRON JOBS (Alternativa: execução via cron ao invés de Docker)
setup-cron:
	@chmod +x scripts/setup_cron.py
	@python3 scripts/setup_cron.py install
	@echo "✓ Cron jobs instalados. Execute: make cron-logs"

remove-cron:
	@chmod +x scripts/setup_cron.py
	@python3 scripts/setup_cron.py remove
	@echo "✓ Cron jobs removidos"

list-cron:
	@chmod +x scripts/setup_cron.py
	@python3 scripts/setup_cron.py list

cron-logs:
	@tail -f logs/cron.log logs/data_generator.log logs/dbt_runner.log 2>/dev/null || echo "Logs not found. Run 'make setup-cron' first"

validate-sidecars:
	@chmod +x scripts/validate_sidecars.py
	@python3 scripts/validate_sidecars.py

.DEFAULT_GOAL := help
