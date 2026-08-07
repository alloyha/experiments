.PHONY: help start stop health clean logs register-connector logs-postgres logs-kafka logs-debezium logs-flink

help:
	@echo "CDC Local Stack - Makefile Commands"
	@echo ""
	@echo "  make start              - Inicia a stack completa"
	@echo "  make stop               - Para a stack"
	@echo "  make health             - Verifica saúde dos serviços"
	@echo "  make clean              - Para e remove volumes (reset)"
	@echo "  make register-connector - Registra o connector Debezium"
	@echo "  make status-connector   - Verifica status do connector"
	@echo "  make logs               - Mostra logs de todos os serviços"
	@echo "  make logs-postgres      - Logs do PostgreSQL"
	@echo "  make logs-kafka         - Logs do Kafka"
	@echo "  make logs-debezium      - Logs do Debezium"
	@echo "  make logs-flink         - Logs do Flink"
	@echo "  make psql               - Conecta ao PostgreSQL"
	@echo "  make kafka-topics       - Lista tópicos Kafka"
	@echo "  make kafka-consume      - Consome tópico (use TOPIC=customers)"
	@echo "  make flink-shell        - Acessa shell do Flink"
	@echo "  make dbt-run            - Executa models dbt"
	@echo "  make dbt-test           - Executa testes dbt"
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

.DEFAULT_GOAL := help
