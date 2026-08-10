#!/bin/bash

# Referência rápida de comandos do CDC Stack
# Uso: ./scripts/quick-ref.sh (ou 'source' para manter no shell atual)

cat <<'EOF'
================================================================================
📖 CDC STACK - REFERÊNCIA RÁPIDA
================================================================================

🚀 Ciclo de vida
  make start                Sobe a stack (+ sidecars)
  make stop                 Para a stack
  make clean                Para e remove volumes (reset completo)
  make health               Verifica saúde dos serviços

🔌 Debezium
  make register-connector   Registra o connector Postgres → Kafka
  make status-connector     Mostra status do connector

💻 Acesso direto
  make psql                 Conecta ao PostgreSQL (cdc_db)
  make kafka-topics         Lista tópicos Kafka
  TOPIC=customers make kafka-consume   Consome um tópico

🔄 dbt
  make dbt-run              Executa os models
  make dbt-test             Executa os testes
  make dbt-docs             Gera a documentação

🤖 Sidecars (Docker)
  make build-sidecars       Reconstrói as imagens dos sidecars
  make logs-generator       Logs do data generator em tempo real
  make logs-dbt-runner      Logs do dbt runner em tempo real
  make stop-sidecars        Para só os sidecars

📌 Alternativa via cron (sem Docker)
  make setup-cron           Instala os cron jobs
  make remove-cron          Remove os cron jobs
  make list-cron            Lista os cron jobs instalados
  make cron-logs            Acompanha os logs dos cron jobs
  make validate-sidecars    Valida se tudo está funcionando

📊 Monitoramento
  make monitor              Dashboard em tempo real
  docker compose logs -f    Logs de todos os serviços
================================================================================
EOF
