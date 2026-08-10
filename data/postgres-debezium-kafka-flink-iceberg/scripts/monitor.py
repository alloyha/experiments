#!/usr/bin/env python3
"""
CDC Stack Real-Time Monitor (Python)
Dashboard em tempo real do status da stack, sidecars e fluxo de dados
"""

import subprocess
import os
import time
from datetime import datetime

CONTAINERS = [
    ("cdc-postgres", "PostgreSQL"),
    ("cdc-kafka", "Kafka"),
    ("cdc-debezium", "Debezium"),
    ("cdc-data-generator", "Data Gen"),
    ("cdc-dbt-runner", "dbt Runner"),
]


def run(cmd, timeout=5):
    """Executar comando shell e retornar stdout (ou None em erro/timeout)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return None


def container_status(name):
    """Retornar 'running', 'stopped' ou 'unknown' para um container"""
    output = run(f"docker inspect -f '{{{{.State.Status}}}}' {name}")
    if not output:
        return "unknown"
    if output == "running":
        return "running"
    return output


def status_icon(status):
    if status == "running":
        return "🟢"
    if status in ("exited", "stopped", "dead"):
        return "🔴"
    return "⚪"


def print_banner():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("╔" + "=" * 78 + "╗")
    print("║" + "🚀 CDC STACK REAL-TIME MONITOR".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║  " + now.ljust(76) + "║")
    print("╚" + "=" * 78 + "╝")


def print_container_status():
    print("\n📦 CONTAINER STATUS:\n")
    for container, label in CONTAINERS:
        status = container_status(container)
        icon = status_icon(status)
        print(f"   {icon}  {label:<20} {status}")


def print_data_flow_metrics():
    print("\n\n📊 DATA FLOW METRICS:\n")

    customer_count = run(
        "docker exec cdc-postgres psql -U postgres -d cdc_db -t -c "
        "\"SELECT COUNT(*) FROM cdc_source.customers;\""
    )
    customer_count = customer_count.strip() if customer_count else "?"
    print(f"   📝 PostgreSQL Customers:     {customer_count} rows")

    topics = run("docker exec cdc-kafka kafka-topics --list --bootstrap-server kafka:29092")
    topic_count = len(topics.splitlines()) if topics else "?"
    print(f"   📨 Kafka Topics:             {topic_count} topics")


def print_sidecars_status():
    print("\n\n🤖 SIDECARS STATUS:\n")

    generator_log = run("docker logs --tail 5 cdc-data-generator")
    if generator_log and ("ciclo" in generator_log.lower() or "customer inserido" in generator_log.lower()):
        last_line = generator_log.strip().splitlines()[-1][:60]
        print(f"   💾 Data Generator:           {last_line}...")
    else:
        print("   💾 Data Generator:           Aguardando...")

    dbt_log = run("docker logs --tail 5 cdc-dbt-runner")
    if dbt_log and "dbt run" in dbt_log.lower():
        print("   🔄 dbt Runner:               Transformando")
    else:
        print("   🔄 dbt Runner:               Aguardando...")


def print_services_status():
    print("\n\n🔗 SERVICES STATUS:\n")

    connector_status = run(
        "curl -s http://localhost:8083/connectors/postgres-cdc-connector/status "
        "| grep -o '\"state\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4"
    )
    connector_status = connector_status if connector_status else "?"
    print(f"   🎯 Debezium:                 {connector_status}")


def print_footer():
    print("\n\n" + "=" * 80)
    print("📖 COMANDOS ÚTEIS:\n")
    print("   make logs-generator       # Ver logs do data generator em tempo real")
    print("   make logs-dbt-runner      # Ver logs do dbt runner em tempo real")
    print("   make kafka-topics         # Ver tópicos Kafka")
    print("   make psql                 # Acessar PostgreSQL")
    print("   make health               # Verificar saúde da stack")
    print("   Ctrl+C para sair")
    print("\n" + "=" * 80)


def render():
    os.system("cls" if os.name == "nt" else "clear")
    print_banner()
    print_container_status()
    print_data_flow_metrics()
    print_sidecars_status()
    print_services_status()
    print_footer()


def main():
    try:
        while True:
            render()
            time.sleep(5)  # Atualizar a cada 5 segundos
    except KeyboardInterrupt:
        print("\n\n✓ Monitor encerrado.")


if __name__ == "__main__":
    main()
