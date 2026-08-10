#!/usr/bin/env python3
"""
One-Shot Setup
Sobe a stack completa + sidecars, aguarda os serviços ficarem saudáveis
e registra o connector Debezium automaticamente. Rodar uma única vez.
"""

import subprocess
import sys
import time

import requests

PROJECT_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip() or "."

DEBEZIUM_URL = "http://localhost:8083"
CONNECTOR_CONFIG = "config/debezium/postgres-source.json"


def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def wait_for_debezium(timeout=120):
    print("Aguardando Debezium ficar pronto...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{DEBEZIUM_URL}/connectors", timeout=3)
            if r.status_code == 200:
                print("✓ Debezium está pronto")
                return True
        except Exception:
            pass
        time.sleep(3)
    print("✗ Timeout aguardando Debezium")
    return False


def register_connector():
    print("Registrando connector Debezium...")
    try:
        import json
        with open(CONNECTOR_CONFIG) as f:
            config = json.load(f)

        r = requests.post(
            f"{DEBEZIUM_URL}/connectors",
            json=config,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code == 201:
            print(f"✓ Connector '{config['name']}' registrado com sucesso")
            return True
        if r.status_code == 409:
            print("✓ Connector já estava registrado")
            return True
        print(f"✗ Erro ao registrar connector: {r.text}")
        return False
    except Exception as e:
        print(f"✗ Erro ao registrar connector: {e}")
        return False


def main():
    print("\n" + "=" * 80)
    print("🚀 CDC STACK - ONE-SHOT SETUP")
    print("=" * 80 + "\n")

    print("Passo 1/4: Subindo containers (build + sidecars)...")
    if not run("docker compose up -d --build"):
        print("✗ Falha ao subir a stack")
        sys.exit(1)

    print("\nPasso 2/4: Aguardando serviços ficarem saudáveis...")
    time.sleep(15)

    print("\nPasso 3/4: Aguardando e registrando connector Debezium...")
    if wait_for_debezium():
        register_connector()

    print("\nPasso 4/4: Setup concluído!")
    print("\n" + "=" * 80)
    print("✓ Stack rodando com sidecars automáticos (data-generator + dbt-runner)")
    print("  Data Generator insere dados a cada ~10s")
    print("  dbt Runner transforma dados a cada ~60s")
    print("\nPróximo passo, em outro terminal:")
    print("  make monitor")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
