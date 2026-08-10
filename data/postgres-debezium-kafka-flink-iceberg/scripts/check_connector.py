#!/usr/bin/env python3
"""
Check Connector
Verificação rápida e independente do status do connector Debezium
(usado por validate_sidecars.py / health checks).
"""

import json
import os
import sys

import requests

DEBEZIUM_URL = os.environ.get("DEBEZIUM_URL", "http://localhost:8083")
CONNECTOR_NAME = os.environ.get("CONNECTOR_NAME", "postgres-cdc-connector")


def main():
    try:
        response = requests.get(f"{DEBEZIUM_URL}/connectors/{CONNECTOR_NAME}/status", timeout=5)
    except Exception as e:
        print(f"✗ Não foi possível contatar o Debezium em {DEBEZIUM_URL}: {e}")
        sys.exit(1)

    if response.status_code != 200:
        print(f"✗ Connector '{CONNECTOR_NAME}' não encontrado (HTTP {response.status_code})")
        sys.exit(1)

    status = response.json()
    connector_state = status.get("connector", {}).get("state", "UNKNOWN")
    tasks = status.get("tasks", [])
    task_states = [task.get("state", "UNKNOWN") for task in tasks]

    print(f"Connector: {CONNECTOR_NAME}")
    print(f"  State:  {connector_state}")
    print(f"  Tasks:  {task_states}")

    healthy = connector_state == "RUNNING" and all(s == "RUNNING" for s in task_states)

    if healthy:
        print("✓ Connector saudável")
        sys.exit(0)
    else:
        print("✗ Connector com problemas")
        print(json.dumps(status, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
