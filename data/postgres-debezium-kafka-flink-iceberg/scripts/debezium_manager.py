#!/usr/bin/env python3
"""
CDC Stack Management Tool
Facilita configuração e monitoramento da stack CDC local
"""

import requests
import json
import sys
import time
from typing import Dict, Optional
import argparse

DEBEZIUM_URL = "http://localhost:8083"
POSTGRES_CONNECTOR = "postgres-cdc-connector"

class DebeziumManager:
    """Gerenciar conectores Debezium"""
    
    def __init__(self, base_url: str = DEBEZIUM_URL):
        self.base_url = base_url
    
    def register_connector(self, config_file: str) -> bool:
        """Registrar um novo connector"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            response = requests.post(
                f"{self.base_url}/connectors",
                json=config,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 201:
                print(f"✓ Connector '{config['name']}' registrado com sucesso")
                return True
            else:
                print(f"✗ Erro ao registrar connector: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Erro: {e}")
            return False
    
    def list_connectors(self) -> Optional[list]:
        """Listar todos os conectores"""
        try:
            response = requests.get(f"{self.base_url}/connectors")
            if response.status_code == 200:
                connectors = response.json()
                print(f"✓ Conectores encontrados: {connectors}")
                return connectors
            else:
                print(f"✗ Erro ao listar conectores: {response.text}")
                return None
        except Exception as e:
            print(f"✗ Erro: {e}")
            return None
    
    def get_connector_status(self, name: str = POSTGRES_CONNECTOR) -> Optional[Dict]:
        """Obter status de um connector"""
        try:
            response = requests.get(f"{self.base_url}/connectors/{name}/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Status do '{name}':")
                print(json.dumps(status, indent=2))
                return status
            else:
                print(f"✗ Connector '{name}' não encontrado")
                return None
        except Exception as e:
            print(f"✗ Erro: {e}")
            return None
    
    def delete_connector(self, name: str = POSTGRES_CONNECTOR) -> bool:
        """Deletar um connector"""
        try:
            response = requests.delete(f"{self.base_url}/connectors/{name}")
            if response.status_code == 204:
                print(f"✓ Connector '{name}' deletado com sucesso")
                return True
            else:
                print(f"✗ Erro ao deletar: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Erro: {e}")
            return False
    
    def pause_connector(self, name: str = POSTGRES_CONNECTOR) -> bool:
        """Pausar um connector"""
        try:
            response = requests.put(f"{self.base_url}/connectors/{name}/pause")
            if response.status_code == 202:
                print(f"✓ Connector '{name}' pausado")
                return True
            else:
                print(f"✗ Erro ao pausar: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Erro: {e}")
            return False
    
    def resume_connector(self, name: str = POSTGRES_CONNECTOR) -> bool:
        """Resumir um connector"""
        try:
            response = requests.put(f"{self.base_url}/connectors/{name}/resume")
            if response.status_code == 202:
                print(f"✓ Connector '{name}' resumido")
                return True
            else:
                print(f"✗ Erro ao resumir: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Erro: {e}")
            return False
    
    def wait_for_ready(self, timeout: int = 60) -> bool:
        """Aguardar Debezium estar pronto"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = requests.get(f"{self.base_url}/")
                if response.status_code == 200:
                    print("✓ Debezium está pronto")
                    return True
            except:
                pass
            time.sleep(2)
        
        print(f"✗ Timeout aguardando Debezium ({timeout}s)")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="CDC Stack Management Tool"
    )
    parser.add_argument(
        "command",
        choices=[
            "register",
            "list",
            "status",
            "delete",
            "pause",
            "resume",
            "wait"
        ],
        help="Comando a executar"
    )
    parser.add_argument(
        "--config",
        default="config/debezium/postgres-source.json",
        help="Arquivo de configuração do connector"
    )
    parser.add_argument(
        "--name",
        default=POSTGRES_CONNECTOR,
        help="Nome do connector"
    )
    
    args = parser.parse_args()
    
    manager = DebeziumManager()
    
    if args.command == "register":
        success = manager.register_connector(args.config)
        sys.exit(0 if success else 1)
    
    elif args.command == "list":
        manager.list_connectors()
    
    elif args.command == "status":
        manager.get_connector_status(args.name)
    
    elif args.command == "delete":
        success = manager.delete_connector(args.name)
        sys.exit(0 if success else 1)
    
    elif args.command == "pause":
        success = manager.pause_connector(args.name)
        sys.exit(0 if success else 1)
    
    elif args.command == "resume":
        success = manager.resume_connector(args.name)
        sys.exit(0 if success else 1)
    
    elif args.command == "wait":
        success = manager.wait_for_ready()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
