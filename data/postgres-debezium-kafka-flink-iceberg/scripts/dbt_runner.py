#!/usr/bin/env python3
"""
dbt Runner Sidecar
Executa um ciclo dbt (run, test, docs) continuamente, em intervalo configurável
"""

import subprocess
import logging
import socket
import time
from datetime import datetime
import os

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'dbt_runner.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DBTRunner:
    def __init__(self, dbt_dir=None):
        # Try env variable first, then auto-detect
        if dbt_dir is None:
            dbt_dir = os.environ.get('DBT_DIR')

        # If still none, find relative to script
        if dbt_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dbt_dir = os.path.join(os.path.dirname(script_dir), 'dbt')

        self.dbt_dir = dbt_dir
        logger.info(f"Using dbt directory: {self.dbt_dir}")

    def run_dbt(self, command="run"):
        """Executar dbt command"""
        try:
            logger.info(f"{'='*80}")
            logger.info(f"Executando: dbt {command}")
            logger.info(f"{'='*80}")

            # Executar dbt CLI diretamente (sem --select para rodar todos os modelos)
            result = subprocess.run(
                ["dbt", command],
                cwd=self.dbt_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.stdout:
                logger.info(result.stdout)

            if result.returncode == 0:
                logger.info(f"✓ dbt {command} executado com sucesso!")
                return True
            else:
                logger.error(f"✗ Erro ao executar dbt {command}:")
                if result.stderr:
                    logger.error(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ dbt {command} timeout após 5 minutos")
            return False
        except Exception as e:
            logger.error(f"✗ Erro ao executar dbt: {e}")
            return False

    def run_tests(self):
        """Executar dbt tests"""
        logger.info(f"{'='*80}")
        logger.info(f"Executando: dbt test")
        logger.info(f"{'='*80}")

        try:
            result = subprocess.run(
                ["dbt", "test"],
                cwd=self.dbt_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.stdout:
                logger.info(result.stdout)

            if result.returncode == 0:
                logger.info(f"✓ dbt tests executados com sucesso!")
                return True
            else:
                logger.warning(f"⚠️  Alguns testes falharam:")
                if result.stderr:
                    logger.warning(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ dbt test timeout após 5 minutos")
            return False
        except Exception as e:
            logger.error(f"✗ Erro ao executar testes: {e}")
            return False

    def run_docs(self):
        """Gerar documentação dbt"""
        logger.info(f"{'='*80}")
        logger.info(f"Gerando: dbt docs")
        logger.info(f"{'='*80}")

        try:
            result = subprocess.run(
                ["dbt", "docs", "generate"],
                cwd=self.dbt_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                logger.info(f"✓ Documentação gerada com sucesso!")
                return True
            else:
                logger.warning(f"⚠️  Erro ao gerar docs: {result.stderr if result.stderr else 'Unknown error'}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"✗ dbt docs timeout após 5 minutos")
            return False
        except Exception as e:
            logger.error(f"✗ Erro ao gerar docs: {e}")
            return False

    def run_cycle(self):
        """Executar ciclo completo: run, test, docs"""
        logger.info(f"\n🔄 Ciclo dbt - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executar models
        if not self.run_dbt("run"):
            logger.warning("⚠️  Pulando testes e docs devido a erro no run")
            return False

        # Executar testes
        self.run_tests()

        # Gerar documentação
        self.run_docs()

        logger.info("\n✓ Ciclo dbt concluído!\n")
        return True


def wait_for_port(host, port, label, retries=30, delay=2):
    """Aguardar até uma porta TCP responder"""
    for attempt in range(retries):
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info(f"✓ {label} está pronto!")
                return True
        except OSError:
            logger.info(f"Aguardando {label} ({attempt + 1}/{retries})...")
            time.sleep(delay)
    logger.warning(f"⚠️  Timeout aguardando {label}, seguindo mesmo assim")
    return False


def main():
    logger.info("=" * 80)
    logger.info("🚀 DBT RUNNER SIDECAR - Loop Contínuo")
    logger.info("=" * 80)

    postgres_host = os.environ.get('POSTGRES_HOST', 'postgres')
    postgres_port = int(os.environ.get('POSTGRES_PORT', 5432))
    kafka_host = os.environ.get('KAFKA_HOST', 'kafka')
    kafka_port = int(os.environ.get('KAFKA_PORT', 29092))

    wait_for_port(postgres_host, postgres_port, "PostgreSQL")
    wait_for_port(kafka_host, kafka_port, "Kafka")

    interval = int(os.environ.get('DBT_RUN_INTERVAL', '60'))
    logger.info(f"Intervalo configurado: {interval}s")

    runner = DBTRunner()

    # Aguarda um pouco antes do primeiro ciclo, para os demais serviços estabilizarem
    time.sleep(5)

    try:
        while True:
            runner.run_cycle()
            logger.info(f"Próximo ciclo em {interval}s...")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("\n✓ Encerrando dbt Runner...")


if __name__ == "__main__":
    main()
