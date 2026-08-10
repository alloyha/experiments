#!/usr/bin/env python3
"""
dbt Runner - Single Execution
Executa um ciclo dbt (run, test, docs) e sai (para ser usado com cron)
"""

import subprocess
import logging
from datetime import datetime
import sys
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

def main():
    logger.info("="*80)
    logger.info("🚀 DBT RUNNER - Single Execution")
    logger.info("="*80)
    
    runner = DBTRunner()
    
    try:
        # Executar ciclo completo
        success = runner.run_cycle()
        logger.info("✓ Execução concluída com sucesso")
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"✗ Erro durante execução: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
