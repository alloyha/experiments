#!/usr/bin/env python3
"""
Iceberg Maintenance - Expiracao de snapshots (Bronze + Silver)

Por que isso existe: Bronze faz um commit Iceberg a cada checkpoint do Flink
(30s), e Silver faz um INSERT OVERWRITE do estado inteiro a cada ciclo (60s).
O Iceberg, por padrao, mantem TODOS os snapshots antigos indefinidamente (para
suportar time-travel) -- isso significa que mesmo uma tabela logicamente
pequena (poucas centenas de linhas) acumula milhares de snapshots com o tempo,
sem nunca ser limpa. Confirmado em producao: slv_customers chegou a 85
snapshots em poucas horas de execucao continua.

O que este script faz: para cada tabela em bronze/silver, expira (remove os
PONTEIROS de metadado de) todo snapshot mais antigo que ICEBERG_SNAPSHOT_RETENTION_HOURS,
sempre preservando o snapshot atual (current) e qualquer branch/tag.

Limitacao conhecida (pyiceberg 0.11.1): expire_snapshots() remove apenas os
ponteiros de metadado -- reduz o overhead de metadado/planejamento de query,
mas NAO deleta fisicamente os arquivos Parquet dos snapshots expirados do
MinIO (essa e a acao "remove_orphan_files"/compactacao, que o Spark tem via
procedures Iceberg mas o pyiceberg ainda nao implementa nesta versao). Ou
seja: isso resolve o crescimento de METADADO, mas nao o de ARQUIVOS DE DADOS
orfaos -- vale deixar isso explicito em vez de prometer mais do que entrega.
"""

import os
import sys
import logging
from datetime import datetime, timedelta, timezone

from pyiceberg.catalog.sql import SqlCatalog

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CATALOG_URI = os.environ.get(
    'ICEBERG_CATALOG_URI',
    'postgresql+psycopg2://postgres:postgres@postgres:5432/iceberg_catalog',
)
WAREHOUSE = os.environ.get('ICEBERG_WAREHOUSE', 's3://iceberg-warehouse/')
S3_ENDPOINT = os.environ.get('MINIO_ENDPOINT_URL', 'http://minio:9000')
S3_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
S3_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
RETENTION_HOURS = float(os.environ.get('ICEBERG_SNAPSHOT_RETENTION_HOURS', '6'))
NAMESPACES = ['bronze', 'silver']


def get_catalog():
    return SqlCatalog(
        'iceberg_catalog',
        **{
            'uri': CATALOG_URI,
            'warehouse': WAREHOUSE,
            's3.endpoint': S3_ENDPOINT,
            's3.access-key-id': S3_ACCESS_KEY,
            's3.secret-access-key': S3_SECRET_KEY,
            's3.path-style-access': 'true',
        },
    )


def run_once():
    catalog = get_catalog()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RETENTION_HOURS)
    total_tables = 0
    total_expired = 0
    had_error = False

    for namespace in NAMESPACES:
        try:
            identifiers = catalog.list_tables(namespace)
        except Exception as e:
            logger.warning(f"Namespace '{namespace}' inacessivel neste ciclo ({e})")
            continue

        for identifier in identifiers:
            total_tables += 1
            table_name = '.'.join(identifier)
            try:
                table = catalog.load_table(identifier)
                before = len(list(table.snapshots()))
                table.maintenance.expire_snapshots().older_than(cutoff).commit()
                table.refresh()
                after = len(list(table.snapshots()))
                expired = before - after
                total_expired += expired
                if expired > 0:
                    logger.info(f"{table_name}: {before} -> {after} snapshots ({expired} expirados)")
                else:
                    logger.info(f"{table_name}: {before} snapshots (nenhum elegivel para expiracao)")
            except Exception as e:
                had_error = True
                logger.error(f"Falha ao expirar snapshots de {table_name}: {e}")

    logger.info(
        f"Ciclo concluido: {total_tables} tabelas verificadas, "
        f"{total_expired} snapshots expirados no total (retencao: {RETENTION_HOURS}h)"
    )
    return had_error


def main():
    logger.info("=" * 80)
    logger.info(f"Iceberg Maintenance - retencao: {RETENTION_HOURS}h, namespaces: {NAMESPACES}")
    logger.info("=" * 80)
    try:
        had_error = run_once()
        return 1 if had_error else 0
    except Exception as e:
        logger.error(f"Erro durante execucao: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
