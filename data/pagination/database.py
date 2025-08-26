import os
import pandas as pd
import time
import multiprocessing
from contextlib import contextmanager
import logging
import gc

import psycopg2
from psycopg2 import OperationalError
from psycopg2.errors import InvalidCatalogName
from psycopg2 import pool
from faker import Faker

from config import DB_CONFIG, BATCH_SIZE, TOTAL_ROWS, logger

fake = Faker()

gc.collect()

# Context manager to handle database connection and cursor using the pool
@contextmanager
def get_db_connection():
    """Create a local connection pool in each process."""
    # Local connection pool within the process
    local_db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, **DB_CONFIG)  # Min 1, Max 20 connections
    conn = local_db_pool.getconn()
    cur = conn.cursor()
    try:
        yield cur  # Yield the cursor to the calling code
    finally:
        cur.close()
        local_db_pool.putconn(conn)  # Return the connection to the pool

def create_database():
    dbname = DB_CONFIG['dbname']
    logger.info(f"📦 Verificando se o banco '{dbname}' existe...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        logger.info(f"✅ Banco '{dbname}' já existe.")
    except OperationalError as e:
        if f'database "{dbname}" does not exist' in str(e):
            logger.warning(f"⚠️ Banco '{dbname}' não encontrado. Criando...")
            conn = psycopg2.connect(**DB_CONFIG)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(f"CREATE DATABASE {dbname};")
            cur.close()
            conn.close()
            logger.info(f"✅ Banco '{dbname}' criado com sucesso.")
        else:
            raise  # Repassa erro desconhecido

def create_table(cur):
    cur.execute("""
        DROP TABLE IF EXISTS users;
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP
        );
    """)

def create_indexes(cur):
    # Create indexes for better query performance
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_id ON users(id);
        CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
    """)
    logger.info("✅ Índices criados: id, created_at")

def generate_batch(start_index, batch_size):
    """Generate a batch of fake data."""
    return [(fake.name(), fake.date_time_between(start_date='-2y', end_date='now')) for _ in range(batch_size)]

def insert_batch(start_index):
    """Insert a batch of data into the database."""
    try:
        with get_db_connection() as cur:
            data = generate_batch(start_index, BATCH_SIZE)
            cur.executemany("INSERT INTO users (name, created_at) VALUES (%s, %s);", data)
            logger.info(f"✔️ Inseridos: {start_index + BATCH_SIZE}")
    except Exception as e:
        logger.error(f"❌ Erro ao inserir batch iniciado em {start_index}: {e}")

def populate_database():
    start = time.time()
    create_database()

    with get_db_connection() as cur:
        logger.info("🧨 Dropando e criando tabela...")
        create_table(cur)
        create_indexes(cur)

    logger.info(f"🚀 Inserindo {N} registros em batches de {BATCH_SIZE}...")
    num_batches = N // BATCH_SIZE
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Using CPU cores for parallelism

    # Distribute the insertion tasks across processes
    pool.map(insert_batch, [i * BATCH_SIZE for i in range(num_batches)])

    pool.close()
    pool.join()

    logger.info(f"✅ População finalizada em {time.time() - start:.2f} segundos.")
