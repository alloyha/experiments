#!/usr/bin/env python3
"""
Data Generator - Single Execution
Executa um ciclo de dados e sai (para ser usado com cron)
"""

import psycopg2
import random
import logging
from datetime import datetime
import names
from decimal import Decimal
import os
import sys

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'data_generator.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, host=None, user=None, password=None, dbname=None):
        # Use environment variables or defaults
        self.host = host or os.environ.get('POSTGRES_HOST', 'localhost')
        self.user = user or os.environ.get('POSTGRES_USER', 'postgres')
        self.password = password or os.environ.get('POSTGRES_PASSWORD', 'postgres')
        self.dbname = dbname or os.environ.get('POSTGRES_DB', 'cdc_db')
        self.conn = None
        self.customer_ids = []
        self.product_ids = []
        
    def connect(self):
        """Conectar ao PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                dbname=self.dbname,
                connect_timeout=5
            )
            logger.info(f"✓ Conectado ao PostgreSQL ({self.host})")
            return True
        except Exception as e:
            logger.error(f"✗ Erro ao conectar: {e}")
            return False
    
    def disconnect(self):
        """Desconectar do PostgreSQL"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Desconectado do PostgreSQL")
    
    def execute_query(self, query, params=None):
        """Executar query e fazer commit"""
        try:
            with self.conn.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                self.conn.commit()
                return True
        except Exception as e:
            self.conn.rollback()
            logger.error(f"✗ Erro na query: {e}")
            return False
    
    def get_customer_ids(self):
        """Obter IDs de clientes existentes"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT id FROM cdc_source.customers ORDER BY id")
                self.customer_ids = [row[0] for row in cursor.fetchall()]
                return self.customer_ids
        except Exception as e:
            logger.error(f"✗ Erro ao obter IDs: {e}")
            return []
    
    def insert_customer(self):
        """Inserir novo customer"""
        name = names.get_full_name()
        email = f"{name.lower().replace(' ', '.')}@example.com"
        phone = f"+55{random.randint(11, 99)}{random.randint(987654321, 999999999)}"
        country = random.choice(['Brazil', 'USA', 'Canada', 'UK', 'Germany', 'France', 'Japan'])
        
        query = """
            INSERT INTO cdc_source.customers (name, email, phone, country)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (name, email, phone, country))
                customer_id = cursor.fetchone()[0]
                self.conn.commit()
                logger.info(f"➕ Customer inserido: ID={customer_id}, Name={name}")
                self.customer_ids.append(customer_id)
                return customer_id
        except Exception as e:
            self.conn.rollback()
            logger.error(f"✗ Erro ao inserir: {e}")
            return None
    
    def update_customer(self):
        """Atualizar customer aleatório"""
        if not self.customer_ids:
            return
        
        customer_id = random.choice(self.customer_ids)
        new_country = random.choice(['Brazil', 'USA', 'Canada', 'UK', 'Germany', 'France', 'Japan'])
        new_phone = f"+55{random.randint(11, 99)}{random.randint(987654321, 999999999)}"
        
        query = """
            UPDATE cdc_source.customers 
            SET country = %s, phone = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        
        if self.execute_query(query, (new_country, new_phone, customer_id)):
            logger.info(f"📝 Customer atualizado: ID={customer_id}, Country={new_country}")
    
    def get_product_ids(self):
        """Obter IDs de produtos existentes"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT id FROM cdc_source.products ORDER BY id")
                self.product_ids = [row[0] for row in cursor.fetchall()]
                return self.product_ids
        except Exception as e:
            logger.error(f"✗ Erro ao obter IDs de produtos: {e}")
            return []

    def insert_product(self):
        """Inserir novo produto"""
        categories = ['Electronics', 'Furniture', 'Apparel', 'Groceries', 'Toys']
        category = random.choice(categories)
        adjective = random.choice(['Pro', 'Lite', 'Max', 'Plus', 'Mini', 'Ultra'])
        noun = random.choice(['Widget', 'Gadget', 'Gizmo', 'Device', 'Kit'])
        name = f"{noun} {adjective} {random.randint(100, 999)}"
        price = round(Decimal(random.uniform(9.90, 499.90)), 2)

        query = """
            INSERT INTO cdc_source.products (name, category, price)
            VALUES (%s, %s, %s)
            RETURNING id
        """

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (name, category, price))
                product_id = cursor.fetchone()[0]
                self.conn.commit()
                logger.info(f"📦 Product inserido: ID={product_id}, Name={name}, Price={price}")
                self.product_ids.append(product_id)
                return product_id
        except Exception as e:
            self.conn.rollback()
            logger.error(f"✗ Erro ao inserir product: {e}")
            return None

    def update_product(self):
        """Atualizar preço/categoria de um produto aleatório (gera historico SCD2)"""
        if not self.product_ids:
            return

        product_id = random.choice(self.product_ids)
        # Reajuste de preco entre -15% e +25%
        factor = Decimal(random.uniform(0.85, 1.25))

        query = """
            UPDATE cdc_source.products
            SET price = ROUND(price * %s, 2), updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """

        if self.execute_query(query, (factor, product_id)):
            logger.info(f"💲 Product atualizado: ID={product_id}, Fator={factor:.2f}")

    def insert_order(self):
        """Inserir novo order"""
        if not self.customer_ids:
            return
        
        customer_id = random.choice(self.customer_ids)
        total_amount = round(Decimal(random.uniform(10, 999)), 2)
        status = random.choice(['PENDING', 'PROCESSING', 'COMPLETED', 'SHIPPED', 'CANCELLED'])
        
        query = """
            INSERT INTO cdc_source.orders (customer_id, total_amount, status)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (customer_id, total_amount, status))
                order_id = cursor.fetchone()[0]
                self.conn.commit()
                logger.info(f"🛒 Order inserido: ID={order_id}, Customer={customer_id}, Amount={total_amount}")
                return order_id
        except Exception as e:
            self.conn.rollback()
            logger.error(f"✗ Erro ao inserir order: {e}")
            return None
    
    def delete_customer(self):
        """Deletar customer aleatório"""
        if len(self.customer_ids) <= 3:  # Manter alguns registros
            return
        
        customer_id = random.choice(self.customer_ids)
        
        # Primeiro deletar orders
        query_orders = "DELETE FROM cdc_source.orders WHERE customer_id = %s"
        self.execute_query(query_orders, (customer_id,))
        
        # Depois deletar customer
        query_customer = "DELETE FROM cdc_source.customers WHERE id = %s"
        if self.execute_query(query_customer, (customer_id,)):
            logger.info(f"🗑️  Customer deletado: ID={customer_id}")
            self.customer_ids.remove(customer_id)
    
    def run_cycle(self):
        """Executar um ciclo de operações"""
        logger.info("="*80)
        logger.info(f"Iniciando ciclo de dados...")
        
        # 70% chance de inserir customer
        if random.random() < 0.7:
            self.insert_customer()

        # 50% chance de inserir order
        if random.random() < 0.5:
            self.insert_order()

        # 20% chance de atualizar customer
        if random.random() < 0.2:
            self.update_customer()

        # 10% chance de deletar customer
        if random.random() < 0.1:
            self.delete_customer()

        # 15% chance de inserir produto
        if random.random() < 0.15:
            self.insert_product()

        # 25% chance de reajustar preco/produto (gera historico SCD2 em dim_products)
        if random.random() < 0.25:
            self.update_product()
        
        logger.info("✓ Ciclo concluído")

def main():
    logger.info("="*80)
    logger.info("🚀 DATA GENERATOR - Single Execution")
    logger.info("="*80)
    
    generator = DataGenerator()
    
    # Conectar
    if not generator.connect():
        logger.error("✗ Falha ao conectar ao PostgreSQL")
        return 1
    
    try:
        # Obter clientes e produtos existentes
        generator.get_customer_ids()
        generator.get_product_ids()
        logger.info(f"✓ {len(generator.customer_ids)} clientes e {len(generator.product_ids)} produtos encontrados no banco")

        # Executar um ciclo
        generator.run_cycle()
        logger.info("✓ Execução concluída com sucesso")
        return 0
    
    except Exception as e:
        logger.error(f"✗ Erro durante execução: {e}")
        return 1
    
    finally:
        generator.disconnect()

if __name__ == "__main__":
    sys.exit(main())
