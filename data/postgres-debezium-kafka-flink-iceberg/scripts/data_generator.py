#!/usr/bin/env python3
"""
Data Generator Sidecar
Insere/atualiza/deleta dados no PostgreSQL continuamente (a cada 10 segundos)
"""

import psycopg2
import random
import sys
import time
from decimal import Decimal
import names
import os

# Garante que os logs apareçam em tempo real em 'docker logs -f' (stdout
# não-tty é bufferizado por bloco por padrão, não por linha)
sys.stdout.reconfigure(line_buffering=True)


class DataGenerator:
    def __init__(self, host=None, user=None, password=None, dbname=None):
        # Use environment variables or defaults
        self.host = host or os.environ.get('POSTGRES_HOST', 'localhost')
        self.user = user or os.environ.get('POSTGRES_USER', 'postgres')
        self.password = password or os.environ.get('POSTGRES_PASSWORD', 'postgres')
        self.dbname = dbname or os.environ.get('POSTGRES_DB', 'cdc_db')
        self.conn = None
        self.customer_ids = []

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
            print(f"✓ Conectado ao PostgreSQL ({self.host})")
            return True
        except Exception as e:
            print(f"✗ Erro ao conectar: {e}")
            return False

    def disconnect(self):
        """Desconectar do PostgreSQL"""
        if self.conn:
            self.conn.close()
            print("✓ Desconectado do PostgreSQL")

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
            print(f"✗ Erro na query: {e}")
            return False

    def get_customer_ids(self):
        """Obter IDs de clientes existentes"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT id FROM cdc_source.customers ORDER BY id")
                self.customer_ids = [row[0] for row in cursor.fetchall()]
                return self.customer_ids
        except Exception as e:
            print(f"✗ Erro ao obter IDs: {e}")
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
                print(f"➕ Customer inserido: ID={customer_id}, Name={name}")
                self.customer_ids.append(customer_id)
                return customer_id
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Erro ao inserir: {e}")
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
            print(f"📝 Customer atualizado: ID={customer_id}, Country={new_country}")

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
                print(f"🛒 Order inserido: ID={order_id}, Customer={customer_id}, Amount={total_amount}")
                return order_id
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Erro ao inserir order: {e}")
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
            print(f"🗑️  Customer deletado: ID={customer_id}")
            self.customer_ids.remove(customer_id)

    def run_cycle(self):
        """Executar um ciclo de operações"""
        print("=" * 80)
        print(f"[{time.strftime('%H:%M:%S')}] Iniciando ciclo de dados...")
        print("=" * 80)

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

        print(f"✓ Ciclo concluído. Próximo em 10 segundos...")


def main():
    print("\n" + "=" * 80)
    print("🚀 DATA GENERATOR SIDECAR - CDC Demo")
    print("=" * 80)
    print("Insere/atualiza/deleta dados no PostgreSQL a cada 10 segundos\n")

    generator = DataGenerator()

    # Tentar conectar com retry
    retry_count = 0
    while not generator.connect() and retry_count < 10:
        print(f"Tentando conectar novamente em 5s... ({retry_count + 1}/10)")
        time.sleep(5)
        retry_count += 1

    if not generator.conn:
        print("✗ Falha ao conectar ao PostgreSQL após 10 tentativas")
        return

    # Obter clientes existentes
    generator.get_customer_ids()
    print(f"✓ {len(generator.customer_ids)} clientes encontrados no banco")

    # Loop contínuo a cada 10 segundos
    try:
        while True:
            generator.get_customer_ids()  # Atualizar lista de clientes
            generator.run_cycle()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n✓ Encerrando Data Generator...")
    finally:
        generator.disconnect()


if __name__ == "__main__":
    main()
