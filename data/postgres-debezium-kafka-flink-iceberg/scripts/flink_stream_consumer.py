#!/usr/bin/env python3
"""
Flink-inspired Stream Consumer for Iceberg
Consome CDC events do Kafka → enriquece → escreve em Iceberg
"""

import json
import logging
from datetime import datetime
from kafka import KafkaConsumer
from pyiceberg.catalog import load_catalog
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlinkStreamConsumer:
    """Consume CDC events from Kafka and write to Iceberg"""
    
    def __init__(self):
        self.setup_kafka()
        self.setup_postgres()
        self.setup_iceberg()
    
    def setup_kafka(self):
        """Configure Kafka consumer"""
        logger.info("Initializing Kafka consumer...")
        try:
            self.kafka_consumer = KafkaConsumer(
                'cdc.cdc_source.customers',
                'cdc.cdc_source.orders',
                bootstrap_servers=['kafka:29092'],
                group_id='flink-iceberg-consumer',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                max_poll_records=100
            )
            logger.info("✓ Kafka consumer ready")
        except Exception as e:
            logger.error(f"✗ Kafka error: {e}")
            raise
    
    def setup_postgres(self):
        """Configure PostgreSQL connection for data enrichment"""
        logger.info("Connecting to PostgreSQL...")
        try:
            self.pg_conn = psycopg2.connect(
                host='postgres',
                user='postgres',
                password='postgres',
                database='cdc_db'
            )
            logger.info("✓ PostgreSQL connected")
        except Exception as e:
            logger.error(f"✗ PostgreSQL error: {e}")
            raise
    
    def setup_iceberg(self):
        """Configure Iceberg catalog"""
        logger.info("Initializing Iceberg catalog...")
        try:
            os.environ["AWS_ENDPOINT_URL_S3"] = "http://minio:9000"
            os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
            os.environ["AWS_REGION"] = "us-east-1"
            os.environ["AWS_S3_USE_PATH_STYLE"] = "true"
            
            # For testing, we'll use PyIceberg if available
            try:
                from pyiceberg.catalog import load_catalog
                self.catalog = load_catalog(
                    "iceberg",
                    warehouse="s3://iceberg-warehouse",
                    s3={
                        "endpoint": "http://minio:9000",
                        "access-key-id": "minioadmin",
                        "secret-access-key": "minioadmin",
                        "path-style-access": "true"
                    }
                )
                logger.info("✓ Iceberg catalog ready (PyIceberg)")
            except:
                logger.warning("⚠️  PyIceberg not available - will use PostgreSQL for now")
                self.catalog = None
        
        except Exception as e:
            logger.error(f"✗ Iceberg error: {e}")
            self.catalog = None
    
    def enrich_event(self, event, topic):
        """Enrich Kafka event with PostgreSQL data"""
        try:
            if 'cdc_source.customers' in topic:
                return self.enrich_customer(event)
            elif 'cdc_source.orders' in topic:
                return self.enrich_order(event)
        except Exception as e:
            logger.error(f"Error enriching event: {e}")
        
        return event
    
    def enrich_customer(self, event):
        """Enrich customer events with staging data"""
        try:
            customer_id = event.get('after', {}).get('id') or event.get('id')
            if not customer_id:
                return event
            
            cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT * FROM dbt_models_staging.stg_customers WHERE id = %s",
                (customer_id,)
            )
            enriched = cursor.fetchone()
            cursor.close()
            
            if enriched:
                event['enriched'] = dict(enriched)
                event['enriched_at'] = datetime.now().isoformat()
            
            return event
        except Exception as e:
            logger.error(f"Error enriching customer: {e}")
            return event
    
    def enrich_order(self, event):
        """Enrich order events with mart data"""
        try:
            customer_id = event.get('after', {}).get('customer_id') or event.get('customer_id')
            if not customer_id:
                return event
            
            cursor = self.pg_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT total_orders, total_spent FROM dbt_models_analytics.mart_customer_orders WHERE customer_id = %s",
                (customer_id,)
            )
            enriched = cursor.fetchone()
            cursor.close()
            
            if enriched:
                event['enriched'] = dict(enriched)
                event['enriched_at'] = datetime.now().isoformat()
            
            return event
        except Exception as e:
            logger.error(f"Error enriching order: {e}")
            return event
    
    def write_to_iceberg(self, table_name, records):
        """Write records to Iceberg table"""
        if not self.catalog:
            logger.debug(f"Skipping Iceberg write (catalog not available). Table: {table_name}, Records: {len(records)}")
            return
        
        try:
            table = self.catalog.load_table(f"analytics.{table_name}")
            # Convert records to proper format
            table.append(records)
            logger.info(f"✓ Wrote {len(records)} records to {table_name}")
        except Exception as e:
            logger.error(f"✗ Error writing to Iceberg: {e}")
    
    def process_batch(self, messages):
        """Process a batch of Kafka messages"""
        customer_events = []
        order_events = []
        
        for message in messages:
            topic = message.topic
            value = message.value
            
            # Enrich event with staging/mart data
            enriched = self.enrich_event(value, topic)
            
            if 'customers' in topic:
                customer_events.append(enriched)
            elif 'orders' in topic:
                order_events.append(enriched)
            
            # Log sample
            if len(customer_events) == 1 or len(order_events) == 1:
                logger.debug(f"Event from {topic}: {json.dumps(enriched, default=str, indent=2)}")
        
        # Write batches to Iceberg (or PostgreSQL for now)
        if customer_events:
            self.write_to_postgres('iceberg_customers_stream', customer_events)
            if self.catalog:
                self.write_to_iceberg('customers_stream', customer_events)
        
        if order_events:
            self.write_to_postgres('iceberg_orders_stream', order_events)
            if self.catalog:
                self.write_to_iceberg('orders_stream', order_events)
    
    def write_to_postgres(self, table_name, records):
        """Temporary: write to PostgreSQL table for validation"""
        try:
            cursor = self.pg_conn.cursor()
            
            # Create table if not exists
            if 'customer' in table_name:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS public.{table_name} (
                        id SERIAL PRIMARY KEY,
                        event_data JSONB,
                        enriched_data JSONB,
                        processed_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            else:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS public.{table_name} (
                        id SERIAL PRIMARY KEY,
                        event_data JSONB,
                        enriched_data JSONB,
                        processed_at TIMESTAMP DEFAULT NOW()
                    )
                """)
            
            # Insert records
            for record in records[:10]:  # Limit to prevent spam
                cursor.execute(
                    f"INSERT INTO public.{table_name} (event_data, enriched_data) VALUES (%s, %s)",
                    (json.dumps(record, default=str), json.dumps(record.get('enriched', {}), default=str))
                )
            
            self.pg_conn.commit()
            logger.info(f"✓ Wrote {len(records)} records to PostgreSQL ({table_name})")
        
        except Exception as e:
            logger.error(f"Error writing to PostgreSQL: {e}")
            self.pg_conn.rollback()
    
    def run(self):
        """Main consumer loop"""
        logger.info("🚀 Starting Flink Stream Consumer...")
        logger.info("Listening for CDC events on Kafka...")
        
        batch_size = 100
        messages_buffer = []
        last_flush = time.time()
        
        try:
            for message in self.kafka_consumer:
                messages_buffer.append(message)
                
                # Flush buffer on size or timeout
                if len(messages_buffer) >= batch_size or (time.time() - last_flush) > 30:
                    if messages_buffer:
                        logger.info(f"Processing batch of {len(messages_buffer)} messages...")
                        self.process_batch(messages_buffer)
                        messages_buffer = []
                        last_flush = time.time()
        
        except KeyboardInterrupt:
            logger.info("🛑 Consumer interrupted")
        except Exception as e:
            logger.error(f"✗ Consumer error: {e}")
        finally:
            if self.kafka_consumer:
                self.kafka_consumer.close()
            if self.pg_conn:
                self.pg_conn.close()
            logger.info("Consumer closed")

if __name__ == "__main__":
    consumer = FlinkStreamConsumer()
    consumer.run()
