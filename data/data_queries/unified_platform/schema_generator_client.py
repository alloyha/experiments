"""
PostgreSQL Schema Generator Python Client
==========================================

This module provides Python wrappers and use cases for the PostgreSQL 
schema generator SQL procedures. It demonstrates practical applications
for dimensional modeling, data warehousing, and ETL pipelines.

Requirements:
    pip install psycopg2-binary pandas sqlalchemy

Usage:
    from schema_generator_client import SchemaGenerator
    
    generator = SchemaGenerator(connection_params)
    generator.create_dimensional_model(metadata)
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    username: str = "postgres"
    password: str = ""
    schema: str = "public"


class SchemaGeneratorError(Exception):
    """Custom exception for schema generator errors"""
    pass


class SchemaGenerator:
    """
    Python client for PostgreSQL Schema Generator procedures
    
    This class provides high-level Python interfaces to the SQL procedures
    for generating DDL/DML, validating metadata, and managing dimensional models.
    """
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection_string = (
            f"host={config.host} port={config.port} "
            f"dbname={config.database} user={config.username} "
            f"password={config.password}"
        )
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.connection_string)
            conn.autocommit = True
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise SchemaGeneratorError(f"Database connection error: {e}")
        finally:
            if conn:
                conn.close()
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate table metadata using the validate_metadata_entry function
        
        Args:
            metadata: Table metadata dictionary
            
        Returns:
            bool: True if valid, raises exception if invalid
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT validate_metadata_entry(%s::jsonb)",
                    (json.dumps(metadata),)
                )
                logger.info(f"✅ Metadata validation passed for table: {metadata.get('name')}")
                return True
            except psycopg2.Error as e:
                raise SchemaGeneratorError(f"Metadata validation failed: {e}")
    
    def generate_table_ddl(self, table_metadata: Dict[str, Any], 
                          target_schema: str = None) -> str:
        """
        Generate DDL for a single table
        
        Args:
            table_metadata: Table metadata dictionary
            target_schema: Target schema name (defaults to config schema)
            
        Returns:
            str: Generated DDL SQL
        """
        schema = target_schema or self.config.schema
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT generate_table_ddl(%s::jsonb, %s)",
                    (json.dumps(table_metadata), schema)
                )
                ddl = cursor.fetchone()[0]
                logger.info(f"✅ Generated DDL for table: {table_metadata.get('name')}")
                return ddl
            except psycopg2.Error as e:
                raise SchemaGeneratorError(f"DDL generation failed: {e}")
    
    def generate_pipeline_ddl(self, pipeline_metadata: Dict[str, Any]) -> str:
        """
        Generate DDL for an entire pipeline
        
        Args:
            pipeline_metadata: Pipeline metadata with tables array
            
        Returns:
            str: Generated DDL SQL for all tables
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT generate_ddl(%s::jsonb)",
                    (json.dumps(pipeline_metadata),)
                )
                ddl = cursor.fetchone()[0]
                table_count = len(pipeline_metadata.get('tables', []))
                logger.info(f"✅ Generated DDL for pipeline with {table_count} tables")
                return ddl
            except psycopg2.Error as e:
                raise SchemaGeneratorError(f"Pipeline DDL generation failed: {e}")
    
    def generate_table_dml(self, table_metadata: Dict[str, Any], 
                          target_schema: str = None) -> str:
        """
        Generate DML (INSERT/UPSERT) for a table
        
        Args:
            table_metadata: Table metadata dictionary
            target_schema: Target schema name
            
        Returns:
            str: Generated DML SQL
        """
        schema = target_schema or self.config.schema
        
        # Prepare metadata with schema
        metadata_with_schema = table_metadata.copy()
        metadata_with_schema['schema'] = schema
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT generate_dml(%s::jsonb, %s)",
                    (json.dumps([metadata_with_schema]), schema)
                )
                dml = cursor.fetchone()[0]
                logger.info(f"✅ Generated DML for table: {table_metadata.get('name')}")
                return dml
            except psycopg2.Error as e:
                raise SchemaGeneratorError(f"DML generation failed: {e}")
    
    def generate_pipeline_dml(self, pipeline_metadata: Dict[str, Any]) -> str:
        """
        Generate DML for an entire pipeline
        
        Args:
            pipeline_metadata: Pipeline metadata with tables array
            
        Returns:
            str: Generated DML SQL for all tables
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT generate_dml(%s::jsonb)",
                    (json.dumps(pipeline_metadata),)
                )
                dml = cursor.fetchone()[0]
                table_count = len(pipeline_metadata.get('tables', []))
                logger.info(f"✅ Generated DML for pipeline with {table_count} tables")
                return dml
            except psycopg2.Error as e:
                raise SchemaGeneratorError(f"Pipeline DML generation failed: {e}")
    
    def create_tables(self, ddl_sql: str) -> None:
        """
        Execute DDL to create tables
        
        Args:
            ddl_sql: DDL SQL statements to execute
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(ddl_sql)
                logger.info("✅ Tables created successfully")
            except psycopg2.Error as e:
                raise SchemaGeneratorError(f"Table creation failed: {e}")
    
    def create_dimensional_model(self, pipeline_metadata: Dict[str, Any], 
                                execute: bool = False) -> Tuple[str, str]:
        """
        Complete workflow to create a dimensional model
        
        Args:
            pipeline_metadata: Complete pipeline metadata
            execute: Whether to execute the DDL (create tables)
            
        Returns:
            Tuple[str, str]: Generated DDL and DML
        """
        logger.info("🚀 Starting dimensional model creation...")
        
        # Validate all table metadata
        for table in pipeline_metadata.get('tables', []):
            self.validate_metadata(table)
        
        # Generate DDL and DML
        ddl = self.generate_pipeline_ddl(pipeline_metadata)
        dml = self.generate_pipeline_dml(pipeline_metadata)
        
        # Optionally execute DDL
        if execute:
            self.create_tables(ddl)
            logger.info("🎉 Dimensional model created successfully!")
        else:
            logger.info("📝 Dimensional model DDL/DML generated (not executed)")
        
        return ddl, dml
    
    def validate_dimensional_model(self, pipeline_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a complete dimensional model
        
        Args:
            pipeline_metadata: Pipeline metadata to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT validate_dimensional_model(%s::jsonb)",
                    (json.dumps(pipeline_metadata),)
                )
                result = cursor.fetchone()[0]
                logger.info("✅ Dimensional model validation completed")
                return result
            except psycopg2.Error as e:
                raise SchemaGeneratorError(f"Dimensional model validation failed: {e}")


# Utility functions for building metadata
def create_dimension_metadata(name: str, grain: str, primary_keys: List[str],
                            physical_columns: List[Dict] = None,
                            scd_type: str = "SCD1") -> Dict[str, Any]:
    """
    Helper function to create dimension table metadata
    
    Args:
        name: Table name
        grain: Business grain description
        primary_keys: List of primary key columns
        physical_columns: Optional physical columns outside JSONB
        scd_type: SCD1 or SCD2
        
    Returns:
        Dict: Dimension metadata
    """
    metadata = {
        "name": name,
        "entity_type": "dimension",
        "grain": grain,
        "primary_keys": primary_keys,
        "scd": scd_type
    }
    
    if physical_columns:
        metadata["physical_columns"] = physical_columns
    
    return metadata


def create_fact_metadata(name: str, grain: str, 
                        dimension_references: List[Dict],
                        measures: List[Dict],
                        physical_columns: List[Dict] = None) -> Dict[str, Any]:
    """
    Helper function to create fact table metadata
    
    Args:
        name: Table name
        grain: Business grain description
        dimension_references: List of dimension FK references
        measures: List of measure definitions
        physical_columns: Optional physical columns
        
    Returns:
        Dict: Fact metadata
    """
    metadata = {
        "name": name,
        "entity_type": "transaction_fact",
        "grain": grain,
        "dimension_references": dimension_references,
        "measures": measures
    }
    
    if physical_columns:
        metadata["physical_columns"] = physical_columns
    
    return metadata


if __name__ == "__main__":
    # Example usage
    config = ConnectionConfig(
        host="localhost",
        database="postgres",
        username="postgres",
        password="postgres"
    )
    
    generator = SchemaGenerator(config)
    
    # Example dimension
    customer_dim = create_dimension_metadata(
        name="dim_customers",
        grain="One row per customer",
        primary_keys=["customer_id"],
        physical_columns=[
            {"name": "customer_id", "type": "text"},
            {"name": "email", "type": "text"},
            {"name": "registration_date", "type": "date"}
        ]
    )
    
    # Validate and generate
    try:
        generator.validate_metadata(customer_dim)
        ddl = generator.generate_table_ddl(customer_dim, "analytics")
        print("Generated DDL:")
        print(ddl)
    except SchemaGeneratorError as e:
        logger.error(f"Error: {e}")