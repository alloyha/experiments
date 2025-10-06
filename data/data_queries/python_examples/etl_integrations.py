"""
ETL Pipeline Integration Examples
================================

This module demonstrates how to integrate the PostgreSQL schema generator
with popular ETL tools and frameworks including:

- Apache Airflow: DAG orchestration and dynamic schema creation
- dbt: Model generation and incremental processing  
- Pandas: DataFrame operations and data profiling
- DuckDB: OLAP analytics and embedded database operations
- Polars: High-performance data processing with lazy evaluation

Each integration showcases:
- Schema validation and generation
- Data quality checks
- Performance optimization techniques
- Cross-tool compatibility patterns
"""

import json
import pandas as pd
import polars as pl
import duckdb
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path

try:
    from .schema_generator_client import SchemaGenerator, ConnectionConfig
except ImportError:
    # Fallback for direct execution
    from schema_generator_client import SchemaGenerator, ConnectionConfig

logger = logging.getLogger(__name__)


class AirflowIntegration:
    """
    Apache Airflow DAG integration examples
    
    Shows how to use schema generator in Airflow workflows for:
    - Dynamic schema creation
    - Data pipeline orchestration
    - Incremental model updates
    """
    
    @staticmethod
    def create_dynamic_schema_task(generator: SchemaGenerator, 
                                  pipeline_config: Dict[str, Any]):
        """
        Create Airflow task for dynamic schema generation
        
        This would be used in an Airflow DAG like:
        
        from airflow import DAG
        from airflow.operators.python import PythonOperator
        
        def schema_generation_task(**context):
            return AirflowIntegration.create_dynamic_schema_task(generator, config)
            
        dag = DAG('dimensional_model_pipeline')
        
        create_schema = PythonOperator(
            task_id='create_schema',
            python_callable=schema_generation_task,
            dag=dag
        )
        """
        try:
            # Validate pipeline configuration
            validation_results = generator.validate_dimensional_model(pipeline_config)
            logger.info(f"✅ Pipeline validation: {validation_results}")
            
            # Generate DDL and DML
            ddl, dml = generator.create_dimensional_model(pipeline_config, execute=True)
            
            # Store generated SQL for downstream tasks
            return {
                "status": "success",
                "ddl_length": len(ddl),
                "dml_length": len(dml),
                "table_count": len(pipeline_config.get('tables', [])),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Schema generation failed: {e}")
            raise
    
    @staticmethod
    def incremental_dimension_update(generator: SchemaGenerator, 
                                   dimension_name: str,
                                   source_data: pd.DataFrame):
        """
        Update SCD2 dimension with new data in Airflow pipeline
        
        Args:
            generator: Schema generator instance
            dimension_name: Name of dimension table to update
            source_data: New data to process
        """
        # Generate SCD2 processing DML
        dimension_metadata = {
            "name": dimension_name,
            "entity_type": "dimension",
            "scd": "SCD2",
            "primary_keys": ["business_key"]
        }
        
        dml = generator.generate_table_dml(dimension_metadata)
        
        # Process data and execute updates
        logger.info(f"Processing {len(source_data)} records for {dimension_name}")
        
        # Execute DML with data
        # (In real implementation, you'd parameterize and execute the DML)
        
        return {
            "dimension": dimension_name,
            "records_processed": len(source_data),
            "dml_generated": True
        }


class DBTIntegration:
    """
    dbt (data build tool) integration examples
    
    Shows how to generate dbt models and macros using schema generator
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
    
    def generate_dbt_model(self, table_metadata: Dict[str, Any], 
                          model_type: str = "table") -> str:
        """
        Generate dbt model SQL from table metadata
        
        Args:
            table_metadata: Table metadata dictionary
            model_type: 'table', 'view', or 'incremental'
            
        Returns:
            str: dbt model SQL
        """
        table_name = table_metadata['name']
        entity_type = table_metadata.get('entity_type', 'dimension')
        
        if entity_type == "dimension" and table_metadata.get('scd') == 'SCD2':
            return self._generate_scd2_dbt_model(table_metadata)
        elif entity_type in ["transaction_fact", "fact"]:
            return self._generate_fact_dbt_model(table_metadata, model_type)
        else:
            return self._generate_dimension_dbt_model(table_metadata)
    
    def _generate_scd2_dbt_model(self, metadata: Dict[str, Any]) -> str:
        """Generate SCD2 dimension dbt model"""
        table_name = metadata['name']
        primary_keys = metadata.get('primary_keys', ['id'])
        
        model_sql = f"""
{{{{
  config(
    materialized='table',
    post_hook="{{{{ grant_select_on_schemas(schemas=[target.schema], type='table') }}}}"
  )
}}}}

-- SCD2 Dimension: {table_name}
-- Generated by PostgreSQL Schema Generator

WITH source_data AS (
  SELECT * FROM {{{{ ref('staging_{table_name.replace("dim_", "")}') }}}}
),

current_records AS (
  SELECT * FROM {{{{ this }}}}
  WHERE valid_to IS NULL
),

changed_records AS (
  SELECT 
    s.*,
    CASE 
      WHEN c.{primary_keys[0]} IS NULL THEN 'INSERT'
      WHEN c.row_hash != s.row_hash THEN 'UPDATE'
      ELSE 'NO_CHANGE'
    END AS change_type
  FROM source_data s
  LEFT JOIN current_records c ON s.{primary_keys[0]} = c.{primary_keys[0]}
)

SELECT * FROM changed_records
WHERE change_type != 'NO_CHANGE'
        """
        
        return model_sql.strip()
    
    def _generate_fact_dbt_model(self, metadata: Dict[str, Any], 
                                model_type: str = "incremental") -> str:
        """Generate fact table dbt model"""
        table_name = metadata['name']
        measures = metadata.get('measures', [])
        
        measure_columns = []
        for measure in measures:
            measure_columns.append(f"    {measure['name']}")
        
        model_sql = f"""
{{{{
  config(
    materialized='{model_type}',
    unique_key='surrogate_key',
    on_schema_change='append_new_columns'
  )
}}}}

-- Fact Table: {table_name}
-- Generated by PostgreSQL Schema Generator

SELECT
  {{{{ dbt_utils.surrogate_key([
    'dimension_key_1',
    'dimension_key_2', 
    'event_timestamp'
  ]) }}}} AS surrogate_key,
  
  -- Dimension foreign keys
  dimension_key_1,
  dimension_key_2,
  
  -- Measures
{chr(10).join(measure_columns)},
  
  -- Audit columns
  event_timestamp,
  created_at
  
FROM {{{{ ref('staging_{table_name.replace("fact_", "")}') }}}}

{{% if is_incremental() %}}
  WHERE event_timestamp > (SELECT MAX(event_timestamp) FROM {{{{ this }}}})
{{% endif %}}
        """
        
        return model_sql.strip()
    
    def _generate_dimension_dbt_model(self, metadata: Dict[str, Any]) -> str:
        """Generate standard dimension dbt model"""
        table_name = metadata['name']
        
        return f"""
{{{{
  config(materialized='table')
}}}}

-- Dimension: {table_name}
-- Generated by PostgreSQL Schema Generator

SELECT
  {table_name}_sk,
  business_key,
  properties,
  row_hash,
  created_at,
  updated_at
  
FROM {{{{ ref('staging_{table_name.replace("dim_", "")}') }}}}
        """.strip()
    
    def generate_dbt_schema_yml(self, pipeline_metadata: Dict[str, Any]) -> str:
        """Generate dbt schema.yml file with tests and documentation"""
        tables = pipeline_metadata.get('tables', [])
        
        schema_yml = {
            "version": 2,
            "models": []
        }
        
        for table in tables:
            table_name = table['name']
            entity_type = table.get('entity_type', 'dimension')
            
            model_config = {
                "name": table_name,
                "description": f"{entity_type.replace('_', ' ').title()}: {table.get('grain', 'N/A')}",
                "columns": []
            }
            
            # Add surrogate key column
            model_config["columns"].append({
                "name": f"{table_name}_sk",
                "description": "Surrogate key",
                "tests": ["unique", "not_null"]
            })
            
            # Add business key columns
            for pk in table.get('primary_keys', []):
                model_config["columns"].append({
                    "name": pk,
                    "description": f"Business key: {pk}",
                    "tests": ["not_null"]
                })
            
            # Add measure columns for facts
            if entity_type in ["transaction_fact", "fact"]:
                for measure in table.get('measures', []):
                    model_config["columns"].append({
                        "name": measure['name'],
                        "description": f"Measure: {measure['name']} ({measure.get('additivity', 'unknown')})",
                        "tests": ["not_null"] if measure.get('required') else []
                    })
            
            schema_yml["models"].append(model_config)
        
        return json.dumps(schema_yml, indent=2)


class PandasIntegration:
    """
    Pandas DataFrame integration for data processing
    
    Shows how to use schema generator with pandas for:
    - Data validation and profiling
    - DataFrame to database mapping
    - ETL data transformations
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
    
    def validate_dataframe_against_schema(self, df: pd.DataFrame, 
                                        table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DataFrame structure against generated schema
        
        Args:
            df: Pandas DataFrame to validate
            table_metadata: Table schema metadata
            
        Returns:
            Dict: Validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        # Get expected columns from metadata
        expected_columns = set()
        
        # Add primary key columns
        for pk in table_metadata.get('primary_keys', []):
            expected_columns.add(pk)
        
        # Add physical columns
        for col in table_metadata.get('physical_columns', []):
            expected_columns.add(col['name'])
        
        # Add measure columns for facts
        if table_metadata.get('entity_type') in ['transaction_fact', 'fact']:
            for measure in table_metadata.get('measures', []):
                expected_columns.add(measure['name'])
        
        # Check for missing columns
        df_columns = set(df.columns)
        missing_columns = expected_columns - df_columns
        extra_columns = df_columns - expected_columns
        
        if missing_columns:
            results["errors"].append(f"Missing columns: {missing_columns}")
            results["valid"] = False
        
        if extra_columns:
            results["warnings"].append(f"Extra columns: {extra_columns}")
        
        # Data quality checks
        results["stats"] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        return results
    
    def generate_insert_statements(self, df: pd.DataFrame, 
                                 table_metadata: Dict[str, Any],
                                 schema: str = "public") -> List[str]:
        """
        Generate SQL INSERT statements from DataFrame
        
        Args:
            df: Source DataFrame
            table_metadata: Table metadata
            schema: Target schema
            
        Returns:
            List[str]: SQL INSERT statements
        """
        table_name = table_metadata['name']
        
        # Get DML template
        dml_template = self.generator.generate_table_dml(table_metadata, schema)
        
        insert_statements = []
        
        for _, row in df.iterrows():
            # Build INSERT statement from template
            # (In real implementation, you'd use proper parameterization)
            values = []
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    values.append("NULL")
                elif isinstance(value, str):
                    values.append(f"'{value.replace("'", "''")}'")
                else:
                    values.append(str(value))
            
            insert_sql = f"INSERT INTO {schema}.{table_name} ({', '.join(df.columns)}) VALUES ({', '.join(values)});"
            insert_statements.append(insert_sql)
        
        return insert_statements
    
    def profile_dataframe_for_schema_design(self, df: pd.DataFrame, 
                                          table_name: str) -> Dict[str, Any]:
        """
        Profile DataFrame to suggest schema design
        
        Args:
            df: DataFrame to profile
            table_name: Proposed table name
            
        Returns:
            Dict: Suggested schema metadata
        """
        suggested_schema = {
            "name": table_name,
            "entity_type": "dimension",  # Default assumption
            "grain": f"One row per {table_name.replace('_', ' ')}",
            "physical_columns": []
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                "name": col,
                "type": self._suggest_postgres_type(df[col]),
                "nullable": df[col].isnull().any(),
                "unique_values": df[col].nunique(),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            
            suggested_schema["physical_columns"].append(col_info)
        
        # Suggest primary keys (columns with unique values)
        potential_keys = [
            col for col in df.columns 
            if df[col].nunique() == len(df) and not df[col].isnull().any()
        ]
        
        if potential_keys:
            suggested_schema["primary_keys"] = potential_keys[:1]  # Take first candidate
        
        return suggested_schema
    
    def _suggest_postgres_type(self, series: pd.Series) -> str:
        """Suggest PostgreSQL data type for pandas Series"""
        if series.dtype == 'object':
            # String data
            max_length = series.astype(str).str.len().max() if not series.empty else 0
            if max_length <= 255:
                return "text"
            else:
                return "text"
        elif series.dtype in ['int64', 'int32']:
            return "integer"
        elif series.dtype in ['float64', 'float32']:
            return "numeric"
        elif series.dtype == 'bool':
            return "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "timestamptz"
        else:
            return "text"  # Fallback


class DuckDBIntegration:
    """
    DuckDB integration for OLAP and analytics workloads
    
    Shows how to use schema generator with DuckDB for:
    - Fast analytical queries
    - Data lake integration
    - Cross-database analytics
    - Embedded analytics
    """
    
    def __init__(self, generator: SchemaGenerator, duckdb_path: Optional[str] = None):
        self.generator = generator
        self.duckdb_path = duckdb_path or ":memory:"
        self.conn = None
    
    def __enter__(self):
        self.conn = duckdb.connect(self.duckdb_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def create_schema_from_postgres(self, schema_name: str, 
                                  postgres_config: ConnectionConfig,
                                  tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Replicate PostgreSQL schema structure in DuckDB
        
        Args:
            schema_name: Schema name in PostgreSQL
            postgres_config: PostgreSQL connection configuration
            tables: Specific tables to replicate (None for all)
            
        Returns:
            Dict: Replication results
        """
        results = {
            "schema": schema_name,
            "tables_created": [],
            "errors": []
        }
        
        try:
            # Install and load postgres extension
            self.conn.execute("INSTALL postgres")
            self.conn.execute("LOAD postgres")
            
            # Attach PostgreSQL database
            postgres_dsn = f"postgresql://{postgres_config.username}:{postgres_config.password}@{postgres_config.host}:{postgres_config.port}/{postgres_config.database}"
            self.conn.execute(f"ATTACH '{postgres_dsn}' AS pg_db (TYPE postgres)")
            
            # Create schema in DuckDB
            self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            
            # Get table list from PostgreSQL
            if tables is None:
                table_query = f"""
                SELECT table_name 
                FROM pg_db.information_schema.tables 
                WHERE table_schema = '{schema_name}'
                """
                tables = [row[0] for row in self.conn.execute(table_query).fetchall()]
            
            # Replicate each table
            for table in tables:
                try:
                    # Create table in DuckDB with data
                    create_sql = f"""
                    CREATE TABLE {schema_name}.{table} AS 
                    SELECT * FROM pg_db.{schema_name}.{table}
                    """
                    self.conn.execute(create_sql)
                    results["tables_created"].append(table)
                    
                except Exception as e:
                    results["errors"].append(f"Table {table}: {str(e)}")
            
            return results
            
        except Exception as e:
            results["errors"].append(f"Schema replication failed: {str(e)}")
            return results
    
    def generate_analytical_queries(self, table_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate common analytical queries for dimensional tables
        
        Args:
            table_metadata: Table metadata from schema generator
            
        Returns:
            Dict: Named analytical queries
        """
        table_name = table_metadata['name']
        entity_type = table_metadata.get('entity_type', 'dimension')
        
        queries = {}
        
        if entity_type in ['transaction_fact', 'fact']:
            # Fact table analytics
            measures = table_metadata.get('measures', [])
            
            # Aggregation query
            measure_aggs = []
            for measure in measures:
                measure_name = measure['name']
                if measure.get('additivity') == 'additive':
                    measure_aggs.extend([
                        f"SUM({measure_name}) as total_{measure_name}",
                        f"AVG({measure_name}) as avg_{measure_name}",
                        f"COUNT({measure_name}) as count_{measure_name}"
                    ])
                else:
                    measure_aggs.append(f"COUNT({measure_name}) as count_{measure_name}")
            
            if measure_aggs:
                queries['daily_aggregates'] = f"""
                SELECT 
                    DATE_TRUNC('day', event_timestamp) as date,
                    {', '.join(measure_aggs)}
                FROM {table_name}
                WHERE event_timestamp >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', event_timestamp)
                ORDER BY date DESC
                """
            
            # Top N analysis
            if measures:
                queries['top_performers'] = f"""
                SELECT 
                    *,
                    ROW_NUMBER() OVER (ORDER BY {measures[0]['name']} DESC) as rank
                FROM {table_name}
                WHERE event_timestamp >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY {measures[0]['name']} DESC
                LIMIT 100
                """
        
        elif entity_type == 'dimension':
            # Dimension analytics
            queries['dimension_profile'] = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT {table_metadata.get('primary_keys', ['id'])[0]}) as unique_keys,
                MIN(created_at) as oldest_record,
                MAX(updated_at) as newest_record
            FROM {table_name}
            """
            
            # SCD2 analysis if applicable
            if table_metadata.get('scd') == 'SCD2':
                queries['scd2_analysis'] = f"""
                SELECT 
                    COUNT(*) as total_versions,
                    COUNT(CASE WHEN valid_to IS NULL THEN 1 END) as current_records,
                    COUNT(CASE WHEN valid_to IS NOT NULL THEN 1 END) as historical_records,
                    AVG(EXTRACT(days FROM COALESCE(valid_to, CURRENT_TIMESTAMP) - valid_from)) as avg_version_days
                FROM {table_name}
                """
        
        return queries
    
    def optimize_for_analytics(self, table_name: str, 
                             table_metadata: Dict[str, Any]) -> List[str]:
        """
        Generate DuckDB optimization commands for analytical workloads
        
        Args:
            table_name: Name of table to optimize
            table_metadata: Table metadata
            
        Returns:
            List[str]: Optimization SQL commands
        """
        optimizations = []
        entity_type = table_metadata.get('entity_type', 'dimension')
        
        # Create indexes for common query patterns
        if entity_type in ['transaction_fact', 'fact']:
            # Time-based index for facts
            optimizations.append(f"CREATE INDEX idx_{table_name}_time ON {table_name} (event_timestamp)")
            
            # Dimension foreign key indexes
            for fk in table_metadata.get('foreign_keys', []):
                optimizations.append(f"CREATE INDEX idx_{table_name}_{fk} ON {table_name} ({fk})")
        
        elif entity_type == 'dimension':
            # Primary key index
            for pk in table_metadata.get('primary_keys', []):
                optimizations.append(f"CREATE INDEX idx_{table_name}_{pk} ON {table_name} ({pk})")
            
            # SCD2 optimization
            if table_metadata.get('scd') == 'SCD2':
                optimizations.append(f"CREATE INDEX idx_{table_name}_scd2 ON {table_name} (business_key, valid_from, valid_to)")
        
        return optimizations
    
    def export_to_parquet(self, table_name: str, output_path: str, 
                         partition_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Export DuckDB table to Parquet for data lake storage
        
        Args:
            table_name: Table to export
            output_path: Output directory path
            partition_by: Column to partition by
            
        Returns:
            Dict: Export results
        """
        try:
            if partition_by:
                export_sql = f"""
                COPY {table_name} TO '{output_path}' 
                (FORMAT PARQUET, PARTITION_BY ({partition_by}))
                """
            else:
                export_sql = f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)"
            
            self.conn.execute(export_sql)
            
            return {
                "status": "success",
                "table": table_name,
                "output_path": output_path,
                "partitioned": bool(partition_by),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "table": table_name
            }


class PolarsIntegration:
    """
    Polars integration for high-performance data processing
    
    Shows how to use schema generator with Polars for:
    - Lightning-fast ETL operations
    - Lazy evaluation pipelines
    - Memory-efficient processing
    - Type-safe data operations
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
    
    def dataframe_from_metadata(self, table_metadata: Dict[str, Any], 
                               sample_data: Optional[List[Dict]] = None) -> pl.DataFrame:
        """
        Create Polars DataFrame with schema from table metadata
        
        Args:
            table_metadata: Table metadata from schema generator
            sample_data: Optional sample data to populate DataFrame
            
        Returns:
            pl.DataFrame: Polars DataFrame with correct schema
        """
        # Build schema from metadata
        schema = {}
        
        # Add primary keys
        for pk in table_metadata.get('primary_keys', []):
            schema[pk] = pl.Utf8  # Default to string
        
        # Add physical columns
        for col in table_metadata.get('physical_columns', []):
            col_type = self._postgres_to_polars_type(col.get('type', 'text'))
            schema[col['name']] = col_type
        
        # Add measures for facts
        if table_metadata.get('entity_type') in ['transaction_fact', 'fact']:
            for measure in table_metadata.get('measures', []):
                measure_type = self._postgres_to_polars_type(measure.get('type', 'numeric'))
                schema[measure['name']] = measure_type
        
        # Add audit columns
        schema['created_at'] = pl.Datetime
        schema['updated_at'] = pl.Datetime
        
        # Create DataFrame
        if sample_data:
            df = pl.DataFrame(sample_data, schema=schema)
        else:
            # Create empty DataFrame with schema
            df = pl.DataFrame(schema=schema)
        
        return df
    
    def _postgres_to_polars_type(self, postgres_type: str) -> pl.DataType:
        """Convert PostgreSQL type to Polars type"""
        type_mapping = {
            'text': pl.Utf8,
            'varchar': pl.Utf8,
            'integer': pl.Int64,
            'bigint': pl.Int64,
            'smallint': pl.Int32,
            'numeric': pl.Float64,
            'decimal': pl.Float64,
            'real': pl.Float32,
            'double precision': pl.Float64,
            'boolean': pl.Boolean,
            'timestamp': pl.Datetime,
            'timestamptz': pl.Datetime,
            'date': pl.Date,
            'json': pl.Utf8,  # JSON as string in Polars
            'jsonb': pl.Utf8,
            'uuid': pl.Utf8
        }
        
        return type_mapping.get(postgres_type.lower(), pl.Utf8)
    
    def create_scd2_pipeline(self, source_df: pl.LazyFrame, 
                           current_df: pl.LazyFrame,
                           business_key: str,
                           compare_columns: List[str]) -> pl.LazyFrame:
        """
        Create SCD2 processing pipeline using Polars lazy evaluation
        
        Args:
            source_df: New source data
            current_df: Current dimension data
            business_key: Business key column
            compare_columns: Columns to compare for changes
            
        Returns:
            pl.LazyFrame: Lazy frame with SCD2 processing logic
        """
        # Add row hash for change detection
        source_with_hash = source_df.with_columns([
            pl.concat_str([pl.col(col) for col in compare_columns], separator='|')
            .hash()
            .alias('row_hash')
        ])
        
        # Get current records (valid_to IS NULL)
        current_records = current_df.filter(pl.col('valid_to').is_null())
        
        # Join to detect changes
        changes = source_with_hash.join(
            current_records.select([business_key, 'row_hash', f'{business_key}_sk']),
            on=business_key,
            how='left'
        ).with_columns([
            pl.when(pl.col(f'{business_key}_sk').is_null())
            .then(pl.lit('INSERT'))
            .when(pl.col('row_hash') != pl.col('row_hash_right'))
            .then(pl.lit('UPDATE'))
            .otherwise(pl.lit('NO_CHANGE'))
            .alias('change_type')
        ])
        
        # Filter for actual changes
        changed_records = changes.filter(pl.col('change_type') != 'NO_CHANGE')
        
        return changed_records
    
    def create_fact_aggregation_pipeline(self, fact_df: pl.LazyFrame,
                                       dimensions: List[str],
                                       measures: List[Dict[str, Any]],
                                       time_grain: str = 'day') -> pl.LazyFrame:
        """
        Create fact table aggregation pipeline
        
        Args:
            fact_df: Fact table DataFrame
            dimensions: Dimension columns to group by
            measures: Measure definitions with aggregation rules
            time_grain: Time granularity ('day', 'week', 'month')
            
        Returns:
            pl.LazyFrame: Aggregated fact data
        """
        # Time dimension processing
        time_expr = {
            'day': pl.col('event_timestamp').dt.truncate('1d'),
            'week': pl.col('event_timestamp').dt.truncate('1w'),
            'month': pl.col('event_timestamp').dt.truncate('1mo')
        }
        
        time_column = time_expr.get(time_grain, time_expr['day']).alias('time_grain')
        
        # Build aggregation expressions
        agg_exprs = [time_column]
        
        for measure in measures:
            measure_name = measure['name']
            additivity = measure.get('additivity', 'additive')
            
            if additivity == 'additive':
                agg_exprs.extend([
                    pl.col(measure_name).sum().alias(f'sum_{measure_name}'),
                    pl.col(measure_name).mean().alias(f'avg_{measure_name}'),
                    pl.col(measure_name).count().alias(f'count_{measure_name}')
                ])
            elif additivity == 'semi_additive':
                # For semi-additive measures, use last value
                agg_exprs.extend([
                    pl.col(measure_name).last().alias(f'last_{measure_name}'),
                    pl.col(measure_name).count().alias(f'count_{measure_name}')
                ])
            else:  # non-additive
                agg_exprs.append(
                    pl.col(measure_name).count().alias(f'count_{measure_name}')
                )
        
        # Group by dimensions and time
        group_by_cols = dimensions + ['time_grain']
        
        aggregated = fact_df.group_by(group_by_cols).agg(agg_exprs)
        
        return aggregated
    
    def validate_data_quality(self, df: pl.LazyFrame, 
                            table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform data quality validation using Polars
        
        Args:
            df: DataFrame to validate
            table_metadata: Table metadata for validation rules
            
        Returns:
            Dict: Validation results
        """
        # Collect for analysis (only if DataFrame is not too large)
        try:
            df_collected = df.collect()
        except Exception as e:
            return {"error": f"Failed to collect DataFrame: {e}"}
        
        results = {
            "table": table_metadata['name'],
            "row_count": len(df_collected),
            "column_count": len(df_collected.columns),
            "checks": {}
        }
        
        # Null checks for required columns
        for pk in table_metadata.get('primary_keys', []):
            if pk in df_collected.columns:
                null_count = df_collected.select(pl.col(pk).is_null().sum()).item()
                results["checks"][f"{pk}_nulls"] = {
                    "test": "not_null",
                    "passed": null_count == 0,
                    "null_count": null_count
                }
        
        # Uniqueness checks
        for pk in table_metadata.get('primary_keys', []):
            if pk in df_collected.columns:
                unique_count = df_collected.select(pl.col(pk).n_unique()).item()
                total_count = len(df_collected)
                results["checks"][f"{pk}_unique"] = {
                    "test": "unique",
                    "passed": unique_count == total_count,
                    "unique_count": unique_count,
                    "total_count": total_count
                }
        
        # Data type validation
        schema_check = {}
        for col_name, col_type in df_collected.schema.items():
            schema_check[col_name] = str(col_type)
        
        results["schema"] = schema_check
        
        return results
    
    def optimize_for_performance(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply performance optimizations to Polars DataFrame
        
        Args:
            df: Input LazyFrame
            
        Returns:
            pl.LazyFrame: Optimized LazyFrame
        """
        # Get schema efficiently
        schema = df.collect_schema()
        
        # Common optimizations
        string_columns = [col for col, dtype in schema.items() if dtype == pl.Utf8]
        
        if string_columns:
            optimized = df.with_columns([
                # Convert string columns to categorical if low cardinality
                pl.when(pl.col(col).n_unique() < 1000)
                .then(pl.col(col).cast(pl.Categorical))
                .otherwise(pl.col(col))
                for col in string_columns
            ])
        else:
            optimized = df
        
        return optimized
    
    def export_to_postgres(self, df: pl.DataFrame, 
                         table_metadata: Dict[str, Any],
                         postgres_config: ConnectionConfig,
                         schema: str = "public",
                         if_exists: str = "append") -> Dict[str, Any]:
        """
        Export Polars DataFrame to PostgreSQL
        
        Args:
            df: Polars DataFrame
            table_metadata: Table metadata
            postgres_config: PostgreSQL connection config
            schema: Target schema
            if_exists: 'append', 'replace', or 'fail'
            
        Returns:
            Dict: Export results
        """
        try:
            import psycopg2
            import io
            
            # Convert to Pandas for PostgreSQL export (until Polars has native support)
            pandas_df = df.to_pandas()
            
            # Create connection
            conn = psycopg2.connect(
                host=postgres_config.host,
                database=postgres_config.database,
                user=postgres_config.username,
                password=postgres_config.password,
                port=postgres_config.port
            )
            
            table_name = table_metadata['name']
            
            # Use COPY for efficient bulk insert
            output = io.StringIO()
            pandas_df.to_csv(output, sep='\t', header=False, index=False)
            output.seek(0)
            
            with conn.cursor() as cursor:
                cursor.copy_from(output, f"{schema}.{table_name}", columns=list(pandas_df.columns))
                conn.commit()
            
            conn.close()
            
            return {
                "status": "success",
                "rows_exported": len(df),
                "table": f"{schema}.{table_name}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "table": table_metadata['name']
            }


def main():
    """Demonstrate ETL integrations"""
    config = ConnectionConfig(
        host="localhost",
        database="postgres",
        username="postgres",
        password="postgres"
    )
    
    generator = SchemaGenerator(config)
    
    print("🔄 ETL PIPELINE INTEGRATIONS")
    print("=" * 50)
    
    # Airflow Integration Demo
    print("\n📅 Airflow Integration")
    print("-" * 30)
    
    sample_pipeline = {
        "schema": "airflow_demo",
        "tables": [
            {
                "name": "dim_products",
                "entity_type": "dimension",
                "grain": "One row per product",
                "primary_keys": ["product_id"]
            }
        ]
    }
    
    airflow_result = AirflowIntegration.create_dynamic_schema_task(
        generator, sample_pipeline
    )
    print(f"✅ Airflow task result: {airflow_result}")
    
    # dbt Integration Demo
    print("\n🏗️ dbt Integration")
    print("-" * 30)
    
    dbt_integration = DBTIntegration(generator)
    
    fact_metadata = {
        "name": "fact_sales",
        "entity_type": "transaction_fact",
        "grain": "One row per sale",
        "measures": [
            {"name": "amount", "type": "numeric"},
            {"name": "quantity", "type": "integer"}
        ]
    }
    
    dbt_model = dbt_integration.generate_dbt_model(fact_metadata, "incremental")
    print("✅ Generated dbt model:")
    print(dbt_model[:200] + "...")
    
    # Pandas Integration Demo
    print("\n🐼 Pandas Integration")
    print("-" * 30)
    
    pandas_integration = PandasIntegration(generator)
    
    # Create sample DataFrame
    sample_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003'],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
        'signup_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    
    # Profile the data
    schema_suggestion = pandas_integration.profile_dataframe_for_schema_design(
        sample_data, "dim_customers"
    )
    
    print("✅ Schema suggestion from DataFrame:")
    print(f"  - Table: {schema_suggestion['name']}")
    print(f"  - Columns: {len(schema_suggestion['physical_columns'])}")
    print(f"  - Suggested keys: {schema_suggestion.get('primary_keys', 'None')}")
    
    # DuckDB Integration Demo
    print("\n🦆 DuckDB Integration")
    print("-" * 30)
    
    try:
        with DuckDBIntegration(generator, ":memory:") as duckdb_integration:
            # Generate analytical queries
            fact_metadata = {
                "name": "fact_sales",
                "entity_type": "transaction_fact",
                "measures": [
                    {"name": "amount", "type": "numeric", "additivity": "additive"},
                    {"name": "quantity", "type": "integer", "additivity": "additive"}
                ]
            }
            
            analytical_queries = duckdb_integration.generate_analytical_queries(fact_metadata)
            print(f"✅ Generated {len(analytical_queries)} analytical queries")
            for query_name in analytical_queries.keys():
                print(f"  - {query_name}")
            
            # Export demo (create a dummy table first)
            duckdb_integration.conn.execute("""
                CREATE TABLE fact_sales AS 
                SELECT 
                    'S001' as sale_id,
                    100.50 as amount,
                    5 as quantity,
                    CURRENT_DATE as event_date,
                    CURRENT_TIMESTAMP as event_timestamp
            """)
            
            import tempfile
            import os
            temp_dir = tempfile.mkdtemp()
            parquet_path = os.path.join(temp_dir, "fact_sales.parquet")
            
            export_result = duckdb_integration.export_to_parquet(
                "fact_sales", 
                parquet_path
            )
            print(f"✅ Parquet export: {export_result.get('status', 'unknown')}")
            
            # Cleanup
            try:
                os.remove(parquet_path)
                os.rmdir(temp_dir)
            except:
                pass
            
    except ImportError:
        print("⚠️ DuckDB not installed - install with: pip install duckdb")
    except Exception as e:
        print(f"⚠️ DuckDB demo error: {e}")
    
    # Polars Integration Demo
    print("\n⚡ Polars Integration")
    print("-" * 30)
    
    try:
        polars_integration = PolarsIntegration(generator)
        
        # Create DataFrame from metadata
        dimension_metadata = {
            "name": "dim_customers",
            "entity_type": "dimension",
            "primary_keys": ["customer_id"],
            "physical_columns": [
                {"name": "customer_name", "type": "text"},
                {"name": "email", "type": "text"},
                {"name": "signup_date", "type": "date"}
            ]
        }
        
        sample_data = [
            {
                "customer_id": "C001", 
                "customer_name": "John Doe", 
                "email": "john@example.com",
                "signup_date": datetime(2023, 1, 1).date(),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            },
            {
                "customer_id": "C002", 
                "customer_name": "Jane Smith", 
                "email": "jane@example.com",
                "signup_date": datetime(2023, 1, 2).date(),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
        ]
        
        polars_df = polars_integration.dataframe_from_metadata(
            dimension_metadata, 
            sample_data
        )
        
        print(f"✅ Created Polars DataFrame:")
        print(f"  - Shape: {polars_df.shape}")
        print(f"  - Schema: {list(polars_df.schema.keys())}")
        
        # Data quality validation
        lazy_df = polars_df.lazy()
        quality_results = polars_integration.validate_data_quality(lazy_df, dimension_metadata)
        print(f"✅ Data quality checks: {len(quality_results.get('checks', {}))}")
        
        # Performance optimization
        optimized_df = polars_integration.optimize_for_performance(lazy_df)
        print("✅ Applied performance optimizations")
        
    except ImportError:
        print("⚠️ Polars not installed - install with: pip install polars")
    except Exception as e:
        print(f"⚠️ Polars demo error: {e}")
    
    print("\n🎯 ETL Integration Summary")
    print("=" * 50)
    print("✅ Airflow: Dynamic schema generation and pipeline orchestration")
    print("✅ dbt: Model generation and incremental processing")  
    print("✅ Pandas: DataFrame operations and data profiling")
    print("✅ DuckDB: OLAP analytics and data lake integration")
    print("✅ Polars: High-performance data processing with lazy evaluation")
    print("\nAll integrations support:")
    print("- Schema validation and generation")
    print("- Data quality checks")
    print("- Performance optimization")
    print("- Cross-tool compatibility")


if __name__ == "__main__":
    main()