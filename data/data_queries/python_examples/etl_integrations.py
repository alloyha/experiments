"""
ETL Pipeline Integration Examples
================================

This module demonstrates how to integrate the PostgreSQL schema generator
with popular ETL tools and frameworks like Apache Airflow, dbt, and Pandas.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

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
                    values.append(f"'{value.replace(\"'\", \"''\")}'")
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


def main():
    """Demonstrate ETL integrations"""
    config = ConnectionConfig(
        host="localhost",
        database="postgres",
        username="postgres"
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


if __name__ == "__main__":
    main()