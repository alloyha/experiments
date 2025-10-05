# PostgreSQL Schema Generator - Python Examples

This directory contains comprehensive Python examples and use cases for the PostgreSQL Schema Generator SQL procedures.

## 📁 Files Overview

| File | Description | Use Case |
|------|-------------|----------|
| `schema_generator_client.py` | Core Python client library | Main interface to SQL procedures |
| `use_cases.py` | Real-world business scenarios | E-commerce, SaaS, Financial, Analytics |
| `etl_integrations.py` | ETL tool integrations | Airflow, dbt, Pandas, DuckDB, Polars |
| `advanced_examples.py` | Advanced schema management | Versioning, Multi-tenant, Performance |
| `setup.py` | Installation and setup | Quick start guide |

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   python setup.py
   ```

2. **Configure Database Connection**
   ```python
   from schema_generator_client import SchemaGenerator, ConnectionConfig
   
   config = ConnectionConfig(
       host="localhost",
       database="postgres",
       username="postgres",
       password="your_password"
   )
   
   generator = SchemaGenerator(config)
   ```

3. **Create Your First Dimension**
   ```python
   from schema_generator_client import create_dimension_metadata
   
   customer_dim = create_dimension_metadata(
       name="dim_customers",
       grain="One row per customer",
       primary_keys=["customer_id"],
       physical_columns=[
           {"name": "customer_id", "type": "text"},
           {"name": "email", "type": "text"}
       ]
   )
   
   # Validate and generate DDL
   generator.validate_metadata(customer_dim)
   ddl = generator.generate_table_ddl(customer_dim, "analytics")
   print(ddl)
   ```

## 📊 Business Use Cases

### E-commerce Data Warehouse
```python
from use_cases import ECommerceDataWarehouse

ecommerce = ECommerceDataWarehouse(generator)
ddl, dml = ecommerce.deploy_complete_model(execute=True)
```

### SaaS Metrics Platform
```python
from use_cases import SaaSMetricsDataMart

saas = SaaSMetricsDataMart(generator)
ddl, dml = saas.deploy_saas_metrics(execute=True)
```

### Financial Services Pipeline
```python
from use_cases import FinancialDataPipeline

financial = FinancialDataPipeline(generator)
account_dim = financial.create_account_dimension()
transactions_fact = financial.create_transactions_fact()
```

## 🔄 ETL Integrations

### Apache Airflow
```python
from etl_integrations import AirflowIntegration

# Use in Airflow DAG
def schema_task(**context):
    return AirflowIntegration.create_dynamic_schema_task(
        generator, pipeline_config
    )
```

### dbt Integration
```python
from etl_integrations import DBTIntegration

dbt = DBTIntegration(generator)
model_sql = dbt.generate_dbt_model(fact_metadata, "incremental")
schema_yml = dbt.generate_dbt_schema_yml(pipeline_metadata)
```

### Pandas DataFrames
```python
from etl_integrations import PandasIntegration

pandas_int = PandasIntegration(generator)
validation = pandas_int.validate_dataframe_against_schema(df, table_metadata)
insert_statements = pandas_int.generate_insert_statements(df, table_metadata)
```

### DuckDB Analytics
```python
from etl_integrations import DuckDBIntegration

# OLAP and embedded analytics
with DuckDBIntegration(generator) as duckdb:
    # Replicate PostgreSQL schema to DuckDB
    result = duckdb.create_schema_from_postgres("analytics", postgres_config)
    
    # Generate analytical queries
    queries = duckdb.generate_analytical_queries(fact_metadata)
    
    # Export to data lake
    duckdb.export_to_parquet("fact_sales", "/data/lake/sales.parquet")
```

### Polars High-Performance Processing
```python
from etl_integrations import PolarsIntegration

polars_int = PolarsIntegration(generator)

# Create schema-aware DataFrames
df = polars_int.dataframe_from_metadata(table_metadata, sample_data)

# SCD2 processing with lazy evaluation
scd2_pipeline = polars_int.create_scd2_pipeline(
    source_df.lazy(), current_df.lazy(), "business_key", ["name", "email"]
)

# Fast aggregations for fact tables
fact_aggs = polars_int.create_fact_aggregation_pipeline(
    fact_df.lazy(), ["dim1_key", "dim2_key"], measures, "day"
)

# Data quality validation
quality_results = polars_int.validate_data_quality(df.lazy(), table_metadata)
```

## 🔧 Advanced Features

### Schema Versioning
```python
from advanced_examples import SchemaVersionManager

version_manager = SchemaVersionManager(generator)
v1 = version_manager.create_version(pipeline, "1.0", "Initial schema")
migration = version_manager.generate_migration_script("1.0", "2.0")
```

### Multi-tenant Management
```python
from advanced_examples import MultiTenantSchemaManager

multi_tenant = MultiTenantSchemaManager(generator)
multi_tenant.set_base_pipeline(base_pipeline)
tenant_ddl = multi_tenant.create_tenant_schema("tenant_123")
```

### Performance Optimization
```python
from advanced_examples import PerformanceOptimizer

optimizer = PerformanceOptimizer(generator)
analysis = optimizer.analyze_table_performance(table_metadata, query_patterns)
optimization_sql = optimizer.generate_optimization_sql(analysis)
```

### Data Governance
```python
from advanced_examples import DataGovernanceManager

governance = DataGovernanceManager(generator)
tagged_pipeline = governance.tag_sensitive_data(pipeline)
lineage = governance.generate_data_lineage(pipeline)
compliance_report = governance.generate_compliance_report(pipeline)
```

## 🎯 Common Patterns

### Complete Dimensional Model Deployment
```python
# Define your dimensional model
pipeline = {
    "schema": "analytics",
    "tables": [
        create_dimension_metadata("dim_customers", "One row per customer", ["customer_id"]),
        create_dimension_metadata("dim_products", "One row per product", ["product_id"]),
        create_fact_metadata(
            name="fact_orders",
            grain="One row per order",
            dimension_references=[
                {"dimension": "dim_customers", "fk_column": "customer_sk"},
                {"dimension": "dim_products", "fk_column": "product_sk"}
            ],
            measures=[
                {"name": "order_amount", "type": "numeric(10,2)", "additivity": "additive"}
            ]
        )
    ]
}

# Deploy complete model
ddl, dml = generator.create_dimensional_model(pipeline, execute=True)
```

### Data Pipeline with Validation
```python
# Validate all metadata first
for table in pipeline['tables']:
    try:
        generator.validate_metadata(table)
        print(f"✅ {table['name']} validated")
    except Exception as e:
        print(f"❌ {table['name']} validation failed: {e}")

# Generate and review DDL before execution
ddl = generator.generate_pipeline_ddl(pipeline)
print("Generated DDL:")
print(ddl)

# Execute if validation passes
generator.create_tables(ddl)
```

## 🛠️ Error Handling

```python
from schema_generator_client import SchemaGeneratorError

try:
    generator.validate_metadata(table_metadata)
    ddl = generator.generate_table_ddl(table_metadata)
    generator.create_tables(ddl)
    print("✅ Table created successfully")
    
except SchemaGeneratorError as e:
    print(f"❌ Schema generation error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

## 📋 Requirements

- Python 3.7+
- PostgreSQL 12+
- psycopg2-binary
- pandas (for DataFrame integration)
- sqlalchemy (for advanced database operations)

## 🔗 Integration Examples

The Python client integrates seamlessly with:
- **Apache Airflow** - Dynamic schema generation in data pipelines
- **dbt** - Generate dbt models and schema.yml files
- **Pandas** - Validate DataFrames against schemas
- **SQLAlchemy** - ORM integration
- **FastAPI/Django** - Web application backends
- **Jupyter Notebooks** - Interactive data analysis

## 📈 Performance Tips

1. **Batch Operations**: Use `generate_pipeline_ddl()` for multiple tables
2. **Connection Pooling**: Reuse SchemaGenerator instances
3. **Validation First**: Always validate metadata before generation
4. **Schema Caching**: Cache generated DDL for repeated deployments
5. **Async Processing**: Use async patterns for large deployments

## 🔒 Security Best Practices

1. **Credential Management**: Use environment variables for database credentials
2. **SQL Injection Prevention**: The schema generator validates all identifiers
3. **Access Control**: Use database roles and permissions
4. **Audit Logging**: Enable database audit logs for schema changes
5. **Data Classification**: Use the governance features to tag sensitive data

Start with `use_cases.py` for practical examples, then explore the advanced features as your needs grow!