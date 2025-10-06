# ETL Integrations Expansion: DuckDB + Polars

## Overview

The ETL integrations module has been expanded with two powerful new integrations:

- **🦆 DuckDB**: Embedded OLAP database for fast analytics
- **⚡ Polars**: High-performance DataFrame library with lazy evaluation

## New Capabilities Added

### DuckDB Integration (`DuckDBIntegration` class)

**Key Features:**
- Schema replication from PostgreSQL to DuckDB
- Analytical query generation for OLAP workloads
- Data lake integration with Parquet export
- Performance optimization for analytical queries

**Methods:**
- `create_schema_from_postgres()`: Replicate PostgreSQL schemas in DuckDB
- `generate_analytical_queries()`: Create fact/dimension analytics
- `optimize_for_analytics()`: Generate performance optimizations
- `export_to_parquet()`: Export to data lake formats

**Use Cases:**
```python
with DuckDBIntegration(generator) as duckdb:
    # Replicate production schema for analytics
    result = duckdb.create_schema_from_postgres("analytics", postgres_config)
    
    # Generate pre-built analytical queries
    queries = duckdb.generate_analytical_queries(fact_metadata)
    
    # Export aggregated data to data lake
    duckdb.export_to_parquet("fact_sales", "/data/lake/sales.parquet")
```

### Polars Integration (`PolarsIntegration` class)

**Key Features:**
- Schema-aware DataFrame creation from metadata
- SCD2 processing with lazy evaluation
- High-performance fact table aggregations
- Data quality validation at scale
- Memory-efficient data processing

**Methods:**
- `dataframe_from_metadata()`: Create DataFrames with correct schema
- `create_scd2_pipeline()`: Lazy SCD2 change detection
- `create_fact_aggregation_pipeline()`: High-speed aggregations
- `validate_data_quality()`: Comprehensive data validation
- `optimize_for_performance()`: Memory and speed optimizations
- `export_to_postgres()`: Bulk data export

**Use Cases:**
```python
polars_int = PolarsIntegration(generator)

# Create schema-aware DataFrame
df = polars_int.dataframe_from_metadata(table_metadata, sample_data)

# Process SCD2 changes with lazy evaluation
scd2_pipeline = polars_int.create_scd2_pipeline(
    source_df.lazy(), current_df.lazy(), "business_key", ["name", "email"]
)

# High-performance fact aggregations
fact_aggs = polars_int.create_fact_aggregation_pipeline(
    fact_df.lazy(), ["dim1_key", "dim2_key"], measures, "day"
)
```

## Integration Benefits

### DuckDB Advantages:
- **Embedded Analytics**: No separate database server required
- **Fast OLAP**: Columnar storage optimized for analytics
- **Data Lake Integration**: Native Parquet support
- **Cross-Database Queries**: Query PostgreSQL + files together
- **Zero-ETL Analytics**: Analyze data where it lives

### Polars Advantages:
- **Lightning Speed**: 10-100x faster than Pandas for large data
- **Lazy Evaluation**: Optimal query planning and execution
- **Memory Efficient**: Process datasets larger than RAM
- **Type Safety**: Strong typing prevents runtime errors
- **Modern API**: Expressive and intuitive syntax

## Installation

### Basic Setup:
```bash
# Core dependencies (existing)
pip install psycopg2-binary pandas sqlalchemy

# New optional dependencies
pip install duckdb>=0.8.0 polars>=0.19.0
```

### Quick Setup:
```bash
python setup.py install_optional
```

## Usage Patterns

### 1. OLAP Analytics Pipeline
```python
# Use DuckDB for fast analytical queries
with DuckDBIntegration(generator) as duckdb:
    # Replicate dimensional model to DuckDB
    duckdb.create_schema_from_postgres("analytics", config)
    
    # Generate and run analytical queries
    queries = duckdb.generate_analytical_queries(fact_metadata)
    daily_sales = duckdb.conn.execute(queries['daily_aggregates']).fetchall()
```

### 2. High-Performance ETL
```python
# Use Polars for fast data processing
polars_int = PolarsIntegration(generator)

# Process large datasets efficiently
processed_df = (
    raw_df.lazy()
    .pipe(lambda df: polars_int.create_scd2_pipeline(df, current_df.lazy(), "id", ["name"]))
    .pipe(lambda df: polars_int.optimize_for_performance(df))
    .collect()
)
```

### 3. Hybrid Workflows
```python
# Combine tools for optimal performance
# 1. Use Polars for data preparation
cleaned_df = polars_int.validate_and_clean(raw_df)

# 2. Export to PostgreSQL via bulk insert
polars_int.export_to_postgres(cleaned_df, table_metadata, config)

# 3. Use DuckDB for analytics
with DuckDBIntegration(generator) as duckdb:
    analytics = duckdb.generate_analytical_queries(table_metadata)
```

## Performance Comparisons

| Operation | Pandas | Polars | DuckDB | Use Case |
|-----------|--------|--------|---------|----------|
| Large CSV Processing | 1x | 10-50x | 5-20x | Data ingestion |
| Aggregations | 1x | 5-30x | 10-100x | Analytics |
| Joins | 1x | 5-15x | 20-50x | Data integration |
| Memory Usage | High | Low | Medium | Resource efficiency |

## Architecture Integration

```
PostgreSQL (OLTP)
    ↓ Schema Generator
    ↓ Metadata
    ├── Polars (ETL Processing)
    │   ├── Data Quality
    │   ├── SCD2 Processing  
    │   └── Transformations
    │
    └── DuckDB (OLAP Analytics)
        ├── Aggregations
        ├── Data Lake Queries
        └── Cross-DB Analytics
```

## Migration Guide

### Existing Pandas Code:
```python
# Before (Pandas)
df = pd.read_csv('large_file.csv')
result = df.groupby(['dim1', 'dim2']).agg({'measure': 'sum'})

# After (Polars)
polars_int = PolarsIntegration(generator)
df = pl.read_csv('large_file.csv').lazy()
result = polars_int.create_fact_aggregation_pipeline(
    df, ['dim1', 'dim2'], [{'name': 'measure', 'additivity': 'additive'}], 'day'
).collect()
```

### Existing SQL Analytics:
```sql
-- Before (Manual SQL)
SELECT date_trunc('day', event_time), SUM(amount)
FROM fact_sales 
GROUP BY date_trunc('day', event_time);

-- After (Generated DuckDB)
with DuckDBIntegration(generator) as duckdb:
    queries = duckdb.generate_analytical_queries(fact_metadata)
    # Automatically generates optimized analytical queries
```

## Best Practices

1. **Use Polars for**: Large-scale ETL, data cleaning, SCD processing
2. **Use DuckDB for**: Ad-hoc analytics, data exploration, OLAP queries  
3. **Use Pandas for**: Small datasets, prototyping, existing integrations
4. **Combine tools**: Leverage each tool's strengths in the same pipeline

## Future Enhancements

Potential future additions:
- **Apache Arrow**: Zero-copy data sharing between tools
- **Spark**: Big data ecosystem integration
- **ClickHouse**: Time-series analytics
- **Delta Lake**: ACID transactions for data lakes