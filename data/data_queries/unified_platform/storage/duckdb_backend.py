"""
DuckDB Storage Backend Implementation
===================================

Implementation of DuckDB storage backend for embedded analytics.
"""

from typing import Dict, List, Any
from .abstract_backend import StorageBackend


class DuckDBBackend(StorageBackend):
    """DuckDB storage backend implementation"""
    
    def generate_ddl(self, table_metadata: Dict[str, Any]) -> str:
        """Generate DuckDB DDL"""
        table_name = table_metadata['name']
        entity_type = table_metadata.get('entity_type', 'dimension')
        columns = table_metadata.get('columns', [])
        
        if entity_type in ['fact', 'transaction_fact']:
            ddl = f"""
-- DuckDB Fact Table DDL (Optimized for Analytics)
CREATE TABLE {self.schema_name}.{table_name} (
"""
            # Add fact columns
            for col in columns:
                col_name = col.get('name', 'unknown_column')
                data_type = self._map_data_type(col.get('type', 'DECIMAL'))
                nullable = "" if col.get('nullable', True) else " NOT NULL"
                ddl += f"    {col_name} {data_type}{nullable},\n"
            
            ddl += f"""    transaction_date DATE NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for analytical queries
CREATE INDEX idx_{table_name}_date ON {self.schema_name}.{table_name}(transaction_date);
CREATE INDEX idx_{table_name}_created ON {self.schema_name}.{table_name}(created_at);
"""
        
        elif entity_type == 'dimension':
            ddl = f"""
-- DuckDB Dimension Table DDL
CREATE TABLE {self.schema_name}.{table_name} (
    {table_name}_sk BIGINT NOT NULL,
    {table_name}_id VARCHAR NOT NULL,
"""
            # Add dimension columns
            for col in columns:
                col_name = col.get('name', 'unknown_column')
                data_type = self._map_data_type(col.get('type', 'VARCHAR'))
                nullable = "" if col.get('nullable', True) else " NOT NULL"
                ddl += f"    {col_name} {data_type}{nullable},\n"
            
            # Add SCD2 columns
            ddl += f"""    valid_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_to TIMESTAMP,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY ({table_name}_sk)
);

-- Unique index on natural key for current records
CREATE UNIQUE INDEX idx_{table_name}_current 
ON {self.schema_name}.{table_name}({table_name}_id) 
WHERE is_current = TRUE;
"""
        
        return ddl.strip()
    
    def generate_dml(self, table_metadata: Dict[str, Any], operation: str) -> str:
        """Generate DuckDB DML"""
        table_name = table_metadata['name']
        
        if operation == 'insert':
            return f"""
-- DuckDB Insert
INSERT INTO {self.schema_name}.{table_name} 
({", ".join([col['name'] for col in table_metadata.get('columns', [])])})
VALUES 
({", ".join(['?' for _ in table_metadata.get('columns', [])])});
"""
        
        elif operation == 'analytical_query':
            return f"""
-- DuckDB Analytical Query
SELECT 
    DATE_TRUNC('month', transaction_date) as month,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MEDIAN(amount) as median_amount,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) as p95_amount
FROM {self.schema_name}.{table_name}
WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', transaction_date)
ORDER BY month DESC;
"""
        
        elif operation == 'parquet_export':
            return f"""
-- DuckDB Parquet Export
COPY (
    SELECT * FROM {self.schema_name}.{table_name}
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
) TO '{table_name}_export.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY);
"""
        
        elif operation == 'csv_import':
            return f"""
-- DuckDB CSV Import
CREATE TABLE {self.schema_name}.{table_name}_staging AS 
SELECT * FROM read_csv_auto('{table_name}_import.csv');

-- Insert with validation
INSERT INTO {self.schema_name}.{table_name}
SELECT * FROM {self.schema_name}.{table_name}_staging
WHERE amount IS NOT NULL AND amount > 0;

DROP TABLE {self.schema_name}.{table_name}_staging;
"""
        
        elif operation == 'window_analysis':
            return f"""
-- DuckDB Window Function Analysis
SELECT 
    transaction_date,
    amount,
    ROW_NUMBER() OVER (PARTITION BY transaction_date ORDER BY amount DESC) as daily_rank,
    LAG(amount) OVER (ORDER BY transaction_date, created_at) as prev_amount,
    SUM(amount) OVER (ORDER BY transaction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as rolling_7day_sum,
    AVG(amount) OVER (ORDER BY transaction_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as rolling_30day_avg
FROM {self.schema_name}.{table_name}
WHERE transaction_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY transaction_date DESC, amount DESC;
"""
        
        return f"-- DuckDB {operation} for {table_name}"
    
    def get_optimal_data_types(self) -> Dict[str, str]:
        """Get DuckDB optimal data types"""
        return {
            'id': 'VARCHAR',
            'name': 'VARCHAR',
            'description': 'TEXT',
            'amount': 'DECIMAL(18,2)',
            'quantity': 'BIGINT',
            'rate': 'DECIMAL(8,4)',
            'percentage': 'DECIMAL(5,2)',
            'flag': 'BOOLEAN',
            'date': 'DATE',
            'timestamp': 'TIMESTAMP',
            'created_at': 'TIMESTAMP',
            'updated_at': 'TIMESTAMP',
            'json_field': 'JSON',
            'array_field': 'VARCHAR[]',
            'struct_field': 'STRUCT(field1 VARCHAR, field2 BIGINT)'
        }
    
    def estimate_performance(self, query: str, table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate DuckDB performance"""
        entity_type = table_metadata.get('entity_type', 'dimension')
        estimated_size = table_metadata.get('estimated_size', 'medium')
        
        # DuckDB is very fast for analytical queries
        if any(keyword in query.upper() for keyword in ['GROUP BY', 'COUNT', 'SUM', 'AVG']):
            latency = '50ms - 2 seconds'  # Analytical aggregations
        elif 'WINDOW' in query.upper() or 'OVER' in query.upper():
            latency = '100ms - 5 seconds'  # Window functions
        elif 'JOIN' in query.upper():
            latency = '10ms - 1 second'  # Joins
        else:
            latency = '1-50ms'  # Simple queries
        
        return {
            'estimated_latency': latency,
            'throughput': 'millions of rows/second',
            'scalability': 'single_node_optimized',
            'memory_efficiency': 'excellent',
            'optimal_for': 'embedded_analytics',
            'columnar_processing': True,
            'vectorized_execution': True
        }
    
    def get_optimal_workloads(self) -> List[str]:
        """Get optimal workloads for DuckDB"""
        return [
            'embedded_analytics',
            'data_science_workflows',
            'etl_processing',
            'analytical_prototyping',
            'file_format_conversion',
            'data_exploration',
            'medium_scale_analytics',
            'local_data_processing',
            'csv_parquet_processing'
        ]
    
    def supports_feature(self, feature: str) -> bool:
        """Check DuckDB feature support"""
        supported_features = {
            'acid_transactions': True,
            'time_travel': False,
            'schema_evolution': True,
            'partition_pruning': False,
            'columnar_storage': True,
            'distributed_storage': False,
            'real_time_ingestion': True,
            'batch_processing': True,
            'window_functions': True,
            'json_support': True,
            'array_support': True,
            'regex_support': True,
            'file_format_support': True,
            'vectorized_execution': True,
            'memory_efficient': True
        }
        return supported_features.get(feature, False)
    
    def _map_data_type(self, generic_type: str) -> str:
        """Map generic data type to DuckDB type"""
        mapping = {
            'STRING': 'VARCHAR',
            'TEXT': 'TEXT',
            'INTEGER': 'INTEGER',
            'BIGINT': 'BIGINT',
            'DECIMAL': 'DECIMAL(18,2)',
            'FLOAT': 'REAL',
            'DOUBLE': 'DOUBLE',
            'BOOLEAN': 'BOOLEAN',
            'DATE': 'DATE',
            'TIMESTAMP': 'TIMESTAMP',
            'JSON': 'JSON',
            'UUID': 'UUID'
        }
        return mapping.get(generic_type.upper(), 'VARCHAR')