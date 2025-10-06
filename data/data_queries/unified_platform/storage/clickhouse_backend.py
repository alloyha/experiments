"""
ClickHouse Storage Backend Implementation
=======================================

Implementation of ClickHouse storage backend for OLAP workloads.
"""

from typing import Dict, List, Any
from .abstract_backend import StorageBackend


class ClickHouseBackend(StorageBackend):
    """ClickHouse storage backend implementation"""
    
    def generate_ddl(self, table_metadata: Dict[str, Any]) -> str:
        """Generate ClickHouse DDL"""
        table_name = table_metadata['name']
        entity_type = table_metadata.get('entity_type', 'dimension')
        columns = table_metadata.get('columns', [])
        
        if entity_type in ['fact', 'transaction_fact']:
            ddl = f"""
-- ClickHouse Fact Table DDL (Optimized for Analytics)
CREATE TABLE {self.schema_name}.{table_name}
(
"""
            # Add fact columns
            for col in columns:
                col_name = col.get('name', 'unknown_column')
                data_type = self._map_data_type(col.get('type', 'Decimal64'))
                nullable = " Nullable" if col.get('nullable', True) else ""
                ddl += f"    {col_name} {data_type}{nullable},\n"
            
            ddl += f"""    transaction_date Date,
    transaction_hour UInt8,
    created_at DateTime,
    updated_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(transaction_date)
ORDER BY (transaction_date, transaction_hour, {columns[0].get('name', 'id') if columns else 'transaction_date'})
SETTINGS index_granularity = 8192;

-- Materialized view for real-time aggregations
CREATE MATERIALIZED VIEW {self.schema_name}.{table_name}_hourly_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(transaction_date)
ORDER BY (transaction_date, transaction_hour)
AS SELECT
    transaction_date,
    transaction_hour,
    count() as transaction_count,
    sum(amount) as total_amount,
    avg(amount) as avg_amount
FROM {self.schema_name}.{table_name}
GROUP BY transaction_date, transaction_hour;
"""
        
        elif entity_type == 'dimension':
            ddl = f"""
-- ClickHouse Dimension Table DDL
CREATE TABLE {self.schema_name}.{table_name}
(
    {table_name}_sk UInt64,
    {table_name}_id String,
"""
            # Add dimension columns
            for col in columns:
                col_name = col.get('name', 'unknown_column')
                data_type = self._map_data_type(col.get('type', 'String'))
                nullable = " Nullable" if col.get('nullable', True) else ""
                ddl += f"    {col_name} {data_type}{nullable},\n"
            
            # Add SCD2 columns
            ddl += f"""    valid_from DateTime,
    valid_to Nullable(DateTime),
    is_current UInt8,
    created_at DateTime,
    updated_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY ({table_name}_id, valid_from)
SETTINGS index_granularity = 8192;

-- Create dictionary for fast lookups
CREATE DICTIONARY {self.schema_name}.{table_name}_dict
(
    {table_name}_id String,
    {table_name}_sk UInt64
)
PRIMARY KEY {table_name}_id
SOURCE(CLICKHOUSE(
    HOST 'localhost'
    PORT 9000
    USER 'default'
    TABLE '{table_name}'
    DB '{self.schema_name}'
    WHERE 'is_current = 1'
))
LAYOUT(HASHED())
LIFETIME(MIN 300 MAX 600);
"""
        
        return ddl.strip()
    
    def generate_dml(self, table_metadata: Dict[str, Any], operation: str) -> str:
        """Generate ClickHouse DML"""
        table_name = table_metadata['name']
        
        if operation == 'insert':
            return f"""
-- ClickHouse Insert (Batch optimized)
INSERT INTO {self.schema_name}.{table_name} 
({", ".join([col['name'] for col in table_metadata.get('columns', [])])})
VALUES 
({", ".join(['?' for _ in table_metadata.get('columns', [])])});
"""
        
        elif operation == 'analytical_query':
            return f"""
-- ClickHouse Analytical Query
SELECT 
    toStartOfDay(transaction_date) as day,
    toHour(created_at) as hour,
    count() as transaction_count,
    sum(amount) as total_amount,
    avg(amount) as avg_amount,
    median(amount) as median_amount,
    quantile(0.95)(amount) as p95_amount
FROM {self.schema_name}.{table_name}
WHERE transaction_date >= today() - INTERVAL 30 DAY
GROUP BY day, hour
ORDER BY day DESC, hour;
"""
        
        elif operation == 'time_series_analysis':
            return f"""
-- ClickHouse Time Series Analysis
WITH time_buckets AS (
    SELECT 
        toStartOfInterval(created_at, INTERVAL 1 HOUR) as time_bucket,
        count() as events,
        sum(amount) as total_amount
    FROM {self.schema_name}.{table_name}
    WHERE created_at >= now() - INTERVAL 7 DAY
    GROUP BY time_bucket
)
SELECT 
    time_bucket,
    events,
    total_amount,
    lagInFrame(events) OVER (ORDER BY time_bucket) as prev_events,
    (events - lagInFrame(events) OVER (ORDER BY time_bucket)) / lagInFrame(events) OVER (ORDER BY time_bucket) * 100 as growth_rate
FROM time_buckets
ORDER BY time_bucket;
"""
        
        elif operation == 'real_time_aggregation':
            return f"""
-- ClickHouse Real-time Aggregation
SELECT 
    transaction_date,
    transaction_hour,
    transaction_count,
    total_amount,
    avg_amount
FROM {self.schema_name}.{table_name}_hourly_mv
WHERE transaction_date = today()
ORDER BY transaction_hour DESC;
"""
        
        return f"-- ClickHouse {operation} for {table_name}"
    
    def get_optimal_data_types(self) -> Dict[str, str]:
        """Get ClickHouse optimal data types"""
        return {
            'id': 'String',
            'name': 'String',
            'description': 'String',
            'amount': 'Decimal64(2)',
            'quantity': 'UInt64',
            'rate': 'Decimal32(4)',
            'percentage': 'Decimal32(2)',
            'flag': 'UInt8',
            'date': 'Date',
            'timestamp': 'DateTime',
            'created_at': 'DateTime',
            'updated_at': 'DateTime',
            'enum_field': 'Enum8',
            'array_field': 'Array(String)',
            'map_field': 'Map(String, String)',
            'nested_field': 'Nested(field1 String, field2 UInt64)'
        }
    
    def estimate_performance(self, query: str, table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate ClickHouse performance"""
        entity_type = table_metadata.get('entity_type', 'dimension')
        estimated_size = table_metadata.get('estimated_size', 'medium')
        
        # ClickHouse excels at analytical queries
        if any(keyword in query.upper() for keyword in ['GROUP BY', 'COUNT', 'SUM', 'AVG']):
            latency = '100ms - 2 seconds'  # Analytical aggregations
        elif 'WHERE' in query.upper():
            latency = '50-500ms'  # Filtered queries
        else:
            latency = '10-100ms'  # Point queries
        
        return {
            'estimated_latency': latency,
            'throughput': 'billions of rows/second',
            'scalability': 'horizontal',
            'compression_ratio': '10:1 typical',
            'optimal_for': 'analytical_workloads',
            'columnar_efficiency': True,
            'real_time_analytics': True
        }
    
    def get_optimal_workloads(self) -> List[str]:
        """Get optimal workloads for ClickHouse"""
        return [
            'olap_analytics',
            'time_series_analysis',
            'real_time_dashboards',
            'business_intelligence',
            'log_analytics',
            'event_analysis',
            'aggregation_heavy_queries',
            'large_scale_reporting',
            'data_warehouse_queries'
        ]
    
    def supports_feature(self, feature: str) -> bool:
        """Check ClickHouse feature support"""
        supported_features = {
            'acid_transactions': False,  # Eventually consistent
            'time_travel': False,
            'schema_evolution': True,
            'partition_pruning': True,
            'columnar_storage': True,
            'distributed_storage': True,
            'real_time_ingestion': True,
            'batch_processing': True,
            'materialized_views': True,
            'dictionaries': True,
            'compression': True,
            'parallel_processing': True,
            'approximate_functions': True,
            'window_functions': True
        }
        return supported_features.get(feature, False)
    
    def _map_data_type(self, generic_type: str) -> str:
        """Map generic data type to ClickHouse type"""
        mapping = {
            'STRING': 'String',
            'TEXT': 'String',
            'INTEGER': 'Int64',
            'BIGINT': 'Int64',
            'DECIMAL': 'Decimal64(2)',
            'FLOAT': 'Float32',
            'DOUBLE': 'Float64',
            'BOOLEAN': 'UInt8',
            'DATE': 'Date',
            'TIMESTAMP': 'DateTime',
            'UUID': 'UUID'
        }
        return mapping.get(generic_type.upper(), 'String')