"""
PostgreSQL Storage Backend Implementation
========================================

Implementation of PostgreSQL storage backend for OLTP workloads.
"""

from typing import Dict, List, Any
from .abstract_backend import StorageBackend


class PostgreSQLBackend(StorageBackend):
    """PostgreSQL storage backend implementation"""
    
    def generate_ddl(self, table_metadata: Dict[str, Any]) -> str:
        """Generate PostgreSQL DDL"""
        table_name = table_metadata['name']
        entity_type = table_metadata.get('entity_type', 'dimension')
        columns = table_metadata.get('columns', [])
        
        if entity_type == 'dimension':
            ddl = f"""
-- PostgreSQL Dimension Table DDL
CREATE TABLE {self.schema_name}.{table_name} (
    {table_name}_sk BIGSERIAL PRIMARY KEY,
    {table_name}_id VARCHAR(50) NOT NULL UNIQUE,
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
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_{table_name}_id ON {self.schema_name}.{table_name}({table_name}_id);
CREATE INDEX idx_{table_name}_current ON {self.schema_name}.{table_name}(is_current) WHERE is_current = TRUE;
CREATE INDEX idx_{table_name}_valid_from ON {self.schema_name}.{table_name}(valid_from);
"""
        
        elif entity_type in ['fact', 'transaction_fact']:
            ddl = f"""
-- PostgreSQL Fact Table DDL
CREATE TABLE {self.schema_name}.{table_name} (
    {table_name}_sk BIGSERIAL PRIMARY KEY,
"""
            # Add fact columns
            for col in columns:
                col_name = col.get('name', 'unknown_column')
                data_type = self._map_data_type(col.get('type', 'DECIMAL'))
                nullable = "" if col.get('nullable', True) else " NOT NULL"
                ddl += f"    {col_name} {data_type}{nullable},\n"
            
            ddl += f"""    transaction_date DATE NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Partitioning for large fact tables
CREATE INDEX idx_{table_name}_date ON {self.schema_name}.{table_name}(transaction_date);
"""
        
        return ddl.strip()
    
    def generate_dml(self, table_metadata: Dict[str, Any], operation: str) -> str:
        """Generate PostgreSQL DML"""
        table_name = table_metadata['name']
        
        if operation == 'insert':
            return f"""
-- PostgreSQL Insert
INSERT INTO {self.schema_name}.{table_name} 
({", ".join([col['name'] for col in table_metadata.get('columns', [])])})
VALUES 
({", ".join(['%s' for _ in table_metadata.get('columns', [])])});
"""
        
        elif operation == 'update':
            return f"""
-- PostgreSQL Update
UPDATE {self.schema_name}.{table_name} 
SET updated_at = CURRENT_TIMESTAMP,
    {{column_updates}}
WHERE {table_name}_id = %s;
"""
        
        elif operation == 'scd2_update':
            return f"""
-- PostgreSQL SCD2 Update
BEGIN;

-- Expire current record
UPDATE {self.schema_name}.{table_name} 
SET valid_to = CURRENT_TIMESTAMP, 
    is_current = FALSE,
    updated_at = CURRENT_TIMESTAMP
WHERE {table_name}_id = %s AND is_current = TRUE;

-- Insert new record
INSERT INTO {self.schema_name}.{table_name} 
({table_name}_id, {{columns}}, valid_from, is_current, created_at)
VALUES (%s, {{values}}, CURRENT_TIMESTAMP, TRUE, CURRENT_TIMESTAMP);

COMMIT;
"""
        
        return f"-- PostgreSQL {operation} for {table_name}"
    
    def get_optimal_data_types(self) -> Dict[str, str]:
        """Get PostgreSQL optimal data types"""
        return {
            'id': 'VARCHAR(50)',
            'name': 'VARCHAR(255)',
            'description': 'TEXT',
            'amount': 'DECIMAL(18,2)',
            'quantity': 'INTEGER',
            'rate': 'DECIMAL(8,4)',
            'percentage': 'DECIMAL(5,2)',
            'flag': 'BOOLEAN',
            'date': 'DATE',
            'timestamp': 'TIMESTAMP',
            'created_at': 'TIMESTAMP',
            'updated_at': 'TIMESTAMP'
        }
    
    def estimate_performance(self, query: str, table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate PostgreSQL performance"""
        entity_type = table_metadata.get('entity_type', 'dimension')
        estimated_size = table_metadata.get('estimated_size', 'medium')
        
        if 'SELECT' in query.upper() and 'WHERE' in query.upper():
            latency = '< 100ms' if entity_type == 'dimension' else '100-500ms'
        else:
            latency = '< 50ms'
        
        return {
            'estimated_latency': latency,
            'throughput': '1000+ TPS',
            'scalability': 'vertical',
            'acid_compliance': True,
            'optimal_for': 'OLTP workloads'
        }
    
    def get_optimal_workloads(self) -> List[str]:
        """Get optimal workloads for PostgreSQL"""
        return [
            'oltp',
            'transactional_processing',
            'customer_management',
            'order_processing',
            'real_time_updates',
            'dimension_tables',
            'small_to_medium_analytics'
        ]
    
    def supports_feature(self, feature: str) -> bool:
        """Check PostgreSQL feature support"""
        supported_features = {
            'acid_transactions': True,
            'time_travel': False,
            'schema_evolution': True,  # via migrations
            'partition_pruning': True,
            'columnar_storage': False,
            'distributed_storage': False,
            'real_time_ingestion': True,
            'batch_processing': True,
            'concurrent_access': True,
            'full_text_search': True,
            'json_support': True,
            'window_functions': True
        }
        return supported_features.get(feature, False)
    
    def _map_data_type(self, generic_type: str) -> str:
        """Map generic data type to PostgreSQL type"""
        mapping = {
            'STRING': 'VARCHAR(255)',
            'TEXT': 'TEXT',
            'INTEGER': 'INTEGER',
            'BIGINT': 'BIGINT',
            'DECIMAL': 'DECIMAL(18,2)',
            'FLOAT': 'REAL',
            'DOUBLE': 'DOUBLE PRECISION',
            'BOOLEAN': 'BOOLEAN',
            'DATE': 'DATE',
            'TIMESTAMP': 'TIMESTAMP',
            'JSON': 'JSONB'
        }
        return mapping.get(generic_type.upper(), 'VARCHAR(255)')