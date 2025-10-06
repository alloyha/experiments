"""
Unified Platform Storage Backends
=================================

This module provides storage backend implementations for the unified data platform.
Supports multiple storage systems with consistent interfaces.
"""

from .abstract_backend import StorageBackend, StorageType
from .factory import (
    StorageBackendFactory, 
    create_storage_backend, 
    list_storage_backends,
    compare_storage_backends
)

# Storage backend implementations
from .postgresql_backend import PostgreSQLStorageBackend
from .iceberg_backend import IcebergStorageBackend
from .clickhouse_backend import ClickHouseStorageBackend
from .duckdb_backend import DuckDBStorageBackend
from .bigquery_backend import BigQueryStorageBackend
from .snowflake_backend import SnowflakeStorageBackend
from .delta_lake_backend import DeltaLakeStorageBackend
from .parquet_backend import ParquetStorageBackend

__all__ = [
    # Abstract base classes
    'StorageBackend',
    'StorageType',
    
    # Factory functions
    'StorageBackendFactory',
    'create_storage_backend',
    'list_storage_backends', 
    'compare_storage_backends',
    
    # Backend implementations
    'PostgreSQLStorageBackend',
    'IcebergStorageBackend',
    'ClickHouseStorageBackend',
    'DuckDBStorageBackend',
    'BigQueryStorageBackend',
    'SnowflakeStorageBackend',
    'DeltaLakeStorageBackend',
    'ParquetStorageBackend',
]

# Convenience functions for quick access
def get_backend(storage_type: str, **kwargs):
    """Get a storage backend instance"""
    return create_storage_backend(storage_type, **kwargs)

def list_backends():
    """List all available storage backends"""
    return list_storage_backends()

def compare_backends(*backend_types):
    """Compare multiple storage backends"""
    return compare_storage_backends(list(backend_types))