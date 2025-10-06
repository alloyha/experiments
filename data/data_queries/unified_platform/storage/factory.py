"""
Storage Backend Factory
======================

Factory pattern for creating storage backend instances.
"""

from typing import Dict, Any, Optional
from .abstract_backend import StorageBackend, StorageType

# Import all backend implementations
from .postgresql_backend import PostgreSQLStorageBackend
from .iceberg_backend import IcebergStorageBackend
from .clickhouse_backend import ClickHouseStorageBackend
from .duckdb_backend import DuckDBStorageBackend
from .bigquery_backend import BigQueryStorageBackend
from .snowflake_backend import SnowflakeStorageBackend
from .delta_lake_backend import DeltaLakeStorageBackend
from .parquet_backend import ParquetStorageBackend


class StorageBackendFactory:
    """Factory for creating storage backend instances"""
    
    _backends = {
        StorageType.POSTGRESQL: PostgreSQLStorageBackend,
        StorageType.ICEBERG: IcebergStorageBackend,
        StorageType.CLICKHOUSE: ClickHouseStorageBackend,
        StorageType.DUCKDB: DuckDBStorageBackend,
        StorageType.BIGQUERY: BigQueryStorageBackend,
        StorageType.SNOWFLAKE: SnowflakeStorageBackend,
        StorageType.DELTA_LAKE: DeltaLakeStorageBackend,
        StorageType.PARQUET: ParquetStorageBackend,
    }
    
    @classmethod
    def create_backend(cls, storage_type: StorageType, 
                      **kwargs) -> StorageBackend:
        """Create a storage backend instance"""
        
        if storage_type not in cls._backends:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        backend_class = cls._backends[storage_type]
        
        # Handle specific configuration for each backend
        if storage_type == StorageType.POSTGRESQL:
            return backend_class(
                host=kwargs.get('host', 'localhost'),
                database=kwargs.get('database', 'analytics'),
                schema=kwargs.get('schema', 'public')
            )
        
        elif storage_type == StorageType.ICEBERG:
            return backend_class(
                warehouse_path=kwargs.get('warehouse_path', '/warehouse/'),
                catalog_name=kwargs.get('catalog_name', 'iceberg_catalog')
            )
        
        elif storage_type == StorageType.CLICKHOUSE:
            return backend_class(
                host=kwargs.get('host', 'localhost'),
                database=kwargs.get('database', 'analytics')
            )
        
        elif storage_type == StorageType.DUCKDB:
            return backend_class(
                database_path=kwargs.get('database_path', 'analytics.duckdb')
            )
        
        elif storage_type == StorageType.BIGQUERY:
            return backend_class(
                project_id=kwargs.get('project_id', 'your-project'),
                dataset_id=kwargs.get('dataset_id', 'analytics')
            )
        
        elif storage_type == StorageType.SNOWFLAKE:
            return backend_class(
                warehouse_name=kwargs.get('warehouse_name', 'ANALYTICS_WH'),
                database_name=kwargs.get('database_name', 'ANALYTICS_DB'),
                schema_name=kwargs.get('schema_name', 'PUBLIC')
            )
        
        elif storage_type == StorageType.DELTA_LAKE:
            return backend_class(
                storage_path=kwargs.get('storage_path', '/delta/tables/'),
                catalog_name=kwargs.get('catalog_name', 'main')
            )
        
        elif storage_type == StorageType.PARQUET:
            return backend_class(
                storage_path=kwargs.get('storage_path', '/data/parquet/'),
                compression=kwargs.get('compression', 'snappy')
            )
        
        else:
            # Fallback for any missing implementations
            return backend_class(**kwargs)
    
    @classmethod
    def list_available_backends(cls) -> Dict[str, str]:
        """List all available storage backends"""
        return {
            storage_type.value: backend_class.__doc__ or backend_class.__name__
            for storage_type, backend_class in cls._backends.items()
        }
    
    @classmethod
    def get_backend_features(cls, storage_type: StorageType) -> Dict[str, Any]:
        """Get features supported by a specific backend"""
        if storage_type not in cls._backends:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        backend_class = cls._backends[storage_type]
        
        # Create a temporary instance to check features
        try:
            # Use minimal configuration for feature checking
            if storage_type == StorageType.BIGQUERY:
                temp_backend = backend_class(project_id="temp")
            elif storage_type == StorageType.POSTGRESQL:
                temp_backend = backend_class(host="temp")
            elif storage_type == StorageType.CLICKHOUSE:
                temp_backend = backend_class(host="temp")
            else:
                temp_backend = backend_class()
            
            # Common features to check
            features_to_check = [
                'partitioning', 'clustering', 'materialized_views', 'time_travel',
                'change_tracking', 'acid_transactions', 'schema_evolution',
                'compression', 'encryption', 'row_level_security'
            ]
            
            supported_features = {}
            for feature in features_to_check:
                try:
                    supported_features[feature] = temp_backend.supports_feature(feature)
                except (AttributeError, NotImplementedError):
                    supported_features[feature] = False
            
            # Add performance optimizations if available
            try:
                optimizations = temp_backend.get_performance_optimizations()
                supported_features['performance_optimizations'] = optimizations
            except (AttributeError, NotImplementedError):
                supported_features['performance_optimizations'] = []
            
            return supported_features
            
        except Exception as e:
            return {'error': f"Could not determine features: {str(e)}"}


def create_storage_backend(storage_type_str: str, **kwargs) -> StorageBackend:
    """Convenience function to create backend from string"""
    try:
        storage_type = StorageType(storage_type_str.lower())
        return StorageBackendFactory.create_backend(storage_type, **kwargs)
    except ValueError:
        available = list(StorageType)
        raise ValueError(f"Invalid storage type '{storage_type_str}'. "
                        f"Available types: {[t.value for t in available]}")


def list_storage_backends() -> Dict[str, str]:
    """Convenience function to list available backends"""
    return StorageBackendFactory.list_available_backends()


def compare_storage_backends(backend_types: list) -> Dict[str, Any]:
    """Compare features across multiple storage backends"""
    comparison = {}
    
    for backend_type_str in backend_types:
        try:
            storage_type = StorageType(backend_type_str.lower())
            features = StorageBackendFactory.get_backend_features(storage_type)
            comparison[backend_type_str] = features
        except ValueError:
            comparison[backend_type_str] = {'error': 'Invalid storage type'}
    
    return comparison