"""
Storage Backend Factory
======================

Factory for creating and managing storage backend instances.
"""

from typing import Dict, Type, List
from .abstract_backend import StorageBackend, StorageConfig, StorageType
from .postgresql_backend import PostgreSQLBackend
from .iceberg_backend import IcebergBackend
from .clickhouse_backend import ClickHouseBackend
from .duckdb_backend import DuckDBBackend


class StorageBackendFactory:
    """Factory for creating storage backend instances"""
    
    _backends: Dict[StorageType, Type[StorageBackend]] = {
        StorageType.POSTGRESQL: PostgreSQLBackend,
        StorageType.ICEBERG: IcebergBackend,
        StorageType.CLICKHOUSE: ClickHouseBackend,
        StorageType.DUCKDB: DuckDBBackend,
    }
    
    @classmethod
    def create_backend(cls, config: StorageConfig) -> StorageBackend:
        """Create a storage backend instance"""
        backend_class = cls._backends.get(config.storage_type)
        if not backend_class:
            raise ValueError(f"Unsupported storage type: {config.storage_type}")
        
        return backend_class(config)
    
    @classmethod
    def get_supported_backends(cls) -> List[StorageType]:
        """Get list of supported storage backends"""
        return list(cls._backends.keys())
    
    @classmethod
    def register_backend(cls, storage_type: StorageType, backend_class: Type[StorageBackend]):
        """Register a new storage backend type"""
        cls._backends[storage_type] = backend_class
    
    @classmethod
    def get_backend_capabilities(cls) -> Dict[StorageType, Dict[str, any]]:
        """Get capabilities matrix for all backends"""
        capabilities = {}
        
        for storage_type, backend_class in cls._backends.items():
            # Create a temporary instance to get capabilities
            temp_config = StorageConfig(
                storage_type=storage_type,
                connection_params={},
                schema_name="temp"
            )
            temp_backend = backend_class(temp_config)
            
            capabilities[storage_type] = {
                'optimal_workloads': temp_backend.get_optimal_workloads(),
                'feature_matrix': temp_backend.get_feature_matrix(),
                'data_types': temp_backend.get_optimal_data_types()
            }
        
        return capabilities