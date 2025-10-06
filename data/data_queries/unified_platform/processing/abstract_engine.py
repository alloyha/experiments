"""
Processing Engine Abstraction
=============================

Abstract interfaces and core types for processing engines in the unified data platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from ..storage.abstract_backend import StorageType


class ProcessingEngine(Enum):
    """Supported processing engines"""
    POSTGRESQL = "postgresql"
    DUCKDB = "duckdb"
    POLARS = "polars"
    SPARK = "spark"
    TRINO = "trino"
    CLICKHOUSE = "clickhouse"


@dataclass
class ProcessingConfig:
    """Configuration for a processing engine"""
    engine_type: ProcessingEngine
    connection_params: Dict[str, Any]
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class QueryWorkload:
    """Defines a query workload pattern"""
    workload_type: str  # 'oltp', 'olap', 'streaming', 'batch'
    query_patterns: List[str]
    performance_requirements: Dict[str, Any]
    data_sources: List[StorageType]


class ProcessingEngineBackend(ABC):
    """Abstract base class for processing engines"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.engine_type = config.engine_type
    
    @abstractmethod
    def generate_query(self, table_metadata: Dict[str, Any], 
                      operation: str, **kwargs) -> str:
        """Generate query for this processing engine"""
        pass
    
    @abstractmethod
    def supports_storage_backend(self, storage_type: StorageType) -> bool:
        """Check if engine can work with storage backend"""
        pass
    
    @abstractmethod
    def estimate_query_performance(self, query: str, 
                                 table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate query performance characteristics"""
        pass
    
    @abstractmethod
    def get_optimal_query_patterns(self) -> List[str]:
        """Get optimal query patterns for this engine"""
        pass
    
    def get_sample_connection_config(self) -> Dict[str, Any]:
        """Get sample connection configuration for this engine"""
        return {
            "engine_type": self.engine_type.value,
            "connection_params": {},
            "properties": {}
        }
    
    def get_storage_compatibility(self) -> Dict[StorageType, bool]:
        """Get storage backend compatibility matrix"""
        return {
            storage_type: self.supports_storage_backend(storage_type)
            for storage_type in StorageType
        }