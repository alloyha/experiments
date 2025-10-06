"""
Storage Backend Abstraction
===========================

Abstract interfaces and core types for storage backends in the unified data platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class StorageType(Enum):
    """Enumeration of supported storage backend types"""
    POSTGRESQL = "postgresql"
    ICEBERG = "iceberg"
    CLICKHOUSE = "clickhouse"
    DUCKDB = "duckdb"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    DELTA_LAKE = "delta_lake"
    PARQUET = "parquet"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    DELTA_LAKE = "delta_lake"
    PARQUET_FILES = "parquet_files"


@dataclass
class StorageConfig:
    """Configuration for a storage backend"""
    storage_type: StorageType
    connection_params: Dict[str, Any]
    schema_name: str = "public"
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class TablePlacement:
    """Defines where and how a table should be stored"""
    table_name: str
    storage_backend: StorageType
    rationale: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.storage_type = config.storage_type
        self.schema_name = config.schema_name
    
    @abstractmethod
    def generate_ddl(self, table_metadata: Dict[str, Any]) -> str:
        """Generate DDL for this storage backend"""
        pass
    
    @abstractmethod
    def generate_dml(self, table_metadata: Dict[str, Any], operation: str) -> str:
        """Generate DML for this storage backend"""
        pass
    
    @abstractmethod
    def get_optimal_data_types(self) -> Dict[str, str]:
        """Get optimal data type mappings for this backend"""
        pass
    
    @abstractmethod
    def estimate_performance(self, query: str, table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate query performance for this backend"""
        pass
    
    @abstractmethod
    def get_optimal_workloads(self) -> List[str]:
        """Get list of optimal workload types for this backend"""
        pass
    
    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """Check if backend supports a specific feature"""
        pass
    
    def get_sample_connection_config(self) -> Dict[str, Any]:
        """Get sample connection configuration for this backend"""
        return {
            "storage_type": self.storage_type.value,
            "connection_params": {},
            "schema_name": "analytics"
        }
    
    def get_feature_matrix(self) -> Dict[str, bool]:
        """Get feature support matrix for this backend"""
        return {
            "acid_transactions": self.supports_feature("acid_transactions"),
            "time_travel": self.supports_feature("time_travel"),
            "schema_evolution": self.supports_feature("schema_evolution"),
            "partition_pruning": self.supports_feature("partition_pruning"),
            "columnar_storage": self.supports_feature("columnar_storage"),
            "distributed_storage": self.supports_feature("distributed_storage"),
            "real_time_ingestion": self.supports_feature("real_time_ingestion"),
            "batch_processing": self.supports_feature("batch_processing")
        }