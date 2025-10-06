"""
Unified Platform Processing Engines
===================================

This module provides processing engine implementations for the unified data platform.
Supports multiple processing systems with consistent interfaces.
"""

from .abstract_engine import ProcessingEngineBackend
from .factory import (
    ProcessingEngineFactory,
    create_processing_engine,
    list_processing_engines,
    compare_processing_engines,
    recommend_best_engine
)

# Processing engine implementations
from .trino_engine import TrinoProcessingEngine
from .spark_engine import SparkProcessingEngine
from .polars_engine import PolarsProcessingEngine
from .duckdb_engine import DuckDBProcessingEngine
from .postgresql_engine import PostgreSQLProcessingEngine
from .clickhouse_engine import ClickHouseProcessingEngine

__all__ = [
    # Abstract base classes
    'ProcessingEngineBackend',
    
    # Factory functions
    'ProcessingEngineFactory',
    'create_processing_engine',
    'list_processing_engines',
    'compare_processing_engines',
    'recommend_best_engine',
    
    # Engine implementations
    'TrinoProcessingEngine',
    'SparkProcessingEngine',
    'PolarsProcessingEngine',
    'DuckDBProcessingEngine',
    'PostgreSQLProcessingEngine',
    'ClickHouseProcessingEngine',
]

# Convenience functions for quick access
def get_engine(engine_type: str, **kwargs):
    """Get a processing engine instance"""
    return create_processing_engine(engine_type, **kwargs)

def list_engines():
    """List all available processing engines"""
    return list_processing_engines()

def compare_engines(*engine_types):
    """Compare multiple processing engines"""
    return compare_processing_engines(list(engine_types))

def recommend_engine(workload_type: str, **kwargs):
    """Get engine recommendation for workload"""
    return recommend_best_engine(workload_type, **kwargs)