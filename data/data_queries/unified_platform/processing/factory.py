"""
Processing Engine Factory
========================

Factory pattern for creating processing engine instances.
"""

from typing import Dict, Any, Optional
from .abstract_engine import ProcessingEngineBackend

# Import all engine implementations
from .trino_engine import TrinoProcessingEngine
from .spark_engine import SparkProcessingEngine
from .polars_engine import PolarsProcessingEngine
from .duckdb_engine import DuckDBProcessingEngine
from .postgresql_engine import PostgreSQLProcessingEngine
from .clickhouse_engine import ClickHouseProcessingEngine


class ProcessingEngineFactory:
    """Factory for creating processing engine instances"""
    
    _engines = {
        'trino': TrinoProcessingEngine,
        'spark': SparkProcessingEngine,
        'polars': PolarsProcessingEngine,
        'duckdb': DuckDBProcessingEngine,
        'postgresql': PostgreSQLProcessingEngine,
        'clickhouse': ClickHouseProcessingEngine,
    }
    
    @classmethod
    def create_engine(cls, engine_type: str, **kwargs) -> ProcessingEngineBackend:
        """Create a processing engine instance"""
        
        engine_type_lower = engine_type.lower()
        
        if engine_type_lower not in cls._engines:
            raise ValueError(f"Unsupported engine type: {engine_type}. "
                           f"Available: {list(cls._engines.keys())}")
        
        engine_class = cls._engines[engine_type_lower]
        
        # Handle specific configuration for each engine
        if engine_type_lower == 'trino':
            return engine_class(
                coordinator_host=kwargs.get('coordinator_host', 'localhost'),
                port=kwargs.get('port', 8080),
                catalog=kwargs.get('catalog', 'hive')
            )
        
        elif engine_type_lower == 'spark':
            return engine_class(
                app_name=kwargs.get('app_name', 'UnifiedPlatformSpark'),
                master=kwargs.get('master', 'local[*]'),
                config=kwargs.get('config', {})
            )
        
        elif engine_type_lower == 'polars':
            return engine_class()  # Polars doesn't need configuration
        
        elif engine_type_lower == 'duckdb':
            return engine_class()  # DuckDB doesn't need configuration
        
        elif engine_type_lower == 'postgresql':
            return engine_class()  # PostgreSQL uses SQL
        
        elif engine_type_lower == 'clickhouse':
            return engine_class()  # ClickHouse uses SQL
        
        else:
            return engine_class(**kwargs)
    
    @classmethod
    def list_available_engines(cls) -> Dict[str, str]:
        """List all available processing engines"""
        return {
            engine_type: engine_class.__doc__ or engine_class.__name__
            for engine_type, engine_class in cls._engines.items()
        }
    
    @classmethod
    def get_engine_capabilities(cls, engine_type: str) -> Dict[str, Any]:
        """Get capabilities of a specific processing engine"""
        engine_type_lower = engine_type.lower()
        
        if engine_type_lower not in cls._engines:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        
        engine_class = cls._engines[engine_type_lower]
        
        # Create temporary instance to check capabilities
        try:
            temp_engine = engine_class()
            
            capabilities = {
                'optimal_query_patterns': temp_engine.get_optimal_query_patterns(),
                'supported_operations': [],
                'storage_compatibility': [],
                'performance_characteristics': {}
            }
            
            # Test common operations
            test_metadata = {'name': 'test_table'}
            common_operations = [
                'analytics', 'aggregation', 'joins', 'time_series',
                'machine_learning', 'streaming'
            ]
            
            for operation in common_operations:
                try:
                    query = temp_engine.generate_query(test_metadata, operation)
                    if query and len(query.strip()) > 50:  # Non-trivial query
                        capabilities['supported_operations'].append(operation)
                except (AttributeError, NotImplementedError):
                    pass
            
            # Test storage backend compatibility
            from ..storage.abstract_backend import StorageType
            for storage_type in StorageType:
                try:
                    if temp_engine.supports_storage_backend(storage_type):
                        capabilities['storage_compatibility'].append(storage_type.value)
                except (AttributeError, NotImplementedError):
                    pass
            
            # Get performance characteristics
            try:
                perf = temp_engine.estimate_query_performance(
                    "SELECT COUNT(*) FROM test", test_metadata
                )
                capabilities['performance_characteristics'] = perf
            except (AttributeError, NotImplementedError):
                pass
            
            return capabilities
            
        except Exception as e:
            return {'error': f"Could not determine capabilities: {str(e)}"}
    
    @classmethod
    def recommend_engine_for_workload(cls, workload_type: str, 
                                    data_size: str = 'medium',
                                    latency_requirement: str = 'medium') -> Dict[str, Any]:
        """Recommend best engine for a specific workload"""
        
        recommendations = {
            'real_time_analytics': {
                'primary': 'clickhouse',
                'alternatives': ['duckdb', 'polars'],
                'reason': 'Optimized for real-time analytical queries'
            },
            'batch_analytics': {
                'primary': 'spark',
                'alternatives': ['trino', 'duckdb'],
                'reason': 'Handles large-scale batch processing efficiently'
            },
            'interactive_analytics': {
                'primary': 'trino',
                'alternatives': ['duckdb', 'clickhouse'],
                'reason': 'Fast interactive queries across multiple data sources'
            },
            'data_preprocessing': {
                'primary': 'polars',
                'alternatives': ['spark', 'duckdb'],
                'reason': 'High-performance DataFrame operations'
            },
            'transactional_processing': {
                'primary': 'postgresql',
                'alternatives': [],
                'reason': 'ACID compliance and transactional guarantees'
            },
            'streaming_analytics': {
                'primary': 'spark',
                'alternatives': ['clickhouse'],
                'reason': 'Native streaming capabilities'
            },
            'federated_queries': {
                'primary': 'trino',
                'alternatives': ['spark'],
                'reason': 'Designed for querying across multiple data sources'
            }
        }
        
        # Adjust recommendations based on data size and latency
        base_rec = recommendations.get(workload_type, {
            'primary': 'duckdb',
            'alternatives': ['polars'],
            'reason': 'General-purpose analytics engine'
        })
        
        # Size-based adjustments
        if data_size == 'small':
            if base_rec['primary'] == 'spark':
                base_rec['primary'] = 'duckdb'
                base_rec['reason'] += ' (DuckDB preferred for small datasets)'
        elif data_size == 'large':
            if base_rec['primary'] in ['duckdb', 'polars']:
                base_rec['primary'] = 'spark'
                base_rec['reason'] += ' (Spark preferred for large datasets)'
        
        # Latency-based adjustments
        if latency_requirement == 'low':
            if base_rec['primary'] == 'spark':
                base_rec['alternatives'].insert(0, 'clickhouse')
                base_rec['reason'] += ' (Consider ClickHouse for lower latency)'
        
        return base_rec


def create_processing_engine(engine_type: str, **kwargs) -> ProcessingEngineBackend:
    """Convenience function to create engine instance"""
    return ProcessingEngineFactory.create_engine(engine_type, **kwargs)


def list_processing_engines() -> Dict[str, str]:
    """Convenience function to list available engines"""
    return ProcessingEngineFactory.list_available_engines()


def compare_processing_engines(engine_types: list) -> Dict[str, Any]:
    """Compare capabilities across multiple processing engines"""
    comparison = {}
    
    for engine_type in engine_types:
        try:
            capabilities = ProcessingEngineFactory.get_engine_capabilities(engine_type)
            comparison[engine_type] = capabilities
        except ValueError:
            comparison[engine_type] = {'error': 'Invalid engine type'}
    
    return comparison


def recommend_best_engine(workload_type: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to get engine recommendation"""
    return ProcessingEngineFactory.recommend_engine_for_workload(
        workload_type, **kwargs
    )