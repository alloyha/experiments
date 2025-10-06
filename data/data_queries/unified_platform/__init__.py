"""
Unified Data Platform

A revolutionary data platform that separates storage backends from processing engines,
enabling unprecedented flexibility and workload optimization.

This package provides:
- Storage abstraction layer supporting multiple backends
- Processing engine abstraction for optimal workload execution  
- Intelligent workload placement and optimization
- Cross-platform query federation capabilities
- Complete implementation framework

Example Usage:
    ```python
    from unified_platform import UnifiedDataPlatform
    from unified_platform.storage import StorageType, StorageConfig
    from unified_platform.processing import ProcessingEngine, ProcessingConfig
    
    # Configure platform
    platform = UnifiedDataPlatform(
        storage_configs=[...],
        processing_configs=[...]
    )
    
    # Create and execute plan
    plan = platform.create_execution_plan(table_metadata, "analytical_query")
    result = platform.execute_plan(plan)
    ```
"""

__version__ = "1.0.0"
__author__ = "Unified Data Platform Team"

# Core exports
from .orchestrator.platform import UnifiedDataPlatform
from .storage.abstract_backend import StorageType, StorageConfig
from .processing.abstract_engine import ProcessingEngine, ProcessingConfig

__all__ = [
    "UnifiedDataPlatform",
    "StorageType", 
    "StorageConfig",
    "ProcessingEngine",
    "ProcessingConfig"
]