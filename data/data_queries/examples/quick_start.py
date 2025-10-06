"""
Quick Start Example for Unified Data Platform
=============================================

This example demonstrates basic usage of the unified data platform.
"""

import sys
import os

# Add the parent directory to the path to import unified_platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_platform import UnifiedDataPlatform, StorageType, StorageConfig, ProcessingEngine, ProcessingConfig


def main():
    """Demonstrate basic platform usage"""
    
    print("🚀 Unified Data Platform - Quick Start Example")
    print("=" * 50)
    
    # Configure storage backends
    storage_configs = [
        StorageConfig(
            storage_type=StorageType.POSTGRESQL,
            connection_params={
                "host": "localhost",
                "port": 5432,
                "database": "analytics",
                "user": "admin"
            },
            schema_name="public"
        ),
        StorageConfig(
            storage_type=StorageType.ICEBERG,
            connection_params={
                "catalog_type": "hive",
                "warehouse_path": "s3://analytics-warehouse/",
                "catalog_uri": "thrift://localhost:9083"
            },
            schema_name="analytics"
        )
    ]
    
    # Configure processing engines
    processing_configs = [
        ProcessingConfig(
            engine_type=ProcessingEngine.POSTGRESQL,
            connection_params={
                "host": "localhost",
                "port": 5432,
                "database": "analytics"
            }
        ),
        ProcessingConfig(
            engine_type=ProcessingEngine.TRINO,
            connection_params={
                "host": "localhost", 
                "port": 8080,
                "catalog": "analytics"
            }
        ),
        ProcessingConfig(
            engine_type=ProcessingEngine.POLARS,
            connection_params={}
        )
    ]
    
    # Create platform
    platform = UnifiedDataPlatform(
        storage_configs=storage_configs,
        processing_configs=processing_configs,
        optimization_enabled=True,
        auto_placement=True
    )
    
    # Show platform overview
    overview = platform.get_platform_overview()
    print("\n📊 Platform Overview:")
    print(f"  Storage Backends: {', '.join(overview['storage_backends'])}")
    print(f"  Processing Engines: {', '.join(overview['processing_engines'])}")
    print(f"  Total Combinations: {overview['total_combinations']}")
    print(f"  Auto Placement: {'✅ Enabled' if overview['auto_placement'] else '❌ Disabled'}")
    
    # Example 1: OLTP workload
    print("\n🔄 Example 1: OLTP Customer Management")
    print("-" * 40)
    
    customer_metadata = {
        "name": "customers",
        "entity_type": "dimension",
        "estimated_size": "medium",
        "access_pattern": "frequent_updates"
    }
    
    oltp_plan = platform.create_execution_plan(customer_metadata, "transactional_query")
    print(f"  Optimal Storage: {oltp_plan.storage_backend.value}")
    print(f"  Optimal Processing: {oltp_plan.processing_engine.value}")
    print(f"  Rationale: {oltp_plan.rationale}")
    
    # Execute plan
    result = platform.execute_plan(oltp_plan)
    print(f"  Execution Status: {result['status']}")
    
    # Example 2: Analytics workload
    print("\n📈 Example 2: Sales Analytics")
    print("-" * 30)
    
    sales_metadata = {
        "name": "fact_sales",
        "entity_type": "fact",
        "estimated_size": "large",
        "access_pattern": "analytical_queries"
    }
    
    analytics_plan = platform.create_execution_plan(sales_metadata, "analytical_query")
    print(f"  Optimal Storage: {analytics_plan.storage_backend.value}")
    print(f"  Optimal Processing: {analytics_plan.processing_engine.value}")
    print(f"  Rationale: {analytics_plan.rationale}")
    
    # Execute plan
    result = platform.execute_plan(analytics_plan)
    print(f"  Execution Status: {result['status']}")
    
    # Example 3: Federated query
    print("\n🌐 Example 3: Federated Customer 360")
    print("-" * 35)
    
    federated_metadata = {
        "name": "customer_360_view",
        "entity_type": "analytical_view",
        "estimated_size": "large",
        "access_pattern": "cross_system_joins"
    }
    
    federated_plan = platform.create_execution_plan(federated_metadata, "federated_join")
    print(f"  Optimal Storage: Multi-backend")
    print(f"  Optimal Processing: {federated_plan.processing_engine.value}")
    print(f"  Rationale: {federated_plan.rationale}")
    
    # Show sample query
    print(f"\n  Sample Query:")
    query_preview = federated_plan.execution_query[:200].strip()
    print(f"  {query_preview}...")
    
    print("\n✅ Quick Start Complete!")
    print("The unified platform successfully demonstrated intelligent")
    print("workload placement across multiple storage and processing systems.")


if __name__ == "__main__":
    main()