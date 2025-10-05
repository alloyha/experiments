"""
Advanced Schema Management Examples
==================================

This module demonstrates advanced use cases for the PostgreSQL schema generator:
- Schema versioning and migration
- Multi-tenant architecture
- Performance optimization
- Data governance and lineage
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from schema_generator_client import SchemaGenerator, ConnectionConfig

class ChangeType(Enum):
    """Types of schema changes"""
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    MODIFY_COLUMN = "modify_column"
    ADD_TABLE = "add_table"
    DROP_TABLE = "drop_table"
    ADD_INDEX = "add_index"
    DROP_INDEX = "drop_index"


@dataclass
class SchemaVersion:
    """Schema version information"""
    version: str
    timestamp: datetime
    description: str
    metadata_hash: str
    changes: List[Dict[str, Any]]


class SchemaVersionManager:
    """
    Manages schema versions and migrations
    
    Tracks schema changes over time and generates migration scripts
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
        self.versions: List[SchemaVersion] = []
    
    def calculate_metadata_hash(self, pipeline_metadata: Dict[str, Any]) -> str:
        """Calculate hash of pipeline metadata for change detection"""
        # Create deterministic hash of metadata
        metadata_str = json.dumps(pipeline_metadata, sort_keys=True)
        return hashlib.sha256(metadata_str.encode()).hexdigest()
    
    def create_version(self, pipeline_metadata: Dict[str, Any], 
                      version: str, description: str) -> SchemaVersion:
        """Create new schema version"""
        metadata_hash = self.calculate_metadata_hash(pipeline_metadata)
        
        # Calculate changes from previous version
        changes = []
        if self.versions:
            changes = self._calculate_changes(
                self.versions[-1], pipeline_metadata
            )
        
        schema_version = SchemaVersion(
            version=version,
            timestamp=datetime.now(),
            description=description,
            metadata_hash=metadata_hash,
            changes=changes
        )
        
        self.versions.append(schema_version)
        return schema_version
    
    def _calculate_changes(self, previous_version: SchemaVersion, 
                          new_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate changes between schema versions"""
        # Simplified change detection
        # In production, this would be more sophisticated
        changes = []
        
        # For now, just note that changes were detected
        if previous_version.metadata_hash != self.calculate_metadata_hash(new_metadata):
            changes.append({
                "type": "schema_modified",
                "description": "Schema metadata changed",
                "timestamp": datetime.now().isoformat()
            })
        
        return changes
    
    def generate_migration_script(self, from_version: str, 
                                to_version: str) -> str:
        """Generate migration script between versions"""
        from_idx = next(i for i, v in enumerate(self.versions) if v.version == from_version)
        to_idx = next(i for i, v in enumerate(self.versions) if v.version == to_version)
        
        if from_idx >= to_idx:
            raise ValueError("Invalid version range")
        
        migration_script = f"""
-- Migration from {from_version} to {to_version}
-- Generated on {datetime.now()}

BEGIN;

-- Add migration logic here
-- This would include ALTER TABLE statements, etc.

COMMIT;
        """
        
        return migration_script.strip()


class MultiTenantSchemaManager:
    """
    Manages schemas for multi-tenant applications
    
    Each tenant gets their own schema with identical structure
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
        self.base_pipeline: Optional[Dict[str, Any]] = None
    
    def set_base_pipeline(self, pipeline_metadata: Dict[str, Any]):
        """Set the base pipeline structure for all tenants"""
        self.base_pipeline = pipeline_metadata
    
    def create_tenant_schema(self, tenant_id: str, 
                           custom_config: Dict[str, Any] = None) -> str:
        """
        Create schema for a specific tenant
        
        Args:
            tenant_id: Unique tenant identifier
            custom_config: Tenant-specific customizations
            
        Returns:
            str: Generated DDL for tenant schema
        """
        if not self.base_pipeline:
            raise ValueError("Base pipeline not set")
        
        # Create tenant-specific pipeline
        tenant_pipeline = self.base_pipeline.copy()
        tenant_pipeline['schema'] = f"tenant_{tenant_id}"
        
        # Apply tenant customizations
        if custom_config:
            tenant_pipeline = self._apply_tenant_customizations(
                tenant_pipeline, custom_config
            )
        
        # Generate DDL
        ddl = self.generator.generate_pipeline_ddl(tenant_pipeline)
        
        return ddl
    
    def _apply_tenant_customizations(self, pipeline: Dict[str, Any], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tenant-specific customizations"""
        # Example customizations:
        # - Additional columns
        # - Different retention policies
        # - Custom business rules
        
        if 'additional_columns' in config:
            for table in pipeline['tables']:
                if table['name'] in config['additional_columns']:
                    table.setdefault('physical_columns', []).extend(
                        config['additional_columns'][table['name']]
                    )
        
        return pipeline
    
    def deploy_to_all_tenants(self, tenant_ids: List[str], 
                             execute: bool = False) -> Dict[str, str]:
        """Deploy schema updates to all tenants"""
        results = {}
        
        for tenant_id in tenant_ids:
            try:
                ddl = self.create_tenant_schema(tenant_id)
                
                if execute:
                    self.generator.create_tables(ddl)
                
                results[tenant_id] = "success"
                
            except Exception as e:
                results[tenant_id] = f"error: {e}"
        
        return results


class PerformanceOptimizer:
    """
    Analyzes and optimizes schema performance
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
    
    def analyze_table_performance(self, table_metadata: Dict[str, Any], 
                                 query_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze table performance and suggest optimizations
        
        Args:
            table_metadata: Table metadata
            query_patterns: Common query patterns
            
        Returns:
            Dict: Performance analysis and recommendations
        """
        analysis = {
            "table_name": table_metadata['name'],
            "entity_type": table_metadata.get('entity_type'),
            "recommendations": [],
            "index_suggestions": [],
            "partitioning_suggestions": []
        }
        
        # Analyze based on entity type
        if table_metadata.get('entity_type') == 'transaction_fact':
            analysis['recommendations'].extend([
                "Consider partitioning by date for time-series data",
                "Add indexes on foreign key columns",
                "Consider columnar storage for analytical queries"
            ])
        
        # Analyze query patterns
        for pattern in query_patterns:
            if pattern.get('type') == 'aggregation':
                analysis['index_suggestions'].append({
                    "columns": pattern.get('group_by_columns', []),
                    "type": "btree",
                    "reason": "Supports GROUP BY operations"
                })
            
            if pattern.get('type') == 'range_scan':
                analysis['index_suggestions'].append({
                    "columns": pattern.get('range_columns', []),
                    "type": "btree", 
                    "reason": "Supports range queries"
                })
        
        # Partitioning suggestions for large tables
        if table_metadata.get('entity_type') in ['transaction_fact', 'fact']:
            analysis['partitioning_suggestions'].append({
                "strategy": "range_partitioning",
                "column": "created_at",
                "interval": "monthly",
                "reason": "Improve query performance on time-based data"
            })
        
        return analysis
    
    def generate_optimization_sql(self, analysis: Dict[str, Any]) -> str:
        """Generate SQL for performance optimizations"""
        table_name = analysis['table_name']
        sql_statements = []
        
        # Generate index creation statements
        for idx, suggestion in enumerate(analysis['index_suggestions']):
            index_name = f"ix_{table_name}_{idx + 1}"
            columns = ", ".join(suggestion['columns'])
            
            sql_statements.append(
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {table_name} ({columns});"
            )
        
        # Generate partitioning statements
        for suggestion in analysis['partitioning_suggestions']:
            if suggestion['strategy'] == 'range_partitioning':
                sql_statements.append(
                    f"-- Consider partitioning {table_name} by {suggestion['column']}"
                )
        
        return "\n".join(sql_statements)


class DataGovernanceManager:
    """
    Manages data governance, lineage, and compliance
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
    
    def tag_sensitive_data(self, pipeline_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tag columns containing sensitive data for compliance
        
        Args:
            pipeline_metadata: Pipeline metadata
            
        Returns:
            Dict: Updated metadata with data classification tags
        """
        # Define sensitive data patterns
        sensitive_patterns = {
            'PII': ['email', 'phone', 'ssn', 'address', 'name'],
            'FINANCIAL': ['salary', 'income', 'credit', 'account_number'],
            'HEALTH': ['diagnosis', 'treatment', 'medical_record']
        }
        
        tagged_pipeline = pipeline_metadata.copy()
        
        for table in tagged_pipeline.get('tables', []):
            # Add data classification to physical columns
            for column in table.get('physical_columns', []):
                column_name = column['name'].lower()
                
                for classification, patterns in sensitive_patterns.items():
                    if any(pattern in column_name for pattern in patterns):
                        column.setdefault('tags', []).append(f"sensitive:{classification}")
                        column.setdefault('security_policy', 'encrypt_at_rest')
        
        return tagged_pipeline
    
    def generate_data_lineage(self, pipeline_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate data lineage information
        
        Args:
            pipeline_metadata: Pipeline metadata
            
        Returns:
            Dict: Data lineage graph
        """
        lineage = {
            "pipeline": pipeline_metadata.get('schema', 'unknown'),
            "tables": [],
            "relationships": []
        }
        
        # Map table relationships
        for table in pipeline_metadata.get('tables', []):
            table_info = {
                "name": table['name'],
                "type": table.get('entity_type'),
                "grain": table.get('grain'),
                "upstream_dependencies": [],
                "downstream_dependencies": []
            }
            
            # Find foreign key relationships
            if table.get('entity_type') in ['transaction_fact', 'fact']:
                for dim_ref in table.get('dimension_references', []):
                    table_info['upstream_dependencies'].append(
                        dim_ref['dimension']
                    )
                    
                    # Add relationship
                    lineage['relationships'].append({
                        "from": dim_ref['dimension'],
                        "to": table['name'],
                        "type": "foreign_key",
                        "column": dim_ref['fk_column']
                    })
            
            lineage['tables'].append(table_info)
        
        return lineage
    
    def generate_compliance_report(self, pipeline_metadata: Dict[str, Any]) -> str:
        """
        Generate compliance report for auditing
        
        Args:
            pipeline_metadata: Pipeline metadata
            
        Returns:
            str: Compliance report
        """
        tagged_pipeline = self.tag_sensitive_data(pipeline_metadata)
        lineage = self.generate_data_lineage(tagged_pipeline)
        
        report = f"""
DATA GOVERNANCE COMPLIANCE REPORT
Generated: {datetime.now()}
Pipeline: {pipeline_metadata.get('schema', 'Unknown')}

SENSITIVE DATA INVENTORY:
"""
        
        for table in tagged_pipeline.get('tables', []):
            sensitive_columns = []
            for column in table.get('physical_columns', []):
                if 'tags' in column:
                    sensitive_tags = [tag for tag in column['tags'] if tag.startswith('sensitive:')]
                    if sensitive_tags:
                        sensitive_columns.append(f"  - {column['name']}: {sensitive_tags}")
            
            if sensitive_columns:
                report += f"\nTable: {table['name']}\n"
                report += "\n".join(sensitive_columns)
        
        report += f"""

DATA LINEAGE SUMMARY:
- Total Tables: {len(lineage['tables'])}
- Total Relationships: {len(lineage['relationships'])}

RECOMMENDATIONS:
- Implement row-level security for tables with PII data
- Enable audit logging for sensitive table access
- Consider data masking for non-production environments
        """
        
        return report.strip()


def main():
    """Demonstrate advanced schema management"""
    config = ConnectionConfig(
        host="localhost",
        database="postgres",
        username="postgres"
    )
    
    generator = SchemaGenerator(config)
    
    print("🔧 ADVANCED SCHEMA MANAGEMENT")
    print("=" * 50)
    
    # Sample pipeline for demonstrations
    sample_pipeline = {
        "schema": "advanced_demo",
        "tables": [
            {
                "name": "dim_customers",
                "entity_type": "dimension",
                "grain": "One row per customer",
                "primary_keys": ["customer_id"],
                "physical_columns": [
                    {"name": "customer_id", "type": "text"},
                    {"name": "email", "type": "text"},
                    {"name": "full_name", "type": "text"}
                ]
            },
            {
                "name": "fact_orders",
                "entity_type": "transaction_fact",
                "grain": "One row per order",
                "dimension_references": [
                    {"dimension": "dim_customers", "fk_column": "customer_sk"}
                ],
                "measures": [
                    {"name": "order_amount", "type": "numeric(10,2)"}
                ]
            }
        ]
    }
    
    # Schema Versioning Demo
    print("\n📋 Schema Versioning")
    print("-" * 30)
    
    version_manager = SchemaVersionManager(generator)
    v1 = version_manager.create_version(sample_pipeline, "1.0", "Initial schema")
    print(f"✅ Created version {v1.version}: {v1.description}")
    print(f"   Hash: {v1.metadata_hash[:8]}...")
    
    # Multi-tenant Demo
    print("\n🏢 Multi-tenant Management")
    print("-" * 30)
    
    multi_tenant = MultiTenantSchemaManager(generator)
    multi_tenant.set_base_pipeline(sample_pipeline)
    
    tenant_ddl = multi_tenant.create_tenant_schema("acme_corp")
    print(f"✅ Generated tenant schema (DDL length: {len(tenant_ddl)} chars)")
    
    # Performance Optimization Demo
    print("\n⚡ Performance Optimization")
    print("-" * 30)
    
    optimizer = PerformanceOptimizer(generator)
    
    query_patterns = [
        {"type": "aggregation", "group_by_columns": ["customer_sk"]},
        {"type": "range_scan", "range_columns": ["created_at"]}
    ]
    
    fact_table = sample_pipeline['tables'][1]  # fact_orders
    analysis = optimizer.analyze_table_performance(fact_table, query_patterns)
    
    print(f"✅ Performance analysis for {analysis['table_name']}:")
    print(f"   - Recommendations: {len(analysis['recommendations'])}")
    print(f"   - Index suggestions: {len(analysis['index_suggestions'])}")
    
    # Data Governance Demo
    print("\n🛡️ Data Governance")
    print("-" * 30)
    
    governance = DataGovernanceManager(generator)
    
    tagged_pipeline = governance.tag_sensitive_data(sample_pipeline)
    lineage = governance.generate_data_lineage(tagged_pipeline)
    
    print(f"✅ Data lineage generated:")
    print(f"   - Tables: {len(lineage['tables'])}")
    print(f"   - Relationships: {len(lineage['relationships'])}")
    
    compliance_report = governance.generate_compliance_report(sample_pipeline)
    print("\n✅ Compliance report generated:")
    print(f"   - Length: {len(compliance_report)} characters")


if __name__ == "__main__":
    main()