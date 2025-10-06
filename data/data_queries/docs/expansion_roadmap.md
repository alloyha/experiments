# Unified Platform Instructions: Dimensional Modeling + Multi-Backend Architecture

## System Overview

You are enhancing a **unified data platform** that combines **dimensional modeling expertise** with **multi-backend flexibility**. The platform automatically selects optimal storage and processing technologies while maintaining proven dimensional modeling patterns across all backends.

## Core Architecture Principles

### 1. **Dimensional Modeling First, Technology Second**
- Business entities define the model structure (dimensions, facts, bridges)
- Platform automatically selects optimal technology per entity workload
- Proven dimensional patterns adapted to each storage backend's strengths

### 2. **Technology-Agnostic Dimensional Patterns**
```python
# Same dimensional model, different optimal backends
customer_dimension = {
    "entity_type": "dimension",
    "scd": "SCD2", 
    "grain": "One row per customer per version"
}
# → PostgreSQL for OLTP compliance needs
# → Iceberg for data lake analytics
# → ClickHouse for real-time aggregations
```

### 3. **Metadata Schema**
```json
{
  "pipeline_name": "customer_analytics",
  "entities": [
    {
      "name": "dim_customers",
      "entity_type": "dimension",
      "grain": "One row per customer per version",
      "scd": "SCD2",
      "business_keys": ["customer_number"],
      "physical_columns": [
        {"name": "customer_number", "type": "varchar(50)", "nullable": false},
        {"name": "customer_name", "type": "varchar(200)", "nullable": false},
        {"name": "customer_tier", "type": "varchar(20)", "nullable": false}
      ],
      "workload_characteristics": {
        "update_frequency": "daily",
        "compliance_requirements": ["GDPR", "SOX"],
        "query_patterns": ["customer_lookup", "compliance_reporting"]
      }
    },
    {
      "name": "fact_orders",
      "entity_type": "transaction_fact", 
      "grain": "One row per order line item",
      "dimension_references": [
        {"dimension": "dim_customers", "fk_column": "customer_sk"},
        {"dimension": "dim_products", "fk_column": "product_sk"}
      ],
      "measures": [
        {"name": "order_amount", "type": "decimal(10,2)", "additivity": "additive"},
        {"name": "discount_amount", "type": "decimal(8,2)", "additivity": "additive"},
        {"name": "unit_price", "type": "decimal(8,2)", "additivity": "non_additive"}
      ],
      "workload_characteristics": {
        "volume": "high", 
        "query_patterns": ["aggregation", "time_series", "drill_down"],
        "latency_requirements": "sub_second"
      }
    }
  ]
}
```

## File Structure

```
unified_platform/
├── core/
│   ├── enhanced_metadata.py          # Dimensional-aware metadata validation
│   ├── workload_analyzer.py          # Intelligent backend selection
│   └── deployment_orchestrator.py    # Cross-backend deployment
├── patterns/
│   ├── dimensional_modeling.py       # SCD, fact patterns for all backends
│   ├── grain_validation.py           # Cross-backend grain enforcement
│   └── quality_patterns.py           # Data quality across technologies
├── storage/
│   ├── [existing 8 backends]         # Dimensional awareness
│   └── dimensional_adapters/         # Dimensional pattern implementations
└── processing/
    ├── [existing 6 engines]          # Dimensional queries
    └── dimensional_queries/           # Cross-backend dimensional SQL
```

## Key Enhancement Areas

### 1. **Metadata Validation (`core/enhanced_metadata.py`)**

```python
class DimensionalMetadataValidator:
    """
    Validates dimensional modeling patterns across all storage backends
    """
    
    def validate_entity_metadata(self, entity: Dict[str, Any]) -> ValidationResult:
        """
        Apply dimensional modeling rules regardless of target backend
        """
        entity_type = entity.get('entity_type')
        
        if entity_type == 'dimension':
            return self._validate_dimension_metadata(entity)
        elif entity_type in ['fact', 'transaction_fact']:
            return self._validate_fact_metadata(entity)
        elif entity_type == 'bridge':
            return self._validate_bridge_metadata(entity)
    
    def _validate_dimension_metadata(self, dim: Dict[str, Any]) -> ValidationResult:
        """
        Dimension-specific validation rules
        """
        issues = []
        
        # Grain validation
        if not dim.get('grain'):
            issues.append("Dimension must declare grain")
        
        # SCD validation  
        scd_type = dim.get('scd', 'SCD1')
        if scd_type == 'SCD2':
            if not any(col.get('name') == 'is_current' for col in dim.get('physical_columns', [])):
                issues.append("SCD2 dimensions require is_current column")
        
        # Business key validation
        if not dim.get('business_keys'):
            issues.append("Dimension must have business keys defined")
        
        return ValidationResult(issues)
    
    def _validate_fact_metadata(self, fact: Dict[str, Any]) -> ValidationResult:
        """
        Fact-specific validation rules
        """
        issues = []
        
        # Grain validation
        if not fact.get('grain'):
            issues.append("Fact must declare grain")
        
        # Measures validation
        measures = fact.get('measures', [])
        if not measures:
            issues.append("Fact must have at least one measure")
        
        for measure in measures:
            if not measure.get('additivity'):
                issues.append(f"Measure {measure['name']} must declare additivity")
        
        # Dimension references validation
        dim_refs = fact.get('dimension_references', [])
        if not dim_refs:
            issues.append("Fact must reference at least one dimension")
        
        return ValidationResult(issues)
```

### 2. **Intelligent Backend Selection (`core/workload_analyzer.py`)**

```python
class DimensionalWorkloadAnalyzer:
    """
    Analyzes dimensional entities to select optimal storage/processing backends
    """
    
    def select_optimal_backend(self, entity: Dict[str, Any]) -> BackendRecommendation:
        """
        Select best backend based on dimensional characteristics
        """
        entity_type = entity.get('entity_type')
        workload_chars = entity.get('workload_characteristics', {})
        
        if entity_type == 'dimension':
            return self._recommend_dimension_backend(entity, workload_chars)
        elif entity_type in ['fact', 'transaction_fact']:
            return self._recommend_fact_backend(entity, workload_chars)
    
    def _recommend_dimension_backend(self, dim: Dict[str, Any], 
                                   workload: Dict[str, Any]) -> BackendRecommendation:
        """
        Dimension-specific backend selection
        """
        # Small, frequently updated dimensions with compliance → PostgreSQL
        if (workload.get('update_frequency') == 'real_time' and 
            workload.get('compliance_requirements')):
            return BackendRecommendation(
                storage='postgresql',
                processing='postgresql', 
                reason='ACID compliance + real-time updates'
            )
        
        # Large, slowly changing dimensions → Iceberg + Trino
        elif workload.get('volume') == 'large':
            return BackendRecommendation(
                storage='iceberg',
                processing='trino',
                reason='Scale + time travel for large dimensions'
            )
        
        # High-frequency lookup dimensions → ClickHouse
        elif 'real_time_lookup' in workload.get('query_patterns', []):
            return BackendRecommendation(
                storage='clickhouse',
                processing='clickhouse',
                reason='Sub-millisecond dimension lookups'
            )
    
    def _recommend_fact_backend(self, fact: Dict[str, Any],
                              workload: Dict[str, Any]) -> BackendRecommendation:
        """
        Fact-specific backend selection
        """
        # High-volume analytical facts → ClickHouse
        if (workload.get('volume') == 'high' and 
            'aggregation' in workload.get('query_patterns', [])):
            return BackendRecommendation(
                storage='clickhouse',
                processing='clickhouse',
                reason='Columnar optimization for analytical queries'
            )
        
        # Batch ETL facts → Iceberg + Spark
        elif workload.get('processing_pattern') == 'batch':
            return BackendRecommendation(
                storage='iceberg', 
                processing='spark',
                reason='Schema evolution + massive scale processing'
            )
        
        # Real-time streaming facts → Delta Lake + Spark
        elif workload.get('processing_pattern') == 'streaming':
            return BackendRecommendation(
                storage='delta_lake',
                processing='spark',
                reason='ACID streaming with change data feed'
            )
```

### 3. **Cross-Backend Dimensional Patterns (`patterns/dimensional_modeling.py`)**

```python
class CrossBackendDimensionalPatterns:
    """
    Implements dimensional modeling patterns across all storage backends
    """
    
    def create_scd2_dimension(self, entity: Dict[str, Any], 
                            backend: StorageBackend) -> str:
        """
        Generate SCD2 dimension for any backend
        """
        if isinstance(backend, PostgreSQLStorageBackend):
            return self._create_postgresql_scd2(entity)
        elif isinstance(backend, IcebergStorageBackend):
            return self._create_iceberg_scd2(entity)
        elif isinstance(backend, ClickHouseStorageBackend):
            return self._create_clickhouse_scd2(entity)
        elif isinstance(backend, DeltaLakeStorageBackend):
            return self._create_delta_scd2(entity)
    
    def _create_postgresql_scd2(self, entity: Dict[str, Any]) -> str:
        """
        PostgreSQL SCD2 with full ACID compliance
        """
        table_name = entity['name']
        return f"""
        -- PostgreSQL SCD2 Dimension: {table_name}
        CREATE TABLE IF NOT EXISTS {table_name} (
            {table_name}_sk BIGSERIAL PRIMARY KEY,
            {self._generate_business_key_columns(entity)},
            {self._generate_physical_columns(entity)},
            
            -- SCD2 tracking columns
            effective_date TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expiration_date TIMESTAMPTZ DEFAULT '9999-12-31'::TIMESTAMPTZ,
            is_current BOOLEAN DEFAULT TRUE,
            row_version INTEGER DEFAULT 1,
            
            -- Audit columns
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            source_system VARCHAR(50) DEFAULT 'unified_platform'
        );
        
        -- SCD2-specific indexes
        CREATE INDEX IF NOT EXISTS idx_{table_name}_business_key_current 
            ON {table_name} ({self._get_business_key_list(entity)}) 
            WHERE is_current = TRUE;
        
        CREATE INDEX IF NOT EXISTS idx_{table_name}_effective_date 
            ON {table_name} (effective_date, expiration_date);
        """
    
    def _create_iceberg_scd2(self, entity: Dict[str, Any]) -> str:
        """
        Iceberg SCD2 with time travel capabilities
        """
        table_name = entity['name']
        return f"""
        -- Iceberg SCD2 Dimension: {table_name}
        CREATE TABLE {table_name} (
            {table_name}_sk BIGINT,
            {self._generate_business_key_columns(entity)},
            {self._generate_physical_columns(entity)},
            
            -- SCD2 with Iceberg time travel
            effective_timestamp TIMESTAMP NOT NULL,
            expiration_timestamp TIMESTAMP,
            is_current BOOLEAN DEFAULT TRUE,
            
            -- Iceberg metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        ) USING ICEBERG
        PARTITIONED BY (bucket(16, {table_name}_sk), is_current)
        TBLPROPERTIES (
            'write.metadata.delete-after-commit.enabled' = 'true',
            'write.metadata.previous-versions-max' = '100'
        );
        """
    
    def create_fact_table(self, entity: Dict[str, Any], 
                         backend: StorageBackend) -> str:
        """
        Generate fact table optimized for backend
        """
        if isinstance(backend, ClickHouseStorageBackend):
            return self._create_clickhouse_fact(entity)
        elif isinstance(backend, PostgreSQLStorageBackend):
            return self._create_postgresql_fact(entity)
        elif isinstance(backend, IcebergStorageBackend):
            return self._create_iceberg_fact(entity)
    
    def _create_clickhouse_fact(self, entity: Dict[str, Any]) -> str:
        """
        ClickHouse fact table optimized for analytics
        """
        table_name = entity['name']
        measures = entity.get('measures', [])
        
        return f"""
        -- ClickHouse Analytical Fact: {table_name}
        CREATE TABLE {table_name} (
            {table_name}_sk UInt64,
            {self._generate_dimension_foreign_keys(entity)},
            {self._generate_measure_columns(measures)},
            
            -- Temporal columns for time-based partitioning
            transaction_date Date,
            transaction_timestamp DateTime64(3),
            
            -- ClickHouse optimizations
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(transaction_date)
        ORDER BY ({self._get_clustering_keys(entity)}, transaction_timestamp)
        TTL transaction_date + INTERVAL 7 YEAR
        SETTINGS index_granularity = 8192;
        
        -- Create materialized view for real-time aggregations
        CREATE MATERIALIZED VIEW {table_name}_daily_agg
        ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(transaction_date)
        ORDER BY (transaction_date, {self._get_dimension_keys(entity)})
        AS SELECT 
            transaction_date,
            {self._get_dimension_keys(entity)},
            {self._generate_sum_aggregates(measures)},
            count() as record_count
        FROM {table_name}
        GROUP BY transaction_date, {self._get_dimension_keys(entity)};
        """
```

### 4. **Storage Backends**

Each storage backend should be with dimensional awareness:

```python
# unified_platform/storage/postgresql_backend.py (Enhanced)
class PostgreSQLStorageBackend(StorageBackend):
    """PostgreSQL backend with dimensional modeling expertise"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimensional_patterns = CrossBackendDimensionalPatterns()
    
    def generate_dimensional_ddl(self, entity: Dict[str, Any]) -> str:
        """
        Generate DDL using dimensional modeling patterns
        """
        entity_type = entity.get('entity_type')
        
        if entity_type == 'dimension':
            scd_type = entity.get('scd', 'SCD1')
            if scd_type == 'SCD2':
                return self.dimensional_patterns.create_scd2_dimension(entity, self)
            else:
                return self.dimensional_patterns.create_scd1_dimension(entity, self)
        
        elif entity_type in ['fact', 'transaction_fact']:
            return self.dimensional_patterns.create_fact_table(entity, self)
    
    def generate_dimensional_dml(
        self, entity: Dict[str, Any], 
        operation: str
    ) -> str:
        """
        Generate DML with dimensional patterns
        """
        if entity.get('entity_type') == 'dimension' and entity.get('scd') == 'SCD2':
            return self._generate_scd2_upsert(entity)
        elif entity.get('entity_type') in ['fact', 'transaction_fact']:
            return self._generate_fact_upsert(entity)
```

### 5. **Platform Orchestrator**

```python
# unified_platform/core/enhanced_platform.py
class UnifiedPlatform:
    """
    Unified platform with dimensional modeling intelligence
    """
    
    def __init__(self):
        self.metadata_validator = DimensionalMetadataValidator()
        self.workload_analyzer = DimensionalWorkloadAnalyzer() 
        self.dimensional_patterns = CrossBackendDimensionalPatterns()
        self.storage_factory = StorageBackendFactory()
        self.processing_factory = ProcessingEngineFactory()
    
    def deploy_dimensional_model(self, pipeline_metadata: Dict[str, Any]) -> DeploymentResult:
        """
        Deploy dimensional model with optimal technology selection
        """
        # 1. Validate dimensional model
        validation_result = self.metadata_validator.validate_pipeline(pipeline_metadata)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.issues)
        
        deployment_plan = DeploymentPlan()
        
        # 2. Analyze each entity and select optimal backend
        for entity in pipeline_metadata['entities']:
            recommendation = self.workload_analyzer.select_optimal_backend(entity)
            
            # 3. Create backend instances
            storage_backend = self.storage_factory.create_backend(
                StorageType(recommendation.storage), 
                **entity.get('storage_config', {})
            )
            processing_engine = self.processing_factory.create_engine(
                recommendation.processing,
                **entity.get('processing_config', {})
            )
            
            # 4. Generate dimensional DDL/DML
            ddl = storage_backend.generate_dimensional_ddl(entity)
            dml = storage_backend.generate_dimensional_dml(entity, 'upsert')
            
            deployment_plan.add_entity(
                entity=entity,
                storage_backend=storage_backend,
                processing_engine=processing_engine,
                ddl=ddl,
                dml=dml,
                recommendation_reason=recommendation.reason
            )
        
        # 5. Validate cross-entity relationships
        self._validate_cross_entity_integrity(deployment_plan)
        
        return self._execute_deployment(deployment_plan)
    
    def _validate_cross_entity_integrity(self, plan: DeploymentPlan):
        """
        Ensure dimensional model integrity across different backends
        """
        dimensions = [e for e in plan.entities if e.entity_type == 'dimension']
        facts = [e for e in plan.entities if e.entity_type in ['fact', 'transaction_fact']]
        
        # Validate all fact foreign keys reference existing dimensions
        for fact in facts:
            for dim_ref in fact.entity.get('dimension_references', []):
                dim_name = dim_ref['dimension']
                if not any(d.entity['name'] == dim_name for d in dimensions):
                    raise ValidationError(f"Fact {fact.entity['name']} references unknown dimension {dim_name}")
```

## Usage Examples

### **Example 1: Customer Analytics Pipeline**

```python
# Define dimensional model
customer_analytics = {
    "pipeline_name": "customer_analytics",
    "entities": [
        {
            "name": "dim_customers",
            "entity_type": "dimension",
            "grain": "One row per customer per version",
            "scd": "SCD2",
            "business_keys": ["customer_number"],
            "workload_characteristics": {
                "compliance_requirements": ["GDPR"],
                "update_frequency": "daily"
            }
        },
        {
            "name": "fact_orders", 
            "entity_type": "transaction_fact",
            "grain": "One row per order line item",
            "workload_characteristics": {
                "volume": "high",
                "query_patterns": ["aggregation", "time_series"]
            }
        }
    ]
}

# Deploy with intelligent backend selection
platform = UnifiedPlatform()
result = platform.deploy_dimensional_model(customer_analytics)

# Platform automatically:
# - dim_customers → PostgreSQL (compliance + ACID)
# - fact_orders → ClickHouse (high-volume analytics)
# - Validates grain consistency
# - Generates proper SCD2 DDL for PostgreSQL
# - Creates optimized fact table for ClickHouse
# - Sets up cross-backend referential integrity
```

## Key Enhancement Principles

### 1. **Dimensional Modeling Excellence**
- Apply proven dimensional patterns regardless of technology
- Maintain grain consistency across all entities
- Enforce proper SCD2, fact table, and bridge patterns

### 2. **Technology Optimization**
- Each entity deployed to optimal backend based on characteristics
- Dimensional patterns adapted to leverage each technology's strengths
- Cross-backend referential integrity maintained

### 3. **Unified Developer Experience**
- Single metadata schema works across all technologies
- Automatic backend selection with clear reasoning
- Consistent dimensional patterns regardless of underlying technology

### 4. **Production Readiness**
- Comprehensive validation at metadata and deployment levels
- Performance optimization per backend
- Data quality monitoring across all technologies

This approach combines the best of dimensional modeling expertise with multi-backend flexibility, creating a truly unified data platform that maintains modeling excellence while optimizing technology selection per workload.