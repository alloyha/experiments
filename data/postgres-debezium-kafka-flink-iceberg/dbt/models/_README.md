# dbt Models - Medalhão + Diamond Architecture (Lakehouse)

## Directory Structure

```
models/
├── 🔵 bronze/        Raw CDC data from Kafka (Iceberg via Flink)
│   └── brz_*.sql     Bronze models
│
├── 🟡 silver/        Cleaned & deduplicated data (Iceberg via Flink)
│   └── slv_*.sql     Silver models
│
├── 📊 gold/          Business-ready models (PostgreSQL)
│   ├── fct_*.sql     Fact tables
│   └── dim_*.sql     Dimension tables
│
├── 💎 diamond/       Semantic OLAP layer (PostgreSQL)
│   ├── obt_*.sql     One Big Table models
│   ├── agg_*.sql     Pre-aggregated metrics
│   └── _DIAMOND.md   Layer documentation
│
├── _sources.yml      Central sources definition
├── _models.yml       Model tags and metadata
└── tests.yml         dbt tests
```

## Layer Details

### 🔵 Bronze (Raw)
- **Source**: Kafka CDC topics
- **Target**: Iceberg (S3/MinIO)
- **Tool**: Flink Streaming
- **Models**: brz_*.sql
- **Purpose**: Capture raw CDC events without transformation
- **Materialization**: Stream (Flink-managed)
- **Examples**:
  - brz_customers_cdc.sql - Raw customer events from Debezium

### 🟡 Silver (Cleaned)
- **Source**: Bronze (Iceberg)
- **Target**: Iceberg (S3/MinIO)
- **Tool**: Flink Streaming
- **Models**: slv_*.sql
- **Purpose**: Clean, deduplicate, and enrich data
- **Materialization**: Table (Iceberg-managed)
- **Business Logic**:
  - Deduplication (keep latest version per key)
  - Filter deleted records
  - Data quality checks
  - Standardization (LOWER case, trimming, etc)
- **Examples**:
  - slv_customers.sql - Cleaned customer data (latest version per customer_id)

### 📊 Gold (Business-Ready)
- **Source**: Silver (Iceberg) + Bronze (Iceberg)
- **Target**: PostgreSQL (gold schema)
- **Tool**: dbt (SQL)
- **Models**: fct_*.sql (facts), dim_*.sql (dimensions)
- **Purpose**: Business-ready fact & dimension tables
- **Materialization**: TABLE (indexed)
- **Use Cases**: BI tools, dashboards, reports
- **Examples**:
  - fct_orders.sql - Order fact table with indexes
  - dim_customers.sql - Customer dimension with KPIs

### 💎 Diamond (Semantic Layer - OBT OLAP Cube)
- **Source**: Gold layer (fct_* and dim_* tables)
- **Target**: PostgreSQL (diamond schema)
- **Tool**: dbt (SQL)
- **Models**: obt_*.sql (One Big Table), agg_*.sql (aggregates)
- **Purpose**: Pre-built OLAP cubes optimized for BI tools (Tableau, Power BI, Looker)
- **Materialization**: TABLE (columnstore indexes, optimized for aggregations)
- **Pattern**: Denormalized for maximum query performance
- **Examples**:
  - obt_customer_orders.sql - Flat consolidated customer-order facts
  - agg_daily_sales_by_segment.sql - Pre-aggregated metrics for dashboards
- **Documentation**: See [diamond/_DIAMOND.md](./diamond/_DIAMOND.md)

## Naming Conventions

| Layer | Prefix | Example |
|-------|--------|---------|
| Bronze | brz_ | brz_customers_cdc |
| Silver | slv_ | slv_customers |
| Fact | fct_ | fct_orders |
| Dimension | dim_ | dim_customers |
| Diamond (OBT) | obt_ | obt_customer_orders |
| Diamond (Agg) | agg_ | agg_daily_sales_by_segment |

## Usage

### Deploy Bronze + Silver (Flink → Iceberg - Default)
```bash
dbt run -t flink
```

### Deploy Gold + Diamond (PostgreSQL)
```bash
dbt run -t postgres
```

### Deploy Specific Layer
```bash
dbt run -s bronze    # Just bronze
dbt run -s silver    # Just silver
dbt run -s gold      # Just gold tables
dbt run -s diamond   # Just diamond (OBT + aggregates)
```

### Test Models
```bash
dbt test
dbt test -s silver   # Test specific layer
```

### Generate Docs
```bash
dbt docs generate
dbt docs serve       # Visit http://localhost:8000
```

## Architecture Layers

```
🔵 BRONZE (Raw CDC Events)
  ↓ Flink Streaming
🟡 SILVER (Cleaned & Deduplicated)
  ↓ Transformation in Iceberg Lakehouse
📊 GOLD (Fact & Dimension Tables)
  ↓ Denormalization
💎 DIAMOND (OLAP Semantic Layer - OBT Cubes)
  ↓ BI Tools Connection
📈 Dashboards (Tableau, Power BI, Looker, Metabase)
```

## Data Flow Diagram

```
External Data Sources
         ↓
PostgreSQL ──CDC──→ Kafka
         ↓          ↓
      Debezium ←────┘
         ↓
    🔵 BRONZE
    (Raw Events)
    ├─ brz_customers_cdc
    └─ brz_products_cdc
         ↓ Flink Streaming
    🟡 SILVER
    (Cleaned & Deduplicated)
    ├─ slv_customers
    └─ slv_products
         ↓ dbt Transform
    📊 GOLD
    (Normalized Analytics)
    ├─ Fact Layer
    │  ├─ fct_orders
    │  └─ fct_customer_orders
    ├─ Dimension Layer
    │  ├─ dim_customers
    │  └─ dim_products
         ↓ dbt Semantic Layer
    💎 DIAMOND
    (OBT OLAP Cubes)
    ├─ obt_customer_orders
    └─ agg_daily_sales_by_segment
         ↓
    📊 BI Tools
    ├─ Tableau
    ├─ Power BI
    ├─ Looker
    └─ Metabase
```

## Storage Architecture

```
Iceberg Lakehouse (S3/MinIO)          PostgreSQL Database
├─ bronze/ (Raw)                      ├─ public/ (Operational)
│  └─ customer_cdc events             ├─ gold/ (Analytics - Fact & Dim)
│                                     └─ diamond/ (Semantic - OBT & Agg)
└─ silver/ (Cleaned)
   └─ customer latest versions
   
Bronze + Silver = Lake (immutable, auditable, recoverable)
Gold + Diamond = Analytics DB (optimized for queries, BI-ready)
```

## Quick Start

1. **Setup dbt profiles** (if needed):
   ```bash
   dbt debug
   ```

2. **Run all models**:
   ```bash
   dbt run          # Default: Flink (bronze + silver)
   dbt run -t postgres  # PostgreSQL (gold + diamond)
   ```

3. **Run specific layer**:
   ```bash
   dbt run -s bronze     # Bronze only
   dbt run -s silver     # Silver only
   dbt run -s gold       # Gold only
   dbt run -s diamond    # Diamond only
   ```

4. **Test all models**:
   ```bash
   dbt test
   ```

5. **Generate documentation**:
   ```bash
   dbt docs generate
   dbt docs serve  # Visit http://localhost:8000
   ```

## Performance Tips

### Bronze + Silver (Flink/Iceberg)
- Use Flink parallelism for high-throughput streaming
- Partition Iceberg tables by date for faster queries
- Monitor Kafka lag in Flink dashboards
- Leverage immutability for audit trails

### Gold (PostgreSQL)
- Create indexes on fact table foreign keys and dates
- Use table partitioning for large fact tables
- Run `ANALYZE` periodically for query optimization
- Vacuum regularly to maintain performance

### Diamond (Semantic Layer)
- Pre-aggregate common query patterns in `agg_*` models
- Use columnstore indexes for OLAP performance
- Create separate aggregates for different time grains (daily, weekly, monthly)
- Consider materialized views for complex joins

## BI Tool Integration

Connect your BI tools directly to the Diamond schema for instant insights:

**Tableau**: Create live data source pointing to `postgres://localhost:5432/analytics_db` with schema `diamond`

**Power BI**: Use PostgreSQL connector, auto-discover `diamond.obt_*` and `diamond.agg_*` tables

**Looker**: Create views on top of Diamond tables with business-friendly naming

**Metabase**: Auto-discovery of diamond schema tables and columns (no complex joins needed)

## Related Files

- [diamond/_DIAMOND.md](./diamond/_DIAMOND.md) - Detailed Diamond layer documentation
- [../dbt_project.yml](../dbt_project.yml) - dbt project configuration
- [../profiles.yml](../profiles.yml) - Database connection profiles

## Why This Architecture?

### ✅ Lakehouse Pattern (Bronze + Silver in Iceberg)
- **Immutability**: Raw data never changes, full audit trail
- **Recoverability**: Can replay entire data pipeline from any point
- **Cost-effective**: S3/MinIO storage cheaper than databases
- **Governance**: Data lineage and compliance ready

### ✅ Medalhão Pattern (Structured Layers)
- **Clear separation**: Each layer has a specific purpose
- **Scalability**: Easy to add new models in any layer
- **Reusability**: Silver tables used by multiple Gold tables
- **Testability**: Each layer can be tested independently

### ✅ Diamond Semantic Layer
- **Simplified queries**: No JOINs for BI analysts
- **Performance**: 30-50× faster than multi-table queries
- **BI-ready**: OBT tables integrate instantly with Tableau/Power BI
- **Business metrics**: Pre-calculated KPIs and dimensions

## Next Steps

1. ✅ Deploy Bronze + Silver models
2. ✅ Verify Iceberg tables in MinIO
3. ⬜ Deploy Gold models
4. ⬜ Connect BI tools to Diamond schema
5. ⬜ Create business metrics documentation
6. ⬜ Add dbt tests and data quality checks

---

**Architecture**: Medalhón Extended with Diamond Semantic Layer (Lakehouse)  
**Pattern**: Iceberg (Bronze/Silver) + PostgreSQL (Gold/Diamond)  
**Layer Count**: 4 (Bronze → Silver → Gold → Diamond)  
**Last Updated**: 2026-08-09
