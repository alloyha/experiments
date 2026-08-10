# 💎 Diamond Layer - Semantic OLAP Cube

## Overview

The **Diamond layer** is a semantic layer built on top of the Gold layer using the **OBT (One Big Table)** approach. It provides pre-built OLAP cubes optimized for analytical queries and BI tools.

```
🟢 GOLD (Business-Ready)
├─ Fact tables (fct_*)
├─ Dimension tables (dim_*)
└─ Indexed for transactional queries

       ↓ DENORMALIZATION
       
💎 DIAMOND (Semantic Layer)
├─ OBT models (obt_*)
├─ Pre-aggregated measures
├─ Flattened dimensions
└─ Optimized for OLAP queries
```

## Diamond Principles

### 1. **One Big Table (OBT) Pattern**
- Consolidate fact + dimensions into a single denormalized table
- Eliminate JOINs for faster query performance
- Optimized for BI tools (Tableau, Power BI, Looker)
- Minimal cognitive load for analysts

### 2. **Semantic Layer**
- Business-friendly naming conventions
- Pre-calculated measures and KPIs
- Clear documentation of metrics
- Role-based access control ready

### 3. **OLAP Optimization**
- Columnstore indexes for aggregation speed
- Materialized aggregations (if needed)
- Grain consistency across all dimensions
- Time-series optimization

## Directory Structure

```
models/diamond/
├── 🎯 obt_*.sql           One Big Table models
├── 📊 agg_*.sql           Pre-aggregated measures
├── 🔗 bridge_*.sql        Bridge tables (if needed)
├── 📝 _DIAMOND.md         This file
├── 📝 _models.yml         Model definitions
└── 📝 _sources.yml        Source definitions
```

## Model Naming Convention

| Type | Prefix | Purpose | Example |
|------|--------|---------|---------|
| One Big Table | `obt_` | Flat denormalized table for OLAP | `obt_customer_orders` |
| Aggregation | `agg_` | Pre-calculated metrics | `agg_daily_sales` |
| Bridge | `bridge_` | Many-to-many relationships | `bridge_customer_channels` |

## Example: OBT Customer Orders

### Business Question
"What are the total orders by customer, product, and month?"

### Without Diamond (Query Hell)
```sql
SELECT 
  dc.customer_name,
  dc.customer_segment,
  dp.product_name,
  dp.product_category,
  DATE_TRUNC('month', fo.order_date) as month,
  COUNT(fo.order_id) as order_count,
  SUM(fo.order_amount) as total_sales,
  AVG(fo.order_amount) as avg_order_value
FROM gold.fct_orders fo
JOIN gold.dim_customers dc ON fo.customer_id = dc.customer_id
JOIN gold.dim_products dp ON fo.product_id = dp.product_id
GROUP BY 1,2,3,4,5
```

### With Diamond (One Query, No Joins)
```sql
SELECT 
  month,
  customer_name,
  customer_segment,
  product_name,
  product_category,
  order_count,
  total_sales,
  avg_order_value
FROM diamond.obt_customer_orders
WHERE month >= DATE_TRUNC('month', NOW() - INTERVAL '12 months')
ORDER BY month DESC, total_sales DESC
```

## Configuration

### dbt_project.yml

```yaml
models:
  cdc_analytics:
    diamond:
      +materialized: table
      +target: postgres
      +schema: diamond
      +post_hook: >
        CREATE INDEX IF NOT EXISTS idx_{{ this.name }}_customer_id
        ON {{ this.schema }}.{{ this.name }}(customer_id);
        CREATE INDEX IF NOT EXISTS idx_{{ this.name }}_date
        ON {{ this.schema }}.{{ this.name }}(order_date);
```

### profiles.yml

Ensure diamond uses the same PostgreSQL target as gold:
```yaml
postgres:
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5432
      user: postgres
      password: postgres
      dbname: analytics_db
      schema: diamond
      threads: 8
  target: dev
```

## Running Diamond Models

```bash
# Run all diamond models
dbt run -s diamond

# Run specific OBT
dbt run -s obt_customer_orders

# Run with full lineage
dbt run -s diamond --select +diamond

# Test diamond models
dbt test -s diamond

# Generate docs
dbt docs generate --select diamond
```

## Performance Considerations

### ✅ Advantages
- **Faster Queries**: No JOINs needed, queries hit single table
- **Easier for BI Tools**: No relationship mapping required
- **Consistent Grain**: Single source of truth for metrics
- **Pre-calculated**: Measures already computed
- **Simple Aggregations**: GROUP BY is trivial

### ⚠️ Trade-offs
- **Larger Tables**: Denormalization = more rows and columns
- **Storage Cost**: Redundant data storage
- **Update Complexity**: Must maintain consistency with Gold
- **Less Flexible**: Hard to change dimension logic

### 🚀 Optimization Tips
1. Use **table partitioning** by date
2. Add **columnstore indexes** for aggregation columns
3. Materialize **daily/monthly aggregates** separately
4. Use **incremental models** for large tables
5. Document **grain** explicitly (e.g., "one row per customer-product-month")

## Data Grain

### obt_customer_orders
- **Grain**: One row per customer + product + order_date
- **Scope**: Last 24 months of orders
- **Volume**: ~100K rows (10K customers × 10 products)
- **Update Frequency**: Real-time (via dbt run cycle)

## Relationship to Gold Layer

```
Gold Layer (Normalized)
├── dim_customers
│   ├── customer_id (PK)
│   ├── customer_name
│   ├── customer_segment
│   └── ...
│
├── dim_products
│   ├── product_id (PK)
│   ├── product_name
│   ├── product_category
│   └── ...
│
└── fct_orders
    ├── order_id (PK)
    ├── customer_id (FK)
    ├── product_id (FK)
    ├── order_date
    ├── order_amount
    └── ...

       ↓ DENORMALIZE
       
Diamond Layer (Semantic)
└── obt_customer_orders (OBT)
    ├── order_id
    ├── customer_id
    ├── customer_name         ← From dim_customers
    ├── customer_segment      ← From dim_customers
    ├── product_id
    ├── product_name          ← From dim_products
    ├── product_category      ← From dim_products
    ├── order_date
    ├── order_amount
    └── ... (all relevant attributes)
```

## Next Steps

1. **Create OBT Models** - Start with `obt_customer_orders`
   ```bash
   dbt run -s diamond
   ```

2. **Add Indexes** - Optimize for BI query patterns
   ```sql
   CREATE INDEX idx_obt_customer_orders_date 
   ON diamond.obt_customer_orders(order_date DESC);
   ```

3. **Connect BI Tools**
   - Tableau: Direct table connection to `diamond.obt_*`
   - Power BI: Import into Power BI Desktop
   - Looker: Create views on top of OBT tables
   - Metabase: Auto-discovery of diamond schema

4. **Create Aggregates** (if needed for performance)
   ```sql
   -- agg_daily_sales.sql
   SELECT 
     order_date,
     customer_segment,
     product_category,
     SUM(order_amount) as daily_sales
   FROM diamond.obt_customer_orders
   GROUP BY 1,2,3
   ```

5. **Document Metrics** - Add business definitions
   ```yaml
   # _models.yml
   models:
     - name: obt_customer_orders
       description: "Semantic layer for customer order analysis"
       metrics:
         - name: total_sales
           description: "Sum of all order amounts"
           formula: "SUM(order_amount)"
   ```

## Related Files

- [dbt/models/_README.md](../_README.md) - All layers documentation
- [dbt_project.yml](../../dbt_project.yml) - dbt configuration
- [profiles.yml](../../profiles.yml) - Database targets

## References

- [One Big Table (OBT) Pattern](https://www.getdbt.com/analytics-engineering/transformation/medalhao-architecture/)
- [Semantic Layer Best Practices](https://www.getdbt.com/blog/semantic-layer-benefits/)
- [OLAP Cube Design](https://en.wikipedia.org/wiki/OLAP_cube)

---

**Status**: 💎 Diamond Layer Ready  
**Pattern**: Medalhão + Diamond (Extended)  
**Layer Count**: 5 (Bronze → Silver → Staging → Gold → Diamond)
