# Meta Metrics Catalog

A comprehensive, research-informed business metric catalog spanning finance, sales, marketing, product, customer success, ecommerce, support, engineering, HR, operations, supply chain, data, security, strategy, and quality. Generates dbt Semantic Layer (MetricFlow) definitions automatically.

## Overview

This project provides:

- **208 metrics** across 15 business domains with standardized definitions, formulas, and lineage
- **DuckDB persistent store** (`metric_catalog.duckdb`) for efficient querying and validation
- **dbt Semantic Layer YAML** generator for immediate MetricFlow integration
- **Mermaid ERD** visualization of the catalog schema
- **Structured metadata** including quality rules, owners, benchmarks, dependencies, and change history

## Quick Start

```bash
# Install dependencies
~/.local/bin/uv sync

# Generate DuckDB from JSON source
.venv/bin/python metric_catalog_to_duckdb.py data/metric_catalog_v1.json data/metric_catalog.duckdb

# Generate dbt Semantic Layer YAML (with custom schema/database)
.venv/bin/python metric_catalog_to_dbt.py data/metric_catalog.duckdb data/dbt_output \
  --schema analytics --database warehouse
```

Or use the installed CLI commands:

```bash
source .venv/bin/activate
load-catalog data/metric_catalog_v1.json
generate-dbt data/metric_catalog.duckdb --schema analytics --database warehouse
```

## Project Structure

```
.
├── main.py                          # Metric definition engine (208 metrics)
├── metric_catalog_to_duckdb.py      # JSON → DuckDB loader with validation
├── metric_catalog_to_dbt.py         # DuckDB → dbt Semantic Layer generator
├── data/
│   ├── metric_catalog_v1.json       # Source: all metrics in pseudocode
│   ├── metric_catalog.duckdb        # Persistent store with 7 tables + views
│   ├── metric_catalog.md            # Mermaid ERD diagram
│   └── dbt_output/                  # Generated dbt YAML structure
│       ├── _semantic_models.yml     # 30 semantic models (one per source)
│       ├── _sources.yml             # dbt source declarations
│       └── <domain>/
│           └── _metrics.yml         # Domain-specific metrics
├── pyproject.toml                   # Python project config with CLI entry points
└── README.md                        # This file
```

## Data Model

### Core Tables

| Table | Purpose | Rows |
|---|---|---|
| `metric` | Canonical metric definitions | 208 |
| `metric_version` | Formula evolution (pseudocode → SQL) | 208 |
| `metric_column` | Column-level lineage (source, role) | 45 |
| `metric_dependency` | Computational edges between metrics | 38 |
| `metric_dimension` | Dimensional attributes (segments, cohorts) | 100+ |
| `metric_quality` | Quality rules (freshness, completeness, accuracy) | 679 |
| `data_source` | Physical tables (invoice, order, customer…) | 30 |

### Key Features

- **Dependencies**: Every metric knows which metrics it depends on (e.g., `finance.arr` depends on `finance.mrr`)
- **Lineage**: Column-level source tracking; joins between tables automatically detected from formulas
- **Quality Rules**: Freshness, completeness, accuracy, consistency rules per metric with thresholds and severity
- **Owners & Tags**: Business ownership, domain tags, related metrics
- **Change History**: Tracked changes and version history for each metric
- **Metadata**: Benchmarks, execution cost, usage context, access controls

## Metric Types

- **Simple metrics** (43): Direct aggregations with lineage (`sum`, `count`, `avg`, `max`, `min`, `count_distinct`)
- **Derived metrics** (165): Computed from other metrics or complex expressions (`ratio`, `custom`, formulas)

### Example Metrics

| ID | Name | Formula | Type |
|---|---|---|---|
| `finance.arr` | Annual Recurring Revenue | `MRR * 12` | `derived` |
| `finance.arpu` | Average Revenue Per User | `net_revenue / NULLIF(distinct_paying_users,0)` | `derived` |
| `product.dau_mau_ratio` | Stickiness | `DAU / NULLIF(MAU,0)` | `derived` |
| `finance.net_revenue` | Net Revenue | `SUM(invoice.net_amount)` | `simple` |
| `ecommerce.orders` | Orders | `COUNT(order.id)` | `simple` |

## dbt Semantic Layer (MetricFlow) Integration

The generator automatically creates:

1. **Semantic Models** (30 total, one per data source)
   - Inferred primary entity from metric grain
   - Measures with aggregation type
   - Dimensions with optional temporal role

2. **Metrics** (208 total, grouped by domain)
   - `type: simple` for direct aggregations
   - `type: derived` for formulas and cross-metric dependencies
   - `meta:` block with quality rules, owners, benchmarks

3. **Sources**
   - dbt source declarations pointing to your warehouse schema/database

### Generated YAML Example

```yaml
# _semantic_models.yml
semantic_models:
  - name: invoice
    model: ref('invoice')
    entities:
      - name: customer
        type: primary
        expr: customer_id
    dimensions:
      - name: region
        type: categorical
        expr: region_id
    measures:
      - name: net_revenue
        agg: sum
        expr: net_amount
        description: Revenue after discounts and adjustments

# finance/_metrics.yml
metrics:
  - name: finance_arr
    label: Annual Recurring Revenue
    type: derived
    type_params:
      expr: finance_mrr * 12
      metrics:
        - name: finance_mrr
    meta:
      quality:
        - dimension: freshness
          rule: max_age_hours
          threshold: "26"
          severity: error
      owners:
        - team: Finance
          contact: finance@empresa.com
```

## Usage

### Query the Catalog

```sql
-- List all metrics in a domain
SELECT metric_id, name, aggregation, grain
FROM metric
WHERE metric_id LIKE 'finance.%'
ORDER BY metric_id;

-- Find metric dependencies
SELECT m1.metric_id, m2.metric_id, type
FROM metric_dependency md
JOIN metric m1 ON m1.metric_id = md.metric_id
JOIN metric m2 ON m2.metric_id = md.depends_on_metric_id
WHERE m1.metric_id = 'finance.arr';

-- Quality rules for a metric
SELECT dimension, rule, threshold, severity
FROM metric_quality
WHERE metric_id = 'finance.net_revenue';

-- Column-level lineage
SELECT mc.metric_id, ds.table_name, mc.column_name, mc.role
FROM metric_column mc
JOIN data_source ds ON ds.source_id = mc.source_id
WHERE mc.metric_id = 'finance.net_revenue';
```

### Generate Outputs

```bash
# Regenerate entire DuckDB (slow, 5-10 seconds; destructive)
python metric_catalog_to_duckdb.py data/metric_catalog_v1.json data/metric_catalog.duckdb

# Regenerate dbt YAML (fast, <1 second; non-destructive)
python metric_catalog_to_dbt.py data/metric_catalog.duckdb data/dbt_output

# Customize warehouse details
python metric_catalog_to_dbt.py data/metric_catalog.duckdb data/dbt_output \
  --schema "analytics" \
  --database "prod_warehouse"
```

### Extend the Catalog

Edit `main.py` to add new metrics:

```python
metric(
    domain="my_domain",
    key="my_metric",
    name="My Metric Display Name",
    desc="Metric description.",
    expr="SUM(my_table.my_column)",
    agg="sum",
    grain="mês",
    unit="BRL",
    tags=["my_domain", "custom"],
    owner_team="My Team",
    owner_contact="my.team@empresa.com"
)
```

Then regenerate:

```bash
python main.py  # Writes data/metric_catalog_v1.json
python metric_catalog_to_duckdb.py data/metric_catalog_v1.json
python metric_catalog_to_dbt.py data/metric_catalog.duckdb data/dbt_output --schema <your_schema>
```

## Domain Coverage

| Domain | Metric Count | Key Metrics |
|---|---|---|
| **Finance** | 25 | ARR, MRR, CAC, LTV, Margin, Cash, Runway |
| **Sales** | 20 | Pipeline, Win Rate, Quota, Average Deal Size |
| **Marketing** | 18 | Lead Gen, MQL→SQL, ROAS, Campaign ROI |
| **Product** | 20 | DAU, MAU, Retention, Activation, Adoption |
| **Customer** | 16 | Churn, Retention, GRR, NRR, Expansion |
| **Ecommerce** | 15 | Orders, AOV, Conversion, Cart Abandonment |
| **Support** | 12 | Tickets, CSAT, Resolution Time, SLA |
| **Engineering** | 12 | Deployment Frequency, Lead Time, Incident Rate |
| **HR** | 12 | Headcount, Turnover, Time-to-Hire, Engagement |
| **Operations** | 12 | Throughput, Cycle Time, Utilization, Defects |
| **Supply Chain** | 12 | Inventory, Fill Rate, Supplier SLA, DIO |
| **Data** | 10 | Freshness, Completeness, Pipeline Success, Validity |
| **Security** | 8 | Incidents, MTTD, MTTC, Vulnerabilities |
| **Strategy** | 8 | Customer Growth, Market Share, KPI Attainment |
| **Quality** | 8 | Defect Density, Complaints, Scrap, Audit |

## Key Concepts

### Grain

Free-text label describing the primary entity (e.g., `"fatura"` → customer_id, `"assinatura"` → subscription_id). Used to infer the primary entity in generated semantic models.

Supported grains: customer, invoice, subscription, order, opportunity, lead, ticket, employee, deployment, incident, campaign, SKU, contract, account, product, etc.

### Aggregation

- `sum`, `count`, `count_distinct`, `avg`, `max`, `min`: Simple metrics (one measure per semantic model)
- `ratio`: Ratio metric (numerator / denominator)
- `custom`: Complex formula (derives from other metrics or complex expressions)

### Quality Dimensions

- **Freshness**: `max_age_hours` — how recently updated
- **Completeness**: `null_rate` (sum/count) or `denominator_not_zero_rate` (ratio)
- **Accuracy**: `value_in_range_0_to_2` (for % ratios)
- **Consistency**: `cross_system_reconciliation` (for audited metrics)

## Architecture

### Load Pipeline

```
main.py (define 208 metrics)
  → data/metric_catalog_v1.json (pseudocode formulas)
    → metric_catalog_to_duckdb.py (validate, extract lineage, populate 7 tables)
      → data/metric_catalog.duckdb (persistent queryable store)
        → metric_catalog_to_dbt.py (extract semantic models, translate formulas)
          → data/dbt_output (30 semantic models + 208 metrics in YAML)
```

### Key Design Decisions

1. **Pseudocode formulas** allow portable metric definitions before warehouse-specific SQL is written
2. **Lineage extraction** via regex parsing of `table.column` references finds joins automatically
3. **Two-pass insertion** into DuckDB (metrics first, then dependencies) avoids FK constraint violations
4. **Formula translation** for `derived` metrics replaces metric-name tokens with MetricFlow-safe slugs
5. **Grain → entity mapping** uses a configurable `GRAIN_TO_ENTITY` dict to infer primary keys

## Performance

- **Load JSON → DuckDB**: ~5–10 seconds (208 metrics, 7 tables, validation)
- **Query catalog**: ~10–100ms (indexed on metric_id, indexed views)
- **Generate dbt YAML**: <1 second (pure Python, no DB writes)

## Troubleshooting

**"Binder Error: FK constraint violation"**
- Usually caused by a metric depending on a non-existent metric. Check `metric_dependency` table.
- Solution: Ensure all `depends_on_metric_id` values exist in the `metric` table.

**"metrics: [] causes MetricFlow validation error"**
- Derived metrics with no input metrics are invalid. Solution: Ensure the formula references at least one valid input metric.

**"TODO_replace_schema / TODO_replace_database in _sources.yml"**
- Override with `--schema` and `--database` flags when running the generator.

**"Grain not recognized"**
- Add it to `GRAIN_TO_ENTITY` in `metric_catalog_to_dbt.py` or use a slug-normalized version.

## Contributing

- **Add metrics**: Edit `main.py`, add to the appropriate domain section
- **Fix formulas**: Update `DEPS` dict in `main.py` or the `expression` field in `metric_version` table
- **Extend schema**: Add columns to any table in `metric_catalog_to_duckdb.py::DDL`, regenerate with `metric_catalog_to_duckdb.py`
- **Customize entity mapping**: Update `GRAIN_TO_ENTITY` in `metric_catalog_to_dbt.py`

## Resources

- [Mermaid ERD](data/metric_catalog.md) — Visual schema overview
- [dbt Semantic Layer Docs](https://docs.getdbt.com/docs/use-dbt-semantic-layer)
- [MetricFlow Specifications](https://github.com/dbt-labs/metricflow)

## License

Internal use only.
