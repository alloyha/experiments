# TSE analytical model DAG

```mermaid
flowchart LR
    RAW["RAW<br/>immutable TSE objects"]
    B["Bronze<br/>source interpretation"]
    S["Silver<br/>canonical grain"]
    P["Physical<br/>execution helper"]
    A[("candidate_fact_partition<br/>Parquet")]
    G["Gold<br/>facts / dimensions / reconciliation"]
    M["Semantic<br/>consumer metrics"]

    RAW --> B
    B --> S
    B --> G
    S --> G
    S --> P
    P -. publish .-> A
    A -. read .-> G
    G --> M
```

Logical dependencies are validated through `ref()`. Physical lineage is declared
with `meta.physical_publish` and `meta.physical_source`, so external persisted
artifacts do not disappear from architectural lineage.

Bronze/Silver leaf models must either feed another model or explicitly declare
`meta.architecture_status='orphan'` with an `architecture_reason`.

## Layer selectors

```bash
make dbt-bronze
make dbt-silver
make dbt-gold
make dbt-semantic-layer
make medallion-contracts
```
