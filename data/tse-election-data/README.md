# TSE election data

Reproducible **local ingestion and analytics** for public Brazilian Superior Electoral Court (TSE) datasets. The pipeline discovers CKAN resources, retains versioned raw sources by SHA-256, and models them with **DuckDB + dbt** (`Bronze → Silver → Gold → Semantic`). Physical Parquet partitions support selected analytical facts.

**Validated cycles:** 2018 (general), 2020 (municipal), 2022 (general), and 2024 (municipal) are frozen; **2026 (general) is provisional**. See [cycle contracts](docs/supported-election-cycles.md).

## Quick start

Requires **Linux/WSL, Python ≥ 3.13, and [`uv`](https://docs.astral.sh/uv/)**. Run from the repository root:

```bash
cd data/tse-election-data
make setup
make list YEARS=2026  # inspect selection; no source downloads
make unit
make medallion-contracts
```

Use `make help` to see the other targets. **Ingestion and production gates download/process data**; choose a storage volume with enough free space before running them.

<details>
<summary><strong>Ingest and build the analytical warehouse</strong></summary>

Run the 2026 incremental pipeline explicitly:

```bash
make pipeline YEARS=2026
```

- `make ingest` downloads and prepares changed raw resources; `make backfill` acquires historical versions.
- `make dbt-build` builds/tests the analytical graph; `make pipeline` combines incremental ingestion with that build.
- `YEARS` defines the analytical cycles; `UPDATE_YEARS` selects the subset to refresh and defaults to `YEARS`. The `general` / `municipal` types come automatically from the election calendar.
- Example: `make pipeline YEARS="2018 2020 2022 2024 2026" UPDATE_YEARS=2026` refreshes 2026 while keeping all five cycles in analytical scope.
- Defaults: `PROFILE=analytics`, `GRANULARITY=brasil`, `DATA_ROOT=./data/tse`, `DUCKDB_PATH=./data/warehouse/tse_analytics.duckdb`. Override with Make variables as needed.
- `make materialize-raw` rehydrates selected local raw cache without downloading it again.

</details>

<details>
<summary><strong>Architecture and local storage</strong></summary>

```text
TSE CKAN → versioned RAW (SHA-256) → dbt Bronze → Silver → Gold → Semantic
                                             ↘ physical Parquet fact partitions
```

- `data/tse/raw/`: versioned physical sources and regenerable extractions.
- `data/tse/_metadata/ingest_manifest.jsonl`: ingestion audit history.
- `data/tse/_metadata/resource_state.json`: durable active resource state.
- `data/tse/_metadata/pending_resources.json`: sources awaiting publication/recovery.
- `data/tse/_metadata/current_objects.jsonl`: derived index of physically available active objects.
- `data/warehouse/`: DuckDB warehouse and materialized Parquet facts.

Raw source versions are identified by content hash. Extracted files are cache, not the durable source of truth. Read [storage policy](docs/raw-storage-policy.md), [architecture ADR](docs/adr/001-medallion-architecture.md), and [model DAG](docs/model-dag.md).

</details>

<details>
<summary><strong>Recovery and safety boundaries</strong></summary>

After a process interruption, ingestion reconciles the active snapshot, pending journal, and published manifest. Rehydrated and orphaned local sources are checked against SHA-256/size before use; an incomplete active materialization is not treated as a successful skip.

**Certified scope:** single coordinator on a local filesystem, including process crashes (`SIGKILL`). **Not certified:** concurrent writers or power-loss durability. Do not treat `current_objects.jsonl` as a substitute for the underlying physical files.

`make prune-cache` is a dry run by default; `APPLY=1` deletes regenerable extracted cache. **Avoid `make clean-all CONFIRM=1` unless you intentionally want to remove generated raw and warehouse state.**

</details>

<details>
<summary><strong>Quality gates, coverage, and specifications</strong></summary>

```bash
make unit                 # Python contracts (no TSE dataset required)
make medallion-contracts   # dbt DAG/lineage structure
make raw-contracts        # requires local raw state
make cycle-regressions    # requires an existing built warehouse
```

Full end-to-end certification (ingests data and builds the warehouse):

```bash
make production-gate YEARS=2026
```

Historical cycles have frozen regression baselines; 2026 is provisional and requires intentional updates after a TSE republication. Known source coverage gaps are explicit, never silently reconciled away.

Read [operational contracts](docs/operational-contracts.md), [source coverage/gaps](docs/source-coverage-and-gaps.md), and [machine-readable cycle baselines](contracts/election_cycles.json).

</details>
