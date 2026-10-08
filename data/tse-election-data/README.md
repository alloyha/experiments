# TSE election data

Local, reproducible ingestion and analytical modeling of Brazilian TSE public
election data with immutable raw sources, DuckDB, and dbt.

## Validated cycles

- 2018 general — frozen
- 2020 municipal — frozen
- 2022 general — frozen
- 2024 municipal — frozen
- 2026 general — provisional snapshot

See:

- `docs/supported-election-cycles.md`
- `docs/source-coverage-and-gaps.md`
- `docs/raw-storage-policy.md`
- `docs/operational-contracts.md`
- `docs/adr/001-medallion-architecture.md`
- `docs/model-dag.md`

## Main commands

```bash
make raw-contracts
make cycle-regressions

make production-gate \
  YEARS="2026" \
  ELECTION_TYPES="general" \
  INCREMENTAL_YEARS="2026" \
  INCREMENTAL_ELECTION_TYPES="general"
```

`current_objects.jsonl` contains only physically available active objects.
`extracted/*` is regenerable cache: pruning removes its active-index references,
and the next ingest can rehydrate it from the retained immutable source without a
network transfer.
