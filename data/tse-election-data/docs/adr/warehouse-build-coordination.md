# ADR — Per-warehouse CTAS + physical publication coordination (PR37)

**Status:** Proposed — local TDD and real-data gates pending.

## Decision

The `candidate-fact-partition` Makefile target invokes a small Python coordinator.
The coordinator takes an exclusive POSIX `flock` on a stable adjacent
`<duckdb-file>.build.lock` inode **before** observing the current published
partition generation. It runs the dbt `+int_candidate_fact_partition` CTAS and
publishes with the existing PR36 per-partition lock and CAS before releasing the
warehouse lock.

The coordinator directly executes the virtualenv's `dbt` console executable
and passes the warehouse-lock file descriptor through `pass_fds`. This is
important: an unexpected `SIGKILL` of the coordinator must not release the
warehouse lock while its dbt child still writes DuckDB. A wrapper such as `uv`
cannot be assumed to preserve arbitrary inherited descriptors.

The sidecar lock is keyed by the normalized absolute DuckDB path, not by the
DuckDB database inode (which may be newly created or replaced). Other DuckDB
warehouses use different lock inodes, so there is no global serialization.

## Failure and recovery

| Event | Required result |
| --- | --- |
| Another cooperating build holds the same warehouse | Wait until timeout or serialize; do not start second CTAS |
| CTAS fails or is interrupted | Do not call publisher or advance published generation |
| Coordinator dies but dbt is still running | dbt retains lock descriptor; second build must not start yet |
| dbt finishes but coordinator dies before publication | Old generation remains; next coordinated run rebuilds and may publish |
| Publisher rejects CAS or fails | Warehouse lock released and previous generation preserved by PR36 |
| Jobs use separate warehouse files | Distinct locks; may run independently |

## Boundaries

This is *cooperative* POSIX coordination for `candidate-fact-partition`. Other
write-capable Makefile targets or external DuckDB clients do not automatically
participate. It is not a cross-filesystem transaction, cross-partition
snapshot, distributed lease, or power-loss durability certificate. Separate
warehouses may still share dbt `target/`, `logs/`, or resource limits; ensure
per-job dbt artifact isolation when running true multi-job production builds.

## Promotion evidence

- RED/GREEN test coverage with actual subprocesses, competing jobs, failure
  injection, parent `SIGKILL`, and two independent warehouses.
- Run 2026/general CTAS + physical publication using a temporary DuckDB and
  physical partition, preserve canonical TSE raw and published data.
- Existing PR36 tests and scoped dbt 162-node gate remain green.
