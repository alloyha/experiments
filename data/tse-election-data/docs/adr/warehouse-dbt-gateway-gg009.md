# ADR — Warehouse-scoped dbt gateway (GG-009, incremental)

Status: proposed; real dbt handoff passed on a disposable 2024 municipal fixture; canonical canary untouched; CI pending.

## Decision

Route dbt executions from the project `Makefile` through
`scripts/run_coordinated_dbt.py`. Warehouse-writing commands acquire the
**existing** `<absolute-duckdb-path>.build.lock` using the POSIX `flock`
implementation in `build_candidate_fact_partition`. The wrapper starts the
project's virtualenv `dbt` executable and passes the held FD via `pass_fds`,
preventing an early unlock when the Python wrapper is SIGKILLed but dbt still
runs. Do not take a second differently named lock or add a global mutex.

`deps`, `compile`, `parse`, `debug`, `ls`/`list` are exempt from the warehouse
**write** lock. All other dbt subcommands use the exclusive warehouse lock.
Each mutating dbt execution gets isolated `target/` and `logs/` under
`<duckdb>.dbt-runs/<uuid>/`. The latest `run_results.json` is published at
`<duckdb>.dbt-runs/latest_run_results.json` and used by `make
performance-report` / `make performance-baseline`. The latest pointer is
removed at the start of a new run and republished **only on success**, so an
unsuccessful or interrupted run cannot masquerade as fresh performance results.
Failed execution artifacts remain available under their run UUID directory.

The guarded graph reader acquires a shared warehouse lock before its partition
lock; coordinated candidate fact publisher uses an exclusive warehouse lock
before its partition lock. The new dbt wrapper acquires *only* the warehouse
lock, because normal dbt transformations do not publish the physical candidate
Parquet pointer. These orders avoid introducing a lock inversion.

## Evidence required before merge

1. Scoped subprocess tests: same-warehouse contention, independent databases,
   inherited lock survives coordinator SIGKILL, nonzero dbt exit propagation.
2. `make -n dbt-{bronze,silver,gold}`, `make -n semantic`, `make -n dbt-build`,
   `make -n dbt-full-refresh`, `make -n test-only`, `make -n pr-gate`; inspect
   that no invocation still uses an unguarded virtualenv dbt executable.
3. `make unit`, Medallion contracts, compile, PR CI. No canonical warehouse
   rebuild to test the CLI wiring.
4. A temporary/synthetic DuckDB graph proof, guarded reader vs running dbt
   writer, and positive post-commit freshness denial after a changed Gold row.

## Remaining limitations

- **Cooperative only.** External `dbt`/DuckDB callers and other standalone
  scripts that write the same warehouse can bypass this gateway.
- The executable or environment may itself launch detached children that do
  not inherit the lock FD; these must be tested separately.
- The certificate directory is not transactionally coupled to DuckDB. Graph
  freshness is revalidated by guarded reads; GG-009 stays *planned* until all
  write surfaces and the end-to-end handoff are covered.
- GG-012 crash/uncertain-publication certification remains independent work.

## Real dbt handoff falsifier (temporary warehouse)

`tests/test_graph_dbt_writer_handoff.py` constructs PR54's original synthetic
2024/municipal graph fixture in a disposable DuckDB file, certifies it through
GG-010, then runs a **real dbt** `run --select dim_party` behind a deterministic
wait/exec launcher. The launcher inherits the writer's `flock` fd and executes
the project's actual dbt binary, replacing the fixture Gold dimension using a
local staging relation.

The test must prove all boundaries: a read lock blocks the writer from starting;
a writer lock blocks a new reader; after dbt commits a changed `dim_party`, the
old certificate is rejected by the guarded read without returning rows.
In CI without `duckdb`/`dbt`, the test skips; run it explicitly in the project
WSL virtualenv before proposing merge and confirm it actually executed (not
skipped). Neither historical TSE Parquets nor the canonical warehouse are used.

This remains a **cooperative** handoff proof only. An ad-hoc DuckDB writer
outside the gateway can bypass advisory locks; the GG-009 catalog entry must
stay `planned` until those direct writing paths are audited and migrated.
