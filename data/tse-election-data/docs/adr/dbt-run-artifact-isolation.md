# ADR — Isolated dbt artifacts per coordinated build (PR38)

**Status:** Proposed; TDD and multi-cycle execution pending.

## Decision

The PR37 coordinator remains the single owner of a cooperative POSIX lock for
one DuckDB warehouse from generation observation through dbt CTAS and fenced
physical publication. PR38 assigns each invocation a unique artifact directory
under a configurable `DBT_ARTIFACTS_ROOT`, defaulting to a sibling of the
warehouse file. Its `target/`, `logs/`, and `duckdb_tmp/` are isolated for each
run, including failed runs. The job explicitly passes absolute
`--target-path`/`--log-path` flags to dbt 1.12 and configures
`TSE_DUCKDB_TEMP_DIRECTORY` for the DuckDB profile.

The run directory is retained for postmortem audit. No automatic retention/GC
policy is added here. Paths must be on writable local storage; the root should
not point into the canonical raw or published physical directories.

This isolation removes shared dbt manifest/compiled SQL/log/spill pathname
collisions for *cooperating* jobs that share the same project source tree while
using independent DuckDB warehouses. The existing warehouse lock still
serializes jobs targeting exactly the same canonical DuckDB path.

## Falsification requirements

- Two successive jobs on one warehouse create distinct artifact directories.
- A failed dbt run retains its artifacts and does not advance publication.
- Two concurrent jobs with independent warehouses and different electoral scopes
  (2026/general and 2024/municipal) both reach dbt before either finishes.
- Both cycles publish separate valid Parquet generations and Gold paths.
- No automatic modification of canonical TSE raw or published partitions.
- Existing unit tests and scoped dbt gate remain green.

## Boundaries

- POSIX local filesystem semantics only; no distributed locks or transactions.
- Other writers and `dbt deps` operations are outside the coordinator's lock.
- Physical power-loss recovery, generation garbage collection and cross-partition
  snapshot atomicity remain unverified.
- Synthetic two-cycle concurrency tests do not certify full TSE historical
  source coverage. Real multi-cycle analysis is a separate gate.
