# Candidate Cube: independent offline dbt-contract gate

This gate is a **read-only contract audit of the already staged clone**. It is
not a call to `dbt run`, does not mutate or promote the main Gold warehouse,
and does not issue production freshness certificates. Keep the existing
`ACTIVE_UNCERTIFIED` promotion journal and all prior backup/certificates.

## Source evidence (2026-10-16)

- Manifest (5/5 cycles) SHA-256:
  `063ddf0510837804815799d010a49d61c7450ca2ef19e3aaa5533f655feddda1`
- Main Gold SHA-256:
  `77e930d91815b89bde1cd225c585f7868cd426405942f38a299500ec4955318e`
- Preview SHA-256:
  `ea53050d0c667cbf7b734e8bd1f342a8892cdb5fffb702650d5a29d8af5189d4`
- Offline cube clone SHA-256:
  `564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a`
- 5 cycles, 22,797,132 rows, 2,547,959,750 nominal votes, 29 columns,
  previously proved zero two-direction `EXCEPT ALL` deviations.

## Preconditions

1. Run unit tests on WSL with the project `.venv` and real DuckDB/Jinja2.
   The DuckDB integration test **must not be skipped**.
2. Stop all external/non-cooperating DuckDB and Parquet writers. POSIX locks
   prevent only properly cooperating publishers. Keep main/preview/clone WAL-free.
3. Use the exact preexisting manifest, stage report and clone. The script
   refuses missing/mismatched identities and changed source Parquets.
4. Keep ample free disk for DuckDB temporary spill. This executes the real dbt
   conservation (`EXCEPT ALL` against published physical fact) and share/rank
   assertions for every cycle, potentially consuming considerable CPU/disk.

## Run from `data/tse-election-data`

```bash
.venv/bin/python -m unittest discover -s tests \
  -p 'test_audit_staged_candidate_cube_contracts.py' -v

ROOT="$PWD/data/warehouse"
EVIDENCE="$ROOT/cube_transfer_evidence"
.venv/bin/python scripts/audit_staged_candidate_cube_contracts.py \
  --main-db "$ROOT/tse_analytics.duckdb" \
  --preview-db "$ROOT/tse_event_registry_preview.duckdb" \
  --clone-db "$ROOT/tse_analytics_cube_v1.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --manifest "$EVIDENCE/five-cycle-eligibility.json" \
  --manifest-sha256 063ddf0510837804815799d010a49d61c7450ca2ef19e3aaa5533f655feddda1 \
  --stage-report "$EVIDENCE/candidate-cube-stage-v1.json" \
  --clone-sha256 564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a \
  --report "$EVIDENCE/candidate-cube-offline-contracts-v1.json" \
  --memory-limit 4GB \
  --ack-quiescent
```

The gate uses **unmodified repository SQL** from:

- `tse_dbt/tests/assert_cube_candidate_vote_conservation.sql`
- `tse_dbt/tests/assert_cube_candidate_share_and_rank.sql`

Those two source files are checksummed and checked again after execution.
It compiles just the narrowly allowlisted `ref()` and cycle predicates via
`StrictUndefined` Jinja2, executes each query on a DuckDB opened in
`read_only=True` mode, checks the full cube inventory, grain uniqueness and
not-null constraints, then rechecks input/output hashes *before writing a
create-only evidence file*. This reproduces the data assertions; it does not
run dbt's CLI or its artifact generation pipeline.

The clone has **no pre-existing stable `.build.lock` guaranteed** by the stage
protocol. The gate protects main, preview and all five physical partitions
using their existing shared locks; it separately pins the clone SHA-256 before
and after and requires operator-acknowledged quiescence. Direct ad-hoc writes
can still bypass cooperative locking.

If interrupted, do **not** use `Ctrl+Z` (suspended process may hold locks).
Use `Ctrl+C`; no PASS report is issued after an incomplete run. Existing
results remain intact. The report name is create-only; use a new one on retry
only after auditing any partial result.

## Promotion gate (not yet implemented)

PASS from this offline audit is **necessary but not sufficient** for promotion.
A successor promotion protocol must preserve the previously successful
promotion journal/backup, explicitly invalidate the main-path freshness
identity through epoch rotation, verify recovery under actual process death,
then reissue **path-bound graph and cube freshness certificates**. A successful
clone audit must never be described as certification of the main Gold.
