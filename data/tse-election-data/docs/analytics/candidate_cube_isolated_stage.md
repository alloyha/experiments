# Candidate Performance Cube — isolated, exact staged copy

**Prototype, never execute against `tse_analytics.duckdb` as output.** This is
not a production promotion or freshness certification. It consumes the 5/5
read-only eligibility manifest and stages a new DuckDB file, preserving the
already-promoted main Gold, preview, Parquets, lifecycle epoch, promotion
journal, backup and existing graph certificate.

## Preconditions

1. Run the full earlier eligibility audit and consolidate the five reports.
   Current approved 2026-10-16 manifest digest:
   `063ddf0510837804815799d010a49d61c7450ca2ef19e3aaa5533f655feddda1`.
2. Explicitly stop all DuckDB and external Parquet writers while staging.
   The script acquires *shared*, existing POSIX warehouse/partition locks and
   will not create missing locks, but it cannot fence non-cooperating writers.
3. Confirm enough disk space for a new copy of main plus materialized cube.
   The preflight reserve is `size(main) + size(preview) + 1 GiB`; this is a
   conservative floor, not a complete upper bound for spill usage. Prefer WSL
   native Linux filesystem and monitor available space.
4. Make an immutable, persistent copy of the manifest and **all five** source
   reports. `/tmp` may be cleared by system housekeeping. Do not edit the JSON.
5. Keep the original `.promotion.json` (phase `ACTIVE_UNCERTIFIED`) and backup.
   The existing `promote_conformed_gold.py` refuses additional promotions while
   its journal exists; a future *successor promotion/lineage* protocol must be
   reviewed and tested. Do not delete or rename that journal to force a swap.

## Run from data/tse-election-data

The following is a safe, non-database-mutating step to persist the manifest
first; `cp -n` will refuse to replace existing evidence:

```bash
ROOT="$PWD/data/warehouse"
EVIDENCE="$ROOT/cube_transfer_evidence"
mkdir -p "$EVIDENCE"
cp -n /tmp/tse-cube-closeout.n3XC5K/candidate-cube-five-cycle-eligibility.json \
  "$EVIDENCE/five-cycle-eligibility.json"
sha256sum "$EVIDENCE/five-cycle-eligibility.json"
# Expected: 063ddf0510837804815799d010a49d61c7450ca2ef19e3aaa5533f655feddda1

# Optional but recommended: preserve the original five independent reports.
cp -n /tmp/tse-cube-transfer.DlhPhd/{2018-general,2020-municipal}.json "$EVIDENCE/"
cp -n /tmp/tse-cube-optimized.QIuz16/2022-general.json "$EVIDENCE/"
cp -n /tmp/tse-cube-gate.vNUySN/eligibility-2024.json "$EVIDENCE/"
cp -n /tmp/tse-cube-2026.Ntkckx/2026-general.json "$EVIDENCE/"
```

First run the *small, disposable DuckDB fixture*:

```bash
.venv/bin/python -m unittest discover -s tests -p test_stage_candidate_cube.py -v
```

On the WSL `.venv` with DuckDB installed, the integration test must **not**
be skipped. It tests actual `ATTACH ... (READ_ONLY)`, CTAS and two-way `EXCEPT
ALL` against small isolated files. Synthetic tests also include real SIGKILL
before publication and verify that the original never changes.

Then, only with quiescent writers and known source hashes:

```bash
.venv/bin/python scripts/stage_candidate_cube.py \
  --main-db "$ROOT/tse_analytics.duckdb" \
  --preview-db "$ROOT/tse_event_registry_preview.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --manifest "$EVIDENCE/five-cycle-eligibility.json" \
  --manifest-sha256 063ddf0510837804815799d010a49d61c7450ca2ef19e3aaa5533f655feddda1 \
  --output-db "$ROOT/tse_analytics_cube_v1.duckdb" \
  --report "$EVIDENCE/candidate-cube-stage-v1.json" \
  --memory-limit 4GB \
  --ack-quiescent
```

The output path and report are **create-only**. The script copies main to a
hidden staging file, imports the complete `BASE TABLE` from preview via
`ATTACH ... READ_ONLY`, checks the 29-column column names/types/order and
full five-cycle counts/votes, compares all columns and all multiplicities with
`EXCEPT ALL` in **both directions per cycle**, then rechecks both database
SHA-256 and every Parquet generation/hash under existing read locks. Only on
full PASS is the staged file hard-linked atomically to the explicitly named
new output and a separate report emitted. No dbt, raw ingestion, cycle
recalculation, certificate or production replacement is invoked.

The full validation is intentionally expensive. It can spend minutes scanning
22.8 million records; monitor progress and avoid concurrent audits. `Ctrl+C`
interrupts; **never `Ctrl+Z`**, which can leave locks held by a suspended
process.

## Failure semantics

- Missing source/lock, mismatched manifest SHA, WAL, wrong inventory, stale
  Parquet, insufficient capacity, destination collision, SQL error or nonzero
  multiset discrepancy: fail closed, **never** overwrite main.
- SIGKILL before publish: hidden `.tse_analytics_cube_v1.incoming-*.duckdb`
  orphan may remain; no final output exists. Inspect before any deliberate
  cleanup; do not delete the warehouse's stable `.build.lock` or lifecycle.
- SIGKILL after clone publication but before report: the new file may exist
  without a PASS report. Treat it as **UNCERTIFIED**, not as promotion-ready.
  Do not simply rerun with the same path or reuse it without forensic checks.
- `--ack-quiescent` acknowledges an *externally enforced* maintenance window;
  it is not a technical fence against arbitrary non-cooperative processes.

## Later, independent gate

The **offline stage report is not a freshness certificate**. Before promotion,
perform separate integrity and crash-recovery tests of a successor promotion
protocol that safely archives/links the previous `ACTIVE_UNCERTIFIED` journal
without discarding its backup. A new promotion must rotate the main lifecycle
and require independent re-issuance of **both** the `2024:municipal` graph
certificate and candidate cube per-cycle certificates. Do not copy existing
certificates across DuckDB paths. `2026:general` remains provisional.
