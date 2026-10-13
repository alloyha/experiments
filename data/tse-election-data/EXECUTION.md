# TSE — local storage audit and evidence collection

**Status:** post-PR39 (durable source recovery) and post-PR40 (`YEARS` / `UPDATE_YEARS`).
This document supersedes the earlier PR32–34 patch-execution guide. Those
changes are already integrated; **do not reapply the old patches**.

## Purpose and safety boundary

Measure the local storage footprint and reconcile it against the TSE control
plane without deleting, moving, extracting, downloading, or changing any payload.
The auditor separates:

- **Durable data**: immutable `raw/**/source/*`, other version-level files
  (e.g., preparation artifacts), `_metadata` state/journal/manifest,
  persisted DuckDB and physical candidate facts.
- **Cache**: `raw/**/extracted/*`, which may be reproducible from source after
  integrity checks and under the materialization contracts.
- **Source-backed *candidates***: extracted bytes that have exactly one
  co-located immutable `source/*` sibling; this is an **estimate**, not a
  proof of integrity or authorization to remove anything.

`current_objects.jsonl` reflects *physically present* active tabular objects;
`resource_state.json` records active version truth; `pending_resources.json`
is the durable unpublished journal. Evicted extracted files can be absent from
the index without invalidating their active source versions.

**Never run `make clean-all`, `make clean-warehouse`, or `make prune-cache APPLY=1`
merely to perform this audit.** Neither `make storage-report` nor its Python
script changes state. No live-data falsification or eviction is required here.

## Before touching the checkout

Keep the original, untracked `EXECUTION.md` and
`scripts/report_local_storage.py` in the original worktree. A separate worktree
for this PR ensures that neither file nor the original raw ZIP/CSV cache is
clobbered. The new implementation must be reviewed against those originals
before it is committed.

## 1. Measure existing local data, using the new read-only command

The report can be executed from a **separate code worktree** while pointing to
the original data paths. Replace the example path if the canonical raw data
resides elsewhere:

```bash
DATA="$HOME/github/experiments/data/tse-election-data/data"

make storage-report \
  DATA_ROOT="$DATA/tse" \
  DUCKDB_PATH="$DATA/warehouse/tse_analytics.duckdb" \
  CANDIDATE_FACT_ROOT="$DATA/warehouse/fact_candidate_votes"

# Capture machine-readable evidence outside the raw state directory.
make storage-report FORMAT=json \
  DATA_ROOT="$DATA/tse" \
  DUCKDB_PATH="$DATA/warehouse/tse_analytics.duckdb" \
  CANDIDATE_FACT_ROOT="$DATA/warehouse/fact_candidate_votes" \
  > /tmp/tse-storage-audit-before.json
```

By default this reads file sizes and the control-plane index but **does not
hash** the source archives. A full integrity pass is explicitly requested with
`VERIFY_HASHES=1`; it reads all active and pending immutable source blobs and
may take substantial time and disk I/O. `STRICT=1` makes unresolved
consistency issues fail with exit code 2. For example:

```bash
make storage-report VERIFY_HASHES=1 STRICT=1 \
  DATA_ROOT="$DATA/tse" \
  DUCKDB_PATH="$DATA/warehouse/tse_analytics.duckdb" \
  CANDIDATE_FACT_ROOT="$DATA/warehouse/fact_candidate_votes"
```

The audit reports index references to missing physical files, active sources
that are missing/unrecognized, materializations unexpectedly absent (as distinct
from intentionally evicted cache), mismatches of recorded source size, and
SHA-256 mismatch *only if* hashing was enabled. A nonzero strict exit status
blocks certification; it does **not** imply that the script has repaired data.

## 2. Inspect and interpret the evidence

Examine source bytes, extracted cache, prepared/other raw, metadata, DuckDB,
physical facts, and the ranking by domain and year. Inspect active and pending
resource counts, missing/evicted materializations and dangling index references.

The reported **logical file sizes** are not `du`'s allocated physical blocks.
Hard links, sparse files, WAL files outside the configured DuckDB path, external
mounts, `.venv`, and dbt package/target caches can make actual filesystem disk
usage differ. The report never estimates savings as guaranteed reclaimable
space. A verified source hash also does not by itself prove every downstream
rehydration contract; tests and the existing materialization path provide that
additional evidence.

## 3. Non-destructive gate and optional next experiment

From the isolated code worktree:

```bash
make unit
make medallion-contracts
make compile
make storage-report FORMAT=json DATA_ROOT="$DATA/tse" \
  DUCKDB_PATH="$DATA/warehouse/tse_analytics.duckdb" \
  CANDIDATE_FACT_ROOT="$DATA/warehouse/fact_candidate_votes" \
  > /tmp/tse-storage-audit-after.json
```

The two storage snapshots should agree when no unrelated ingest/build activity
modifies the live store. If a controlled prune/rehydration experiment is needed,
perform it **in a separate disposable dataset copy**, with a new independent
`DATA_ROOT` and DuckDB path, after validating the dry-run and confirming source
integrity. Never perform this experiment on the only durable local copy.

## Acceptance criteria

- Read-only commands do not change any raw/control-plane/warehouse bytes.
- The report never treats a directory name as a source integrity proof.
- An invalid root cannot produce a misleading successful zero-byte report.
- Missing indexed files, unexpected missing active sources and optional SHA
  mismatches are detectable and fail `STRICT=1`.
- Intentional `materialization_state=evicted` is distinguished from an
  unexpectedly missing extracted object.
- Baseline and final snapshots are retained for later analysis and the
  provenance of measured physical data is clear.

## 4. Known first-baseline anomalies (2026-10-13)

The first physical audit measured **37.9 GiB** (5.7 GiB immutable source,
31.5 GiB extracted cache, 489.8 MiB DuckDB, 139.9 MiB physical facts).
Two `vote_result` outputs were absent for 2018 and 2022, and a `candidate`
resource for 2026 lacked both its expected source and one materialization.
These observations block certification even though the 160 Python tests pass.
The original v1 summary mistakenly displayed `0` for all missing materializations:
that counter represented only *intentionally evicted* files. The v2 counters
separate expected evictions and unexpected missing objects.

Investigate the three resource keys without changing the raw store:

```bash
make storage-report DIAGNOSE=1 \
  DATA_ROOT="$DATA/tse" \
  DUCKDB_PATH="$DATA/warehouse/tse_analytics.duckdb" \
  CANDIDATE_FACT_ROOT="$DATA/warehouse/fact_candidate_votes"
```

Inspect the expected path and local source siblings in the diagnostics. The
presence of a source sibling is insufficient to certify it; only `VERIFY_HASHES=1`
proves the digest matches authoritative metadata, and then controlled offline
rehydration must still be demonstrated before cache eviction. Do not automatically
rewrite `resource_state.json` or `current_objects.jsonl` to hide the anomalies.

## 5. Certification outcome — 2026-10-13

The following is an **observed local-environment result**, not a CI fixture or
an assertion that all historical source versions remain available. It was
collected after an independent recovery/republication workflow, using the
canonical data paths for the original checkout, not the separate code worktree.

| Current-state check | Observed result |
|---|---|
| Active resources / pending | 86 / 0 |
| Active source SHA-256 digests verified | 86 / 86 |
| Active source files missing | 0 |
| Unrecognized active source paths | 0 |
| Unexpectedly missing extracted outputs | 0 |
| Physical current-index rows | 85 |
| Extracted cache, logical bytes | 39.2 GiB (rounded) |
| Persisted DuckDB, original data root | 489.8 MiB (rounded) |
| Physical candidate facts, original data root | 139.9 MiB (rounded) |
| Entire measured logical footprint, original data root | 45.6 GiB (rounded) |
| Strict storage report (`STRICT=1 VERIFY_HASHES=1`) | Exited successfully |
| Raw physical/index contracts | PASS (85 indexed objects; 5 candidate scope objects) |

The `current_objects.jsonl` row count need not equal the count of active
resources: the index enumerates **physically available tabular objects**, not
all control-plane resources. The DuckDB and candidate-fact numbers must be
measured with explicit `DUCKDB_PATH` and `CANDIDATE_FACT_ROOT`; omitting them
in a separate worktree previously produced a misleading 12 KiB/0 B report.

### Provenance: `historico_candidatura_2026`

The 2026 active source initially lacked both its ZIP and expected extracted
CSV, and the stored physical path still used the legacy `domain=candidate`
partition. The current state is valid under `domain=candidate_history`.
The append-only ingestion manifest retains **two distinct published versions**:

- **2026-10-08 10:24:08 UTC:** SHA-256
  `a1f1c1696868acdc0bfb8eedc15090ccaf732ea28af3cba7c0bf7df6895062a4`,
  size 4,414,973 bytes. The original bytes have **not** been recovered.
- **2026-10-13 12:02:32 UTC:** SHA-256
  `16fd5803af84fcb2d02dbecda70999d766c11264e833d354b6c0e88f0f1e0338`,
  size 4,423,762 bytes. This is the **active, physically present, hash-verified**
  version, indexed as `candidate_history`.

The later `metadata-skip` executions observed an already-published version;
they did not themselves acquire or publish the new source. There is no proof
of why the original bytes became unavailable or exactly when the upstream
content changed. The original manifest record must remain immutable. In
particular, never put the new bytes into the old SHA-addressed directory.

### Scope of certification and remaining caveats

The **current active state** has passed the strict physical consistency and
hash-integrity gate. This does **not** certify recoverability of the missing
2026-10-08 snapshot, all historical/orphan blobs, model correctness, or a
future source republication. The physical presence and matching hash of a ZIP
also do not guarantee that every extracted member will be reproducible under
future extractor versions.

The **39.2 GiB cache is not yet authorized for eviction**. Before destructive
experiments, demonstrate the complete prune -> offline rehydrate -> index
repair -> downstream equality cycle on a disposable dataset copy, preserve
version-level truth and backups, and identify which cache can be reclaimed
without violating consumer requirements. Never run this experiment against
the sole copy of the user's data.

Keep JSON and forensic evidence outside the Git repository if they contain
absolute local paths; the rounded aggregate results above are safe to record
in this guide. The historical-retention gap should be tracked independently
of this read-only audit's certification gate.
