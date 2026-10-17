# Candidate cube transfer — one-time legacy publication-lock bootstrap

**Not a data migration.** Older published physical partitions can predate the
`.publish.lock` protocol; the transfer auditor must never silently manufacture
missing locks during a read-only audit or fall back to weaker coordination.

The separate bootstrap command initializes **only** an absent `.publish.lock`
inside one existing candidate fact partition, using `O_CREAT|O_EXCL` and
`O_NOFOLLOW` when available. It holds the *real exclusive* warehouse locks
in sorted order and then the partition lock, fsyncs new lock inode+directory,
verifies the approved SHA-256 of main and preview before/after and fingerprints
published Parquet bytes before/after. It never changes `.duckdb`, `.lifecycle`,
`.promotion.json`, Parquet or freshness certificates. Existing sidecars are
preserved, not replaced. Re-running is idempotent.

**Maintenance window required:** stop all DuckDB processes, dbt runs and
candidate-fact publishers first. A publisher using a different warehouse path
can evade these two warehouse locks; `--ack-quiescent` is an operator assertion,
not an automatically demonstrated guarantee. Do not use during concurrent writes.

```bash
cd ~/github/experiments/data/tse-election-data
ROOT="$PWD/data/warehouse"
PY=.venv/bin/python

# Run the tests without modifying any canonical data.
"$PY" -m unittest discover -s tests -p 'test_bootstrap_candidate_partition_lock.py' -v

# AFTER confirming quiescence, for the known legacy 2018 partition:
"$PY" scripts/bootstrap_candidate_partition_lock.py \
  --main-db "$ROOT/tse_analytics.duckdb" \
  --preview-db "$ROOT/tse_event_registry_preview.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --cycle 2018:general \
  --main-sha256 77e930d91815b89bde1cd225c585f7868cd426405942f38a299500ec4955318e \
  --preview-sha256 ea53050d0c667cbf7b734e8bd1f342a8892cdb5fffb702650d5a29d8af5189d4 \
  --ack-quiescent --confirm-bootstrap
```

If the command reports `created: true`, a new stable sidecar was introduced.
If it reports `created: false`, the existing lock was reused. Never unlink or
recreate a published `.publish.lock`. On a fail-closed result, inspect before
retrying; do not overwrite locks or relax transfer-contract checks.

Use `audit_candidate_cube_transfer.py --cycle 2018:general --max-rows 10000000`
separately, after lock bootstrap. Older cycles are large: ensure adequate
free WSL filesystem space and plan for DuckDB temp spill. A PASS is transfer
**eligibility only**, not a cube freshness certificate, production permission,
or authorization to copy rows into the main warehouse.
