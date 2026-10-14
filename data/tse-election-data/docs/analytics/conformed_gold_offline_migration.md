# Conformed Gold — offline copy migration (post PR #48)

## Objective

Migrate the **persisted** DuckDB warehouse from the pre-PR-48 schema to
`cycle_id = year:type` and `election_id = year:type:code` on an *isolated clone*.
Do **not** read the evicted raw CSV cache, rebuild Bronze/Silver, modify the
source database, mutate immutable ZIPs, or republish Parquet.

## What the command does

1. Requires explicit quiescence and acquires the existing cooperative warehouse
   `.build.lock`; refuses if the source DuckDB `.wal` is present.
2. Verifies the source file SHA-256, copies it into a temporary sibling, and
   re-verifies the unchanged source SHA. Keeps the **original** in place.
3. Within one transaction, adds or repairs `cycle_id` on eight persisted Gold
   dimensions/facts; fixes `dim_candidate.election_id`; reconstructs
   `dim_election_cycle` from `dim_election`; and recreates the
   `fact_candidate_votes` **view** from already-published Parquet.
4. Executes the existing PR-48 singular dbt assertion SQL directly against
   DuckDB (not through a full dbt run) and fails for identity or dimensional
   coverage violations.
5. Compares before/after row counts and sums for the existing facts and key
   semantic products, checkpoints the copy, then atomically publishes the
   **candidate file**, never overwriting the source.

This certifies SQL-level conformance and gross-measure conservation; it is NOT
an exhaustive full dbt regression, partition-rebuild equivalence test, or
per-row cryptographic equality test. The original database SHA is verified
unchanged. A surviving failure can leave a `*.partial-*` file if the process is
killed; inspect/remove those manually after confirming no active writer.

## Procedure in WSL

Use a fresh branch/worktree based on merged main. Copy the migration script to
`data/tse-election-data/scripts/` and test to `data/tse-election-data/tests/`.
Close the `duckdb` CLI, dbt jobs, ingest and any other DuckDB users first.

```bash
MAIN="$HOME/github/experiments/data/tse-election-data"
DB="$MAIN/data/warehouse/tse_analytics.duckdb"
FACT="$MAIN/data/warehouse/fact_candidate_votes"
CANDIDATE="$MAIN/data/warehouse/tse_analytics.conformed-candidate.duckdb"
PY="$MAIN/.venv/bin/python"

"$PY" -m unittest discover -s tests -p test_migrate_conformed_gold.py -v

"$PY" scripts/migrate_conformed_gold.py \
  --source-db "$DB" \
  --candidate-db "$CANDIDATE" \
  --fact-root "$FACT" \
  --project-root "$PWD" \
  --ack-quiescent \
  --report "$HOME/tse-conformed-migration-report.json"
```

The migrated copy remains alongside the original so relative references in
unrelated preexisting views do not change. Make sure disk space is available:
at least `max(2 GiB, 3 × original DuckDB bytes)` free, often more for tests.

## Approval gate

A PASS must include zero identity and dimension-membership violations,
identical measure signatures, no source SHA change, a readable candidate
with the candidate-vote Parquet view resolving, and no unexpected WAL.
The live source is not promoted automatically. Do not delete it.

If the candidate passes broader semantic and reconciliation queries, decide
separately whether and how to promote it (including a rollback plan, exact
file inventory, and a fresh maintenance window). Do not combine promotion with
this preparatory PR.

## Missing election events: evidence-backed recovery (v2)

A real production copy may contain persisted Gold events absent from the
historically built `dim_election` table. This is NOT evidence that the Gold fact
records are fabricated; it means the dimension's observed coverage is smaller
than the union of actual event keys referenced by Gold. Neither invent a TSE
`election_code` nor disable `assert_conformed_dimension_membership.sql`.

**Step 1: audit the original read-only database before any copy:**

```bash
"$PY" scripts/diagnose_conformed_gold_events.py --db "$DB" \
  | tee "$HOME/tse-missing-election-events.json"
```

Inspect each missing event's observed fact/dimension sources and
`election_scope_values`. If the report indicates any `safe_to_reconcile=false`,
DO NOT continue: inspect source scope inconsistencies. The read-only audit
never changes the source or the extracted cache.

**Step 2: opt into audited enrichment of the COPIED `dim_election` only:**

```bash
"$PY" scripts/migrate_conformed_gold.py \
  --source-db "$DB" --candidate-db "$CANDIDATE" \
  --fact-root "$FACT" --project-root "$PWD" \
  --ack-quiescent --reconcile-observed-events \
  --report "$HOME/tse-conformed-migration-report.json"
```

`--reconcile-observed-events` inserts only keys already present in persisted
Gold facts/dimensions and requires exactly one authoritative compatible
`election_scope` per new event. Each inserted event has recorded source names
in the migration report. **The previous source DuckDB is never changed.**
The original count of `dim_election` is allowed to increase only by exactly
those audited additions; all other Gold/semantic row and measure signatures
must remain identical. The original key and membership assertions remain hard
failure gates.

This repairs the *isolated snapshot* and does not solve upstream `dim_election`
source coverage: future full dbt builds must separately reconcile the actual
Bronze/Silver event population and update `dim_election`'s source policy if
necessary. Never introduce dependencies on downstream Gold facts in the
original dbt dimension if doing so creates a DAG cycle.
