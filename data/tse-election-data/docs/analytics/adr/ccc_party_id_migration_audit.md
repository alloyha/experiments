# CCC B2: materialized party identity audit (read-only)

**Intent:** After PR #86 corrected `dim_party.party_id` SQL, audit actual persisted
rows before publishing ODCS `dim_party` or trusting refreshed Semantic views.
This is a diagnostic, not a repair job or publication certificate.

## Safety and boundary

- Open the **explicit** `--db` DuckDB path with `read_only=True`; never create a warehouse.
- Require a materialized `main.dim_party` **BASE TABLE** with all identity columns.
- Report each observed election cycle's row count, null keys, inconsistent
  cycle/event IDs, noncanonical party IDs, recognized legacy IDs and duplicates.
- Optionally require each `--expect-cycle` to exist; missing cycles make the audit FAIL.
- An empty warehouse cannot pass. An unrecognized election type or year string
  cannot pass. An in-progress writer may prevent a read-only DuckDB connection;
  in that case the audit fails closed rather than retrying as a writer.

## Interpretation

| Result | Meaning |
|---|---|
| `PASS` | Observed, persisted `dim_party` rows conform to cycle, event and party key equations; expected cycles are present |
| `FAIL` | Bad, legacy, duplicate or missing expected identity/cycle detected |
| `EMPTY` | The table exists but has no party rows |
| `UNAVAILABLE` | Missing/inaccessible DuckDB or no persistently materialized `dim_party` |

**`PASS` does not certify** that all expected electoral cycles were ingested, that
raw parquet publications are valid, that GG certificates remain fresh, or that
external downstream caches have migrated. It checks only observed rows; supply
all expected cycles explicitly and separately verify row-count/source completeness.

For the authoritative warehouse, invoke the script with the **path to the
persisted DuckDB** rather than a new worktree's empty default. A recommended
snapshot of results is the JSON output redirected to a file outside the
warehouse (e.g. `/tmp/tse_party_id_audit.json`). The script never writes to
DuckDB. Avoid audit and re-materialization concurrently.

## Subsequent work — not automated here

1. Review cycles with legacy or invalid IDs, plus external consumer expectations.
2. Plan an isolated backup/snapshot and a scoped partition-replacement job
   through `scripts/run_coordinated_dbt.py`. Do not run full-refresh on the
   canonical warehouse merely because an audit reported drift.
3. Compare row counts, composite uniqueness and party totals before and after
   each affected cycle; test the two Semantic consumers.
4. Assess GG certificates: a Gold mutation can invalidate prior graph evidence.
5. Only then propose the versioned ODCS `dim_party` baseline and CCC promotion.
