# Conformed Gold: recoverable offline-copy promotion (operator runbook)

**Status: prototype, not production-certified.** The main warehouse is not
modified until `promote` is invoked with `--ack-quiescent`,
`--confirm-promote`, and exact approved input hashes. A successful promotion
finishes as `ACTIVE_UNCERTIFIED`, not a certified graph.

## Evidence motivating the promotion

- Initial `tse_analytics.duckdb`: SHA-256
  `35460201f56147f5ddd4080368a1af88c80814528b71459aff602cb952534cff`.
- Conformed v2: SHA-256
  `77e930d91815b89bde1cd225c585f7868cd426405942f38a299500ec4955318e`.
- Five historical cycle regressions, three event-source dbt tests, eight
  `2024:municipal` graph assertions, independent graph audit, guarded
  vertex/edge samples: PASS **on the v2 path**.
- Persisted views: 28. Twelve schemas identical, fourteen equally fail to
  resolve evicted 2026 raw files in both warehouses, and two are additive
  `cycle_id VARCHAR` projections (`fact_candidate_votes` and
  `party_tally_coverage_gaps`). This is a schema compatibility assessment,
  **not** live Bronze/Silver readiness.
- The candidate performance cube is absent from original and v2, but present
  in `tse_event_registry_preview.duckdb`. Do not silently copy that table.

## Safe local checks (never alter original or candidate)

From the monorepo root:

```bash
cd ~/github/experiments/data/tse-election-data
.venv/bin/python -m unittest discover -s tests -p 'test_promote_conformed_gold*.py' -v

.venv/bin/python scripts/promote_conformed_gold.py preflight \
  --original "$PWD/data/warehouse/tse_analytics.duckdb" \
  --candidate "$PWD/data/warehouse/tse_analytics_conformed_v2.duckdb" \
  --original-sha256 35460201f56147f5ddd4080368a1af88c80814528b71459aff602cb952534cff \
  --candidate-sha256 77e930d91815b89bde1cd225c585f7868cd426405942f38a299500ec4955318e
```

## Actual promotion (separate explicit operator approval required)

Do **not** execute this step without a maintenance window, all DuckDB/CLI/dbt
writers shut down, and a reviewed recovery plan. The CLI action is named
`promote` and requires `--ack-quiescent --confirm-promote`, both hashes,
`--original`, `--candidate`, and availability of the committed cycle contracts.

The promoter takes exclusive stable `<db>.build.lock` locks for both databases;
checks hashes/WAL; makes durable, hash-verified backup and incoming copies in
the same directory; creates an fsynced journal; rotates the main `.lifecycle`
epoch; atomically renames incoming data to the **original exact filename**;
fsyncs the directory; validates event and cycle counts, source provenance,
Gold view resolution and five cycle regression contracts; and records
`ACTIVE_UNCERTIFIED`. On a postflight failure, it attempts a locked rollback,
rotates epoch again, restores from backup, and retains journal/backup. On
I/O failure of rollback itself, the journal must be recovered explicitly.

**SIGKILL cases**: journal `PREPARED` or `EPOCH_ROTATED` with old hash means the
swap did not happen; with v2 hash means the swap happened. The `rollback`
action verifies the expected hashes and recovers the old main warehouse from
the durable backup. Unknown content hashes fail closed. Orphan staging files
left before journal creation must be inspected manually.

**Do not remove or recreate `.build.lock`, `.lifecycle` or `.promotion.json`.**
Do not discard backup or journal after `ACTIVE_UNCERTIFIED` until independent
post-promotion certification and operator sign-off. No automatized journal
cleanup is implemented in this prototype.

## After successful atomic activation

1. While services remain quiescent, verify original-path SHA and read-only
   historical regressions. The promoter executes these smoke checks itself,
   but independent checks add useful evidence.
2. Reissue graph proof for the **main** warehouse path using an isolated
   attestations directory such as `graph_attestations_main`. Existing v2
   certificates are not valid for main; main lifecycle rotation invalidates
   prior main certificates.
3. Independently audit the new proof and run guarded reads at the main path.
4. Only then decide whether production consumers may resume, considering
   cooperative-only locking and raw/cube scope limitations.

## Rollback and limits

Explicit rollback requires:

```bash
.venv/bin/python scripts/promote_conformed_gold.py status \
  --original "$PWD/data/warehouse/tse_analytics.duckdb"

# With writers stopped and operator approval:
.venv/bin/python scripts/promote_conformed_gold.py rollback \
  --original "$PWD/data/warehouse/tse_analytics.duckdb" \
  --ack-quiescent --confirm-rollback
```

The rollback is refused if main differs from both recorded original and v2
hashes, the backup digest differs, or WAL is active. This is deliberate: no
blind overwrite of later writes. Use forensic recovery instead of force.

`status` obtains the main cooperative warehouse lock and reports both the
journal phase and observed active SHA-256 (`observed_content`), lifecycle
identity, and WAL presence. A process killed after rename but before writing
the new journal phase can leave the journal at `EPOCH_ROTATED` while the v2
bytes are already active. The diagnostic flags this discrepancy and sets
`recovery_required=true`; it does not silently declare the warehouse active.
An otherwise consistent `ACTIVE_UNCERTIFIED` state still does **not** establish
production certification, reader isolation, or signed attestation.

POSIX `flock` is **cooperative**. Ad hoc writers and readers can bypass the
locks. File-level atomic rename does not make a sequence of DuckDB writes,
Parquet replacements, view reads, certificate issues or client connections
transactional. No claim of live concurrency safety or production certification
is made. Crash fault-injection tests cover process-level interruption at
important boundaries; power-loss/filesystem-level fault tolerance must still
be proved on the real deployment environment.
