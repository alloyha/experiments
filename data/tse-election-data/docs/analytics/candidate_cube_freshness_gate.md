# Candidate cube — versioned input freshness and invalidation (post-PR #52)

## Purpose and boundary

`certify_candidate_cube.py audit` checks *coverage*: each published candidate-vote
Parquet cycle has a nonempty persistent candidate cube partition. Coverage does
**not** prove the cube matches the current source or current dimensions.

A separate, fail-closed `audit-freshness` gate checks a per-cycle certificate
that is created **only after** targeted dbt materialization, multiset vote
conservation and share/rank tests have passed. A missing or stale certificate
fails the gate, even when `audit` passes.

## Provenance included in each certificate

- Candidate fact: active `election_type=.../year=.../data.parquet` publication
  generation plus SHA-256 of its actual bytes. SHA-256 is required even for
  `legacy` regular-file publications that lack an immutable generation token.
- `dim_candidate` and `dim_geography`: SHA-256 over all scoped rows and column
  types, with deterministic SQL sorting and duplicate-sensitive streaming.
- Candidate cube: row count, sum of nominal votes and two aggregate DuckDB
  hashes covering all columns (a fast change detector, **not** cryptographic
  attestation). The DuckDB package version is also part of the certificate.
- Scope: resolved *canary* database path, physical fact root and cycle. Do not
  copy these certificates to a different warehouse or treat them as portable.

A dimension-only change or re-publication of a source partition invalidates
that cycle immediately **when the freshness gate runs**. The system cannot
prevent consumers from querying the table directly without running this gate.
There is no automatic event-triggered database rollback or eager refresh.

## Explicit operator commands (run from data/tse-election-data)

```bash
MAIN="$HOME/github/experiments/data/tse-election-data"
PY="$MAIN/.venv/bin/python"
DBT="$MAIN/.venv/bin/dbt"
CANARY="$MAIN/data/warehouse/tse_event_registry_preview.duckdb"
FACTS="$MAIN/data/warehouse/fact_candidate_votes"
ATTEST="$MAIN/data/warehouse/candidate_cube_attestations"

# Coverage alone: existing PR #52 behavior; does not mean fresh.
"$PY" scripts/certify_candidate_cube.py \
  --db "$CANARY" --source-root "$FACTS" audit

# Fail closed: all five cycles must have current attestations.
"$PY" scripts/certify_candidate_cube.py \
  --db "$CANARY" --source-root "$FACTS" \
  --attestations-dir "$ATTEST" audit-freshness

# Only when a cycle has genuinely changed: refresh and certify that cycle.
# This mutates the named DuckDB, revokes its previous active certificate before
# dbt run, validates conservation and ranking, then publishes new authority.
"$PY" scripts/certify_candidate_cube.py \
  --db "$CANARY" --source-root "$FACTS" \
  --attestations-dir "$ATTEST" \
  --report "$(mktemp -d)/candidate-cube-2024.json" \
  certify --year 2024 --election-type municipal --dbt "$DBT" \
  --memory-limit 4GB --run-timeout 600 --test-timeout 600
```

The attestation folder should be persistent and access controlled. It is
separate from the create-only `--report` evidence and from the Git repository;
never place warehouse-specific certificates under version control.
A failed run or failed tests leave the old certificate under `revoked/`,
so an interrupted replacement cannot be reported as current.

`--attestations-dir` is **required for every `certify` invocation**. An
unattested mutating run is rejected before any source or warehouse changes.

**Migration of the existing canary:** The five cycles certified manually in
PR #52 do **not** have these new certificates. Therefore the first
`audit-freshness` must fail until each cycle has been recertified using the new
mode. Do not issue retrospective attestations merely from the former coverage
report. Recertify gradually only when needed; avoid blindly rerunning all
historic heavy conservation tests. A promotion policy must explicitly require
`audit-freshness` rather than the weaker coverage-only audit.

## Failure / recovery matrix

| Change | Expected effect | Recovery |
|---|---|---|
| Published Parquet generation changed | stale source SHA/generation | Rebuild and certify affected cycle |
| Legacy Parquet bytes changed in place | stale source SHA | Rebuild and certify affected cycle |
| Candidate/geography dimension changed | stale dimension signature | Rebuild and certify affected cycle |
| Candidate cube data modified | stale output signature | Rebuild and certify affected cycle |
| Missing / altered certificate | hard failure | Reissue through successful certification |
| dbt run/test interrupted | active certificate revoked | Recover DuckDB, rerun certification |
| New electoral cycle published | coverage/freshness failure | Load and certify new cycle |

## Limitations and future work

1. This is an **on-demand gate**, not a publisher event subscription. Wire the
   gate into downstream consumption/promotion workflows; otherwise consumers
   can still read stale persisted rows.
2. The cube fingerprint uses DuckDB's version-specific `hash`, not cryptographic
   proof. Database files/certificates must remain access controlled.
3. The input is hashed before and after the mutable certification and checked
   once more at audit. Concurrent uncoordinated writers may cause certification
   to fail; external single-writer coordination is still required.
4. An attestation captures **one particular source+dimension state**; changing a
   source means its past data may remain physically in the cube until rebuilt.
5. Snapshotting large dimension partitions may be costly. Benchmark this gate
   on the canary before making it a synchronous request-path prerequisite.
