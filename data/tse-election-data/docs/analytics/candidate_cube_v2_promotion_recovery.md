# Candidate Cube — chained V2 promotion, admission, recovery, and authority

This protocol is distinct from `promote_conformed_gold.py` (the V1 migration).
Never delete/overwrite/rename the existing `<main>.promotion.json` or its
backup. A V2 admission is grounded in the previously generated read-only
`candidate-cube-v2-preflight-v1.json` and the 5/5 offline SQL contract proof.

## Important properties

- **`prepare` does not modify the live DuckDB or its lifecycle epoch.** It
  creates a create-only state directory `<main>.promotion-v2`, writes the
  initial immutable admission event, copies the old Gold into a uniquely
  named SHA-verified V2 backup and copies the approved clone into an incoming
  file. Then it writes `PREPARED`. Re-running `prepare` is refused.
- **`status` is read-only**, requiring existing lock inodes. It reports whether
  the currently witnessed main SHA matches the append-only journal state.
- **`activate` is independently gated**, requiring explicit operator quiescence
  and confirmation; under exclusive locks it rechecks all evidence/physical
  publication SHA values. Before swapping bytes, it durably records
  `EPOCH_ROTATION_INTENT`, rotates the main epoch, then durably records
  `SWAP_INTENT`. An atomic same-directory rename activates exact approved clone
  bytes; live-path smoke must pass before `ACTIVE_UNCERTIFIED` is recorded.
- **`recover` is explicit**, requires the intact V2 baseline backup for
  started/prepared activations and restores the baseline after interrupted
  activation. For a cancelled `ADMITTED` or `PREPARED` state it leaves main and
  epoch untouched. Recovering after activation rotates the epoch before
  reinstating the baseline, preventing stale certificate resurrection.
- V2 historical state is **one create-only JSON per transition**, linked by
  SHA-256 of the previous record. Each record is fsynced, hard-linked into
  place and the directory fsynced. The initial record is prepared in a hidden
  directory then atomically renamed to the authoritative location. Incomplete
  copied artifacts and hidden directories are preserved for manual inspection;
  they are never silently deleted/reused.
- The original V1 journal+backup remain immutable. V2 proof and ledger cannot
  replace proper cube freshness or production graph certification. Promotion
  invalidates the lifecycle-bound proofs; rebuild both independent proofs in
  the **new main path** before guarded production reads.
- Advisory `flock` protects **cooperating** workers only. Direct DuckDB
  connections and external publishers bypassing locks are NOT fenced. Stop all
  direct writers/readers and confirm quiescence externally before any prepare,
  activate, or recover. V2 does not implement a service routing/traffic drain.

## First: tests (on real WSL filesystem)

```bash
cd ~/github/experiments/data/tse-election-data
.venv/bin/python -m unittest discover -s tests \
  -p 'test_promote_candidate_cube_v2.py' -v
```

These tests exercise synthetic filesystem fixtures and real `SIGKILL` process
termination at seven activation boundaries. They **do not** prove large-file
read/write performance or a complete DuckDB production integration. The live
DuckDB smoke requires `duckdb` installed in the actual environment.

## Stage 1: PREPARE, without touching the main DuckDB

After tests, adequate disk space, confirmed quiescence, and an already-valid
read-only V2 preflight report:

```bash
ROOT="$PWD/data/warehouse"
EVIDENCE="$ROOT/cube_transfer_evidence"
APPROVAL="$EVIDENCE/candidate-cube-v2-preflight-v1.json"

.venv/bin/python scripts/promote_candidate_cube_v2.py prepare \
  --main-db "$ROOT/tse_analytics.duckdb" \
  --preflight-report "$APPROVAL" \
  --preflight-sha256 "$(sha256sum "$APPROVAL" | awk '{print $1}')" \
  --ack-quiescent --confirm-prepare

.venv/bin/python scripts/promote_candidate_cube_v2.py status \
  --main-db "$ROOT/tse_analytics.duckdb"

sha256sum "$ROOT/tse_analytics.duckdb"
```

Expect `phase=PREPARED` and **old main SHA unchanged**
`77e930d91815b89bde1cd225c585f7868cd426405942f38a299500ec4955318e`.
No certified reader should assume that a `PREPARED` ledger is an activated
product. The status field `recovery_required=true` at this phase means the
activation state machine is not yet terminal; it is not evidence of a corrupt
Gold database.

## Stage 2: review activation before doing it

**Do not activate solely from this runbook.** First run the new file-backed
crash falsifiers in WSL, examine the `PREPARED` record, confirm the exact
sha256s of the old Gold, V2 backup, incoming snapshot, and the V1 anchor,
review the independent live-path smoke, and arrange a read-routing drain.
This patch delivers the action but does not declare production activation
approved. There is no automatic production switch in `prepare`.

`recover` is a rollback, not an automatic resumption of activation. It can
invalidate active freshness certificates by rotating the lifecycle epoch, so
invoke it only to restore a known baseline after a known incomplete/active
V2 attempt. If the live SHA does not match either approved file, or if the
ledger/backup integrity check fails, the action **refuses recovery** and
requires manual forensic inspection.

If a preparatory copy was interrupted, `status` can show `ADMITTED`; only an
explicit recovery can close it. If state was never published but a hidden
`.building` workdir exists, inspect it; do not remove it blindly.

## Following activation (future, separate certification run)

1. Verify the main SHA is the exact V2 clone SHA, epoch differs from V1, WAL
   is absent, and V2 ledger is `ACTIVE_UNCERTIFIED`.
2. Execute independent graph assertions and per-cycle candidate cube checks
   on the **live main path**, under the agreed cooperative read fencing.
3. Issue new graph/cube freshness certificates bound to the new lifecycle
   epoch. The offline report and V1 certificates are **not transferable**.
4. Maintain append-only V1 and V2 history with both backups; never repurpose
   earlier journals or silently rotate lock inodes.
