# Candidate Cube: V2 chained-promotion admission (read-only)

The previous `promote_conformed_gold.py` promotion produced an active v1
`<main>.promotion.json`. It intentionally refuses a second promotion. **Do not
remove, overwrite, rename or edit that journal, its backup or the lifecycle marker.**

`preflight_candidate_cube_promotion_v2.py` is a **read-only input gate** for a
future independent, versioned promotion protocol. It is not a replacement for
recoverable activation. A `PASS` never authorizes manually replacing the live
DuckDB, and it is not a production cube/graph freshness certificate.

## Dependencies

- Five-cycle eligibility manifest (5/5 exact multiset PASS).
- Stage report for the independently copied, persisted 29-column cube.
- Offline `assert_cube_candidate_vote_conservation` and
  `assert_cube_candidate_share_and_rank` PASS for **every** cycle.
- Immutable SHA-256 identities of all evidence/warehouse files.
- v1 journal exactly `ACTIVE_UNCERTIFIED`, active candidate SHA equal to the
  current main SHA, intact v1 backup and unchanged lifecycle epoch.
- Existing main/preview/partition lock inodes; all warehouses without WAL.
- Enough free space for the next promoter to copy the old and new DB files.
- No previous v2 promotion state (`<main>.promotion-v2`).

## Run on the WSL where all files exist

```bash
cd ~/github/experiments/data/tse-election-data
ROOT="$PWD/data/warehouse"
EVIDENCE="$ROOT/cube_transfer_evidence"
MANIFEST="$EVIDENCE/five-cycle-eligibility.json"
STAGE="$EVIDENCE/candidate-cube-stage-v1.json"
OFFLINE="$EVIDENCE/candidate-cube-offline-contracts-v1.json"
OUTPUT="$EVIDENCE/candidate-cube-v2-preflight-v1.json"

# Do not create or overwrite any existing report.
test ! -e "$OUTPUT" || { echo "Evidence already exists"; exit 2; }

.venv/bin/python -m unittest discover -s tests \
  -p 'test_preflight_candidate_cube_promotion_v2.py' -v

.venv/bin/python scripts/preflight_candidate_cube_promotion_v2.py \
  --main-db "$ROOT/tse_analytics.duckdb" \
  --preview-db "$ROOT/tse_event_registry_preview.duckdb" \
  --clone-db "$ROOT/tse_analytics_cube_v1.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --manifest "$MANIFEST" \
  --manifest-sha256 063ddf0510837804815799d010a49d61c7450ca2ef19e3aaa5533f655feddda1 \
  --stage-report "$STAGE" \
  --stage-report-sha256 cf7c31b71abdd1483015073b28afc046f96ae2413320c502d48793744635077d \
  --offline-contract-report "$OFFLINE" \
  --offline-contract-report-sha256 "$(sha256sum "$OFFLINE" | cut -d' ' -f1)" \
  --clone-sha256 564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a \
  --report "$OUTPUT"

jq '{result, scope, legacy_anchor, cycle_count, total_rows,
     total_nominal_votes, required_free_bytes_at_audit,
     observed_free_bytes_at_audit}' "$OUTPUT"
sha256sum "$OUTPUT"
```

This command **never** rotates an epoch, modifies or checkpoints a warehouse,
creates an output DuckDB, or archives the v1 journal. The report is created
only if every assertion succeeds and is exclusively created on disk.

A future versioned promoter MUST revalidate the full report, every SHA and
publication pointer while holding exclusive warehouse + physical partition
locks. It must durably write a new v2 journal **before** rotating lifecycle
or swapping bytes; maintain an append-only history linking the immutable v1
journal and backup to each v2 generation; and provide `status`, explicit
recovery/rollback, and real SIGKILL tests at every fsync/rename boundary.

After activation, previous graph/cube certificates must not be reused. New
warehouse-path/epoch-bound certifications must be independently issued. No
uncoordinated DuckDB writer/reader is fenced by POSIX `flock`; maintenance
quiescence and reader coordination are separate deployment requirements.
