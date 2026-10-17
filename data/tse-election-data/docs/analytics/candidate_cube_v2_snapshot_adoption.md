# V2 Candidate Cube: adopt certified bytes, never rematerialize

`adopt_candidate_cube_snapshot_v2.py` bridges the successful offline SQL
contract proof and the existing `candidate_cube_freshness` certificate format.
It **never calls dbt run, never rebuilds the cube and never writes DuckDB**.

## Phase safety

1. While the real Gold is still `PREPARED`, use `check` to validate the full
   immutable admission/Parquet/SQL chain. It returns `ready_for_adoption=false`.
2. The real Gold activation is a separate, explicit operator decision using
   `promote_candidate_cube_v2.py activate`. Do not infer permission from `check`.
3. Only after `ACTIVE_UNCERTIFIED` with exact approved clone bytes, rotated
   lifecycle epoch, intact backup and unchanged Parquet can `adopt` issue the
   five per-cycle candidate-cube freshness certificates.
4. The adoption action requires operator acknowledgement and **exclusive stable
   warehouse and partition locks**. It captures `input_signature()` and
   `output_signature()` from `candidate_cube_freshness` and cross-checks live
   source SHA, 29 columns, counts and nominal votes against the pinned manifest.
   The two SQL contracts are not re-executed here: the offline 5/5 exact SQL
   proof is revalidated by SHA, with the activated DuckDB bytes identical to
   the certified offline clone.
5. Certificates are issued under a private `.building` directory. A Linux
   atomic `renameat2(RENAME_NOREPLACE)` publishes the complete set *only*
   after every cycle passes and a second post-fingerprint provenance check
   succeeds. On failure, private orphan evidence is deliberately retained;
   it must **not** be interpreted as an active certificate directory.
6. Use a **new** certificate directory, e.g.
   `data/warehouse/cube_transfer_evidence/candidate_cube_attestations_v2`.
   Do not reuse or overwrite the historical graph/cube attestation directories.
7. After adoption, use `certify_candidate_cube.py audit-freshness` against the
   new directory. Separately recertify the graph for the new warehouse epoch
   and issue graph evidence in a new directory. The old graph proof cannot
   authorize the new epoch. Readers must be switched only after both gates.
8. Recovery to the V2 baseline rotates the lifecycle epoch. Already published
   V2 cube certificates then **fail closed**. Recertification is required.

## WSL verification before real activation

```bash
cd ~/github/experiments/data/tse-election-data
.venv/bin/python -m unittest discover -s tests -p test_adopt_candidate_cube_snapshot_v2.py -v
.venv/bin/python -m unittest discover -s tests -p test_adopt_candidate_cube_snapshot_v2_real_duckdb.py -v
ROOT="$PWD/data/warehouse"
.venv/bin/python scripts/adopt_candidate_cube_snapshot_v2.py check \
  --main-db "$ROOT/tse_analytics.duckdb"
```

Expected current state: `PREPARED`, `ready_for_adoption: false`. **Do not
run the `adopt` action now**. The graph and cube attestation paths must be
configured before production activation. No implication of uncoordinated
reader/writer isolation: external quiescence remains a requirement.
