# Candidate performance cube — partition certification runbook

The candidate cube is an **incremental table with partition replacement**.
Its data source is the published immutable candidate-vote Parquet; the party
cube remains an independent view. Do **not** run full-refresh, rebuild evicted
raw CSV files, or implicitly target the production DuckDB.

## Full-history coverage audit (no dbt run/test)

From `data/tse-election-data` in the feature worktree:

```sh
MAIN="$HOME/github/experiments/data/tse-election-data"
PREVIEW="$MAIN/data/warehouse/tse_event_registry_preview.duckdb"
FACTS="$MAIN/data/warehouse/fact_candidate_votes"
PY="$MAIN/.venv/bin/python"

"$PY" scripts/certify_candidate_cube.py \
  --db "$PREVIEW" --source-root "$FACTS" \
  --report "$(mktemp -d)/candidate-cube-history.json" audit
```

The audit discovers published `data.parquet` partitions (not historical
unpublished generation files), confirms `BASE TABLE`, and requires **exactly**
that set of nonempty cycles in the cube. It reports row/vote counts; it does
not rescan every Parquet, prove vote conservation, establish dimension freshness
or provide an external cryptographic attestation.

## Certify a **changed** individual cycle

The following **mutates the explicitly named DuckDB**, unlike `audit`.
It captures fingerprints of all other cycles, performs one incremental dbt
run, checks they are unchanged, ensures the selected cycle exists and has rows,
then executes just the two cycle-scoped SQL certification tests.

```sh
DBT="$MAIN/.venv/bin/dbt"
"$PY" scripts/certify_candidate_cube.py \
  --db "$PREVIEW" --source-root "$FACTS" \
  --report "$(mktemp -d)/candidate-cube-2024.json" \
  certify --year 2024 --election-type municipal --dbt "$DBT" \
  --memory-limit 4GB --run-timeout 600 --test-timeout 600
```

On dbt failure, timeout, missing/empty partition, or altered unselected cycle,
the workflow stops **without claiming certification**. It does not perform an
automatic rollback of a failed run. Evidence files are create-only. Do not
recertify already approved 2018–2026 just to run the coverage audit.

## Local certification evidence (14–15 Oct 2026)

| Cycle | Incremental run | Rank/share | Multiset conservation |
|---|---:|---:|---:|
| 2026:general | 219.86 s | PASS, 20.34 s | PASS, 239.37 s |
| 2024:municipal | 8.10 s | PASS, 0.80 s | PASS, 2.42 s |
| 2020:municipal | 7.18 s | PASS, 2.65 s | PASS, 8.58 s |
| 2018:general | 154.65 s | PASS, 13.81 s | PASS, 309.50 s |
| 2022:general | 230.10 s | PASS, 12.67 s | PASS, 411.36 s |

All listed targeted dbt commands returned PASS with no errors. Between loads,
non-selected cycle fingerprints were unchanged. These observations came from
manual user-run commands; the new certifier was **not** used to generate those
past results. The 2026 cube contains 5,965,381 rows and 653,808,509 nominal
votes; the 2024 cube contains 428,827 rows and 241,421,242 votes.

### Complete-history coverage and replacement idempotency

The `audit` command completed successfully on the canary database for all five
physical cycles, with **22,797,132 materialized rows** and the published
Parquet partition set matching the cube exactly. Evidence was written to
`/tmp/tse-pr52-history-audit.json` in the operator's WSL environment.

On 15 Oct 2026, the certifier successfully reloaded the already-materialized
`2024:municipal` cycle, checked preservation of the four other cycles and
passed both targeted SQL tests. An independent before/after fingerprint of
**all cube columns** for `2024:municipal` was identical (428,827 rows,
241,421,242 nominal votes). The test process and snapshots were recorded at
`/tmp/tse-pr52-idempotence.2B49EI/` in the operator's WSL environment.
These are local validation observations, not CI evidence and not cryptographic
proof of row-by-row equality.

## Promotion gates not covered here

- Prove freshness against **physical Parquet generation**, `dim_candidate` and
  `dim_geography`; dim-only edits can leave a previously certified cube stale.
- Record canary machine, memory, DuckDB/dbt versions and run artifacts in PR.
- Verify the repo CI, selective dbt schema tests and PR diff; do not mark a
  historical manual PASS as a CI result.

The public PR number is assigned by GitHub after the branch is pushed and a PR
is created; this runbook does not assume a remote PR #52 already exists.
