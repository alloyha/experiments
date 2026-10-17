# Candidate Cube — five-cycle transfer eligibility closeout

This is a **read-only, snapshot-specific eligibility manifest**, not a freshness
certificate, transfer, or production authorization. It binds the five existing
29-column, duplicate-sensitive audit reports to the current main and preview
DuckDB byte digests and the current Parquet publication generation/digests.

**Do not run any mutating dbt command against the main warehouse as part of this
closeout.** Main Graph `2024:municipal` remains a separate certification.

## Operator steps (WSL / POSIX)

From `data/tse-election-data`, collect the already-produced report paths:

```bash
find /tmp -maxdepth 2 -type f \( \
  -name 'eligibility-2024.json' -o \
  -name '2020-municipal.json' -o \
  -name '2018-general.json' -o \
  -name '2022-general.json' -o \
  -name '2026-general.json' \) -print
```

Choose **one** complete `PASS` report for each cycle. A failed/interrupted
process produces no accepted report. Do not invent/mutate historical JSON.

```bash
ROOT="$PWD/data/warehouse"
MANIFEST_DIR="$(mktemp -d /tmp/tse-cube-closeout.XXXXXX)"
# Replace the five /absolute/report paths below with actual existing files.
.venv/bin/python scripts/consolidate_candidate_cube_transfer.py \
  --main-db "$ROOT/tse_analytics.duckdb" \
  --preview-db "$ROOT/tse_event_registry_preview.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --main-sha256 77e930d91815b89bde1cd225c585f7868cd426405942f38a299500ec4955318e \
  --preview-sha256 ea53050d0c667cbf7b734e8bd1f342a8892cdb5fffb702650d5a29d8af5189d4 \
  --report /absolute/report/2018-general.json \
  --report /absolute/report/2020-municipal.json \
  --report /absolute/report/2022-general.json \
  --report /absolute/report/eligibility-2024.json \
  --report /absolute/report/2026-general.json \
  --output "$MANIFEST_DIR/candidate-cube-five-cycle-eligibility.json"
```

The tool **requires exactly five** reports: one per `2018:general`,
`2020:municipal`, `2022:general`, `2024:municipal`, `2026:general`.
Every report must confirm `PASS`, full 29-column multiset equality,
zero differences and the same complete cube row inventory. It checks the
source generation, SHA-256, and length of *each published* Parquet file and
SHA-256 of both DuckDB files **before and after** the verification under
cooperative shared read locks. No lock file is created and all locks must
already be present. The output is a create-only JSON file.

### Exit gates

- `PASS`, `cycle_count=5`, `total_rows=22797132`, `mismatch_rows=0` for every cycle.
- Main and preview SHA-256 must remain the approved values.
- Every Parquet generation and SHA-256 must match its source report.
- Any missing report, changed data, WAL, absent lock, symlink lock, duplicate
  cycle, inconsistent inventory or existing output path **fails closed**.

*Limitations*: POSIX `flock` is cooperative. Ad hoc writers can bypass it;
stronger production guarantees require fenced writes and immutable snapshot
isolation. Even PASS does not turn the historic cycle comparisons into new
per-cycle freshness certificates. `2026:general` is provisional. Do not
remove promotion backup/journal or graph evidence on account of this manifest.

## Subsequent implementation

Only after closeout should a separate, explicit operation copy the preview's
candidate-cube table into a **disposable clone** of the conformed Gold. Validate
its schema, exact row multiset, candidate-vote conservation and rank/share
contracts; protect the main warehouse from writes. Only then design a new
recoverable promotion plus independent graph and candidate-cube recertification.
