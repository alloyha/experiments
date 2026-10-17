# GG: candidate-vote conservation — read-only exact benchmark

Goal: falsify the idea that the **30.2 minute** 2018 general-election
`candidate_vote_conservation` check is dominated by cardinality-sensitive
aggregations and/or two-way `EXCEPT ALL` and temp spill. This experiment does
not change the graph models, assertions, Gold DB or lifecycle certificates.

## What is held equal

- Original `expected` and `actual` CTEs are reused from the **unmodified**
  `tse_dbt/tests/assert_graph_candidate_vote_conservation.sql` using the
  same `certify_gold_graph_cycle.setup_scoped_views` and physical published
  Parquet file. The script refuses to run after an unreviewed SQL-tail change.
- Original arm: two-way `EXCEPT ALL`, `COUNT(*)` over the full difference.
  This differs intentionally from certification's early-violation `LIMIT 1`:
  for correct cycles (zero violations), a complete scan is required in both.
- Signed arm: `UNION ALL` rows tagged `+1/-1`, group by **all ten original
  contract fields** (including NULLs), sum signed multiplicities and
  `SUM(ABS(delta))`. This yields the **same total mismatch multiplicity** as
  two-way `EXCEPT ALL`, not an approximate hash or set comparison.
- Default resource controls match the production diagnostic: 4GB DuckDB
  buffer limit, two threads. Temperature/spill capped at 6GB by default,
  configurable without changing production SQL. `DuckDBPyConnection.interrupt()` is invoked by a Python watchdog
  after `--max-seconds` for each SQL query. This is **cooperative**, not a hard
  OS-level kill deadline; use an external process supervisor for hard limits. Variant ordering alternates AB / BA over repeats.
- Output: create-only run directory with `.plan.txt`, `.profile.json`,
  `.result.json` per trial; final `summary.json`. A killed trial preserves
  earlier results. JSON profile includes `system_peak_temp_dir_size`,
  `system_peak_buffer_memory`, `cpu_time`, `latency`, cardinality and I/O
  counters (version-dependent). **Do not equate RSS and DuckDB buffer usage.**
- Each run checks that the published Parquet's path, hash, size and contract
  SQL hash did not drift. The entire experiment holds existing shared POSIX
  locks (warehouse before partitions), verifies old/new warehouse SHA and epoch,
  and refuses any WAL. It never creates new warehouse/partition lock inodes.

## Critical sequencing

**Do not run until the current `graph_freshness.py` process completes**. The
benchmark defaults to zero lock-wait time and fails rather than competing
with the real certifier. Run after the five cycles have been certified, or
in a maintenance window with all graph writers/readers quiescent. The shared
locks are cooperative; stop direct unmanaged DuckDB/Parquet writers as well.

Before starting, check free space in `/tmp`, CPU/RAM in WSL, and the absence
of direct unmanaged writers. Run the unit test first:

```bash
cd ~/github/experiments/data/tse-election-data
.venv/bin/python -m unittest discover -s tests \
  -p 'test_benchmark_graph_candidate_conservation.py' -v
```

Use an **explicit path outside data/warehouse**. Keep the approved activated
Gold sha256 pinned. Start with small municipal data to smoke the profiler:

```bash
ROOT="$PWD/data/warehouse"
GOLD_SHA=564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a
.venv/bin/python scripts/benchmark_graph_candidate_conservation.py \
  --db "$ROOT/tse_analytics.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --expected-main-sha256 "$GOLD_SHA" \
  --cycles 2020:municipal --probe-cardinality \
  --output-dir "$(mktemp -d /tmp/tse-graph-benchmark-parent.XXXXXX)/trial"
```

After confirming the profiler and both variants return zero differences,
repeat with a **new output** (avoid heavy overlap) for general elections:

```bash
.venv/bin/python scripts/benchmark_graph_candidate_conservation.py \
  --db "$ROOT/tse_analytics.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --expected-main-sha256 "$GOLD_SHA" \
  --cycles 2018:general 2022:general 2026:general --probe-cardinality \
  --repeats 2 --threads 2 --memory-limit 4GB \
  --max-seconds 2400 --max-temp-gb 6 \
  --output-dir "$(mktemp -d /tmp/tse-graph-benchmark-parent.XXXXXX)/general"
```

Caution: 2018 took 1,812 seconds for ONE arm; 3 cycles x 2 variants x
2 repeats may be **hours**, and may hit the per-arm timeout. Start with
`--repeats 1` or just `2018:general` if time/disk is tight. If an arm times
out, inspect its partial profile and never call that a performance win.

## Interpretation and promotion gate

Compare within the **same cycle, repeat, thread count, memory cap and SQL hash**:

- `diff_rows` must be exactly 0 for both arms. For synthetic mutation cases,
  their exact mismatch multiplicities must agree.
- `speedup = except_all_seconds / signed_seconds`; report both arms and the
  absolute difference, not just the ratio. With ≥2 repeats, report the
  median and ordering sensitivity; OS file cache is still uncontrolled.
- Large `system_peak_temp_dir_size` and disk-write counters strongly support
  spill as a contributor; zero temp use suggests cardinality, redundant
  rollups or optimizer strategy rather than a memory-spill explanation.
- If `--probe-cardinality` is enabled, compare exact `source_rows`,
  `groups`, `rows_per_group` and `largest_group_rows` from the physical Parquet.
  This probe itself performs an extra potentially expensive `COUNT(DISTINCT zone)`
  group aggregation; keep its timing separate from the two compared arms.
- Inspect plan/operator cardinalities to decide whether source `GROUP BY` or
  `EXCEPT ALL` dominates. This benchmark alone does not isolate the cost of
  materializing an intermediate candidate rollup; that is a separate arm.
- **Do not rewrite canonical SQL or publish graph certificates based solely on
  a faster benchmark.** First add exact equivalence tests and an independent
  certification on disposable physical fixtures; changing a graph SQL hash
  invalidates any existing certificate under `graph_freshness.contract_fingerprint`.

Profiles are for research only, never freshness certificates. No GitHub CI
is expected to execute the full multi-hour production-sized benchmark.

## DuckDB compatibility: query deadline

DuckDB versions before the setting appears in their configuration registry
reject `SET max_execution_time` outright. The benchmark therefore does not
use this setting. Instead, a daemon watchdog waits `--max-seconds` and calls
the documented Python `DuckDBPyConnection.interrupt()` method when necessary.
An interrupted trial fails closed; it is not counted as zero mismatches.
The limit is best-effort and is **not** a hard process wall-clock timeout.
Use external process-level limits if a hung native query must be forcibly
terminated. Never kill the unrelated Gold certifier.

## DuckDB compatibility: profiling output extension

DuckDB validates the profiling-output file extension against its currently
selected profiler format. Setting `profiling_output = '...profile.json'` before
`enable_profiling = 'json'` fails because the default format is `query_tree`.
The benchmark now calls `configure_json_profiler()` in both the cardinality
probe and each comparison trial, enabling JSON format **first** and then
setting the unique create-only JSON output path. A regression test exercises
this ordering; an additional optional integration test writes and parses a
real DuckDB JSON profile. Plan-only mode does not enable the profiler.

A failed previous benchmark output directory is intentionally left untouched;
run again using a new, empty output directory. The summary exists only after
all trials succeed.

## DuckDB profiling and Python result finalization

The Python `fetchone()` API does **not** finalize a query after its first row.
DuckDB reports that the result/transaction is completed only when a further
`fetchone()` returns `None` (or when `fetchall()` exhausts it). Consequently,
profiling can remain unpublished even for `SELECT count(*)` / scalar aggregate
queries when the benchmark fetches just the value. `fetchone_with_timeout()`
now requires exactly one row, consumes EOF under the same query watchdog and
closes the in-memory connection **before** reading the `.profile.json`.
This does not run the expensive query twice, nor does it modify the Gold.

If the profile still does not exist after EOF and close, the benchmark fails
closed instead of assuming there was no temporary disk use. Run the DuckDB
profiler integration test in WSL before any large election cycles.

## Fast single-arm diagnostic after an expensive baseline

When a certified canonical `except_all` baseline has already taken 30–41
minutes, isolate `signed` without re-executing the expensive comparison:

```bash
ROOT="$PWD/data/warehouse"
OUT="$(mktemp -d /tmp/tse-graph-signed.XXXXXX)"
.venv/bin/python scripts/benchmark_graph_candidate_conservation.py \
  --db "$ROOT/tse_analytics.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --expected-main-sha256 564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a \
  --cycles 2018:general \
  --variants signed --repeats 1 \
  --memory-limit 4GB --threads 2 --max-seconds 3600 \
  --output-dir "$OUT/results"
```

This explicitly reports `SINGLE_ARM_PASS` and
`paired_comparison_complete=false`. The single-arm result must not be
presented as a fresh A/B speedup. A historical *certification* time may be
included as separate context, but it is not an equivalent paired benchmark.
Use a NEW directory. Never overwrite partial evidence from interrupted runs.
Cancelling a *foreground* long-running comparison with Ctrl+C is an operator
choice; it does not modify the live DuckDB, Parquets, or certificates, though
interrupted trials have no valid performance result. Do not launch a second
heavy benchmark while the original process still holds shared locks.

## Guarded COUNT(*) candidate-zone experiment (October 2026)

The independent five-cycle `benchmark_graph_zone_count_ablation.py` measured
zero NULL zones and zero duplicate `(the eight candidate-contest rollup keys,
zone)` rows in ALL five pinned Parquet publications. `COUNT(*)` therefore
matches `COUNT(DISTINCT zone)` **for those pinned bytes**, not generally.
Observed *isolated source-rollup* speedups removing DISTINCT were:
2018 general 3.0951x, 2020 municipal 2.1291x, 2022 general 2.5394x,
2024 municipal 2.5175x, 2026 general 2.2862x (one sequential run each).
This does not yet establish an end-to-end graph-conservation speedup.

The optional `--variants signed signed_countstar` adds a **guarded**, read-only
end-to-end comparison. Before running any `signed_countstar` SQL, a separate
DuckDB in-memory connection independently groups the **physical selected
Parquet** by all eight candidate-contest keys **and** `zone`, then rejects
*any* NULL zone or duplicate non-NULL `(group, zone)` occurrence. The guard
writes its own JSON/SQL-SHA/profile with the publication generation/digest.
A failed guard cannot enter the `signed_countstar` trial. Every trial must
refer to exactly the same publication as the guard; the benchmark also checks
warehouse SHA, lifecycle epoch and WAL after the sequence under existing
shared locks. Non-cooperating external writers are outside these locks.

Only in an **in-memory** DuckDB connection, `signed_countstar` changes the
expected rollup's single `COUNT(DISTINCT zone) AS zones` and the graph model's
candidate rollup's single `COUNT(DISTINCT zone) AS zone_count` to `COUNT(*)`.
It does **not** rewrite the party rollup, any Gold relation, checked-in model,
assertion, or existing freshness certificate. The candidate graph model and
expected source are still evaluated separately; all ten equality columns,
including zone_count, remain under exact signed multiset comparison. A
structurally different checked-in SQL is refused rather than rewritten by
guessing. A **single** guarded arm reports `SINGLE_ARM_PASS`, not a speedup;
a paired `signed` / `signed_countstar` run reports its own paired measurements.

Run first on a cheaper cycle (all measurements are per query and output paths
are create-only):

```bash
cd ~/github/experiments/data/tse-election-data
ROOT="$PWD/data/warehouse"
GOLD_SHA=564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a
OUT="$(mktemp -d /tmp/tse-graph-countstar.XXXXXX)"
.venv/bin/python scripts/benchmark_graph_candidate_conservation.py \
  --db "$ROOT/tse_analytics.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --expected-main-sha256 "$GOLD_SHA" \
  --cycles 2020:municipal --variants signed signed_countstar \
  --repeats 2 --memory-limit 4GB --threads 2 \
  --max-seconds 900 --max-temp-gb 6 --output-dir "$OUT/results"
jq '.independent_countstar_guards, .performance_comparison' "$OUT/results/summary.json"
```

Only after the complete synthetic/integration test suite and this cheap
falsifier have passed, repeat on `2018:general` with a new output directory
and `--max-seconds 3600`; keep original canonical graph proofs untouched.
If the independent uniqueness guard fails for ANY future partition, revert
to the original `COUNT(DISTINCT zone)` semantics: do not silently infer or
manufacture group-zone uniqueness from a successful comparison of two
implementations that share the same simplification.
