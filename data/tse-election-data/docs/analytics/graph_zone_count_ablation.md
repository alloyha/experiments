# Candidate graph: `COUNT(DISTINCT zone)` ablation (read-only)

The 2018 general `signed` profile attributes 263.447 and 147.113 CPU seconds
respectively to the independent expected/actual eight-key rollups, each of which
computes `sum(votes)`, `count(*)`, and `count(distinct zone)`. Attribution to
`count(distinct zone)` specifically is **not** established by that profile.

`benchmark_graph_zone_count_ablation.py` profiles an isolated *physical-source*
rollup twice. `no_distinct` retains identical grouping, `sum(votes)`, `count(*)`
and `count(zone)`; `with_distinct` adds just `count(distinct zone)`. This measures
an incremental distinct-count contribution, not a proof that removing it from
the graph is permissible. No graph model, Gold, Parquet or certificates change.

Run from `data/tse-election-data` on Linux/WSL, **after other graph operations
finish** (the script rejects busy cooperative locks):

```bash
ROOT="$PWD/data/warehouse"
SHA=564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a
OUT="$(mktemp -d /tmp/tse-zone-ablation-parent.XXXXXX)"

# Small first falsifier: only AC within the physical 2018 Parquet publication.
.venv/bin/python research/benchmark_graph_zone_count_ablation.py \
  --db "$ROOT/tse_analytics.duckdb" \
  --source-root "$ROOT/fact_candidate_votes" \
  --expected-main-sha256 "$SHA" \
  --cycle 2018:general --uf AC --memory-limit 4GB --threads 2 \
  --max-seconds 900 --max-temp-gb 6 --output-dir "$OUT/AC"

jq '.shared_comparison' "$OUT/AC/summary.json"
```

If that passes, execute a new output directory with the same command but omit
`--uf` to measure the **whole** 2018 general publication. The script runs
`no_distinct` first and then `with_distinct`, profiling both independently.
Order and OS cache may bias individual elapsed times. Profile JSON, per-arm
reports, source identity and query SHA-256 are retained even if a later arm
fails; only both passing produces `summary.json`.

Safety / semantic prerequisites:

- Global `SUM(count(*)) == SUM(count(distinct zone))` proves that **each eight-key
  group** has no null zone and no duplicate non-null zone: every groupwise
  difference is nonnegative. Only then may `count(*)` substitute for
  `count(distinct zone)` for the checked exact publication and grouping.
- Global `SUM(count(zone)) == SUM(count(distinct zone))` proves no duplicate
  non-null zone *per group*, but does not prove the absence of null zones.
- If either flag is false, **do not** replace `count(distinct zone)` with that
  count in canonical SQL. Even if true, a future publication could violate the
  property, so a deployable rewrite needs a durable, fail-closed source-grain
  invariant and adversarial tests.
- Shared group count, source-row count, vote total and nonnull-zone count must
  agree exactly between the two arms; otherwise the ablation fails closed.
- The experiment uses the same actual published Parquet path and cooperative
  locks, but these do not fence arbitrary external direct writers. It does not
  confer freshness or production-read permissions and does not estimate
  per-operator spill from a root profile metric alone.
