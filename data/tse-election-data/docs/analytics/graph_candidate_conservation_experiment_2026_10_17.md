# Graph candidate vote conservation — benchmark evidence (2026-10-17)

**Scope:** read-only, cycle-scoped benchmark of the currently active Gold V2 publication.
This file records WSL observations supplied by the operator; the results were **not**
reproduced by a second independent host. It is not an execution authority, freshness
certificate, or approval to rewrite the canonical models.

## Reproducibility boundary

- Gold V2 main DuckDB SHA-256: `564ba3f7778c3483704bb435ab26eb3ac1209e4f5b5b9d62c39ac03d4bd64c8a`.
- Read-only DuckDB attached to in-memory scoped views, physical candidate publication
  selected by `(election_type, year)`; no `dbt run` and no Gold mutation.
- DuckDB `memory_limit=4GB`, `threads=2`, `max_temp_gb=6` for integrated experiments.
- `signed_countstar` substitutes `COUNT(*)` for `COUNT(DISTINCT zone)` in the
  **candidate** rollup of the in-memory graph view and in the separately computed
  expected rollup. The party rollup is unchanged.
- Independent, full-source `(cycle_id, election_id, round_number, uf,
  municipality_code, office_code, is_transit_vote, candidate_id, zone)` guard
  denies the experiment on NULL zone or duplicated non-NULL group/zone.
- `SINGLE_ARM_PASS` means the requested optimized arm returned `diff_rows=0` and
  its local source guard passed; it **does not** mean a paired performance trial.

## Observed integrated timing (seconds)

The canonical reference timings came from earlier independent graph certificate
runs with different execution conditions. The guarded optimized timings come
from separate read-only single-arm benchmarks, except 2020 (one paired trial).
**Do not interpret the ratios or totals below as randomized, cold-cache,
paired scientific speedup estimates.**

| Cycle | Earlier canonical reference | Signed + COUNT(*) query | Independent guard | Optimized including guard | Source guard: NULL / duplicated | Query diff rows |
|:--|--:|--:|--:|--:|:--:|--:|
| 2018 general | 1811.730 | 73.152 | 9.539 | 82.691 | 0 / 0 | 0 |
| 2020 municipal | 18.313 | 2.104 | 0.316 | 2.420 | 0 / 0 | 0 |
| 2022 general | 2455.235 | 69.942 | 12.135 | 82.077 | 0 / 0 | 0 |
| 2024 municipal | 4.442 | 6.739 | 2.849 | 9.588 | 0 / 0 | 0 |
| 2026 general | 619.076 | 58.684 | 7.516 | 66.200 | 0 / 0 | 0 |
| **Sum of independent observations** | **4908.796** | **210.621** | **32.355** | **242.976** | — | — |

Important counterexample to uniform performance improvement: **2024 municipal**
was slower in its optimized trial (even without the guard) than the historical
canonical run. For a controlled conclusion, re-run both versions in matched
conditions with alternating order and repeated trials; preserve the per-query
JSON profiler output.

The source cardinality probe for 2018 general reported 8,680,108 physical rows,
7,582,674 candidate rollup groups and maximum multiplicity 58. The integrated
`2018 general` signed baseline took 352.670 seconds with 1,420,787,712 bytes
peak temporary storage; guarded `signed_countstar` took 73.152 seconds with
494,698,496 bytes peak temporary storage. These are non-paired observations.

## Physical-zone ablation (independent isolated rollup)

| Cycle | With DISTINCT (s) | Without DISTINCT (s) | Reported ratio |
|:--|--:|--:|--:|
| 2018 general | 34.124 | 11.025 | 3.0951 |
| 2020 municipal | 3.315 | 1.557 | 2.1291 |
| 2022 general | 32.809 | 12.920 | 2.5394 |
| 2024 municipal | 1.007 | 0.400 | 2.5175 |
| 2026 general | 15.402 | 6.737 | 2.2862 |

All five complete ablation reports asserted zero NULL zones, zero duplicated
non-NULL group/zone keys, and equality of shared `SUM`/`COUNT(*)` aggregates.
The timing order was sequential, not randomized.

## Experimental controls and limitations

1. This evidence establishes `COUNT(*) = COUNT(DISTINCT zone)` only for the
   **five selected physical publications** under their observed byte identities.
   It is not a timeless assumption about TSE data or future rehydration.
2. Retain an independently evaluated, fail-closed guard bound to the exact
   source Parquet SHA-256, physical partition identity and version/epoch.
   Reject any publication lacking this proof, or retain the distinct algorithm.
3. Because both expected and actual are generated from the same physical fact
   source, the independent guard is essential: without it, replacing DISTINCT
   on both sides could hide a common-mode zone-count error.
4. A benchmark-only, in-memory rewrite is **not** an admissible rewrite of the
   lazily evaluated canonical `graph_edges.sql` without a production publication
   gate and tests proving its enforcement along every write/recovery path.
5. Preserve the historical Gold, candidate cube, and five existing graph
   attestations. SQL/model edits invalidate previous code hashes and require
   fresh certification; never relabel existing artifacts as if re-certified.
6. DuckDB profile timings and peak buffer memory are query-reported metrics,
   not process RSS or attribution of spill to one specific operator.
7. Preserve A/B order, cold/warm cache state, DuckDB version, system load,
   profile and source file hashes before making generalized speedup claims.

## Evidence locations on the original WSL host

Paths here are reproducibility pointers, **not guaranteed permanent artifacts**.
Copy summaries, plans and JSON profiles to a versioned evidence store before
cleaning `/tmp`.

- 2018 signed baseline: `/tmp/tse-graph-signed.9iAyfm/results/summary.json`.
- 2018 countstar: `/tmp/tse-countstar-general.XjQ15F/results/summary.json`.
- 2020 paired integrated comparison: `/tmp/tse-countstar-full.C77wM3/municipal/summary.json`.
- Remaining integrated cycles: `/tmp/tse-countstar-remaining.u0Gr1j/results/summary.json`.
- 2018 zone ablation: `/tmp/tse-zone-general.cN2KDI/general/summary.json`.
- Other four zone ablations: `/tmp/tse-zone-all-cycles.0lZLTK/{cycle}/summary.json`.

## Proposed staged integration

**PR A — experiment only.** Merge reproducible scripts, tests, benchmark
protocol and evidence manifest; do not edit the production graph model,
canonical assertion, published Gold, or certificate files.

**PR B — production enforcement + optimization.** First enforce the exact
physical uniqueness invariant on every candidate publication, including
replacement/recovery flows and stable SHA/epoch binding. Add adversarial tests
for NULL zones, duplicates, concurrent publication changes, stale proof reuse
and common-mode errors. Only then consider replacing `COUNT(DISTINCT zone)` in
the candidate graph rollup and the independent expected assertion, and changing
the multiset comparison to signed multiplicities. Keep the party rollup intact.
Re-run the five canonical graph certificates with fresh code hashes and strict
zero-difference checks before promoting.
