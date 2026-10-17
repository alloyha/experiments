# Candidate performance cube — phase 1: read-only transfer eligibility

**No copying, dbt runs, certification, warehouse mutation or new production permission occurs in this phase.**

The main warehouse `tse_analytics.duckdb` has been promoted to conformed Gold,
with a separately certified `2024:municipal` graph. The historic candidate
cube has 22,797,132 materialized rows across five cycles in
`tse_event_registry_preview.duckdb`, not in main. Because preview Gold dimensions
may differ from the current conformed main, copying the cube without rebuilding
its enrichment against current dimensions would be unsound.

The read-only gate reconstructs the exact checked-in semantic model for **one
specified cycle** from the *physically pinned Parquet partition* and the
*currently deployed* `main.dim_candidate` and `main.dim_geography`. It compares
all 29 projected fields against the preview cube with exact duplicate-sensitive
signed multiplicity grouping (or the original two-way `EXCEPT ALL` fallback),
including identity, vote conservation, rank/share and
all descriptive fields. It also checks full-cycle coverage, hashes input
Parquet before and after, hashes both DuckDB files before and after, checks
WAL absence, and holds cooperative shared locks over both warehouses and the selected
physical fact partition only. The complete published-cycle inventory is checked
before and after; older partitions are not required to have ".publish.lock"
sidecars. This relies on cooperating publishers acquiring a warehouse writer
lock before modifying the physical inventory. Only temporary in-memory SQL objects are created.

```bash
cd ~/github/experiments/data/tse-election-data
.venv/bin/python -m unittest discover -s tests \
  -p 'test_audit_candidate_cube_transfer.py' -v

.venv/bin/python scripts/audit_candidate_cube_transfer.py \
  --main-db "$PWD/data/warehouse/tse_analytics.duckdb" \
  --preview-db "$PWD/data/warehouse/tse_event_registry_preview.duckdb" \
  --source-root "$PWD/data/warehouse/fact_candidate_votes" \
  --cycle '2024:municipal' \
  --main-sha256 77e930d91815b89bde1cd225c585f7868cd426405942f38a299500ec4955318e \
  --memory-limit 4GB
```

**PASS** is a narrow finding that the preview's 2024 partition represents the
current main input model and is eligible for designing a separate, recoverable
promotion on a new isolated Gold copy. It does **not** issue a cube freshness
certificate: the existing `candidate_cube_freshness` contract requires a
cycle-scoped `dbt run` and two successful dbt tests before issuing authority.
A non-zero result must remain a blocker; do not patch away schema differences,
copy preview certificates across DB paths or overwrite the main warehouse.

If the 2024 gate passes, run it on the remaining cycles **individually**,
raising `--max-rows` explicitly for larger cycles. If a cycle fails, compare
preview and main dimension provenance before deciding to rebuild it, using
persisted Parquet and no raw ingestion. Once all chosen cycles pass, implement
an isolated copy/attach plus transactional insert, full conservation test and
atomic main warehouse promotion with fresh main-path graph + cube attestations.

Current certificate and lifecycle sidecars of the main warehouse must remain
untouched during this audit. Cooperative locks do not protect against direct
uncoordinated writers, including direct changes to other partitions; do not run during concurrent warehouse maintenance.

## Performance: exact single-group multiset comparison (2022/2026)

The initial two-way `EXCEPT ALL` can be CPU-intensive for 8 million rows and
29 wide columns. The auditor now defaults to `--compare-mode signed`: it
`UNION ALL`s expected rows tagged +1 and preview rows tagged -1, groups by
**all 29 actual column values**, and sums nonzero absolute multiplicities.
This is the same exact duplicate-sensitive multiset test as two-way
`EXCEPT ALL`, NOT probabilistic hashing or sampling. The original implementation
remains available with `--compare-mode except-all` for comparison.

The CLI emits timestamped progress lines on stderr before and after expected
cube materialization, comparison and checksum verification. JSON output is
written only after the whole audit succeeds. `--memory-limit 4GB` is advised
for WSL with 11 GiB of RAM; an 8GB memory cap is not a speed guarantee.
This rewrite is an optimization hypothesis, not a claimed performance win
until validated by a real DuckDB run with the same input hashes.

If cancelling a running audit, use Ctrl+C in its foreground terminal. Do not
Ctrl+Z (it leaves cooperative file locks held), do not delete .build.lock or
.publish.lock, and do not run concurrent comparison processes. This audit does
not change either warehouse or the published Parquet.
