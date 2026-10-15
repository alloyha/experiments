# ADR 0007 — Municipality candidate and party performance cubes

**Status:** Proposed for PR #52.

## Decision

Provide two independent semantic marts on the published Gold facts. The candidate mart is an **incrementally materialized table (partition-replaced derived snapshots)** because DuckDB would otherwise re-execute its large Parquet aggregation for every dbt schema test. The party mart remains a **view**:

1. `cube_candidate_performance_municipality`: event × round × UF × municipality × office × transit × candidate. Sum candidate nominal valid votes over zone; compute observed candidate share and per-municipality nominal rank *after* aggregation.
2. `cube_party_performance_municipality`: event × round × UF × municipality × office × transit × party. Sum nominal and legend measures separately; compute observed party-valid share and local rank.

Only after aggregation, left-join descriptive `dim_candidate` / `dim_party` and `dim_geography`. Preserve unmatched rows as `candidate_matched=false`, `party_matched=false` and `geography_matched=false`; never coalesce missing metadata to a fake member. Use `EXCEPT ALL` tests to detect loss and duplicate fanout, supplemented by uniqueness, not-null, share, and ranking tests.

## Semantic caveats

- Party total-legend votes already include converted nominal-to-legend components; party valid votes equal nominal valid + **total** legend valid, not nominal + base legend + converted + total legend.
- Candidate ranking based on nominal valid votes is **not** official elected status.
- Candidate share denominators cover only observed candidate rows and must not be presented as share of all TSE valid votes without independent reconciliation and verified source coverage.
- Candidate and party cubes are not joined at candidate/party grain. Their comparison belongs in the independent reconciliation mart and must distinguish source coverage gaps.
- Neither cube joins cycle-level electorate or sums turnout across offices.

## Operations

Deploy these semantic marts only on the separately verified canary DuckDB, with exact `--select` (no leading `+`) and no Bronze/Silver rebuild. The first candidate-table materialization requires one expensive Parquet read; the fact-to-cube conservation test requires an independent second read. Respect the configured WSL memory budget. The production DuckDB is unchanged. Materializing all cuboids or expanding arbitrary dimensions is out of scope.

## Materialization, completeness and freshness (PR #52)

The candidate cube is **partition-replace incremental**, one `(election_year,
 election_type)` per targeted invocation, and *not* a monolithic CTAS or a
live view. The model filters the Parquet source even for **initial table
creation** and refreshes the selected partition with the project's existing
`partition_replace_pre_hook()` on subsequent incremental runs. Preserve all
other published cycles. No Bronze/Silver rebuild is required.

The first load is deliberately **partial**. Only the selected year/type is
present after that run. The candidate share/rank and conservation tests are
scoped to the same explicit cycle; they do **not** prove the entire history has
been loaded. A test PASS against an empty cycle must not authorize promotion.
Track exactly which years/types have been loaded and independently certify all
published Parquet cycles before production promotion. A complete history
requires loading *all* available cycles and running full-history coverage.

**Explicit `--vars` selection (no `--full-refresh`)**:

```sh
dbt run --select cube_candidate_performance_municipality \
  --vars '{incremental_years: [2026], incremental_election_types: [general]}'
dbt test --select assert_cube_candidate_vote_conservation assert_cube_candidate_share_and_rank \
  --vars '{incremental_years: [2026], incremental_election_types: [general]}'
```

After that partition is certified, progressively load and certify 2018, 2020,
2022 and 2024 using their appropriate types. In particular, **do not run a
full-refresh**: it scans and groups the unbounded historical Parquet. A changed
candidate vote Parquet generation, `dim_candidate`, or `dim_geography` requires
refreshing and re-certifying every affected cycle. A snapshot missing that
refresh is stale and must not be promoted. The party cube remains a view.

For heavy WSL scans, record wall time and peak memory/spill and tune only on
the disposable canary. The standard 2 GB profile memory limit remains a
conservative default; changing it requires measuring available system memory.
