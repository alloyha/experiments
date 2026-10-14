# PRD / TRD — Electorate OBT and Participation Cube (PR 51)

## Decision

Provide two **semantic views** over conformed, persisted Gold facts; do not
materialize redundant copies until performance measurement warrants them.
These views can be built against the separately certified 198-event canary
DuckDB, without scanning evicted raw CSVs or publishing candidate-vote Parquet.

## Data products

1. `obt_electorate_municipality`: one record for each `(cycle_id, uf,
   municipality_code)`; contains electorate, conformed municipality label,
   geography-match flag, previous **observed** cycle, absolute electorate
   change, and growth ratio. Electorate is a snapshot, not a voter migration
   measure; comparisons across general/municipal cycles reflect available
   registry snapshots and coverage.
2. `cube_participation_municipality`: sum the zonal `fact_turnout` only within
   `(election_id, round_number, uf, municipality_code, office_code,
   is_transit_vote)`. Keep `office_scope` and election attributes. Rates are
   weighted via `sum(turnout)/sum(eligible_voters)`; zero denominator yields
   `NULL`. Geographical enrichment uses an exact cycle/UF/municipality join.

## Constraints

- No cross-grain electorate/turnout joins; never multiply cycle-level electorate
  by election event, round, office, candidate or party.
- Do not sum turnout over different offices, rounds, elections or transit flags.
  A statewide turnout query must **select one compatible event/round/office**.
- `dim_geography` must be unique by cycle/UF/municipality. Singular conservation
  tests and dbt uniqueness tests make fanout a hard failure.
- `is_transit_vote` remains a distinct slice. Missing geo labels remain visible
  through `geography_matched=false`; rows are never dropped.
- Keep both views independently usable in DuckDB; no raw cache access.

## Gate

1. Python synthetic tests include office repetition, multiple rounds, multiple
   election codes, transit votes, zero denominators, missing geography, previous
   cycle history, and a seeded duplicate geography.
2. `dbt run --select obt_electorate_municipality cube_participation_municipality`
   on an isolated canary copy (no leading `+`).
3. `dbt test --select obt_electorate_municipality cube_participation_municipality
   assert_obt_electorate_conservation assert_cube_participation_conservation
   assert_cube_participation_rates` on the same preview.
4. Compare representative real aggregates for 2024 GO and previous known
   results; no unexplained drift in electorate, turnout or abstentions.
5. Merge only after CI and local canary gates pass; no promotion of the
   original DuckDB in this PR.
