# ADR 0004 — Distinguish Election Cycle, Event and Contest Identity

**Status:** Accepted for implementation in PR 1

## Context

The existing `dim_election` identity includes `election_code`, consistent
with ADR 0003 (e.g. distinct general election codes 297 and 339 in 2018).
`dim_candidate` instead used the shorter year/type `election_id`. This
violation risks nonconformed BI joins. `fact_electorate_municipality` and
`dim_geography` intentionally have no election code.

## Decision

Use two deterministic public identities, with no fictional keys:

- `cycle_id = year:type` for election-cycle snapshots;
- `election_id = year:type:election_code` for each TSE election event;
- contest identity requires election_id + round + office + appropriate unit.

Add `cycle_id` to each Gold dimension/fact; add uniquely grained
`dim_election_cycle`; correct `dim_candidate.election_id`.
Do not alter the existing row grains or reinterpret vote totals as elected status.
Do not denormalize turnout and electorate into different-grain rows.

## Consequences

The migration is additive except for correcting `dim_candidate.election_id`.
Model SQL alone does not update persisted DuckDB relations; data rebuild is a
separate operational activity, especially after raw cache eviction. An explicit
singular dbt test guards identity consistency across Gold, and Python tests
exercise that test against synthetic DuckDB snapshots. OBTs and cubes should
only be implemented after conforming these keys and resolving denominator
policies for each consumer grain.
