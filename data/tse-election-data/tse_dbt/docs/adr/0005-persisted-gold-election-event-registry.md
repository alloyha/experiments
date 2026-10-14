# ADR 0005 — Persist `dim_election` coverage from published Gold snapshots

**Status:** Proposed for PR #50 (event coverage repair)

## Context

The previous `dim_election` materialization was `table`, sourced directly
from `bronze_candidates`, `bronze_party_votes_raw`, and `silver_tally_munzona`.
`dim_candidate` and other Gold snapshots use partition replacement and retain
historical partitions across incremental runs. This mismatch allows event
coverage to diverge when source visibility changes or raw CSVs are evicted.

The offline migration of the real warehouse found 26 missing election events:
all were represented by persisted `dim_candidate`, with unambiguous scopes.
A separate migration reconciled them in a copy, not in the production database.
The precise history of source-refresh operations that caused the divergence has
not been proven.

## Decision

Build the event-grain `dim_election` table from four persisted Gold event
producers: `dim_candidate`, `dim_party`, `fact_tally_munzona`, and
`fact_party_votes`. No one of these may depend on `dim_election` or
`dim_election_cycle`; the DAG checker and regression tests enforce acyclicity.
Do not use the candidate-vote Parquet as a discovery source. Existing strict
event membership tests continue to cover it.

Reject inconsistent scope and malformed identity at materialization time,
without guessing a scope or filling event keys from the calendar. Reconcile
facts and dimension membership using dbt singular tests.

## Consequences and limitations

- A targeted rebuild of `dim_election` requires only already published Gold,
  making recovery possible with raw extracted cache evicted.
- Incremental runs build updated source snapshots before the event dimension.
  Full-refresh rebuilds the upstream Gold within the selected scope and then
  derives the event dimension. Nothing is appended merely to keep stale rows.
- A full dbt DAG rebuild still requires the upstream raw resources of those
  source models; this change does not remove their dependencies.
- Events with no representation in any of the four published sources must be
  investigated and will fail the existing membership test if referenced by
  another fact. This is preferable to a fabricated event member.
- Promotion of the offline migrated DuckDB is a separate, explicitly approved
  operation with backup and rollback.
