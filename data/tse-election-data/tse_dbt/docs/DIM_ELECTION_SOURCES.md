# `dim_election` — published Gold event registry

`dim_election` has one row per `(election_year, election_type, election_code)`.
`dim_election_cycle` has one row per `(election_year, election_type)`.
Neither `cycle_id` nor `election_id` alone identifies an office/round contest.

## Source policy

Use **persisted, authoritative Gold evidence**, not volatile raw-cache views:

- `dim_candidate` (persisted candidate election snapshots)
- `dim_party` (persisted party election snapshots)
- `fact_tally_munzona` (published authoritative tally)
- `fact_party_votes` (published party vote fact)

The original `bronze_candidates`, `bronze_party_votes_raw`, and
`silver_tally_munzona` are upstream of those Gold sources. The old implementation
rebuilt this dimension from the raw-oriented sources even when their current
coverage differed from previously published Gold snapshots. This could erase
event references in partial updates or after cache eviction.

Every row records a sorted `evidence_sources` string with the published Gold
relations that witnessed that election event (for example `dim_candidate`).
The values are evidence provenance, **not** independent TSE source certification.

The new model does **not** depend on `fact_candidate_votes` Parquet as a source:
reading all physical vote rows merely to discover election keys is unnecessary.
`assert_conformed_dimension_membership` still checks its event references and
must remain an enforced gate: a truly unrepresented event must fail, not be
silently synthesized. No election codes or scopes are invented.

## DAG and validation

The dimension depends on Gold snapshots but none of the four parents depends on
`dim_election` or `dim_election_cycle`; this is an intentional acyclic Gold-to-Gold
edge. dbt therefore runs the published source models before `dim_election` when
building the complete graph. `dim_election_cycle` runs afterwards.

An observed null/empty key, unknown scope, mutually inconsistent scopes, or
scope incompatible with the election type makes the model **fail explicitly**.
The model does not use `MAX(election_scope)` to hide disagreements.

After dbt rebuilding of `dim_election` and `dim_election_cycle`, run the following
singular tests (on a disposable DuckDB copy):

- `assert_dim_election_gold_source_coverage`
- `assert_conformed_election_identity`
- `assert_conformed_dimension_membership`

Note: rebuilding only `dim_election` against an already migrated warehouse
requires no extracted CSVs. A **full DAG rebuild** still requires the active
upstream raw inputs for other models. This change does not authorize rehydrating
or removing raw sources, altering Parquet, or promoting a candidate database.
