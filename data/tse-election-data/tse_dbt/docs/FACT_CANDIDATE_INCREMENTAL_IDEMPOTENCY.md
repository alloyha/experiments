# fact_candidate_votes incremental idempotency

The 2018 fact was observed with stale geography rows left by the previous
incremental behavior:

- canonical rows: 8,680,108
- stale noncanonical rows: 382,657
- duplicate grains after canonicalization: 382,657
- max copies per canonical grain: 2

`fact_candidate_votes` is an authoritative snapshot by
`(election_year, election_type)`, so selected partitions must be replaced, not
appended.

The model now uses dbt `delete+insert` with
`unique_key = [election_year, election_type]`.

The expensive snapshot completeness assertion is retained under the
`integrity_heavy` gate. Normal builds keep a cheap canonical municipality guard.
