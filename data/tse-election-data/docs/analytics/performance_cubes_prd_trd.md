# PRD / TRD — Candidate and party performance cubes

## User outcome

Explore municipal candidate and party voting by election event, round and office, with geography and candidate/party labels, without changing the underlying total votes. The data support descriptive analysis and rankings, **not** automatic legal determination of winners, seats or elected candidates.

## Contracts and keys

| Model | Exact row grain | Additive measures |
|---|---|---|
| `cube_candidate_performance_municipality` | `election_id`, `round_number`, `uf`, `municipality_code`, `office_code`, `is_transit_vote`, `candidate_id` | `nominal_valid_votes`, `zone_rows` within defined slice |
| `cube_party_performance_municipality` | same first six dimensions plus `party_number` | `nominal_valid_votes`, `legend_valid_votes`, `total_legend_valid_votes`, `party_valid_votes` |

Both are municipality-level rollups from zone-grained facts. Never remove event, office, round, or transit dimensions when adding denominators. All rates are computed as numerator / `sum(numerator)` within a compatible event/round/municipality/office/transit slice, and undefined when the denominator is nonpositive.

## Requirements

- Read already published Gold, not evicted raw CSVs. The candidate cube is a **partition-replace incremental derived snapshot** refreshed by one explicitly selected election cycle at a time, after publishing changed fact/dimension data; the party cube remains a view. A partly loaded cube is not a complete-history dataset. Never assume candidate snapshots update automatically.
- Aggregate before descriptive joins; left-join and mark unmatched dimensions.
- Reject fanout using duplicate-sensitive `EXCEPT ALL` tests, including negative fixtures with duplicate dimension rows and identical duplicate output rows.
- Preserve exact candidate nominal valid and party nominal / legend / total legend / party-valid vote counts.
- Rank deterministically by votes descending and stable ID ascending.
- Do not equate candidate and party vote totals without coverage-sensitive reconciliation.
- No production database promotion or Parquet republishing.

## Acceptance

Synthetic Python/DuckDB tests, Medallion DAG contract, dbt compile, targeted incremental candidate and party view builds and cycle-scoped candidate tests pass on the canary. For the full DuckDB, run quantitative comparisons against underlying facts. CI success is necessary but not a substitute for the local canary data tests. Document test memory/time and any gaps.
