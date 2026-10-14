# PRD — Conformed Election Dimensions (PR 1)

## Problem

Gold uses `election_id=year:type:code` in `dim_election`, party and tally facts, while
`dim_candidate` previously used `year:type`. Aggregating votes with electorate
can silently duplicate turnout denominators because electorate is a cycle-level
municipality snapshot, not an event/candidate-level fact.

## Outcome

1. Publish deterministic, non-null `cycle_id=year:type` across all Gold facts
   and conformed dimensions, without changing existing fact grains. Expose a
   unique `dim_election_cycle` so event-level `dim_election` cannot fan out
   cycle-level joins.
2. Conform `dim_candidate.election_id=year:type:code` to `dim_election`.
3. Define contest identity as event + round + office + electoral unit, not
   as `election_id` alone.
4. Reject identity mismatches in dbt tests; preserve immutable Parquet sources.
5. Keep all analytical products usable after extracted CSV cache eviction,
   without automatically rebuilding Bronze/Silver.

## Non-goals

- Constructing OBTs/cubes (PRs 2–4).
- Introducing fictional election codes in electorate/geography.
- Rewriting production DuckDB or force-rehydrating the 29.87 GiB CSV cache.
- Converting candidate/party snapshots to synthetic SCD2.

## Acceptance

Model code, schema metadata and contract tests validate consistent IDs; the
local Python synthetic-contract tests pass. Full dbt data tests pass after
controlled Gold partition rebuild, with staging available. Physical Parquet
publication and original vote facts remain unchanged.
