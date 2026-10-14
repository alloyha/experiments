# TRD — Cycle, Election Event and Contest Keys

## Key model

- Cycle: `cycle_id = <election_year>:<election_type>`; join to the unique
  `dim_election_cycle` table, not to event-level `dim_election` by cycle alone.
- Election event: `election_id = <cycle_id>:<election_code>`.
- Contest: composite `(election_id, round_number, office_code, electoral_unit)`;
  geography depends on office scope. It is *not* unique on `election_id` alone.
- Electorate / geography: `(cycle_id, uf, municipality_code)`; never add
  `election_code` to this grained snapshot.
- Candidate candidature: `(election_id, candidate_id)` (existing snapshot keys
  include year/type/code for incremental replacement).
- Party election snapshot: `(election_id, party_number)`.

## Conformed dimension applicability

| Gold relation | Cycle | Event | Candidate | Party | Geography |
|---|---|---|---|---|---|
| fact_electorate_municipality | yes | no | no | no | municipality |
| fact_turnout / fact_tally_munzona | yes | yes | no | no | municipality/zone |
| fact_candidate_votes (Parquet-backed) | yes | yes | yes | via candidate | municipality/zone |
| fact_party_votes | yes | yes | no | yes | municipality/zone |

## Additivity and joins

- Never join a cycle-level electorate directly to several event/office/round
  rows then `SUM(electorate)`; roll up facts separately at a shared grained slice.
- Turnout counts repeat across offices; canonicalize office before aggregating
  the electorate/participation denominator.
- Candidate and party totals must be reconciled only at their common grain
  and with explicit source coverage (no `coalesce(unknown,0)` interpreted as PASS).
- Percentages and rates are recomputed as sum(numerator)/sum(denominator).

## Migration and validation

1. This PR changes SQL model definitions and dbt tests only. It makes no data
   mutations and does not automatically repopulate the evicted CSV cache.
2. `fact_candidate_votes` is a Parquet-backed view: derive `cycle_id` on read;
   do not republish physical partitions.
3. Persistent Gold tables/dimensions need a planned partition/full rebuild to
   materialize new columns and correct historical `dim_candidate.election_id`.
   That rebuild may require source rehydration; do not run it implicitly.
4. After rebuild, execute `dbt test --select tag:gold` and the new singular
   assertion, then run semantic and reconciliation tests on actual DuckDB data.
5. The existing database cannot be treated as migrated simply because SQL
   definitions changed. Never mutate the production warehouse without a backup.
