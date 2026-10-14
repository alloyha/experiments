# PRD / TRD — Published election-event coverage

## Goal

Make the `dim_election` catalog cover **all observed, valid election events**
from the current published Gold snapshots, independently of extracted raw CSV
cache and of the order in which historical incremental partitions were loaded.

## Scope

1. One canonical event-grain dimension and a separate cycle-grain dimension.
2. Source event IDs and scopes strictly from four persisted Gold snapshots.
3. Materialization fails on missing keys or contradictory scopes.
4. Synthetic regression tests exercise candidate-only historical events,
   tally-only events, party-only events, changes to one cycle, snapshot removal,
   and source conflict.
5. dbt tests enforce source-to-dimension equality and broad event membership.

## No-go conditions

- An event with missing evidence is not invented.
- `MAX(election_scope)` is not used to hide inconsistencies.
- Candidate vote Parquet is not scanned simply to enumerate event codes.
- No implicit Gold database promotion or raw cache rehydration.

## Operational gate

On a **separate DuckDB preview copied from the validated conformed candidate**:

1. Confirm original and candidate hashes and ensure writers have stopped.
2. Run dbt `dim_election dim_election_cycle` **without `+`**, so parents are read
   from materialized Gold and Bronze/Silver are not rebuilt.
3. Execute `assert_dim_election_gold_source_coverage`,
   `assert_conformed_election_identity`, and
   `assert_conformed_dimension_membership` against the preview.
4. Verify 198 observed events and five cycles for the current dataset; this
   count is a **snapshot-specific regression expectation**, not a universal
   hardcoded model limit.
5. Compare the same electorate, turnout, party, and candidate aggregates against
   the original and conformed candidate; do not promote until all pass.

`dbt build --full-refresh` on fresh scopes requires the existing raw source
materialization procedures for Gold parents; only the isolated event-dimension
rebuild is raw-cache independent.
