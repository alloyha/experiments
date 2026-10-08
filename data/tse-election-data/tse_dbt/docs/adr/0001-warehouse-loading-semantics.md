# ADR-0001 — Warehouse loading and history semantics

**Status:** Accepted  
**Scope:** TSE analytical warehouse

## Decision

We distinguish four load semantics:

1. **full_replace** — small/static relations are rebuilt completely.
2. **partition_replace** — authoritative TSE snapshots replace the complete
   `(election_year, election_type)` partition on an incremental run.
3. **scd2_merge** — reserved for persistent business entities whose attributes
   change independently of an election grain.
4. **append** — reserved for genuine event/audit streams.

## Why row-key delete+insert is not enough

TSE resources are commonly complete republished snapshots.  If a row existed in
snapshot A and disappears from snapshot B, a row-key-driven incremental merge
cannot discover that deletion because the missing key is absent from B.

Therefore persistent snapshot models use:

1. `pre_hook`: delete the selected election partition;
2. model query: read the complete current snapshot for exactly that partition;
3. insert the replacement partition;
4. tests: compare source and target for exact snapshot completeness.

This makes removals, corrections, and additions deterministic.

## Current dimensional history

### dim_election — Type 0

Election cycles are canonical reference data.  They are rebuilt as a small
table and treated as immutable except for explicit corrections.

### dim_candidate — election snapshot, not SCD2

The current dimension represents a **candidature**, with grain:

`election_year × election_type × election_code × candidate_id`

Party, office, education, occupation, etc. are attributes of that candidature.
A candidate appearing again in another election is a new election snapshot.
Using SCD2 here would duplicate temporal semantics already present in the grain.

A future persistent `dim_person` may use SCD2.

### dim_geography — election snapshot for now

The current geography dimension is conformed to an election snapshot.  A future
persistent geography dimension may become SCD2 once multiple cycles let us
observe and validate administrative changes.

### dim_party — first planned SCD2

Party is the first strong SCD2 candidate because name/abbreviation and legal
identity can evolve independently of a particular candidature.  We will not
activate it from 2018 alone.  It requires multiple cycles and stable-key
validation first.

## Full vs incremental execution

Normal extension to a new cycle:

- `election_years`: all raw cycles available to the build;
- `incremental_years`: only partitions that changed or were newly added.

Example after 2018 when 2022 becomes available:

```text
election_years      = [2018, 2022]
incremental_years   = [2022]
```

If a 2018 source is republished:

```text
incremental_years   = [2018]
```

Only the 2018 partitions are replaced.

`--full-refresh` is intentionally stronger: it replaces the whole relation with
the selected `election_years/election_types` scope.

## Cube invalidation

Cube maintenance follows fact invalidation.  If 2018 candidate votes are
replaced, only canonical cuboids dependent on the 2018 candidate-vote
partition need recomputation.  We do not precompute all dimension combinations.
