# ADR 0008 — Gold graph projections and falsifiable invariant coverage

**Status:** Proposed (PR54 foundational slice)

## Context

The Gold layer has conformed election/cycle identifiers and both candidate and
party vote facts, but multi-hop electoral context cannot yet be browsed as a
typed graph. The existing candidate cube has per-cycle freshness attestations;
this is **not** an attestation of a new graph projection.

## Decision

1. Model graph **vertices from Gold dimensions plus observed fact identities**;
   model factual and association **edges from Gold facts/dimensions**. The
   canonical Gold relational facts and dimensions remain authoritative.
2. Use type-tagged, JSON-encoded tuple IDs, including full election and contest
   grain. Never borrow textbook-style offset IDs, collapse event into cycle,
   or create a cross-election personal/party identity.
3. Do not lose voting records when descriptive dimension metadata is absent:
   retain placeholder *vertices with an explicit unmatched flag*, not invented
   attributes. Reject irreconcilable missing cycle/event parentage at contract
   validation.
4. Begin with **views** tagged expensive. The graph depends on candidate fact
   Parquet and is **not** freshness-certified or transactionally pinned. Avoid
   graph materialization and live consumption claims until benchmarks and
   persistence/fencing design are proven.
5. Require an invariant catalog with executable positive/falsifying examples;
   keep every unresolved critical gap visible to an optional strict gate.

## Consequences

- No new OLTP or graph database, no rebuilding the five candidate Parquets.
- Fast CI covers static/rendering/in-memory graph contracts, not historical
  multiset certification. `assert_graph_*` SQL tests require explicit canary
  execution and measured resource use.
- The graph does not establish individual voter behavior, elected status,
  candidates' identity across elections, party continuity, or causal ties.
- A dedicated follow-up must implement freshness/consumption fencing and prove
  absence of time-of-check/time-of-use races.
