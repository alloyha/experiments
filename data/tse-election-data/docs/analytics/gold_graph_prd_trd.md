# PRD/TRD — Gold electoral property graph and invariant registry (PR54 foundation)

## Product decision

Offer typed electoral graph relationships backed only by conformed Gold facts
and dimensions. Keep the relational Gold authoritative; graph vertices and
edges are **lazy, read-only projections**, not a new master data store and not
an externally certified graph product. This follows the lecture at
`data/data-modelling-lectures/aulas/09_modelagem_grafos`, retaining its
dimension → vertex, fact → event edge, bridge → association edge distinction.

### Vertex types and identities

| Vertex | Canonical key fields | Source |
|---|---|---|
| `cycle` | `cycle_id` | `dim_election_cycle` |
| `election` | `election_id` | `dim_election` |
| `municipality` | `cycle_id, uf, municipality_code` | geography + observed vote facts |
| `candidate` | `election_id, candidate_id` | candidate dimension + observed candidate fact |
| `party` | `election_id, party_number` | party dimension + observed party fact + candidate affiliation |
| `contest` | `election_id, round_number, uf, municipality_code, office_code, is_transit_vote` | candidate and party vote facts |

All vertex/edge IDs use **typed JSON-encoded tuples**, with each component cast
explicitly to text. No arithmetic offsets, artificial person identities or
cross-cycle links. A candidate is an *election candidacy*, not a person tracked
longitudinally; a party is an election-specific observed identifier. No party
identity continuity is inferred across elections.

### Edge types

| Edge | Direction | Measures and provenance |
|---|---|---|
| `CONTAINS_ELECTION` | cycle → election | dimension-derived; no measures |
| `HAS_CONTEST` | election → contest | fact-derived; no measures |
| `LOCATED_IN` | contest → municipality | fact-derived; no measures |
| `HAS_CANDIDATE` | election → candidacy | dimension-derived; no measures |
| `HAS_PARTY` | election → party | dimension-derived; no measures |
| `AFFILIATED_WITH` | candidacy → party | candidate dimension; only known affiliations |
| `VOTED_FOR_CANDIDATE` | contest → candidacy | nominal valid, input rows, distinct zones |
| `VOTED_FOR_PARTY` | contest → party | nominal, legend, total legend, party-valid, rows, zones |

Vote edges aggregate **within the full contest key**. Nominal candidate totals
and party-reported nominal totals are not assumed identical; reconciliation
requires source-coverage analysis. The `total_legend_valid_votes` measure already
contains converted-to-legend votes and must not be added twice.

Unknown descriptive metadata is not silently turned into fake parties or
candidates. The observed fact identity is still present as a vertex, marked
`metadata_matched = false`. Missing elections/cycles are **not** auto-created:
referential certification must fail if Gold event registries contradict votes.

## Data and runtime boundaries

- Paths: `tse_dbt/models/gold/graph/graph_vertices.sql` and
  `graph_edges.sql`; direct `ref()` only to Gold. No semantic → Gold cycle.
- `materialized='view'` and `tag:expensive`. These queries include candidate
  Parquet scans; **never** run full-history schema tests just to validate a PR.
- New `assert_graph_*` dbt SQL checks are tagged `expensive`; use a small canary
  with explicit `incremental_years` and `incremental_election_types` for the
  first physical certification. `make unit` uses tiny in-memory fixtures.
- `make compile` and the static DAG check must pass before any data scans.
- The graph has **no freshness attestation** in this first PR; PR53's
  `candidate_cube` attestation is not a license to serve the graph. A protected
  consumer, graph freshness proof and a reader/writer boundary are separate
  acceptance gates, intentionally still red in the invariant registry.
- Publishing a candidate Parquet generation can make an existing graph read
  reflect the new version. Do not promise transactionally stable traversal
  across separate SQL statements without a pinned snapshot.

## Invariant coverage and evidence

The source of truth is `contracts/invariants/gold_graph.yml`. Every
`status: implemented` record must resolve a SQL assertion and both a positive
fixture and a **deliberately failing** falsifier. The registry checker rejects
missing/duplicate/invalid records; its `--strict` mode intentionally fails if
any critical domain invariant is still planned.

This first wave explicitly covers graph node/edge identity, in-cycle endpoint
reference, candidate/party vote multiset conservation, missing-metadata vote
retention, and separation of measures from topology. It does **not** claim
complete invariants across the entire TSE pipeline. Open gaps include
cross-context traversals, concurrency, graph freshness, temporal-affiliation
falsification and publication crash recovery.

### Developer loop (feature worktree, no canonical writes)

```bash
cd data/tse-election-data
PY="$HOME/github/experiments/data/tse-election-data/.venv/bin/python"
"$PY" -m unittest discover -s tests -p test_gold_graph_invariants.py -v
"$PY" -m unittest discover -s tests -p test_invariant_registry.py -v
"$PY" scripts/check_invariant_registry.py
"$PY" scripts/check_medallion_dag.py
# dbt compile in the feature worktree after dbt deps; avoid dbt build --full-refresh.
```

`--strict` is a **promotion gate**, not a fast CI gate: it is expected to fail
until all critical gaps and physical certification are closed. Do not relabel
`planned` invariants as implemented to make a metric green.

## Explicit next-stage acceptance

1. Dry-run/EXPLAIN canary queries and measure peak spill/time; add physically
   scoped graph publication if lazy scan costs are unacceptable.
2. Execute each `assert_graph_*` SQL check on a small certified cycle, then
   perform deliberate DB-level mutations and show the check fails.
3. Pin facts/dimensions and graph output to a generation-locked proof; invalidate
   graph output when any dependency changes.
4. Expose guarded graph reads with snapshot-safe validation and bounded
   traversal (`max_depth`, allowed edge types and cycle/contest filters).
5. Raise the strict invariant gate to green only after every **critical** open
   item is covered by passing positive and falsifying tests and real evidence.

## Identity and event membership checks (follow-up)

`assert_graph_edge_context` rejects unknown edge types, incorrect vertex types,
cross-election endpoint links **within the same cycle**, and mislabeled
contest rounds/offices/transit flags. `assert_graph_affiliation_conservation`
uses bidirectional `EXCEPT ALL` to prove candidate-party associations are exactly
the current event-scoped dimensional snapshot, rather than a persistent person
affiliation. These are expensive, explicitly selected tests; no new warehouse
materialization is introduced.

This does **not** certify arbitrary recursive traversals; any traversal API
must independently enforce a fixed path budget and event/contest boundaries.
