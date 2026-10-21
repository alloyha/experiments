# CCC classification review: 25 discovered Gold/Semantic resources

**Status: proposals only, NOT approved classifications.** No entry below has been
added to `contracts/coverage_policy.yml`. The checker must continue to report
these 25 as `unclassified` until public exposure / consumer use is reviewed.

The first scan discovered 42 resources: two existing ODCS-public models,
15 technical-layer internals, and 25 unclassified Gold/Semantic models.
Historical policy aliases do not create extra active SQL models.

## Candidate interfaces to review for PUBLIC ODCS (13)

| Resource | Layer | Question before promotion |
|---|---|---|
| `fact_candidate_votes` | Gold | Exact candidate/zone/round grain; persisted parquet obligations remain independent? |
| `fact_tally_munzona` | Gold | Office/zone/round measures and tally conservation? |
| `fact_turnout` | Gold | No duplicate electorate denominators across office? |
| `candidate_ranking` | Semantic | Ranking vs elected status explicitly disclaimed? |
| `candidate_summary` | Semantic | Consumer query shape, unique grain, and joins identified? |
| `candidate_vote_summary` | Semantic | Vote totals and scope of aggregation certified? |
| `cube_candidate_performance_municipality` | Semantic | Municipality-grain candidate KPIs and weighting? |
| `cube_participation_municipality` | Semantic | Participation/turnout denominators and no double count? |
| `cube_party_performance_municipality` | Semantic | Party aggregates and source coverage? |
| `election_results` | Semantic | Top-ranked candidate is not necessarily elected? |
| `electoral_participation` | Semantic | Official turnout vs constructed participation measures? |
| `obt_electorate_municipality` | Semantic | Flattening cardinality and join amplification? |
| `party_performance` | Semantic | Performance measure semantics, grouping and filters? |

**No automatic publication.** A decision of `public` must include a versioned
ODCS contract, its baseline, schema/owned dbt metadata, grain and test mapping.

## Candidate INTERNAL quality and physical diagnostics (7)

| Resource | Review hypothesis |
|---|---|
| `candidate_result_coverage` | Gold diagnostic of candidate resource availability; may be consumer-visible |
| `candidate_tally_coverage_gaps` | Audit of source gaps, not a stable analytical promise |
| `candidate_tally_reconciliation` | Quality assertion input; keep GG/sql checks independent |
| `party_tally_coverage_gaps` | Audit of party-data coverage, not published facts |
| `party_tally_reconciliation` | Vote conservation diagnostic, not a substitute for certification |
| `graph_edges` | Internal graph structure; GG physical certificate governs semantics |
| `graph_vertices` | Internal graph structure; GG physical certificate governs semantics |

**Important:** an explicit internal decision requires a genuine, reviewed
rationale in `coverage_policy.yml`. If any diagnostic is used externally as a
supported interface, move it to the public queue instead.

## Review questions / acceptance

1. Is there an external report, downstream notebook, API, dashboard or other
   named consumer relying on this resource? If yes, prefer a public contract.
2. Can the declared grain be demonstrated by a dbt uniqueness assertion?
3. Does the consumer expect freshness / source coverage / physical GG attestation?
4. Does an upstream schema change impact this resource's public meaning?
5. No public item may be marked `internal` just to achieve CCC PASS.

Promotion can proceed in bounded waves (dimensions/facts, then semantic
surfaces); until then the audit is intentionally advisory, with 25 unresolved
classification decisions. `make check` stays on the previous four enforced
stages, while `make contract-coverage-plan` surfaces outstanding decisions.

## CCC v1.2 — source-reviewed internal decisions

Seven Gold quality/graph models have been individually classified as **internal**
in `contracts/coverage_policy.yml` after inspection of their SQL and lineage.
The policy includes a concrete implementation rationale for each model; a
source-evidence guard requires re-review if its defining behavior changes.
These classifications do not prove that no external consumer exists:
if such a consumer is discovered, promote that interface with a versioned ODCS
contract. All other 18 Gold/Semantic candidates remain unclassified.

The CCC gate is still advisory. The existing four-stage CI remains intact;
do not declare closure until all 18 public decisions and obligations are resolved.

## CCC v1.3 — first public dimensions wave

`dim_candidate` and `dim_geography` are promoted to versioned ODCS public interfaces with composite-grain tests and isolated dbt ownership. The remaining 16 Gold/Semantic candidates are still unclassified. `dim_party` stays pending: its current SQL concatenates `election_code` twice into `party_id` and must be reviewed before freezing the key.

## CCC B2 — public election-scoped party dimension

Following canonical SQL (#86) and guarded transactional migration (#87), all 717 observed parties across 2018, 2020, 2022, 2024, and 2026 have canonical election_id:party_number identity. Both in-repository Semantic queries preserved row count, matched keys and party-vote totals in the production before/after audit. This contract does not assert persistent party/SCD2 continuity, external consumer migration, Parquet completeness, or GG freshness; a changed lifecycle epoch requires separate graph recertification. The municipal Semantic cube remains unmaterialized in the observed warehouse. The remaining 15 Gold/Semantic decisions stay advisory.

## CCC C1 — public municipality electorate fact

The public electorate fact has cycle × UF × municipality grain and a pinned v1.0.0 ODCS baseline. Its electorate measure is additive only across disjoint municipalities within a given cycle. Do not sum electorate after event/round/office joins. This contract does not certify raw-source completeness or GG physical candidate Parquet publication. The remaining 14 Gold/Semantic decisions remain advisory.

## CCC C2 — public party-vote fact

The public party-vote fact is a snapshot at election-event × round × UF × municipality × zone × office × party × transit-flag grain. Its immutable ODCS v1.0.0 contract exposes nominal and legend components separately. Do not add converted nominal-to-legend votes a second time to `party_valid_votes`; no general identity `total_legend = legend + converted` is asserted. Independent party-versus-tally reconciliation, raw-source completeness and GG physical proof remain separate gates. The other 13 Gold/Semantic decisions remain advisory.
