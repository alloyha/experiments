# ADR-001: Medallion architecture for TSE analytics

Status: Accepted

## Context

The TSE pipeline already separates immutable source acquisition from analytical
modeling, but `stg_*`, `int_*`, `marts` do not make transformation ownership
explicit enough for long-run lineage and governance.

The immutable raw store remains outside dbt and is not Bronze.

## Decision

- **RAW** — immutable ZIPs, manifests, active-object index and provenance.
- **Bronze** — faithful source interpretation, schema drift handling and minimal typing.
- **Silver** — canonical grains, keys, semantic duplicate handling and reusable transformations.
- **Gold** — conformed dimensions, additive facts, reconciliation and explicit coverage gaps.
- **Semantic** — consumer-facing summaries and metrics.
- **Physical** — implementation-only materialization helpers; not a business layer.

## Invariants

1. RAW objects remain immutable by SHA.
2. Bronze does not silently alter business grain.
3. Silver owns canonical grain and semantic deduplication.
4. Gold facts and dimensions have stable declared grains and conformed keys.
5. Reconciliation gaps remain explicit instead of being patched away.
6. Semantic models never repair upstream data quality.
7. `party_valid_votes = nominal_valid_votes + total_legend_valid_votes` remains canonical.
8. Party duplicate collapse remains measure-wise MAX, never SUM.
9. Candidate coverage keeps its exact election/round/UF/municipality/zone/office/transit grain.
10. Frozen cycle regressions remain unchanged; 2026 stays provisional until explicitly promoted.
