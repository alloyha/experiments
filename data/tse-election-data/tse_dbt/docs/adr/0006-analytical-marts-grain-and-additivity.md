# ADR 0006 — Keep electorate and participation grains independently additive

**Status:** Proposed (PR 51)

## Context

Electorate is a cycle-level municipal snapshot. TSE turnout counts are reported
at the zone/office/round/event grain and can be repeated across offices. A
naive joined OBT creates false electorates and attendance; averaging rates
at the row or zone level creates biased totals.

## Decision

Build two semantic views on existing Gold with deliberately different grains.
Never join voter population to office/event votes before their independent
aggregation. Compute rates from summed components for a single compatible
contest slice. Keep transit-vote flag and explicit geography-match status.
Use dbt singular tests for conservation and rate identities plus uniqueness
contracts and adversarial synthetic regression tests.

## Consequences

- `obt_electorate_municipality` exposes historical comparison of **observed**
  snapshots, not migration or individual longitudinal identity.
- `cube_participation_municipality` is safe for zone-to-municipality and
  municipality-to-UF rollups **within one fixed event, round, office and
  transit-vote slice**. No broad cross-office sums are certified.
- No implicit rehydration, full-refresh, database promotion or source mutation.
- Promote views into tables only after evidence of repeated high query cost.
