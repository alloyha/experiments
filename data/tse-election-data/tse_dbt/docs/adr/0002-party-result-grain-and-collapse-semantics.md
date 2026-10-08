# ADR 0002 — Party result grain and deterministic collapse semantics

## Status

Accepted.

## Evidence

For the 2018 general-election party resource:

- 607,022 duplicate analytical grains;
- 74 grains differ in nominal-valid votes;
- all 74 are exactly `0 + positive`;
- 0 grains differ in `legend_valid_votes`;
- 0 grains differ in `total_legend_valid_votes`;
- measure-wise MAX collapse yields 0 nominal mismatches and 0 legend mismatches against tally.

## Decision

`silver_party_votes_munzona` collapses alternate/repeated source representations
with measure-wise `MAX()` at the analytical grain.

Do not use arbitrary `row_number()` selection.
Do not SUM representational duplicates.

Party↔tally and candidate↔tally are strict zero-delta invariants.
