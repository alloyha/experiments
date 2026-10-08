# ADR 0003 — Canonical geography keys and candidate-resource coverage

## Status

Accepted.

## Context

The 2018 candidate-vote resource encodes `CD_MUNICIPIO` with variable-width
numeric strings in some rows, while party/tally resources preserve five-character
municipality identifiers.

Examples:

- `4154` vs `04154`
- `1392` vs `01392`
- `6050` vs `06050`
- `35` vs `00035`

Before normalization, candidate↔tally reconciliation produced 7,335 mismatching
common-grain rows.

Applying five-character canonicalization reduced this to exactly one row.

That remaining row is not a reconciliation error. It belongs to:

- election code `339`
- `Eleição Conselho Distrital 2018 FN`
- Fernando de Noronha / PE
- office code `25` — Conselheiro Distrital
- 1,836 nominal valid votes

The tally resource contains this election/office, while the candidate resource
contains no corresponding candidate rows.

## Decision

1. Municipality codes are canonical textual identifiers of exactly five digits.
2. Canonicalization occurs at staging boundaries.
3. Candidate↔tally is a strict zero-delta invariant only inside slices for which
   the candidate resource has coverage.
4. Missing candidate-resource coverage is represented explicitly in
   `candidate_tally_coverage_gaps`.
5. `dim_election` grain includes `election_code`; `year + election_type` alone is
   insufficient to distinguish all elections in a cycle.

## Consequences

- geography joins are deterministic across TSE products;
- source coverage gaps are never hidden as vote discrepancies;
- election 339 remains queryable through tally even without candidate coverage;
- downstream facts use election-aware keys.
