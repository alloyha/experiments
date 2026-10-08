# Candidate test cost profile

On the 2018 candidate result branch (~8.68M rows), generic tests on
`silver_candidate_votes_munzona` can dominate wall-clock time even though the model
itself is a view.

Observed examples:

- `not_null round_number`: ~12m41s
- `not_null uf`: ~12m44s
- `not_null nominal_valid_votes`: ~9m33s
- unique combination at candidate grain: ~11m10s
- accepted office_scope values: ~7m50s

The model itself materializes in under a second.

These full-scan staging validations should be migrated from the normal build
into an `integrity_heavy` gate, keeping lightweight semantic/unit checks in the
default development loop.
