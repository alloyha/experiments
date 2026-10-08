# Candidate vote staging test strategy

`stg_candidate_votes_munzona` is a view over approximately 8.68 million rows.

Previously, generic dbt tests caused repeated complete Parquet scans, dominating
development build time.

Observed 2018 costs included approximately:

- not_null(round_number): 12m41s
- not_null(uf): 12m44s
- candidate-grain uniqueness: 11m10s
- not_null(nominal_valid_votes): 9m33s
- accepted_values(office_scope): 7m50s

The normal build therefore keeps the semantic reconciliation gates but moves
large source-integrity scans into an explicit heavy audit.

Heavy audit:

1. assert_stg_candidate_votes_munzona_integrity_heavy
   - one scan for null and accepted-value checks.

2. assert_stg_candidate_votes_munzona_grain_unique_heavy
   - one grouped scan for exact analytical-grain uniqueness.

Both require `run_heavy_integrity_tests=true`.
