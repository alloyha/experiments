# CCC · Contract Coverage Closure

## Why this gate exists

The ODCS registry is closed-world *only for already published ODCS documents*.
It cannot detect a new consumer-facing dbt resource that no one registered. CCC
closes the **discovery/classification gap** without claiming that every dbt table
needs its own ODCS contract.

## Scope and decisions

`contracts/coverage_policy.yml` classifies every discovered resource. SQL dbt
models in Bronze, Silver and Physical, plus raw source declarations and seeds,
are internal by an explicit **layer-level** architectural decision with a
reviewed reason. Gold/Semantic models and dbt exposures have **no automatic
visibility default**: each requires an individually reviewed decision.

- `public` means a Gold/Semantic consumer interface with a same-named registered
  ODCS contract and a documented, dbt-asserted domain grain.
- `internal` means a reviewed resource-specific reason, not a silent omission.
- `excluded` is a time-bounded waiver with an owner, reason and expiration date;
  overdue exceptions fail.

A model published in ODCS **must** remain classified public, and a classified
public model **must** have a registered ODCS contract. New/deleted models fail
until the inventory is reviewed. Planned domain models do not count as deployed
surfaces. This is an *exhaustive classification* of the discovered universe, not
a claim of complete data correctness or freshness.

## Commands

From `data/tse-election-data`:

```bash
make contract-coverage-plan     # show unclassified nodes and review suggestions
make contract-coverage-report   # read-only inventory and unresolved gaps
make contract-coverage-check    # fail-closed: every gap blocks CI
make check                      # CI includes CCC as a separate short stage
```

Optional reconciliation against a fresh dbt artifact:

```bash
.venv/bin/python -m scripts.check_contract_coverage \
  --mode check --manifest /path/to/fresh/manifest.json
```

The manifest is an **additional check**; the default gate discovers tracked
SQL, seed files, dbt YAML sources and exposures directly. It does not require
raw data, write to DuckDB, or issue GG certificates. External non-dbt APIs,
standalone Parquet exports and other publication endpoints must be added to
the discovery universe explicitly before claiming *system-wide* closure.

## Admission steps

1. Run `contract-coverage-plan`. Review each Gold/Semantic model and exposure,
   recording a `public`, `internal` or time-limited `excluded` decision.
2. For each public model, approve its ODCS baseline/checksum and isolated dbt
   owner, include the domain grain in `model_contracts.yml`, and retain the
   composite uniqueness test. Existing `make data-contracts-check` enforces
   consumer compatibility and drift on disposable dbt trees.
3. Run the falsifier tests, `make contract-coverage-check`, and `make check`.
   `unclassified=0`, `invalid_contracts=0` and
   `unmapped_semantic_obligations=0` are the structural acceptance criteria.

Never infer GG graph certification from CCC. GG-009/GG-012 physical identity,
parquet publication, source conservation, freshness and cycle attestations
remain separate, opt-in operational controls.

## Rollout boundary (CCC v1.1)

The CCC auditor is deliberately NOT a fifth mandatory `make check` stage until
all 25 pending Gold/Semantic resources are reviewed and the gate passes.
Historical `stg_*`/`int_*` names are explicitly marked `superseded` in
`tse_dbt/model_contracts.yml` with current dbt SQL successors.
That does not automatically classify new interfaces as internal.
See `docs/analytics/ccc_classification_review.md` for proposed decisions.
