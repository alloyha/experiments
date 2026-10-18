# ODCS Data Contract CLI: narrow dbt pilot

## Motivation and responsibility boundary

This opt-in pilot uses Data Contract CLI 1.2.3 (ODCS v3.2.0) for a single
well-defined **public interface**, `dim_election_cycle`. The existing
`tse_dbt/model_contracts.yml` remains the domain-grain/lifecycle policy,
`models/schema.yml` carries other dbt quality checks, and the independent
GG-* SQL + read-only certification controls are **not** replaced.

The ODCS contract owns only the declared **three-column consumer-facing
shape** and the cycle-ID primary key for the pilot. Its synchronized model
metadata lives in `tse_dbt/models/gold/dimensions/dim_election_cycle.yml`,
not in the shared `models/schema.yml`. The existing year/type combination
uniqueness remains a separate dbt test. Keep descriptions and grain aligned. No ODCS files are automatically inferred
from a dbt manifest and treated as authoritative.

## Commands (run from `data/tse-election-data`)

```bash
make data-contracts-lint
make data-contracts-dbt-plan
make data-contracts-dbt-check
```

Both targets run `datacontract-cli==1.2.3` in an isolated `uv tool run`
environment, using the project's Python. `lint` validates the ODCS document.
The second command calls `datacontract dbt sync --dry-run --no-prune` and
**does not persist model YAML or tests**. CI verifies that the CLI recognizes the contract and dbt project, then
`make data-contracts-dbt-check` runs real sync on a **disposable project
copy** and rejects semantic drift in dbt model declarations, as well as byte
drift in the ODCS-managed model YAML. The commands do not mutate the real
dbt project or the warehouse.

`datacontract dbt test` and `dbt sync --run-tests` are intentionally not part
of this PR: invoking dbt through that separate process bypasses the project's
`run_coordinated_dbt.py` GG-009 writer gateway, and may also create expensive
unscoped assertions. Never enable automatic prune or write generated SQL
tests over the freshly certified canonical graph.

## Promotion policy

1. This PR isolates the `dim_election_cycle` model in its own YAML and
   promotes the ODCS-managed column types and checks. Existing dbt year/type
   uniqueness remains independent. Compile and test via coordinated dbt.
2. Add a compatibility gate comparing versioned contracts with
   `datacontract breaking <old> <new>` in a follow-up PR, after defining
   the baseline/source version contract and exit code semantics.
3. Broaden to other consumer contracts one bounded model at a time. Schema contracts alone **cannot** prove fact grain uniqueness,
   cross-table vote conservation, physical Parquet identity or freshness.

Because `dim_election_cycle` contains only a cycle key/year/type, it is a
small, verifiable pilot with no warehouse or certificate side effects.
