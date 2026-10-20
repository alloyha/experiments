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
make data-contracts-check     # all registered contracts
make data-contracts-dbt-plan  # optional preview only
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

## Version-compatibility gate

Run `make data-contracts-breaking-check` before changing the published ODCS
interface. The check does not execute dbt or read electoral data. It verifies
the frozen release baseline and semantic version progression, then runs the
explicit consumer-perspective CLI compatibility classifier.

## Multi-contract registry

`contracts/odcs/registry.yaml` is a closed-world inventory of published ODCS
interfaces and their isolated dbt metadata. `make data-contracts-registry-check`
rejects undocumented contracts, unregistered baselines, path traversal,
symlinked model metadata, duplicate ownership, baseline checksum mismatches,
contract identity drift and unversioned changes. The checksum trust anchors
remain pinned in Python code, not solely in the editable registry file.

The registry contains `dim_election_cycle` (cycle grain) and `dim_election`
(event grain); `make data-contracts-check`
iterates **all** registered contracts with CLI lint, consumer compatibility and
sandboxed dbt drift verification. Targeted commands remain available.
Publishing another contract requires its own baseline, reviewed checksum and
isolated dbt YAML; registry admission is not a physical GG certificate.

## Promotion policy

1. This PR isolates the `dim_election_cycle` model in its own YAML and
   promotes the ODCS-managed column types and checks. Existing dbt year/type
   uniqueness remains independent. Compile and test via coordinated dbt.
2. CI compares the contract against the immutable published baseline at
   `contracts/odcs/baselines/dim_election_cycle.v1.0.0.odcs.yaml` using
   `make data-contracts-breaking-check` (read-only). The baseline SHA-256 is
   pinned in `scripts/check_odcs_contract_version.py`. A semantic contract
   change must increase the semantic version, and a change to the contract ID,
   name, kind or ODCS API version is rejected.
   `datacontract-cli==1.2.7` performs the **consumer-perspective** breaking
   check; errors (exit 1) block CI, while warnings (exit 0) need review.
   This CLI pin is *only* for compatibility: dbt sync stays pinned at 1.2.3.
   To replace the published baseline, require a dedicated reviewed policy PR
   that explicitly updates the baseline and recorded checksum.
3. Broaden to other consumer contracts one bounded model at a time. Schema contracts alone **cannot** prove fact grain uniqueness,
   cross-table vote conservation, physical Parquet identity or freshness.

Because `dim_election_cycle` contains only a cycle key/year/type, it is a
small, verifiable pilot with no warehouse or certificate side effects.

## Second consumer interface: dim_election

`dim_election` publishes a six-column, event-grain ODCS surface keyed by
`election_id = year:type:election_code`. The shared `cycle_id` is not
unique. `evidence_sources` and `cycle_label` remain noncontractual.
The original dbt year/type/code uniqueness assertion and
`election_type` accepted-values test live in the isolated
`models/gold/dimensions/dim_election.yml` and are preserved by
synchronization. The original SQL and physical GG certificates
are unchanged. Both baselines are SHA-pinned and the registry-driven
runner checks both interfaces without altering the warehouse.
