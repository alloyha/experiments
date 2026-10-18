SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# DuckDB and the project environment are shared mutable state.
# Parallelism belongs inside ingestion/dbt, not between make targets.
.NOTPARALLEL:

# ---------------------------------------------------------------------------
# Configuration — override with make VAR=value
# ---------------------------------------------------------------------------
UV ?= uv
PYTHON ?= python3
PROJECT_PYTHON ?= .venv/bin/python

DBT_DIR ?= tse_dbt
DATA_ROOT ?= ./data/tse
DUCKDB_PATH ?= ./data/warehouse/tse_analytics.duckdb
CANDIDATE_FACT_ROOT ?= ./data/warehouse/fact_candidate_votes
CYCLE_CONTRACTS ?= ./contracts/election_cycles.json

# Cycle-specific graph certification is explicit and opt-in.
GRAPH_CYCLE ?=
GRAPH_MEMORY_LIMIT ?= 4GB
GRAPH_ATTESTATIONS_DIR ?= ./data/warehouse/graph_attestations

# Years used by dbt and years refreshed in the current run.
YEARS ?= 2018
UPDATE_YEARS ?= $(YEARS)

# Removed public inputs must fail rather than be silently ignored.
ifneq ($(filter command line,$(origin ELECTION_TYPES)),)
$(error ELECTION_TYPES is derived automatically from YEARS)
endif
ifneq ($(filter command line,$(origin INCREMENTAL_YEARS)),)
$(error Use UPDATE_YEARS instead of INCREMENTAL_YEARS)
endif
ifneq ($(filter command line,$(origin INCREMENTAL_ELECTION_TYPES)),)
$(error INCREMENTAL_ELECTION_TYPES is derived automatically from UPDATE_YEARS)
endif

# Read the canonical calendar shared with dbt. The scope contract also
# cross-checks these types against tse_ingest.ELECTION_CALENDAR.
ELECTION_CALENDAR_CSV := $(DBT_DIR)/seeds/election_calendar.csv
GENERAL_YEARS := $(shell awk -F, 'NR>1 && $$2=="general" {print $$1}' '$(ELECTION_CALENDAR_CSV)')
MUNICIPAL_YEARS := $(shell awk -F, 'NR>1 && $$2=="municipal" {print $$1}' '$(ELECTION_CALENDAR_CSV)')
SCOPE_TYPES := $(strip $(if $(filter $(YEARS),$(GENERAL_YEARS)),general) $(if $(filter $(YEARS),$(MUNICIPAL_YEARS)),municipal))
UPDATE_TYPES := $(strip $(if $(filter $(UPDATE_YEARS),$(GENERAL_YEARS)),general) $(if $(filter $(UPDATE_YEARS),$(MUNICIPAL_YEARS)),municipal))

# Validate even standalone commands (e.g. print-dbt-vars, backfill), not just
# pipelines with the Python scope-contract prerequisite.
ifeq ($(strip $(YEARS)),)
$(error YEARS must not be empty)
endif
ifeq ($(strip $(UPDATE_YEARS)),)
$(error UPDATE_YEARS must not be empty)
endif
ifneq ($(strip $(filter-out $(GENERAL_YEARS) $(MUNICIPAL_YEARS),$(YEARS))),)
$(error Unknown YEARS: $(filter-out $(GENERAL_YEARS) $(MUNICIPAL_YEARS),$(YEARS)))
endif
ifneq ($(strip $(filter-out $(YEARS),$(UPDATE_YEARS))),)
$(error UPDATE_YEARS must be a subset of YEARS: $(filter-out $(YEARS),$(UPDATE_YEARS)))
endif

PROFILE ?= analytics
GRANULARITY ?= brasil
UF ?=

INGEST_WORKERS ?= 8
CHECKPOINT_EVERY ?= 10
PRUNE_DOMAINS ?=
MATERIALIZE_DOMAINS ?=
MATERIALIZE_EXTRACTOR ?= auto
MATERIALIZE_CHECKPOINT_EVERY ?= 1

PROJECT_ROOT_ABS := $(abspath .)
RAW_ROOT_ABS := $(abspath $(DATA_ROOT))
DUCKDB_PATH_ABS := $(abspath $(DUCKDB_PATH))
CANDIDATE_FACT_ROOT_ABS := $(abspath $(CANDIDATE_FACT_ROOT))
DBT_ARTIFACTS_ROOT ?= $(DUCKDB_PATH).dbt-runs
DBT_ARTIFACTS_ROOT_ABS := $(abspath $(DBT_ARTIFACTS_ROOT))
GRAPH_ATTESTATIONS_DIR_ABS := $(abspath $(GRAPH_ATTESTATIONS_DIR))

ENV_STAMP := .venv/.tse-ready
DBT_DEPS_STAMP := $(DBT_DIR)/dbt_packages/.tse-deps-ready

RUN := env -u VIRTUAL_ENV $(UV) run --python '$(PROJECT_PYTHON)'
PY := $(RUN) python
INGEST := $(PY) tse_ingest.py

# Central Makefile gateway for warehouse writers (GG-009); all dbt recipes
# must invoke this macro rather than an uncoordinated CLI.
DBT := \
	$(PY) scripts/run_coordinated_dbt.py \
	--db '$(DUCKDB_PATH_ABS)' \
	--dbt-executable '$(abspath $(dir $(PROJECT_PYTHON))dbt)' \
	--artifacts-root '$(DBT_ARTIFACTS_ROOT_ABS)' \
	--

DBT_PROJECT_ARGS := \
	--project-dir $(DBT_DIR) \
	--profiles-dir $(DBT_DIR)

INGEST_COMMON_ARGS := \
	--root '$(RAW_ROOT_ABS)' \
	--profile $(PROFILE) \
	--granularity $(GRANULARITY)

ifneq ($(strip $(UF)),)
INGEST_COMMON_ARGS += --uf $(UF)
endif

INGEST_COMMON_ARGS += \
	--workers $(INGEST_WORKERS) \
	--checkpoint-every $(CHECKPOINT_EVERY)

BACKFILL_YEAR_ARGS := \
	$(foreach year,$(YEARS),--year $(year))

UPDATE_YEAR_ARGS := \
	$(foreach year,$(UPDATE_YEARS),--year $(year))
PRUNE_DOMAIN_ARGS := $(foreach domain,$(PRUNE_DOMAINS),--domain $(domain))
MATERIALIZE_YEAR_ARGS := $(foreach year,$(YEARS),--year $(year))
MATERIALIZE_TYPE_ARGS := $(foreach type,$(SCOPE_TYPES),--election-type $(type))
MATERIALIZE_DOMAIN_ARGS := $(foreach domain,$(MATERIALIZE_DOMAINS),--domain $(domain))

DBT_VARS_CMD = \
	RAW_ROOT='$(RAW_ROOT_ABS)' \
	CANDIDATE_FACT_ROOT='$(CANDIDATE_FACT_ROOT_ABS)' \
	YEARS='$(YEARS)' \
	ELECTION_TYPES='$(SCOPE_TYPES)' \
	INCREMENTAL_YEARS='$(UPDATE_YEARS)' \
	INCREMENTAL_ELECTION_TYPES='$(UPDATE_TYPES)' \
	$(PYTHON) -c 'import json, os; print(json.dumps({ \
	        "tse_raw_root": os.environ["RAW_ROOT"], \
	        "candidate_fact_root": os.environ["CANDIDATE_FACT_ROOT"], \
	        "election_years": [int(x) for x in os.environ["YEARS"].split()], \
	        "election_types": os.environ["ELECTION_TYPES"].split(), \
	        "incremental_years": [int(x) for x in os.environ["INCREMENTAL_YEARS"].split()], \
	        "incremental_election_types": os.environ["INCREMENTAL_ELECTION_TYPES"].split() \
	}))'
# ---------------------------------------------------------------------------
# Command interface — short daily help; all specialized operations: help-all
# ---------------------------------------------------------------------------
.PHONY: help help-all doctor setup dbt-deps scope-contracts print-dbt-vars \
	list ingest backfill materialize-raw candidate-fact-partition unit \
	raw-contracts cycle-regressions medallion-contracts dbt-bronze dbt-silver \
	dbt-gold semantic storage-report prune-cache pr-gate production-gate dbt-build \
	dbt-full-refresh test-only compile performance-report performance-baseline \
	pipeline pipeline-backfill clean clean-warehouse clean-all \
	invariant-registry-strict graph-cycle-certify data-contracts-lint data-contracts-dbt-plan

help:
	@printf 'TSE analytics\n'
	@printf '\n  Setup\n'
	@printf '    %-23s %s\n' 'setup' 'Prepare Python and dbt environment'
	@printf '    %-23s %s\n' 'doctor' 'Check prerequisites'
	@printf '\n  Data\n'
	@printf '    %-23s %s\n' 'list' 'Preview UPDATE_YEARS resources (no downloads)'
	@printf '    %-23s %s\n' 'ingest' 'Download only changed resources'
	@printf '    %-23s %s\n' 'backfill' 'Acquire historical resource versions'
	@printf '    %-23s %s\n' 'pipeline' 'Ingest and build analytics'
	@printf '\n  Analytics & quality\n'
	@printf '    %-23s %s\n' 'dbt-build' 'Build/test analytics on available raw data'
	@printf '    %-23s %s\n' 'unit' 'Run Python tests'
	@printf '    %-23s %s\n' 'medallion-contracts' 'Validate DAG and lineage'
	@printf '    %-23s %s\n' 'data-contracts-lint' 'Validate ODCS data contracts'
	@printf '    %-23s %s\n' 'data-contracts-dbt-plan' 'Preview ODCS/dbt sync without writing'
	@printf '    %-23s %s\n' 'invariant-registry-strict' 'Check 12/12 critical graph contracts'
	@printf '    %-23s %s\n' 'graph-cycle-certify' 'Certify one prepared real graph cycle (opt-in)'
	@printf '    %-23s %s\n' 'pr-gate' 'PR contract and dbt gate'
	@printf '    %-23s %s\n' 'production-gate' 'Warehouse checks; downloads data (graph separate)'
	@printf '\n  More: make help-all (advanced targets and maintenance)\n'
	@printf '  Scope: YEARS=%s UPDATE_YEARS=%s\n' '$(YEARS)' '$(UPDATE_YEARS)'
	@printf '  YEARS=analysis; UPDATE_YEARS=refreshed years (defaults to YEARS).\n'

help-all:
	@awk 'BEGIN {FS = ":.*## "; printf "TSE analytics — all commands\n\n"} \
	  /^[a-zA-Z0-9_.-]+:.*## / \
	  {printf "  %-26s %s\n", $$1, $$2}' \
	  $(MAKEFILE_LIST)
# ---------------------------------------------------------------------------
# Environment and dependencies
# ---------------------------------------------------------------------------
doctor: ## Check prerequisites and environment
	@command -v $(UV) >/dev/null || \
	  { echo 'Missing uv'; exit 1; }
	@command -v $(PYTHON) >/dev/null || \
	  { echo 'Missing $(PYTHON)'; exit 1; }
	@test -x '$(PROJECT_PYTHON)' || \
	  { echo 'Missing project Python: $(PROJECT_PYTHON)'; exit 1; }
	@echo "uv:             $$(command -v $(UV))"
	@echo "python:         $$(command -v $(PYTHON))"
	@echo "project python: $(PROJECT_PYTHON)"
	@echo "raw:            $(RAW_ROOT_ABS)"
	@echo "duckdb:         $(DUCKDB_PATH_ABS)"
$(ENV_STAMP): pyproject.toml uv.lock $(DBT_DIR)/requirements.txt
	@command -v $(UV) >/dev/null || \
	  { echo 'Missing uv'; exit 1; }
	@command -v $(PYTHON) >/dev/null || \
	  { echo 'Missing $(PYTHON)'; exit 1; }
	@mkdir -p '$(dir $(DUCKDB_PATH_ABS))'
	@mkdir -p '$(CANDIDATE_FACT_ROOT_ABS)'
	@if ! test -x '$(PROJECT_PYTHON)'; then \
	  rm -rf .venv; \
	  env -u VIRTUAL_ENV $(UV) venv .venv; \
	fi
	@env -u VIRTUAL_ENV $(UV) sync --inexact
	@env -u VIRTUAL_ENV $(UV) pip install \
	  --python '$(PROJECT_PYTHON)' \
	  -r $(DBT_DIR)/requirements.txt
	@touch '$@'
setup: $(ENV_STAMP) ## Ensure the local environment is ready
	@mkdir -p '$(dir $(DUCKDB_PATH_ABS))'
	@mkdir -p '$(CANDIDATE_FACT_ROOT_ABS)'
	@test -x '$(PROJECT_PYTHON)' || { \
	  rm -f '$(ENV_STAMP)'; \
	  $(MAKE) --no-print-directory '$(ENV_STAMP)'; \
	}
$(DBT_DEPS_STAMP): \
	$(DBT_DIR)/packages.yml \
	$(DBT_DIR)/package-lock.yml \
	$(ENV_STAMP)
	$(DBT) deps $(DBT_PROJECT_ARGS)
	@touch '$@'
dbt-deps: $(DBT_DEPS_STAMP) ## Ensure dbt packages are installed
# ---------------------------------------------------------------------------
# Raw ingestion and materialization
# ---------------------------------------------------------------------------
list: setup ## List incremental ingestion selection
	$(INGEST) \
	  --mode incremental \
	  $(UPDATE_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS) \
	  --list-only
ingest: scope-contracts ## Incrementally ingest changed resources
	$(INGEST) \
	  --mode incremental \
	  $(UPDATE_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS)
backfill: setup ## Backfill immutable resource versions
	$(INGEST) \
	  --mode backfill \
	  $(BACKFILL_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS)
materialize-raw: setup ## Rehydrate selected raw cache offline
	$(PY) scripts/ensure_materialized.py \
	  --root '$(RAW_ROOT_ABS)' \
	  $(MATERIALIZE_YEAR_ARGS) \
	  $(MATERIALIZE_TYPE_ARGS) \
	  $(MATERIALIZE_DOMAIN_ARGS) \
	  --extractor '$(MATERIALIZE_EXTRACTOR)' \
	  --checkpoint-every '$(MATERIALIZE_CHECKPOINT_EVERY)'
# ---------------------------------------------------------------------------
# Analytical models and dbt
# ---------------------------------------------------------------------------
print-dbt-vars: ## Print dbt vars JSON
	@$(DBT_VARS_CMD)
candidate-fact-partition: dbt-deps ## Build one physical candidate partition
	@test "$(words $(UPDATE_YEARS))" -eq 1 || \
	  { echo "requires exactly one UPDATE_YEARS value"; exit 1; }
	@test "$(words $(UPDATE_TYPES))" -eq 1 || \
	  { echo "requires exactly one derived update election type"; exit 1; }

	$(PY) scripts/build_candidate_fact_partition.py \
	  --db '$(DUCKDB_PATH_ABS)' \
	  --root '$(CANDIDATE_FACT_ROOT_ABS)' \
	  --raw-root '$(RAW_ROOT_ABS)' \
	  --project-dir '$(abspath $(DBT_DIR))' \
	  --profiles-dir '$(abspath $(DBT_DIR))' \
	  --dbt-executable '$(abspath $(dir $(PROJECT_PYTHON))dbt)' \
	  --artifacts-root '$(DBT_ARTIFACTS_ROOT_ABS)' \
	  --year '$(UPDATE_YEARS)' \
	  --election-type '$(UPDATE_TYPES)'
dbt-bronze: scope-contracts dbt-deps ## Build/test Bronze
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --select 'tag:bronze' --vars "$$vars"
dbt-silver: scope-contracts dbt-deps ## Build/test Silver and upstreams
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --select '+tag:silver' --vars "$$vars"
dbt-gold: scope-contracts dbt-deps ## Build/test Gold and upstreams
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --select '+tag:gold' --vars "$$vars"
semantic: scope-contracts dbt-deps ## Build semantic models and upstreams
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --select '+path:models/semantic' \
	  --vars "$$vars"
dbt-build: scope-contracts raw-contracts dbt-deps ## Build/test normal dbt graph
	@vars="$$($(DBT_VARS_CMD))"; \
	  echo "dbt vars: $$vars"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --vars "$$vars"
dbt-full-refresh: dbt-deps ## Full-refresh dbt graph
	@vars="$$($(DBT_VARS_CMD))"; \
	  echo "dbt vars: $$vars"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --full-refresh \
	  --vars "$$vars"
test-only: scope-contracts dbt-deps ## Test existing warehouse
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) test $(DBT_PROJECT_ARGS) \
	  --vars "$$vars"
compile: dbt-deps ## Compile graph using fixtures
	@vars="$$($(DBT_VARS_CMD))"; \
	  compile_vars="$$($(PY) -c \
	    'import json,sys; d=json.loads(sys.argv[1]); d["compile_only"]=True; d["dbt_fixture_root"]="$(abspath $(DBT_DIR)/fixtures)"; print(json.dumps(d))' \
	    "$$vars")"; \
	  $(DBT) compile $(DBT_PROJECT_ARGS) \
	  --vars "$$compile_vars"
# ---------------------------------------------------------------------------
# Quality contracts and certification
# ---------------------------------------------------------------------------
# Standalone and read-only preview: never run dbt tests outside the coordinated
# warehouse writer gateway. Tool dependencies remain outside the project venv.
DATACONTRACT := $(UV) tool run --python '$(PROJECT_PYTHON)' --from datacontract-cli==1.2.3 datacontract
ODCS_CYCLE_CONTRACT := contracts/odcs/dim_election_cycle.odcs.yaml

data-contracts-lint: ## Validate the ODCS pilot contract without reading warehouse data
	$(DATACONTRACT) lint '$(ODCS_CYCLE_CONTRACT)'

data-contracts-dbt-plan: data-contracts-lint ## Preview non-destructive dbt model synchronization
	$(DATACONTRACT) dbt sync '$(ODCS_CYCLE_CONTRACT)' --project-dir '$(DBT_DIR)' --dry-run --no-prune

# Re-run real synchronization ONLY on a disposable project copy, then diff all
# models byte-for-byte. Dry-run alone does not detect committed schema drift.
.PHONY: data-contracts-dbt-check
data-contracts-dbt-check: data-contracts-dbt-plan ## Assert ODCS/dbt schema has no drift
	@set -eu; \
	  scratch=$$(mktemp -d /tmp/tse-odcs-dbt-check.XXXXXX); \
	  trap 'rm -rf "$$scratch"' EXIT; \
	  mkdir -p "$$scratch/tse_dbt"; \
	  cp '$(DBT_DIR)/dbt_project.yml' "$$scratch/tse_dbt/"; \
	  cp -a '$(DBT_DIR)/models' "$$scratch/tse_dbt/"; \
	  $(DATACONTRACT) dbt sync '$(abspath $(ODCS_CYCLE_CONTRACT))' --project-dir "$$scratch/tse_dbt" --no-prune; \
	  $(PY) scripts/check_odcs_dbt_sync.py \
	    --committed '$(DBT_DIR)/models' --generated "$$scratch/tse_dbt/models"

scope-contracts: setup ## Validate analytical/incremental scope
	$(PY) -m scripts.check_scope_contracts \
	  --years "$(YEARS)" \
	  --election-types "$(SCOPE_TYPES)" \
	  --incremental-years "$(UPDATE_YEARS)" \
	  --incremental-election-types "$(UPDATE_TYPES)"
unit: setup ## Run Python contracts
	$(PY) -m unittest discover -s tests -p 'test_*.py'
raw-contracts: setup ## Validate raw/index contracts
	$(PY) scripts/check_raw_contracts.py \
	  --root '$(RAW_ROOT_ABS)'
cycle-regressions: setup ## Validate election-cycle baselines
	$(PY) scripts/check_cycle_regressions.py \
	  --db '$(DUCKDB_PATH_ABS)' \
	  --contracts '$(abspath $(CYCLE_CONTRACTS))'
medallion-contracts: setup ## Validate medallion DAG
	$(PY) scripts/check_medallion_dag.py
invariant-registry-strict: setup ## Verify all 12 critical graph invariants are implemented
	$(PY) scripts/check_invariant_registry.py --strict

# Requires an already-built DuckDB and a published physical Parquet partition.
# Intentionally separate from production-gate: certification runs expensive SQL
# only for GRAPH_CYCLE and must never silently prepare or overwrite the source.
graph-cycle-certify: invariant-registry-strict ## Issue a cycle-scoped Gold graph certificate
	@test -n '$(GRAPH_CYCLE)' || { \
	  echo 'GRAPH_CYCLE is required (YYYY:general or YYYY:municipal)'; \
	  exit 2; \
	}
	$(PY) scripts/graph_freshness.py \
	  --db '$(DUCKDB_PATH_ABS)' \
	  --source-root '$(CANDIDATE_FACT_ROOT_ABS)' \
	  --attestations-dir '$(GRAPH_ATTESTATIONS_DIR_ABS)' \
	  --cycle '$(GRAPH_CYCLE)' \
	  --memory-limit '$(GRAPH_MEMORY_LIMIT)' \
	  certify
pr-gate: scope-contracts unit medallion-contracts dbt-deps ## Fast PR gate
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --exclude 'tag:expensive' \
	  --vars "$$vars"
production-gate: scope-contracts invariant-registry-strict ## Warehouse checks (graph cycle separate)
	$(MAKE) --no-print-directory \
	  unit \
	  ingest \
	  raw-contracts \
	  medallion-contracts \
	  dbt-build \
	  cycle-regressions
# ---------------------------------------------------------------------------
# Performance and coordinated workflows
# ---------------------------------------------------------------------------
performance-report: setup ## Report latest dbt execution costs
	@test -f '$(DBT_ARTIFACTS_ROOT_ABS)/latest_run_results.json' || \
	  { echo 'Missing coordinated dbt run_results.json'; exit 1; }
	$(PY) scripts/report_build_performance.py \
	  --run-results '$(DBT_ARTIFACTS_ROOT_ABS)/latest_run_results.json' \
	  --output data/warehouse/performance/latest.json
performance-baseline: setup ## Persist latest performance baseline
	@test -f '$(DBT_ARTIFACTS_ROOT_ABS)/latest_run_results.json' || \
	  { echo 'Missing coordinated dbt run_results.json'; exit 1; }
	$(PY) scripts/report_build_performance.py \
	  --run-results '$(DBT_ARTIFACTS_ROOT_ABS)/latest_run_results.json' \
	  --write-baseline contracts/performance/2026-general.json
pipeline: ingest dbt-build ## Incremental ingest through semantic layer
	@echo 'Pipeline complete: $(DUCKDB_PATH_ABS)'
pipeline-backfill: backfill dbt-full-refresh ## Historical backfill + full refresh
	@echo 'Backfill complete: $(DUCKDB_PATH_ABS)'
# ---------------------------------------------------------------------------
# Read-only storage audit (no setup or data writes)
# ---------------------------------------------------------------------------
storage-report: ## Audit local raw/cache/warehouse storage (read-only)
	@$(PYTHON) scripts/report_local_storage.py \
	  --raw-root '$(RAW_ROOT_ABS)' \
	  --duckdb '$(DUCKDB_PATH_ABS)' \
	  --candidate-fact-root '$(CANDIDATE_FACT_ROOT_ABS)' \
	  $(if $(filter 1 true yes,$(VERIFY_HASHES)),--verify-hashes,) \
	  $(if $(filter 1 true yes,$(STRICT)),--strict,) \
	  $(if $(filter json,$(FORMAT)),--json,) \
	  $(if $(filter 1 true yes,$(DIAGNOSE)),--diagnose,) \
	  $(if $(TOP),--top $(TOP),)
# ---------------------------------------------------------------------------
# Disposable cache roundtrip — never evict the canonical raw tree
# ---------------------------------------------------------------------------
.PHONY: cache-roundtrip
cache-roundtrip: ## Certify one active resource using disposable offline prune/rehydrate
	@test -n '$(ROUNDTRIP_RESOURCE_ID)' || { echo 'ROUNDTRIP_RESOURCE_ID is required'; exit 2; }
	@test -n '$(ROUNDTRIP_YEAR)' || { echo 'ROUNDTRIP_YEAR is required'; exit 2; }
	@test -n '$(ROUNDTRIP_DOMAIN)' || { echo 'ROUNDTRIP_DOMAIN is required'; exit 2; }
	@$(PROJECT_PYTHON) scripts/certify_cache_roundtrip.py \
	  --raw-root '$(RAW_ROOT_ABS)' \
	  --resource-id '$(ROUNDTRIP_RESOURCE_ID)' \
	  --year '$(ROUNDTRIP_YEAR)' \
	  --domain '$(ROUNDTRIP_DOMAIN)' \
	  $(if $(ROUNDTRIP_JSON),--output-json '$(ROUNDTRIP_JSON)',)
# ---------------------------------------------------------------------------
# Disposable REAL dbt downstream roundtrip (PR43, no canonical writes)
# ---------------------------------------------------------------------------
.PHONY: dbt-cache-roundtrip
dbt-cache-roundtrip: ## Certify bronze/silver dbt outputs around offline cache recovery
	@$(PROJECT_PYTHON) scripts/certify_dbt_cache_roundtrip.py \
	  --raw-root '$(RAW_ROOT_ABS)' \
	  --year '$(or $(DBT_ROUNDTRIP_YEAR),2026)' \
	  $(if $(DBT_ROUNDTRIP_RESOURCE_ID),--resource-id '$(DBT_ROUNDTRIP_RESOURCE_ID)',) \
	  $(if $(DBT_ROUNDTRIP_MAX_SOURCE_MIB),--max-source-mib '$(DBT_ROUNDTRIP_MAX_SOURCE_MIB)',) \
	  $(if $(DBT_ROUNDTRIP_MAX_EXTRACTED_MIB),--max-extracted-mib '$(DBT_ROUNDTRIP_MAX_EXTRACTED_MIB)',) \
	  $(if $(DBT_ROUNDTRIP_MAX_ROWS),--max-rows '$(DBT_ROUNDTRIP_MAX_ROWS)',) \
	  $(if $(DBT_ROUNDTRIP_JSON),--output-json '$(DBT_ROUNDTRIP_JSON)',)
# ---------------------------------------------------------------------------
# Gold incremental + persisted Semantic recovery (PR44; isolated scratch only)
# ---------------------------------------------------------------------------
.PHONY: gold-semantic-cache-roundtrip
gold-semantic-cache-roundtrip: ## Compare persisted Gold/Semantic under control and offline recovery
	@$(PROJECT_PYTHON) scripts/certify_gold_semantic_roundtrip.py \
	  --raw-root '$(RAW_ROOT_ABS)' \
	  --year '$(or $(GS_ROUNDTRIP_YEAR),2026)' \
	  $(if $(GS_ROUNDTRIP_CANDIDATE_RESOURCE_ID),--candidate-resource-id '$(GS_ROUNDTRIP_CANDIDATE_RESOURCE_ID)',) \
	  $(if $(GS_ROUNDTRIP_ASSETS_RESOURCE_ID),--assets-resource-id '$(GS_ROUNDTRIP_ASSETS_RESOURCE_ID)',) \
	  $(if $(GS_ROUNDTRIP_MAX_SOURCE_MIB),--max-source-mib '$(GS_ROUNDTRIP_MAX_SOURCE_MIB)',) \
	  $(if $(GS_ROUNDTRIP_MAX_EXTRACTED_MIB),--max-extracted-mib '$(GS_ROUNDTRIP_MAX_EXTRACTED_MIB)',) \
	  $(if $(GS_ROUNDTRIP_MAX_ROWS),--max-rows '$(GS_ROUNDTRIP_MAX_ROWS)',) \
	  $(if $(GS_ROUNDTRIP_MIN_FREE_GIB),--min-free-gib '$(GS_ROUNDTRIP_MIN_FREE_GIB)',) \
	  $(if $(GS_ROUNDTRIP_JSON),--output-json '$(GS_ROUNDTRIP_JSON)',)
# ---------------------------------------------------------------------------
# PR45 synthetic physical Parquet publication and consumer recovery (scratch only)
# ---------------------------------------------------------------------------
.PHONY: physical-publication-recovery
physical-publication-recovery: ## Certify SIGKILL-safe Parquet publish + persisted candidate_ranking
	@$(PROJECT_PYTHON) scripts/certify_candidate_fact_recovery.py \
	  $(if $(PHYSICAL_RECOVERY_MAX_ROWS),--max-rows '$(PHYSICAL_RECOVERY_MAX_ROWS)',) \
	  $(if $(PHYSICAL_RECOVERY_MIN_FREE_GIB),--min-free-gib '$(PHYSICAL_RECOVERY_MIN_FREE_GIB)',) \
	  $(if $(PHYSICAL_RECOVERY_JSON),--output-json '$(PHYSICAL_RECOVERY_JSON)',)
# ---------------------------------------------------------------------------
# Storage maintenance (may alter local state)
# ---------------------------------------------------------------------------
prune-cache: setup ## Read-only inventory; destructive cache removal requires safe_cache_evict.py
	@test "$(filter 1 true yes,$(APPLY))" = "" || { echo "BLOCKED: use scripts/safe_cache_evict.py with verified plan/canary"; exit 2; }
	$(PY) scripts/prune_extracted_raw.py \
	  --root '$(RAW_ROOT_ABS)/raw' \
	  $(PRUNE_DOMAIN_ARGS) \
	  $(if $(filter 1 true yes,$(APPLY)),--apply,)
clean: ## Remove generated dbt artifacts
	rm -rf \
	  $(DBT_DIR)/target \
	  $(DBT_DIR)/dbt_packages \
	  $(DBT_DIR)/logs
clean-warehouse: ## Delete DuckDB under the warehouse lock and rotate proof epoch
	$(PYTHON) scripts/warehouse_lifecycle.py warehouse \
	  --db '$(DUCKDB_PATH_ABS)' \
	  --project-root '$(PROJECT_ROOT_ABS)'
clean-all: ## DESTRUCTIVE: coordinated local cleanup (CONFIRM=1)
	@test "$(CONFIRM)" = "1" || { \
	  echo 'Refusing clean-all without CONFIRM=1'; \
	  exit 2; \
	}
	$(PYTHON) scripts/warehouse_lifecycle.py all --confirm \
	  --db '$(DUCKDB_PATH_ABS)' \
	  --project-root '$(PROJECT_ROOT_ABS)' \
	  --raw-root '$(RAW_ROOT_ABS)' \
	  --fact-root '$(CANDIDATE_FACT_ROOT_ABS)' \
	  --artifacts-root '$(DBT_ARTIFACTS_ROOT_ABS)' \
	  --dbt-dir '$(abspath $(DBT_DIR))' \
	  --venv '$(abspath .venv)'
