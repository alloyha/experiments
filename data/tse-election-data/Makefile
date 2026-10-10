SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# DuckDB and the project environment are shared mutable state.
# Parallelism belongs inside ingestion/dbt, not between make targets.
.NOTPARALLEL:

UV ?= uv
PYTHON ?= python3
PROJECT_PYTHON ?= .venv/bin/python

DBT_DIR ?= tse_dbt
DATA_ROOT ?= ./data/tse
DUCKDB_PATH ?= ./data/warehouse/tse_analytics.duckdb
CANDIDATE_FACT_ROOT ?= ./data/warehouse/fact_candidate_votes
CYCLE_CONTRACTS ?= ./contracts/election_cycles.json

YEARS ?= 2018
ELECTION_TYPES ?= general
INCREMENTAL_YEARS ?= 2018
INCREMENTAL_ELECTION_TYPES ?= general

PROFILE ?= analytics
GRANULARITY ?= brasil
UF ?=

INGEST_WORKERS ?= 8
CHECKPOINT_EVERY ?= 10

PROJECT_ROOT_ABS := $(abspath .)
RAW_ROOT_ABS := $(abspath $(DATA_ROOT))
DUCKDB_PATH_ABS := $(abspath $(DUCKDB_PATH))
CANDIDATE_FACT_ROOT_ABS := $(abspath $(CANDIDATE_FACT_ROOT))

ENV_STAMP := .venv/.tse-ready
DBT_DEPS_STAMP := $(DBT_DIR)/dbt_packages/.tse-deps-ready

RUN := env -u VIRTUAL_ENV $(UV) run --python '$(PROJECT_PYTHON)'
PY := $(RUN) python
INGEST := $(PY) tse_ingest.py

DBT := \
	TSE_DUCKDB_PATH='$(DUCKDB_PATH_ABS)' \
	$(RUN) dbt

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

INCREMENTAL_YEAR_ARGS := \
	$(foreach year,$(INCREMENTAL_YEARS),--year $(year))

DBT_VARS_CMD = \
	RAW_ROOT='$(RAW_ROOT_ABS)' \
	CANDIDATE_FACT_ROOT='$(CANDIDATE_FACT_ROOT_ABS)' \
	YEARS='$(YEARS)' \
	ELECTION_TYPES='$(ELECTION_TYPES)' \
	INCREMENTAL_YEARS='$(INCREMENTAL_YEARS)' \
	INCREMENTAL_ELECTION_TYPES='$(INCREMENTAL_ELECTION_TYPES)' \
	$(PYTHON) -c 'import json, os; print(json.dumps({ \
		"tse_raw_root": os.environ["RAW_ROOT"], \
		"candidate_fact_root": os.environ["CANDIDATE_FACT_ROOT"], \
		"election_years": [int(x) for x in os.environ["YEARS"].split()], \
		"election_types": os.environ["ELECTION_TYPES"].split(), \
		"incremental_years": [int(x) for x in os.environ["INCREMENTAL_YEARS"].split()], \
		"incremental_election_types": os.environ["INCREMENTAL_ELECTION_TYPES"].split() \
	}))'

.PHONY: \
	help doctor setup dbt-deps scope-contracts \
	print-dbt-vars list ingest backfill \
	candidate-fact-partition unit raw-contracts \
	cycle-regressions medallion-contracts \
	dbt-bronze dbt-silver dbt-gold dbt-semantic-layer \
	prune-cache pr-gate production-gate \
	dbt-build dbt-full-refresh semantic \
	test test-only compile \
	performance-report performance-baseline \
	pipeline pipeline-backfill \
	clean clean-warehouse clean-all

help: ## Show targets
	@awk 'BEGIN {FS = ":.*## "; printf "TSE analytics\n\n"} \
	  /^[a-zA-Z0-9_.-]+:.*## / \
	  {printf "  %-26s %s\n", $$1, $$2}' \
	  $(MAKEFILE_LIST)
	@printf '\nDefaults:\n'
	@printf '  YEARS=%s\n' '$(YEARS)'
	@printf '  ELECTION_TYPES=%s\n' '$(ELECTION_TYPES)'
	@printf '  INCREMENTAL_YEARS=%s\n' '$(INCREMENTAL_YEARS)'
	@printf '  INCREMENTAL_ELECTION_TYPES=%s\n' \
	  '$(INCREMENTAL_ELECTION_TYPES)'

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

scope-contracts: setup ## Validate analytical/incremental scope
	$(PY) -m scripts.check_scope_contracts \
	  --years "$(YEARS)" \
	  --election-types "$(ELECTION_TYPES)" \
	  --incremental-years "$(INCREMENTAL_YEARS)" \
	  --incremental-election-types "$(INCREMENTAL_ELECTION_TYPES)"

print-dbt-vars: ## Print dbt vars JSON
	@$(DBT_VARS_CMD)

list: setup ## List incremental ingestion selection
	$(INGEST) \
	  --mode incremental \
	  $(INCREMENTAL_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS) \
	  --list-only

ingest: scope-contracts ## Incrementally ingest changed resources
	$(INGEST) \
	  --mode incremental \
	  $(INCREMENTAL_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS)

backfill: setup ## Backfill immutable resource versions
	$(INGEST) \
	  --mode backfill \
	  $(BACKFILL_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS)

candidate-fact-partition: dbt-deps ## Build one physical candidate partition
	@test "$(words $(INCREMENTAL_YEARS))" -eq 1 || \
	  { echo "requires exactly one incremental year"; exit 1; }
	@test "$(words $(INCREMENTAL_ELECTION_TYPES))" -eq 1 || \
	  { echo "requires exactly one incremental election type"; exit 1; }

	$(DBT) run $(DBT_PROJECT_ARGS) \
	  --select +int_candidate_fact_partition \
	  --vars '{"tse_raw_root":"$(RAW_ROOT_ABS)","candidate_fact_root":"$(CANDIDATE_FACT_ROOT_ABS)","election_years":[$(INCREMENTAL_YEARS)],"election_types":["$(INCREMENTAL_ELECTION_TYPES)"],"incremental_years":[$(INCREMENTAL_YEARS)],"incremental_election_types":["$(INCREMENTAL_ELECTION_TYPES)"],"use_prepared_candidate_votes":true,"build_candidate_fact_partition":true}'

	$(PY) scripts/write_candidate_fact_partition.py \
	  --db '$(DUCKDB_PATH_ABS)' \
	  --root '$(CANDIDATE_FACT_ROOT_ABS)' \
	  --year '$(INCREMENTAL_YEARS)' \
	  --election-type '$(INCREMENTAL_ELECTION_TYPES)'

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

dbt-semantic-layer: semantic ## Compatibility alias for semantic

prune-cache: setup ## Dry-run raw cache pruning; APPLY=1 deletes
	$(PY) scripts/prune_extracted_raw.py \
	  --root '$(RAW_ROOT_ABS)/raw' \
	  $(if $(filter 1 true yes,$(APPLY)),--apply,)

pr-gate: scope-contracts unit medallion-contracts dbt-deps ## Fast PR gate
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
	  --exclude 'tag:expensive' \
	  --vars "$$vars"

production-gate: scope-contracts ## Full production certification
	$(MAKE) --no-print-directory \
	  unit \
	  ingest \
	  raw-contracts \
	  medallion-contracts \
	  dbt-build \
	  cycle-regressions

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

test: ingest dbt-deps ## Ingest and run dbt build/tests
	@vars="$$($(DBT_VARS_CMD))"; \
	  $(DBT) build $(DBT_PROJECT_ARGS) \
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

performance-report: setup ## Report latest dbt execution costs
	@test -f '$(DBT_DIR)/target/run_results.json' || \
	  { echo 'Missing dbt run_results.json'; exit 1; }
	$(PY) scripts/report_build_performance.py \
	  --run-results $(DBT_DIR)/target/run_results.json \
	  --output data/warehouse/performance/latest.json

performance-baseline: setup ## Persist latest performance baseline
	@test -f '$(DBT_DIR)/target/run_results.json' || \
	  { echo 'Missing dbt run_results.json'; exit 1; }
	$(PY) scripts/report_build_performance.py \
	  --run-results $(DBT_DIR)/target/run_results.json \
	  --write-baseline contracts/performance/2026-general.json

pipeline: ingest dbt-build ## Incremental ingest through semantic layer
	@echo 'Pipeline complete: $(DUCKDB_PATH_ABS)'

pipeline-backfill: backfill dbt-full-refresh ## Historical backfill + full refresh
	@echo 'Backfill complete: $(DUCKDB_PATH_ABS)'

clean: ## Remove generated dbt artifacts
	rm -rf \
	  $(DBT_DIR)/target \
	  $(DBT_DIR)/dbt_packages \
	  $(DBT_DIR)/logs

clean-warehouse: ## Remove generated DuckDB warehouse
	rm -f '$(DUCKDB_PATH_ABS)'
	rm -rf '$(DUCKDB_PATH_ABS).tmp'

clean-all: ## DESTRUCTIVE: remove all generated local state (CONFIRM=1)
	@test "$(CONFIRM)" = "1" || { \
	  echo 'Refusing clean-all without CONFIRM=1'; \
	  exit 2; \
	}
	@root='$(PROJECT_ROOT_ABS)'; \
	for path in \
	  '$(RAW_ROOT_ABS)' \
	  '$(DUCKDB_PATH_ABS)' \
	  '$(CANDIDATE_FACT_ROOT_ABS)'; do \
	  case "$$path" in \
	    "$$root"/*) ;; \
	    *) echo "Refusing path outside project: $$path"; exit 2 ;; \
	  esac; \
	done
	rm -rf \
	  '$(RAW_ROOT_ABS)' \
	  '$(DUCKDB_PATH_ABS)' \
	  '$(CANDIDATE_FACT_ROOT_ABS)' \
	  $(DBT_DIR)/target \
	  $(DBT_DIR)/dbt_packages \
	  $(DBT_DIR)/logs \
	  .venv
