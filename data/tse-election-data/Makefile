SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.NOTPARALLEL: pipeline pipeline-backfill

UV ?= uv
PYTHON ?= python3
PROJECT_PYTHON ?= .venv/bin/python
DBT_DIR ?= tse_dbt
DATA_ROOT ?= ./data/tse
DUCKDB_PATH ?= ./data/warehouse/tse_analytics.duckdb
CANDIDATE_FACT_ROOT ?= ./data/warehouse/fact_candidate_votes

# ----------------------------------------------------------------------
# Warehouse scope
# ----------------------------------------------------------------------
# Cycles that dbt models should expose/process for this invocation.
YEARS ?= 2018
ELECTION_TYPES ?= general

# Cycles expected to have changed in this run.
INCREMENTAL_YEARS ?= 2018
INCREMENTAL_ELECTION_TYPES ?= general

PROFILE ?= analytics
GRANULARITY ?= brasil
UF ?=

# ----------------------------------------------------------------------
# Canonical absolute paths
# ----------------------------------------------------------------------
RAW_ROOT_ABS := $(abspath $(DATA_ROOT))
DUCKDB_PATH_ABS := $(abspath $(DUCKDB_PATH))
CANDIDATE_FACT_ROOT_ABS := $(abspath $(CANDIDATE_FACT_ROOT))

# ----------------------------------------------------------------------
# Executables
# ----------------------------------------------------------------------
INGEST := env -u VIRTUAL_ENV $(UV) run --python '$(PROJECT_PYTHON)' python tse_ingest.py

DBT := \
	TSE_DUCKDB_PATH='$(DUCKDB_PATH_ABS)' \
	env -u VIRTUAL_ENV \
	$(UV) run --python '$(PROJECT_PYTHON)' \
	dbt \
	--project-dir $(DBT_DIR) \
	--profiles-dir $(DBT_DIR)

INGEST_WORKERS ?= 8
CHECKPOINT_EVERY ?= 10

# ----------------------------------------------------------------------
# Ingestion arguments
# ----------------------------------------------------------------------
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

BACKFILL_YEAR_ARGS := $(foreach year,$(YEARS),--year $(year))
INCREMENTAL_YEAR_ARGS := $(foreach year,$(INCREMENTAL_YEARS),--year $(year))

# ----------------------------------------------------------------------
# dbt vars
# ----------------------------------------------------------------------
# Keep JSON generation in one place.
#
# This is deliberately a shell command rather than an eagerly expanded
# Make variable. Targets evaluate it at execution time and receive one
# valid JSON token for dbt --vars.
#
# build_candidate_fact_partition is intentionally NOT present here.
# The expensive physical partition builder is opt-in through the
# candidate-fact-partition target only.
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

# ----------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------
.PHONY: \
	help \
	doctor \
	setup \
	dbt-deps \
	print-dbt-vars \
	list \
	ingest \
	backfill \
	candidate-fact-partition \
	dbt-build \
	dbt-full-refresh \
	semantic \
	test \
	test-only \
	compile \
	pipeline \
	pipeline-backfill \
	clean \
	clean-warehouse \
	clean-all

help: ## Show available targets and configuration knobs
	@printf '%s\n' \
	  'TSE local analytics pipeline' \
	  '' \
	  'Main targets:' \
	  '  make pipeline                  Incremental ingestion -> dbt build' \
	  '  make pipeline-backfill         Historical backfill -> dbt full-refresh' \
	  '  make ingest                    Incremental raw ingestion only' \
	  '  make backfill                  Raw historical backfill only' \
	  '  make candidate-fact-partition  Build/replace one candidate fact partition' \
	  '  make dbt-build                 Transform/test all normal dbt layers' \
	  '  make semantic                  Build semantic models and dependencies' \
	  '  make test                      Ingest current data and run dbt build/tests' \
	  '  make test-only                 Run tests against existing warehouse' \
	  '  make compile                   Compile dbt using schema fixtures' \
	  '  make print-dbt-vars            Print JSON supplied to normal dbt builds' \
	  '' \
	  'Configuration examples:' \
	  '  make pipeline INCREMENTAL_YEARS="2026"' \
	  '  make pipeline-backfill YEARS="2020 2022 2024 2026"' \
	  '  make ingest GRANULARITY=uf UF=GO' \
	  '  make pipeline PROFILE=extended' \
	  '  make candidate-fact-partition INCREMENTAL_YEARS="2022" INCREMENTAL_ELECTION_TYPES="general"' \
	  '' \
	  'Defaults:' \
	  '  YEARS=$(YEARS)' \
	  '  ELECTION_TYPES=$(ELECTION_TYPES)' \
	  '  INCREMENTAL_YEARS=$(INCREMENTAL_YEARS)' \
	  '  INCREMENTAL_ELECTION_TYPES=$(INCREMENTAL_ELECTION_TYPES)' \
	  '  DATA_ROOT=$(DATA_ROOT)' \
	  '  DUCKDB_PATH=$(DUCKDB_PATH)' \
	  '  CANDIDATE_FACT_ROOT=$(CANDIDATE_FACT_ROOT)'

doctor: ## Check local prerequisites
	@command -v $(UV) >/dev/null || { echo 'Missing uv'; exit 1; }
	@command -v $(PYTHON) >/dev/null || { echo 'Missing $(PYTHON)'; exit 1; }
	@test -x '$(PROJECT_PYTHON)' || { echo 'Missing project Python: $(PROJECT_PYTHON)'; exit 1; }
	@echo 'uv:                  ' "$$(command -v $(UV))"
	@echo 'python:              ' "$$(command -v $(PYTHON))"
	@echo 'project python:      ' '$(PROJECT_PYTHON)'
	@echo 'raw:                 ' '$(RAW_ROOT_ABS)'
	@echo 'duckdb:              ' '$(DUCKDB_PATH_ABS)'
	@echo 'candidate fact root: ' '$(CANDIDATE_FACT_ROOT_ABS)'

setup: doctor ## Create local environment and install ingestion/dbt dependencies
	@mkdir -p '$(dir $(DUCKDB_PATH_ABS))'
	@mkdir -p '$(CANDIDATE_FACT_ROOT_ABS)'
	@test -d .venv || env -u VIRTUAL_ENV $(UV) venv .venv
	@if test -f pyproject.toml; then \
	  env -u VIRTUAL_ENV $(UV) sync; \
	else \
	  env -u VIRTUAL_ENV $(UV) pip install \
	    --python '$(PROJECT_PYTHON)' \
	    -r requirements-tse-ingest.txt \
	    -r $(DBT_DIR)/requirements.txt; \
	fi

# dbt_utils is intentionally fetched separately from Python dependencies.
dbt-deps: setup ## Install dbt package dependencies
	$(DBT) deps

print-dbt-vars: ## Print normal dbt --vars JSON
	@$(DBT_VARS_CMD)

list: setup ## Show what incremental ingestion would select without downloading
	$(INGEST) \
	  --mode incremental \
	  $(INCREMENTAL_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS) \
	  --list-only

ingest: setup ## Incrementally ingest only changed/new resources
	$(INGEST) \
	  --mode incremental \
	  $(INCREMENTAL_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS)

backfill: setup ## Ingest all missing immutable resource versions for YEARS
	$(INGEST) \
	  --mode backfill \
	  $(BACKFILL_YEAR_ARGS) \
	  $(INGEST_COMMON_ARGS)

# ----------------------------------------------------------------------
# Candidate fact physical partition replacement
# ----------------------------------------------------------------------
#
# Contract:
#   - exactly one year
#   - exactly one election type
#   - build_candidate_fact_partition=true only here
#
# int_candidate_fact_partition should have:
#
#   enabled=var('build_candidate_fact_partition', false)
#
# so ordinary dbt builds never execute its expensive CTAS.
candidate-fact-partition: dbt-deps ## Build and atomically replace one candidate fact partition
	@test "$(words $(INCREMENTAL_YEARS))" -eq 1 || \
	  { echo "candidate-fact-partition requires exactly one incremental year"; exit 1; }
	@test "$(words $(INCREMENTAL_ELECTION_TYPES))" -eq 1 || \
	  { echo "candidate-fact-partition requires exactly one incremental election type"; exit 1; }

	@echo "Building candidate fact partition:"
	@echo "  year:          $(INCREMENTAL_YEARS)"
	@echo "  election type: $(INCREMENTAL_ELECTION_TYPES)"
	@echo "  root:          $(CANDIDATE_FACT_ROOT_ABS)"

	$(DBT) run \
	  --select +int_candidate_fact_partition \
	  --vars '{"tse_raw_root":"$(RAW_ROOT_ABS)","candidate_fact_root":"$(CANDIDATE_FACT_ROOT_ABS)","election_years":[$(INCREMENTAL_YEARS)],"election_types":["$(INCREMENTAL_ELECTION_TYPES)"],"incremental_years":[$(INCREMENTAL_YEARS)],"incremental_election_types":["$(INCREMENTAL_ELECTION_TYPES)"],"use_prepared_candidate_votes":true,"build_candidate_fact_partition":true}'

	env -u VIRTUAL_ENV $(UV) run --python '$(PROJECT_PYTHON)' \
	  python scripts/write_candidate_fact_partition.py \
	  --db '$(DUCKDB_PATH_ABS)' \
	  --root '$(CANDIDATE_FACT_ROOT_ABS)' \
	  --year '$(INCREMENTAL_YEARS)' \
	  --election-type '$(INCREMENTAL_ELECTION_TYPES)'

dbt-build: dbt-deps ## Build staging -> intermediate -> marts -> semantic and run tests
	@vars="$$($(DBT_VARS_CMD))"; \
	  echo "dbt vars: $$vars"; \
	  $(DBT) build --vars "$$vars"

dbt-full-refresh: dbt-deps ## Full-refresh dbt incremental marts and semantic layer
	@vars="$$($(DBT_VARS_CMD))"; \
	  echo "dbt vars: $$vars"; \
	  $(DBT) build --full-refresh --vars "$$vars"

semantic: dbt-deps ## Build semantic models plus upstream dependencies/tests
	@vars="$$($(DBT_VARS_CMD))"; \
	  echo "dbt vars: $$vars"; \
	  $(DBT) build \
	    --select '+path:models/semantic' \
	    --vars "$$vars"

test: ingest dbt-deps ## Self-contained validation: ingest current data and run dbt build/tests
	@vars="$$($(DBT_VARS_CMD))"; \
	  echo "dbt vars: $$vars"; \
	  $(DBT) build --vars "$$vars"

test-only: dbt-deps ## Run tests only against an already-built warehouse
	@vars="$$($(DBT_VARS_CMD))"; \
	  echo "dbt vars: $$vars"; \
	  $(DBT) test --vars "$$vars"

compile: dbt-deps ## Compile entire dbt graph using schema fixtures
	@vars="$$($(DBT_VARS_CMD))"; \
	  compile_vars="$$($(PYTHON) -c \
	    'import json,sys; d=json.loads(sys.argv[1]); d["compile_only"]=True; d["dbt_fixture_root"]="$(abspath $(DBT_DIR)/fixtures)"; print(json.dumps(d))' \
	    "$$vars")"; \
	  echo "dbt compile vars: $$compile_vars"; \
	  $(DBT) compile --vars "$$compile_vars"

pipeline: ingest dbt-build ## End-to-end incremental pipeline through semantic layer
	@echo
	@echo 'Pipeline complete.'
	@echo 'Semantic DuckDB: $(DUCKDB_PATH_ABS)'

pipeline-backfill: backfill dbt-full-refresh ## Historical load followed by dbt full refresh
	@echo
	@echo 'Backfill pipeline complete.'
	@echo 'Semantic DuckDB: $(DUCKDB_PATH_ABS)'

clean: ## Remove dbt generated artifacts, preserving raw data and warehouse
	rm -rf \
	  $(DBT_DIR)/target \
	  $(DBT_DIR)/dbt_packages \
	  $(DBT_DIR)/logs

clean-warehouse: ## Remove generated DuckDB warehouse only
	rm -f '$(DUCKDB_PATH_ABS)'

clean-all: ## DESTRUCTIVE: remove raw data, warehouse, candidate fact partitions, dbt artifacts and venv
	rm -rf \
	  '$(RAW_ROOT_ABS)' \
	  '$(DUCKDB_PATH_ABS)' \
	  '$(CANDIDATE_FACT_ROOT_ABS)' \
	  $(DBT_DIR)/target \
	  $(DBT_DIR)/dbt_packages \
	  $(DBT_DIR)/logs \
	  .venv

