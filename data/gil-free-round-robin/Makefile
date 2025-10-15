# Makefile for Python 3.14 Free-Threading Benchmark Environment
# Fully parameterized for multi-core testing

# Default parameters (can be overridden)
NUM_CONSUMERS ?= 4
NUM_ITEMS ?= 10000
WORK_ITERATIONS ?= 1000
CPUS ?= 4
MAX_CONSUMERS ?= 16
MEMORY ?= 2g
STRATEGY ?= workstealing
STRATEGIES ?= static workstealing central sharded dynamic throttled dropoldest creditbased adaptive pushpull

.PHONY: help build run-nogil run-gil run-both compare dev scaling stop clean compare-metrics compare-all-strategies compare-all-phase

help: ## Show this help message
	@echo "Python 3.14 Free-Threading Benchmark Commands"
	@echo "=============================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make run-nogil              - Run with default settings (4 cores, 10k items)"
	@echo "  make scaling                - Run automatic scaling test"
	@echo "  make compare                - Compare GIL enabled vs disabled"
	@echo ""
	@echo "Parameterized Runs:"
	@echo "  make run-nogil CPUS=8 NUM_CONSUMERS=8 NUM_ITEMS=20000"
	@echo "  make scaling MAX_CONSUMERS=16 NUM_ITEMS=50000"
	@echo ""
	@echo "Available Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Parameters:"
	@echo "  NUM_CONSUMERS    - Number of consumer threads (default: 4)"
	@echo "  NUM_ITEMS        - Number of items to process (default: 10000)"
	@echo "  WORK_ITERATIONS  - Work iterations per item (default: 1000)"
	@echo "  CPUS             - CPU limit for container (default: 4)"
	@echo "  MAX_CONSUMERS    - Max consumers for scaling test (default: 16)"
	@echo "  MEMORY           - Memory limit (default: 2g)"
	@echo "  STRATEGY         - Strategy to test (default: workstealing)"
	@echo ""
	@echo "Examples:"
	@echo "  make run-nogil NUM_CONSUMERS=8 CPUS=8"
	@echo "  make scaling MAX_CONSUMERS=32 NUM_ITEMS=50000"
	@echo "  make test-2core"
	@echo "  make test-all-cores"

build: ## Build custom Python 3.14 image from source
	docker build -t python314-nogil:latest -f Dockerfile .

run-nogil: ## Run benchmark with GIL disabled
	@echo "🚀 Running with GIL DISABLED (free-threading)"
	@echo "   Consumers: $(NUM_CONSUMERS), Items: $(NUM_ITEMS), Work: $(WORK_ITERATIONS), CPUs: $(CPUS)"
	NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) CPUS=$(CPUS) MEMORY=$(MEMORY) \
	docker-compose run --rm python314-nogil

run-gil: ## Run benchmark with GIL enabled (comparison)
	@echo "🐌 Running with GIL ENABLED (traditional)"
	@echo "   Consumers: $(NUM_CONSUMERS), Items: $(NUM_ITEMS), Work: $(WORK_ITERATIONS), CPUs: $(CPUS)"
	NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) CPUS=$(CPUS) MEMORY=$(MEMORY) \
	docker-compose run --rm python314-gil

run-both: ## Run both GIL disabled and enabled for comparison
	@echo "=== Running with GIL DISABLED ==="
	@$(MAKE) run-nogil
	@echo ""
	@echo "=== Running with GIL ENABLED ==="
	@$(MAKE) run-gil

compare: ## Run comprehensive comparison with timing
	@echo "📊 Comprehensive GIL Comparison"
	@echo "================================"
	@echo "Parameters: Consumers=$(NUM_CONSUMERS), Items=$(NUM_ITEMS), CPUs=$(CPUS)"
	@echo ""
	@$(MAKE) run-nogil > results_nogil.txt 2>&1
	@$(MAKE) run-gil > results_gil.txt 2>&1
	@echo ""
	@echo "✓ Results saved to results_nogil.txt and results_gil.txt"
	@echo ""
	@echo "=== NO-GIL Results ==="
	@tail -20 results_nogil.txt
	@echo ""
	@echo "=== GIL Results ==="
	@tail -20 results_gil.txt

scaling: ## Run automatic scaling test
	@echo "📈 Running Scaling Test"
	@echo "   Max Consumers: $(MAX_CONSUMERS), Items: $(NUM_ITEMS)"
	NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) MAX_CONSUMERS=$(MAX_CONSUMERS) MEMORY=$(MEMORY) \
	docker-compose run --rm python314-scaling

dev: ## Start interactive development container
	CPUS=$(CPUS) MEMORY=$(MEMORY) docker-compose run --rm python314-dev

strategy: ## Run specific strategy test
	@echo "🎯 Testing Strategy: $(STRATEGY)"
	NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) CPUS=$(CPUS) MEMORY=$(MEMORY) STRATEGY=$(STRATEGY) \
	docker-compose run --rm python314-strategy

# Individual core count tests
test-2core: ## Test with 2 cores
	@NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) MEMORY=$(MEMORY) docker-compose run --rm python314-2core

test-4core: ## Test with 4 cores
	@NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) MEMORY=$(MEMORY) docker-compose run --rm python314-4core

test-8core: ## Test with 8 cores
	@NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) MEMORY=$(MEMORY) docker-compose run --rm python314-8core

test-16core: ## Test with 16 cores
	@NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) MEMORY=$(MEMORY) docker-compose run --rm python314-16core

test-all-cores: ## Test all core configurations sequentially
	@echo "🔬 Testing All Core Configurations"
	@NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) MEMORY=$(MEMORY) docker-compose run --rm python314-multicore

# Advanced parameterized tests
stress-test: ## Run intensive stress test
	@echo "💪 Running Stress Test"
	@$(MAKE) run-nogil NUM_CONSUMERS=16 NUM_ITEMS=100000 WORK_ITERATIONS=2000 CPUS=16 MEMORY=4g

quick-test: ## Quick test with minimal parameters
	@echo "⚡ Quick Test"
	@$(MAKE) run-nogil NUM_CONSUMERS=2 NUM_ITEMS=1000 WORK_ITERATIONS=500 CPUS=2

benchmark-suite: ## Run comprehensive benchmark suite
	@echo "📊 Running Comprehensive Benchmark Suite"
	@echo "=========================================="
	@echo ""
	@echo "Test 1: Small workload (2 cores, 5k items)"
	@$(MAKE) run-nogil NUM_CONSUMERS=2 NUM_ITEMS=5000 CPUS=2 2>&1 | tee -a benchmark_suite.log
	@echo ""
	@echo "Test 2: Medium workload (4 cores, 10k items)"
	@$(MAKE) run-nogil NUM_CONSUMERS=4 NUM_ITEMS=10000 CPUS=4 2>&1 | tee -a benchmark_suite.log
	@echo ""
	@echo "Test 3: Large workload (8 cores, 20k items)"
	@$(MAKE) run-nogil NUM_CONSUMERS=8 NUM_ITEMS=20000 CPUS=8 2>&1 | tee -a benchmark_suite.log
	@echo ""
	@echo "Test 4: Extreme workload (16 cores, 50k items)"
	@$(MAKE) run-nogil NUM_CONSUMERS=16 NUM_ITEMS=50000 CPUS=16 MEMORY=4g 2>&1 | tee -a benchmark_suite.log
	@echo ""
	@echo "✓ Full results saved to benchmark_suite.log"

# Strategy comparison
compare-strategies: ## Compare all strategies with current parameters
	@echo "🔍 Comparing All Strategies"
	@echo "Consumers: $(NUM_CONSUMERS), Items: $(NUM_ITEMS), CPUs: $(CPUS)"
	@echo ""
	@for strategy in static workstealing central sharded dynamic; do \
		echo "Testing $strategy..."; \
		$(MAKE) strategy STRATEGY=$strategy 2>&1 | grep -A 3 "Winner:"; \
		echo ""; \
	done

# Scaling analysis
scaling-analysis: ## Run detailed scaling analysis with multiple workloads
	@echo "📈 Detailed Scaling Analysis"
	@echo "=============================="
	@echo ""
	@echo "Light workload (5k items):"
	@$(MAKE) scaling NUM_ITEMS=5000 WORK_ITERATIONS=500 MAX_CONSUMERS=$(MAX_CONSUMERS)
	@echo ""
	@echo "Medium workload (10k items):"
	@$(MAKE) scaling NUM_ITEMS=10000 WORK_ITERATIONS=1000 MAX_CONSUMERS=$(MAX_CONSUMERS)
	@echo ""
	@echo "Heavy workload (25k items):"
	@$(MAKE) scaling NUM_ITEMS=25000 WORK_ITERATIONS=1500 MAX_CONSUMERS=$(MAX_CONSUMERS)

stop: ## Stop all running containers
	docker-compose down

clean: ## Remove containers and images
	docker-compose down -v --rmi all

test: ## Quick test to verify Python 3.14
	@docker-compose run --rm python314-nogil python -c "import sys; print(f'Python {sys.version}'); print(f'GIL enabled: {sys._is_gil_enabled() if hasattr(sys, \"_is_gil_enabled\") else \"Unknown\"}')"

logs: ## Show logs from last run
	docker-compose logs

shell: ## Open bash shell in container
	CPUS=$(CPUS) MEMORY=$(MEMORY) docker-compose run --rm python314-dev bash

profile: ## Run with profiling enabled
	NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) CPUS=$(CPUS) MEMORY=$(MEMORY) \
	docker-compose run --rm python314-nogil python -m cProfile -o profile.stats gil_roundrobin_bench.py --consumers $(NUM_CONSUMERS) --items $(NUM_ITEMS) --work $(WORK_ITERATIONS)

# Result analysis
show-results: ## Display recent benchmark results
	@if [ -f results_nogil.txt ]; then \
		echo "=== Latest NO-GIL Results ==="; \
		tail -30 results_nogil.txt; \
		echo ""; \
	fi
	@if [ -f results_gil.txt ]; then \
		echo "=== Latest GIL Results ==="; \
		tail -30 results_gil.txt; \
	fi

clean-results: ## Remove all result files
	rm -f results_*.txt benchmark_suite.log profile.stats

# Docker stats monitoring
stats: ## Show real-time Docker container stats
	@echo "Press Ctrl+C to stop monitoring"
	docker stats

# Environment info
info: ## Show environment information
	@echo "Environment Configuration"
	@echo "========================"
	@echo "NUM_CONSUMERS:   $(NUM_CONSUMERS)"
	@echo "NUM_ITEMS:       $(NUM_ITEMS)"
	@echo "WORK_ITERATIONS: $(WORK_ITERATIONS)"
	@echo "CPUS:            $(CPUS)"
	@echo "MAX_CONSUMERS:   $(MAX_CONSUMERS)"
	@echo "MEMORY:          $(MEMORY)"
	@echo "STRATEGY:        $(STRATEGY)"
	@echo ""
	@echo "Docker Info:"
	@docker info | grep -E "CPUs|Total Memory"

# Complete workflow
full-benchmark: ## Run complete benchmark workflow
	@echo "🚀 Running Complete Benchmark Workflow"
	@echo "======================================"
	@$(MAKE) clean-results
	@$(MAKE) test
	@$(MAKE) compare
	@$(MAKE) scaling
	@$(MAKE) test-all-cores
	@echo ""
	@echo "✓ Complete benchmark finished!"
	@echo "Check results_*.txt for detailed output"

compare-metrics: ## Run benchmark for GIL and NO-GIL, then compare metrics (outputs in results/)
	@echo "📊 Running compare-metrics (Consumers=$(NUM_CONSUMERS), Items=$(NUM_ITEMS), Work=$(WORK_ITERATIONS), Strategy=$(STRATEGY))"
	@mkdir -p results
	@echo "→ Running NO-GIL..."
	NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) STRATEGY=$(STRATEGY) CPUS=$(CPUS) \
	docker-compose run --rm python314-nogil \
	python gil_roundrobin_bench.py --consumers $(NUM_CONSUMERS) --items $(NUM_ITEMS) --work $(WORK_ITERATIONS) --strategies $(STRATEGY) --json-output /workspace/results/nogil.json \
	> results_nogil.txt 2>&1 || ( echo "NO-GIL run failed — check results_nogil.txt"; exit 1 )
	@echo "→ Running GIL..."
	NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) STRATEGY=$(STRATEGY) CPUS=$(CPUS) \
	docker-compose run --rm python314-gil \
	python gil_roundrobin_bench.py --consumers $(NUM_CONSUMERS) --items $(NUM_ITEMS) --work $(WORK_ITERATIONS) --strategies $(STRATEGY) --json-output /workspace/results/gil.json \
	> results_gil.txt 2>&1 || ( echo "GIL run failed — check results_gil.txt"; exit 1 )
	@echo "✓ Both runs finished. Parsing and comparing results..."
	@python3 tools/compare_metrics.py results/nogil.json results/gil.json --out results/compare.csv
	@echo ""
	@echo "SUMMARY (tail of raw outputs):"
	@echo "---- NO-GIL (results_nogil.txt) ----"
	@tail -n 12 results_nogil.txt || true
	@echo "---- GIL (results_gil.txt) ----"
	@tail -n 12 results_gil.txt || true
	@echo ""
	@echo "Comparison saved to results/compare.csv"

compare-all-strategies: ## Run every strategy for NO-GIL and GIL, then aggregate results
	@echo "🔁 Running all strategies: $(STRATEGIES)"
	@mkdir -p results
	@for strat in $(STRATEGIES); do \
	  echo "=== Strategy: $$strat (NO-GIL) ==="; \
	  NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) CPUS=$(CPUS) \
	    docker-compose run --rm python314-nogil \
	    python gil_roundrobin_bench.py --consumers $(NUM_CONSUMERS) --items $(NUM_ITEMS) --work $(WORK_ITERATIONS) --strategies $$strat --json-output /workspace/results/$${strat}_nogil.json \
	    > results/$${strat}_nogil.txt 2>&1 || ( echo "NO-GIL run failed for $$strat — see results/$${strat}_nogil.txt"; exit 1 ); \
	  sleep 1; \
	  echo "=== Strategy: $$strat (GIL) ==="; \
	  NUM_CONSUMERS=$(NUM_CONSUMERS) NUM_ITEMS=$(NUM_ITEMS) WORK_ITERATIONS=$(WORK_ITERATIONS) CPUS=$(CPUS) \
	    docker-compose run --rm python314-gil \
	    python gil_roundrobin_bench.py --consumers $(NUM_CONSUMERS) --items $(NUM_ITEMS) --work $(WORK_ITERATIONS) --strategies $$strat --json-output /workspace/results/$${strat}_gil.json \
	    > results/$${strat}_gil.txt 2>&1 || ( echo "GIL run failed for $$strat — see results/$${strat}_gil.txt"; exit 1 ); \
	  sleep 1; \
	done
	@echo "✓ All strategy runs finished — aggregating..."
	@python3 tools/compare_all_strategies.py --results-dir results --out results/compare_all.csv --plot results/compare_all.png
	@echo "CSV: results/compare_all.csv  PNG: results/compare_all.png"

