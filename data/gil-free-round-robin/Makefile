# Makefile for Python 3.14 Free-Threading Benchmark Environment
.PHONY: help build run-nogil run-gil run-both compare dev multicore stop clean

help: ## Show this help message
	@echo "Python 3.14 Free-Threading Benchmark Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build custom Python 3.14 image from source
	docker build -t python314-nogil:latest -f Dockerfile .

run-nogil: ## Run benchmark with GIL disabled
	@echo "🚀 Running with GIL DISABLED (free-threading)"
	docker-compose run --rm python314-nogil

run-gil: ## Run benchmark with GIL enabled (comparison)
	@echo "🐌 Running with GIL ENABLED (traditional)"
	docker-compose run --rm python314-gil

run-both: ## Run both GIL disabled and enabled for comparison
	@echo "=== Running with GIL DISABLED ==="
	@docker-compose run --rm python314-nogil
	@echo ""
	@echo "=== Running with GIL ENABLED ==="
	@docker-compose run --rm python314-gil

compare: ## Run comprehensive comparison with timing
	@echo "📊 Comprehensive GIL Comparison"
	@echo "================================"
	@time docker-compose run --rm python314-nogil > results_nogil.txt
	@time docker-compose run --rm python314-gil > results_gil.txt
	@echo ""
	@echo "Results saved to results_nogil.txt and results_gil.txt"

dev: ## Start interactive development container
	docker-compose run --rm python314-dev

multicore: ## Run multi-core scaling test
	docker-compose run --rm python314-multicore

stop: ## Stop all running containers
	docker-compose down

clean: ## Remove containers and images
	docker-compose down -v --rmi all

test: ## Quick test to verify Python 3.14
	@docker-compose run --rm python314-nogil python -c "import sys; print(f'Python {sys.version}'); print(f'GIL enabled: {sys._is_gil_enabled() if hasattr(sys, \"_is_gil_enabled\") else \"Unknown\"}')"

logs: ## Show logs from last run
	docker-compose logs

shell: ## Open bash shell in container
	docker-compose run --rm python314-dev bash

# Advanced commands
stress-test: ## Run intensive stress test
	docker-compose run --rm python314-nogil python -c "\
	from gil_roundrobin_bench import run_benchmark; \
	run_benchmark(num_consumers=8, num_items=50000)"

profile: ## Run with profiling enabled
	docker-compose run --rm python314-nogil python -m cProfile -o profile.stats gil_roundrobin_bench.py

watch: ## Watch for file changes and re-run
	@while true; do \
		make run-nogil; \
		inotifywait -e modify gil_roundrobin_bench.py 2>/dev/null || sleep 5; \
	done