# HIL Agent System - Development Makefile (UV-powered)
# Comprehensive development workflow automation with UV

.PHONY: help install install-dev lint format type-check complexity test test-cov test-fast clean build run docs docker security audit all-checks ci

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
UV := uv
UV_RUN := uv run
PROJECT_NAME := hil-agent-system
SRC_DIR := app
TEST_DIR := tests
DOCS_DIR := docs
COVERAGE_MIN := 85
PORT := 8000

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)HIL Agent System - Development Commands (UV-powered)$(NC)"
	@echo "====================================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies with UV...$(NC)"
	$(UV) pip install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies with UV...$(NC)"
	$(UV) pip install -r requirements-dev.txt

install-all: install install-dev ## Install all dependencies

# Code Quality targets
lint: ## Run comprehensive linting with ruff
	@echo "$(BLUE)Running ruff linter...$(NC)"
	$(UV_RUN) ruff check $(SRC_DIR) $(TEST_DIR) --fix
	@echo "$(GREEN)✓ Linting completed$(NC)"

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	$(UV_RUN) ruff format $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Code formatting completed$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking with mypy...$(NC)"
	$(UV_RUN) mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking completed$(NC)"

complexity: ## Analyze code complexity with radon
	@echo "$(BLUE)Analyzing code complexity...$(NC)"
	@echo "$(YELLOW)Cyclomatic Complexity:$(NC)"
	$(UV_RUN) radon cc $(SRC_DIR) -a -s
	@echo "$(YELLOW)Maintainability Index:$(NC)"
	$(UV_RUN) radon mi $(SRC_DIR) -s
	@echo "$(YELLOW)Raw Metrics:$(NC)"
	$(UV_RUN) radon raw $(SRC_DIR) -s
	@echo "$(GREEN)✓ Complexity analysis completed$(NC)"

security: ## Run security analysis
	@echo "$(BLUE)Running security analysis...$(NC)"
	$(UV_RUN) bandit -r $(SRC_DIR) -f json -o security-report.json || true
	$(UV_RUN) bandit -r $(SRC_DIR)
	@echo "$(GREEN)✓ Security analysis completed$(NC)"

type-vulnerabilities: ## Check for type vulnerabilities with ty
	@echo "$(BLUE)Checking type vulnerabilities with ty...$(NC)"
	$(UV_RUN) ty check $(SRC_DIR) || echo "$(YELLOW)Type vulnerabilities found - see output above$(NC)"
	@echo "$(GREEN)✓ Type vulnerability check completed$(NC)"

audit: ## Audit dependencies for vulnerabilities
	@echo "$(BLUE)Auditing dependencies...$(NC)"
	$(UV_RUN) safety check --json --output audit-report.json || true
	$(UV_RUN) safety check
	@echo "$(GREEN)✓ Dependency audit completed$(NC)"

# Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(UV_RUN) pytest $(TEST_DIR) -v --tb=short

test-fast: ## Run tests without slow integration tests
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(UV_RUN) pytest $(TEST_DIR) -v --tb=short -m "not slow"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(UV_RUN) pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term --cov-fail-under=$(COVERAGE_MIN)
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(UV_RUN) pytest $(TEST_DIR) -v -m "integration"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(UV_RUN) pytest $(TEST_DIR) -v -m "not integration and not slow"

# Database targets
db-init: ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
	@echo "$(GREEN)✓ Database initialized$(NC)"

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Database migrations completed$(NC)"

db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "$(RED)WARNING: This will delete all database data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON) -c "from app.core.database import reset_db; import asyncio; asyncio.run(reset_db())"; \
		echo "$(GREEN)✓ Database reset completed$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

# Workflow targets
workflows-init: ## Initialize workflows from YAML files
	@echo "$(BLUE)Initializing workflows...$(NC)"
	$(PYTHON) -c "from app.services.workflow_integration import initialize_workflow_system; import asyncio; asyncio.run(initialize_workflow_system())"
	@echo "$(GREEN)✓ Workflows initialized$(NC)"

workflows-reload: ## Reload workflows from YAML files
	@echo "$(BLUE)Reloading workflows...$(NC)"
	$(PYTHON) -c "from app.services.workflow_integration import workflow_integration; import asyncio; asyncio.run(workflow_integration.reload_workflows())"
	@echo "$(GREEN)✓ Workflows reloaded$(NC)"

workflows-list: ## List available workflows
	@echo "$(BLUE)Available workflows:$(NC)"
	$(PYTHON) -c "from app.services.workflow_integration import get_available_workflows; import asyncio; import json; workflows = asyncio.run(get_available_workflows()); print(json.dumps(workflows, indent=2))"

# Application targets
run: ## Start the development server
	@echo "$(BLUE)Starting HIL Agent System on port $(PORT)...$(NC)"
	$(UV_RUN) uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT)

run-prod: ## Start production server
	@echo "$(BLUE)Starting production server...$(NC)"
	$(UV_RUN) uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --workers 4

dev: db-init workflows-init run ## Start full development environment

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME) .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -p $(PORT):$(PORT) --env-file .env $(PROJECT_NAME)

docker-compose: ## Start with docker-compose
	@echo "$(BLUE)Starting with docker-compose...$(NC)"
	docker compose up -d

docker-compose-dev: ## Start in development mode with logs
	@echo "$(BLUE)Starting development environment with docker-compose...$(NC)"
	docker compose up --build

docker-compose-prod: ## Start in production mode
	@echo "$(BLUE)Starting production environment...$(NC)"
	docker compose -f docker-compose.yml up -d --build

docker-compose-logs: ## Show docker-compose logs
	@echo "$(BLUE)Showing docker-compose logs...$(NC)"
	docker compose logs -f

docker-compose-down: ## Stop docker-compose services
	@echo "$(BLUE)Stopping docker-compose services...$(NC)"
	docker compose down

docker-compose-reset: ## Reset docker-compose (stop, remove volumes, restart)
	@echo "$(RED)WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		docker compose up -d --build; \
		echo "$(GREEN)✓ Docker environment reset$(NC)"; \
	else \
		echo "$(YELLOW)Reset cancelled$(NC)"; \
	fi

# Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	mkdir -p $(DOCS_DIR)
	$(PYTHON) -c "import app; help(app)" > $(DOCS_DIR)/api.txt
	@echo "$(GREEN)✓ Documentation generated$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd $(DOCS_DIR) && $(PYTHON) -m http.server 8080

# Cleanup targets
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -f security-report.json
	rm -f audit-report.json
	@echo "$(GREEN)✓ Cleanup completed$(NC)"

clean-all: clean ## Deep clean including virtual environment
	@echo "$(BLUE)Deep cleaning...$(NC)"
	rm -rf venv
	rm -rf .venv
	@echo "$(GREEN)✓ Deep cleanup completed$(NC)"

# Comprehensive checks
all-checks: lint type-check complexity security type-vulnerabilities test ## Run all code quality checks

ci: lint type-check test ## Run CI pipeline checks

pre-commit: format lint type-check test-fast ## Pre-commit checks (fast)

# Release targets
version: ## Show current version
	@echo "$(BLUE)Current version:$(NC)"
	$(PYTHON) -c "from app import __version__; print(__version__)"

build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Distribution packages built$(NC)"

# Development workflow
setup: install-all db-init workflows-init ## Complete development setup

status: ## Show project status
	@echo "$(BLUE)HIL Agent System Status$(NC)"
	@echo "======================="
	@echo "$(YELLOW)Python version:$(NC) $$($(PYTHON) --version)"
	@echo "$(YELLOW)Dependencies:$(NC)"
	@$(UV) pip list | grep -E "(ruff|radon|mypy|pytest|fastapi|uvicorn)" || true
	@echo "$(YELLOW)Test status:$(NC)"
	@$(UV_RUN) pytest $(TEST_DIR) --collect-only -q | tail -1
	@echo "$(YELLOW)Code quality:$(NC)"
	@$(UV_RUN) ruff check $(SRC_DIR) --statistics || true

# Performance targets
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	$(UV_RUN) pytest tests/test_performance.py -v --benchmark-only
	@echo "$(GREEN)✓ Benchmarks completed$(NC)"

profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	$(UV_RUN) python -m cProfile -o profile.stats -m app.main
	@echo "$(GREEN)✓ Profiling completed - see profile.stats$(NC)"

# Advanced analysis
analyze: complexity security audit ## Run comprehensive code analysis

lint-report: ## Generate detailed lint report
	@echo "$(BLUE)Generating lint report...$(NC)"
	$(UV_RUN) ruff check $(SRC_DIR) $(TEST_DIR) --output-format=json > lint-report.json
	$(UV_RUN) ruff check $(SRC_DIR) $(TEST_DIR) --statistics
	@echo "$(GREEN)✓ Lint report generated$(NC)"

metrics: ## Show code metrics
	@echo "$(BLUE)Code Metrics$(NC)"
	@echo "============"
	@echo "$(YELLOW)Lines of code:$(NC)"
	@find $(SRC_DIR) -name "*.py" -exec wc -l {} + | tail -1
	@echo "$(YELLOW)Test coverage:$(NC)"
	@$(UV_RUN) pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing | grep TOTAL || echo "Run 'make test-cov' first"
	@echo "$(YELLOW)Complexity:$(NC)"
	@$(UV_RUN) radon cc $(SRC_DIR) -a -s | head -10

# Help for specific areas
help-workflow: ## Show workflow-specific commands
	@echo "$(BLUE)Workflow Commands$(NC)"
	@echo "=================="
	@echo "$(GREEN)workflows-init$(NC)    Initialize workflows from YAML"
	@echo "$(GREEN)workflows-reload$(NC)  Reload workflows"
	@echo "$(GREEN)workflows-list$(NC)    List available workflows"

help-db: ## Show database-specific commands
	@echo "$(BLUE)Database Commands$(NC)"
	@echo "=================="
	@echo "$(GREEN)db-init$(NC)     Initialize database"
	@echo "$(GREEN)db-migrate$(NC)  Run migrations"
	@echo "$(GREEN)db-reset$(NC)    Reset database (destructive)"

help-quality: ## Show code quality commands
	@echo "$(BLUE)Code Quality Commands$(NC)"
	@echo "======================"
	@echo "$(GREEN)lint$(NC)         Run ruff linter"
	@echo "$(GREEN)format$(NC)       Format code with ruff"
	@echo "$(GREEN)type-check$(NC)   Run mypy type checking"
	@echo "$(GREEN)complexity$(NC)   Analyze with radon"
	@echo "$(GREEN)security$(NC)     Security analysis"
	@echo "$(GREEN)all-checks$(NC)   Run everything"