# Makefile

# Define common variables
PYTHON := python
PIP := pip
BROWSER := firefox
DOCKER := docker
SOURCE_FOLDER := "lambda_api/"

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

rel_current_path = sys.argv[1]
abs_current_path = os.path.abspath(rel_current_path)
uri = "file://" + pathname2url(abs_current_path)

webbrowser.open(uri)
endef

export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

regex_pattern = r'^([a-zA-Z_-]+):.*?## (.*)$$'

for line in sys.stdin:
	match = re.match(regex_pattern, line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef

export PRINT_HELP_PYSCRIPT

# Additional rules can be added as needed.
.PHONY: prepare sudo migrate run up down ps purge whos

help: ## Add a rule to list commands
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean-logs: # Add a rule to remove log info
	rm -fr build/ dist/ .eggs/
	find . -name '*.log' -o -name '*.log' -exec rm -fr {} +

clean-pyc: # Add a rule to remove pyc files
	find . -name '*.pyc' -o -name '*.pyo' -o -name '*~' -exec rm -rf {} +

clean-test: # remove test and coverage artifacts
	rm -fr .tox/ .coverage coverage.* htmlcov/ .pytest_cache

clean-cache: # remove test and coverage artifacts
	find . -name '*cache*' -exec rm -rf {} +

clean: clean-logs clean-pyc clean-test clean-cache ## Add a rule to remove unnecessary assets
	$(DOCKER) system prune --volumes -f

env: ## Add a rule to activate environment
	poetry shell

notebok: ## Add a rule to activate environment
	jupyter notebook

allow: ## Add a rule to allow scripts to execute
	chmod +x *

prepare: clean env allow ## Add a rule to activate environment and install dependencies

install: ## Add a rule to install project dependencies.
	poetry install

lint: ## Add a rule to clean up any temporary files
	ruff --fix .
	find $(SOURCE_FOLDER) -name "*.py" -exec autopep8 --in-place --aggressive --aggressive {} \;


test: ## Add a rule to test the application
	coverage run -m pytest

report: test ## Add a rule to generate coverage report
	coverage report --show-missing
	coverage html
	$(BROWSER) htmlcov/index.html

ps: ## Add a rule to list containers
	docker ps -a
