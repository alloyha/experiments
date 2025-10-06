# UV Migration Instructions

## Migration Complete! 🎉

Your project has been successfully migrated from `setup.py` to modern Python packaging with `uv`.

### What Changed

1. **Updated `pyproject.toml`**:
   - Migrated all dependencies from `setup.py`
   - Organized dependencies into logical groups (dev, docs, cloud, test)
   - Added comprehensive tool configurations (ruff, black, mypy, pytest)
   - Included proper project metadata and classifiers

2. **Added `.pre-commit-config.yaml`**:
   - Comprehensive pre-commit hooks for code quality
   - Python: ruff, black, mypy type checking
   - SQL: sqlfluff for SQL formatting and linting
   - Security: bandit for security scanning
   - Documentation: pydocstyle for docstring linting
   - General: file formatting, merge conflict detection

### Next Steps

1. **Install dependencies**:
   ```bash
   # Install core dependencies
   uv sync
   
   # Install with development dependencies
   uv sync --group dev
   
   # Install with all optional groups
   uv sync --group dev --group docs --group cloud --group test
   ```

2. **Set up pre-commit**:
   ```bash
   # Install pre-commit hooks
   uv run pre-commit install
   
   # Install commit-msg hook for conventional commits
   uv run pre-commit install --hook-type commit-msg
   
   # Run on all files to test
   uv run pre-commit run --all-files
   ```

3. **Remove old files** (optional):
   ```bash
   # You can now remove setup.py and requirements.txt
   rm setup.py requirements.txt
   ```

### Using UV Commands

```bash
# Add new dependencies
uv add pandas polars

# Add development dependencies
uv add --dev pytest black

# Add optional dependencies to specific groups
uv add --group cloud boto3
uv add --group docs sphinx

# Run scripts
uv run python examples/enhanced_platform_demo.py
uv run pytest
uv run pre-commit run --all-files

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Build the package
uv build

# Run with specific Python version
uv run --python 3.11 python script.py
```

### Dependency Groups

- **Core**: Required for basic functionality
- **dev**: Development tools (pytest, ruff, pre-commit, etc.)
- **docs**: Documentation generation (sphinx, themes)
- **cloud**: Cloud provider integrations (BigQuery, Snowflake, AWS, Azure)
- **test**: Extended testing tools

### Tool Configurations

All tools are now configured in `pyproject.toml`:
- **Ruff**: Fast linting and formatting
- **Black**: Code formatting
- **MyPy**: Type checking
- **Pytest**: Testing framework with coverage
- **Coverage**: Code coverage reporting

### Pre-commit Hooks

The `.pre-commit-config.yaml` includes:
- **Code Quality**: ruff, black, mypy
- **SQL**: sqlfluff for SQL formatting
- **Security**: bandit, safety checks
- **Documentation**: pydocstyle
- **General**: file formatting, conflict detection
- **Conventional Commits**: Enforces commit message standards

Your unified data platform is now using modern Python tooling! 🚀