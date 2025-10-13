# HIL Agent System - Development Script with UV
param([string]$Command = "help", [string]$Port = "8000")

$SrcDir = "app"
$TestDir = "tests"
$VenvDir = ".venv"

function Test-UV {
    try { 
        $null = python -m uv --version 2>$null
        return $LASTEXITCODE -eq 0
    } catch { 
        return $false 
    }
}

function Install-UV {
    Write-Host "Installing UV..." -ForegroundColor Blue
    try {
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        return $LASTEXITCODE -eq 0
    } catch {
        Write-Host "Failed to install UV. Install manually: pip install uv" -ForegroundColor Red
        return $false
    }
}

function Ensure-UV {
    if (-not (Test-UV)) {
        if (-not (Install-UV)) { exit 1 }
    }
}

function Ensure-Venv {
    if (-not (Test-Path $VenvDir)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Blue
        try {
            python -m uv venv $VenvDir
            Write-Host "Virtual environment created successfully." -ForegroundColor Green
        } catch {
            Write-Host "Failed to create virtual environment: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Virtual environment already exists." -ForegroundColor Green
    }
}

switch ($Command.ToLower()) {
    "help" {
        Write-Host "HIL Agent System - Development Commands (UV-powered)" -ForegroundColor Blue
        Write-Host "====================================================" -ForegroundColor Blue
        Write-Host ""
        Write-Host "Installation:" -ForegroundColor Yellow
        Write-Host "  install-dev      Install development dependencies"
        Write-Host "  setup            Complete development setup"
        Write-Host ""
        Write-Host "Code Quality:" -ForegroundColor Yellow
        Write-Host "  lint             Run ruff linter (UV virtual env)"
        Write-Host "  lint-direct      Run ruff linter (direct)"
        Write-Host "  format           Format code with ruff (UV virtual env)"
        Write-Host "  format-direct    Format code with ruff (direct)"
        Write-Host "  type-check       Run mypy type checking (UV virtual env)"
        Write-Host "  type-check-direct Run mypy type checking (direct)"
        Write-Host "  complexity       Analyze code complexity"
        Write-Host ""
        Write-Host "Testing:" -ForegroundColor Yellow
        Write-Host "  test             Run all tests"
        Write-Host "  test-cov         Run tests with coverage"
        Write-Host ""
        Write-Host "Application:" -ForegroundColor Yellow
        Write-Host "  run              Start development server"
        Write-Host ""
        Write-Host "Utilities:" -ForegroundColor Yellow
        Write-Host "  all-checks       Run all quality checks"
        Write-Host "  status           Show project status"
        Write-Host "  clean            Clean temporary files"
        Write-Host "  clean-all        Deep clean (including venv)"
        Write-Host ""
        Write-Host "Usage:" -ForegroundColor Green
        Write-Host "  powershell -ExecutionPolicy Bypass -File dev.ps1 [command]"
    }
    "install-dev" {
        Ensure-UV
        Ensure-Venv
        Write-Host "Installing development dependencies..." -ForegroundColor Blue
        python -m uv pip install -r requirements-dev.txt --python $VenvDir
    }
    "setup" {
        Write-Host "Setting up development environment..." -ForegroundColor Blue
        Ensure-UV
        Ensure-Venv
        python -m uv pip install -e ".[dev]" --python $VenvDir
        Write-Host "Setup completed!" -ForegroundColor Green
    }
    "complexity" {
        Ensure-UV
        Ensure-Venv
        Write-Host "Analyzing complexity..." -ForegroundColor Blue
        python -m uv run --python $VenvDir radon cc $SrcDir -a -s
        python -m uv run --python $VenvDir radon mi $SrcDir -s
    }
    "test" {
        Ensure-UV
        Ensure-Venv
        Write-Host "Running tests..." -ForegroundColor Blue
        python -m uv run --python $VenvDir pytest $TestDir -v
    }
    "test-cov" {
        Ensure-UV
        Ensure-Venv
        Write-Host "Running tests with coverage..." -ForegroundColor Blue
        python -m uv run --python $VenvDir pytest $TestDir --cov=$SrcDir --cov-report=html --cov-report=term
    }
    "run" {
        Ensure-UV
        Ensure-Venv
        Write-Host "Starting development server on port $Port..." -ForegroundColor Blue
        python -m uv run --python $VenvDir uvicorn app.main:app --reload --host 0.0.0.0 --port $Port
    }
    "lint" {
        Write-Host "Running ruff linter (direct mode)..." -ForegroundColor Blue
        python -m ruff check $SrcDir $TestDir --fix
        Write-Host "Linting completed." -ForegroundColor Green
    }
    "format" {
        Write-Host "Running ruff formatter (direct mode)..." -ForegroundColor Blue
        python -m ruff format $SrcDir $TestDir
        Write-Host "Formatting completed." -ForegroundColor Green
    }
    "type-check" {
        Write-Host "Running mypy type checking (direct mode)..." -ForegroundColor Blue
        python -m mypy $SrcDir --ignore-missing-imports
        Write-Host "Type checking completed." -ForegroundColor Green
    }
    "status" {
        Write-Host "Project Status:" -ForegroundColor Blue
        Write-Host "Python: $(python --version)"
        Write-Host "UV: $(if (Test-UV) { python -m uv --version } else { 'Not installed' })"
    }
    "clean" {
        Write-Host "Cleaning up..." -ForegroundColor Blue
        Get-ChildItem -Recurse -Name "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Name "__pycache__" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        @(".pytest_cache", ".coverage", "htmlcov", ".ruff_cache", ".mypy_cache") | ForEach-Object {
            if (Test-Path $_) { Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue }
        }
        Write-Host "Cleanup completed" -ForegroundColor Green
    }
    "clean-all" {
        Write-Host "Deep cleaning (including virtual environment)..." -ForegroundColor Blue
        Get-ChildItem -Recurse -Name "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Name "__pycache__" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        @(".pytest_cache", ".coverage", "htmlcov", ".ruff_cache", ".mypy_cache", $VenvDir) | ForEach-Object {
            if (Test-Path $_) { Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue }
        }
        Write-Host "Deep cleanup completed" -ForegroundColor Green
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run: powershell -ExecutionPolicy Bypass -File dev.ps1 help" -ForegroundColor Yellow
    }
}