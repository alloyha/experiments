# HIL Agent System

Production-Ready AI Workflow Orchestration with Code Agents

## Features

- **Three Agent Types**: Simple (classification), Reasoning (ReAct), Code (autonomous)
- **1000+ Tool Integrations** via Composio (Shopify, Gmail, Slack, etc.)
- **Intelligent LLM Routing** for 56% cost optimization
- **Hybrid Memory System** (short-term + long-term + episodic)
- **Complete Observability** (metrics, traces, costs)
- **DAG-Based Workflows** with conditional branching
- **Production-Ready** with security, scaling, and monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (for dependency management)

### Development Setup

1. **Clone and setup the project:**
```bash
cd hil-agent-system
cp .env.example .env
# Edit .env with your API keys
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Start services:**
```bash
docker-compose up -d postgres redis
```

4. **Run database migrations:**
```bash
poetry run alembic upgrade head
```

5. **Start the application:**
```bash
poetry run uvicorn app.main:app --reload
```

6. **Access the API:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

### Full Stack (Production-like)

```bash
docker-compose up -d
```

This starts:
- FastAPI application (port 8000)
- PostgreSQL with pgvector (port 5432)
- Redis (port 6379)
- Celery worker & scheduler
- Flower (Celery monitoring, port 5555)
- Prometheus (port 9090)
- Grafana (port 3000, admin/admin)

## API Usage

### Execute a Simple Agent

```bash
curl -X POST "http://localhost:8000/api/v1/agents/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "simple",
    "input_data": {"message": "I want to return my order"},
    "model_profile": "fast"
  }'
```

### Execute a Workflow

```bash
curl -X POST "http://localhost:8000/api/v1/workflows/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "customer_support_advanced",
    "input_data": {"message": "I want to return order #12345"}
  }'
```

### List Available Tools

```bash
curl "http://localhost:8000/api/v1/tools/"
```

## Architecture

```
Graph (Workflow) 📊
  └── Node 🔵
       ├── Agent Type (simple/reasoning/code) 🤖
       │    ├── LLM Router (cost optimization) 💰
       │    │    └── Provider (OpenAI/Anthropic/Local) 🔌
       │    ├── Memory Manager (RAG + context) 🧠
       │    └── Tools (Composio or custom) 🛠️
       │         └── Actions (specific operations) ⚡
       └── Conditional Edges (branching logic) 🔀
```

## Configuration

Key environment variables:

```bash
# LLM Providers
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Composio Integration
COMPOSIO_API_KEY=your-key

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db

# Performance Tuning
DEFAULT_AGENT_TIMEOUT=300
MAX_REASONING_ITERATIONS=10
MAX_PARALLEL_NODES=10
```

## Development

### Project Structure

```
app/
├── core/           # Core configuration and utilities
├── agents/         # Agent implementations
├── tools/          # Tool registry and implementations
├── workflows/      # Workflow orchestration
├── memory/         # Memory management
├── api/           # API endpoints
└── models/        # Database models
```

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
poetry run black .
poetry run isort .
poetry run flake8 .
poetry run mypy .
```

## Monitoring

- **Prometheus Metrics**: http://localhost:9090
- **Grafana Dashboards**: http://localhost:3000
- **Celery Monitoring**: http://localhost:5555
- **Application Logs**: `docker-compose logs -f app`

## Production Deployment

See `docs/deployment.md` for detailed production deployment instructions including:
- Kubernetes manifests
- CI/CD pipeline setup
- Security considerations
- Scaling guidelines

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.