# Sistema HIL (Human-in-the-Loop) - Guia de Implementação Completo

## 🎯 Visão Geral do Sistema

Sistema de conversação híbrido bot-humano com transição suave entre atendimento automatizado e manual. **Uma única flag** (`is_hil`) controla todo o estado.

**Stack Tecnológico:**
- **FastAPI**: API assíncrona e webhooks
- **PostgreSQL**: Armazenamento de conversas
- **Redis**: Cache, rate limiting e filas
- **LangGraph**: Orquestração do agente conversacional
- **PydanticAI**: Agents especializados com outputs estruturados
- **Composio**: 150+ integrações prontas (APIs externas)
- **RQ**: Workers para batch jobs pesados

---

## 📊 Arquitetura em Camadas

```
┌─────────────────────────────────────────────┐
│   Orquestrador (HIL Manager)                │
│   - Webhook Handler (async)                  │
│   - Command Parser                           │
└──────────┬──────────────────────────────────┘
           │
           ├─→ is_hil = false
           │   ┌──────────────────────────────┐
           │   │   LangGraph Agent            │
           │   │   - Entender intenção        │
           │   │   - Planejar ações           │
           │   │   - Executar (Composio)      │
           │   │   - Validar resultado        │
           │   │   - Responder/Escalar        │
           │   └──────────────────────────────┘
           │
           ├─→ Comandos @...
           │   ┌──────────────────────────────┐
           │   │   PydanticAI Agents          │
           │   │   - SummaryAgent             │
           │   │   - AnalysisAgent            │
           │   │   - SuggestionAgent          │
           │   │   - SentimentAgent           │
           │   │   - DraftAgent               │
           │   └──────────────────────────────┘
           │
           ├─→ is_hil = true
           │   ┌──────────────────────────────┐
           │   │   Timeout Monitor            │
           │   │   - Redis Sorted Set         │
           │   │   - Worker polling           │
           │   └──────────────────────────────┘
           │
           └─→ Batch Jobs
               ┌──────────────────────────────┐
               │   RQ Workers                 │
               │   - GDPR cleanup             │
               │   - Reports                  │
               │   - Notifications            │
               └──────────────────────────────┘
```

---

## 🗄️ Schema de Banco de Dados

### Migrations SQL

```sql
-- Migration 001: Core tables
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id TEXT NOT NULL,
  is_hil BOOLEAN DEFAULT false,
  last_activity_at TIMESTAMP NOT NULL DEFAULT NOW(),
  last_bot_message_id UUID,
  escalated BOOLEAN DEFAULT false,
  department TEXT,
  closed BOOLEAN DEFAULT false,
  closed_at TIMESTAMPTZ,
  close_reason TEXT,
  summary JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  version BIGINT DEFAULT 0 NOT NULL
);

CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  sender_type TEXT NOT NULL CHECK (sender_type IN ('customer', 'ai_agent', 'human_agent')),
  sender_id TEXT NOT NULL,
  content TEXT NOT NULL,
  message_origin_id TEXT,
  contains_pii BOOLEAN DEFAULT false,
  deleted_at TIMESTAMPTZ,
  timestamp TIMESTAMP DEFAULT NOW()
);

-- Migration 002: Audit & Compliance
CREATE TABLE conversation_audit (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  old_is_hil BOOLEAN,
  new_is_hil BOOLEAN,
  reason TEXT,
  details JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE pii_usage_audit (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  used_at TIMESTAMPTZ DEFAULT NOW(),
  reason TEXT NOT NULL,
  consent_timestamp TIMESTAMPTZ,
  details JSONB
);

CREATE TABLE gdpr_deletion_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id TEXT NOT NULL,
  requester_email TEXT NOT NULL,
  reason TEXT,
  conversations_affected INT,
  messages_deleted INT,
  requested_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed'))
);

-- Migration 003: System tables
CREATE TABLE customer_budgets (
  customer_id TEXT PRIMARY KEY,
  monthly_budget_usd DECIMAL(10, 2) NOT NULL DEFAULT 100.00,
  current_spend_usd DECIMAL(10, 2) NOT NULL DEFAULT 0,
  budget_reset_at TIMESTAMPTZ NOT NULL DEFAULT date_trunc('month', NOW()) + INTERVAL '1 month',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE support_tickets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  error_type TEXT NOT NULL,
  error_message TEXT,
  context JSONB,
  status TEXT DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
  assigned_to TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  resolved_at TIMESTAMPTZ
);

CREATE TABLE agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('operator', 'supervisor', 'admin')),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_conversations_is_hil ON conversations(is_hil) WHERE is_hil = true;
CREATE INDEX idx_conversations_last_activity ON conversations(last_activity_at) WHERE is_hil = true;
CREATE UNIQUE INDEX messages_conv_origin_idx ON messages (conversation_id, message_origin_id) WHERE message_origin_id IS NOT NULL;
CREATE INDEX idx_messages_deleted ON messages(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX idx_audit_conversation ON conversation_audit(conversation_id, created_at DESC);
CREATE INDEX idx_pii_audit_conversation ON pii_usage_audit(conversation_id, used_at DESC);
CREATE INDEX idx_support_tickets_status ON support_tickets(status, created_at DESC);
```

---

## 🔧 Estrutura de Arquivos do Projeto

```
project/
├── app/
│   ├── main.py                    # FastAPI app principal
│   ├── config.py                  # Configurações e env vars
│   ├── dependencies.py            # Dependency injection
│   │
│   ├── core/
│   │   ├── database.py           # AsyncPG setup
│   │   ├── redis.py              # Redis client
│   │   ├── secrets.py            # Secrets manager
│   │   └── metrics.py            # Prometheus metrics
│   │
│   ├── models/
│   │   ├── conversation.py       # Pydantic models
│   │   ├── message.py
│   │   └── command.py
│   │
│   ├── agents/
│   │   ├── pydantic_agents.py    # PydanticAI agents
│   │   ├── langgraph_agent.py    # LangGraph agent
│   │   └── composio_setup.py     # Composio integration
│   │
│   ├── services/
│   │   ├── conversation.py       # Conversation management
│   │   ├── command_handler.py    # Command execution
│   │   ├── timeout_monitor.py    # Timeout handling
│   │   ├── rate_limiter.py       # Rate limiting
│   │   ├── pii_sanitizer.py      # PII redaction
│   │   ├── gdpr_service.py       # GDPR compliance
│   │   └── error_handler.py      # Error handling
│   │
│   ├── api/
│   │   ├── webhooks.py           # Webhook endpoints
│   │   ├── commands.py           # Command API
│   │   ├── gdpr.py               # GDPR endpoints
│   │   └── health.py             # Health checks
│   │
│   └── workers/
│       ├── rq_worker.py          # RQ worker
│       └── tasks/
│           ├── retention.py      # GDPR cleanup
│           ├── reports.py        # Report generation
│           └── notifications.py  # Batch notifications
│
├── tests/
│   ├── conftest.py               # Pytest fixtures
│   ├── test_agents.py
│   ├── test_commands.py
│   ├── test_concurrency.py
│   └── test_gdpr.py
│
├── migrations/
│   └── 001_initial.sql
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 📦 Dependências (requirements.txt)

```txt
# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
asyncpg==0.29.0
psycopg2-binary==2.9.9

# Redis & Queue
redis==5.0.1
rq==1.15.1
rq-scheduler==0.13.1

# AI/ML
langgraph==0.0.20
pydantic-ai==0.0.7
openai==1.3.7
composio-langgraph==0.3.5

# NLP (PII detection)
spacy==3.7.2

# Observability
prometheus-client==0.19.0
structlog==23.2.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
opentelemetry-exporter-jaeger==1.21.0

# Security
cryptography==41.0.7
python-jose[cryptography]==3.3.0

# Utils
python-multipart==0.0.6
httpx==0.25.2
apscheduler==3.10.4

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
```

---

## ⚙️ Configuração (.env.example)

```bash
# Environment
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/conversations
DATABASE_POOL_MIN_SIZE=10
DATABASE_POOL_MAX_SIZE=50

# Redis
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Composio
COMPOSIO_API_KEY=...
COMPOSIO_BASE_URL=https://api.composio.dev

# Timeouts & Limits
HUMAN_TIMEOUT_MINUTES=30
BOT_RESPONSE_TIMEOUT_SECONDS=30
CUSTOMER_RATE_LIMIT_PER_MINUTE=10
CUSTOMER_RATE_LIMIT_PER_HOUR=100
AGENT_RATE_LIMIT_PER_MINUTE=50

# Features
ENABLE_AUTO_SUMMARY=true
SUMMARY_THRESHOLD_MESSAGES=10
MAX_CONTEXT_MESSAGES=50
ENABLE_PII_REDACTION=true
USE_NER_PII_DETECTION=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
JAEGER_HOST=localhost
JAEGER_PORT=6831

# Security
SECRET_KEY=your-secret-key-here
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=

# GDPR/LGPD
DATA_RETENTION_DAYS_MESSAGES=90
DATA_RETENTION_DAYS_SUMMARIES=730
DATA_RETENTION_DAYS_AUDIT=2555
```

---

## 🚀 Implementação Passo a Passo

### FASE 1: Setup Básico (Dia 1-2)

#### 1.1 - Estrutura do Projeto

```bash
# Criar estrutura
mkdir -p app/{core,models,agents,services,api,workers/tasks}
mkdir -p tests migrations

# Inicializar Git
git init
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```

#### 1.2 - Database Setup (app/core/database.py)

```python
import asyncpg
from typing import Optional
import os

class Database:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Cria connection pool"""
        self.pool = await asyncpg.create_pool(
            dsn=os.getenv("DATABASE_URL"),
            min_size=int(os.getenv("DATABASE_POOL_MIN_SIZE", 10)),
            max_size=int(os.getenv("DATABASE_POOL_MAX_SIZE", 50)),
            command_timeout=60
        )
    
    async def disconnect(self):
        """Fecha connection pool"""
        if self.pool:
            await self.pool.close()
    
    async def execute(self, query: str, *args):
        """Executa query sem retorno"""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Busca múltiplas rows"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Busca uma row"""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Busca um valor"""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

# Singleton
db = Database()
```

#### 1.3 - Redis Setup (app/core/redis.py)

```python
import redis.asyncio as aioredis
import os

class RedisClient:
    def __init__(self):
        self.client = None
    
    async def connect(self):
        """Conecta ao Redis"""
        self.client = await aioredis.from_url(
            os.getenv("REDIS_URL"),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )
    
    async def disconnect(self):
        """Desconecta"""
        if self.client:
            await self.client.close()

# Singleton
redis_client = RedisClient()
```

#### 1.4 - FastAPI Main (app/main.py)

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog
import time

from app.core.database import db
from app.core.redis import redis_client
from app.api import webhooks, health, gdpr
from app.services.timeout_monitor import timeout_monitor
from app.core.metrics import metrics_app

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# Create app
app = FastAPI(title="HIL System", version="1.0.0")

# Lifecycle events
@app.on_event("startup")
async def startup():
    logger.info("starting_application")
    await db.connect()
    await redis_client.connect()
    
    # Start timeout monitor
    import asyncio
    asyncio.create_task(timeout_monitor.worker_loop())
    
    logger.info("application_started")

@app.on_event("shutdown")
async def shutdown():
    logger.info("shutting_down_application")
    await db.disconnect()
    await redis_client.disconnect()

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=int(duration * 1000)
    )
    
    return response

# Include routers
app.include_router(webhooks.router, prefix="/webhook", tags=["webhooks"])
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(gdpr.router, prefix="/api/v1/gdpr", tags=["gdpr"])

# Mount metrics
app.mount("/metrics", metrics_app)

# Root endpoint
@app.get("/")
async def root():
    return {"status": "ok", "service": "HIL System"}
```

---

### FASE 2: Core Services (Dia 3-5)

#### 2.1 - Models (app/models/conversation.py)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal
from uuid import UUID

class IncomingMessage(BaseModel):
    conversation_id: str
    sender_type: Literal["customer", "human_agent"]
    sender_id: str
    content: str
    origin_id: Optional[str] = None
    provider: Optional[str] = None

class Conversation(BaseModel):
    id: UUID
    customer_id: str
    is_hil: bool
    last_activity_at: datetime
    last_bot_message_id: Optional[UUID] = None
    escalated: bool = False
    closed: bool = False
    created_at: datetime

class Message(BaseModel):
    id: UUID
    conversation_id: UUID
    sender_type: str
    sender_id: str
    content: str
    contains_pii: bool = False
    timestamp: datetime
```

#### 2.2 - PII Sanitizer (app/services/pii_sanitizer.py)

```python
import re
from typing import Tuple
import spacy

# Regex patterns
PII_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'phone_br': re.compile(r'\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b'),
    'cpf': re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}[-\s]?\d{2}\b'),
    'cnpj': re.compile(r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}[-\s]?\d{2}\b'),
    'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
}

class PIISanitizer:
    def __init__(self, use_ner: bool = True):
        self.use_ner = use_ner
        if use_ner:
            self.nlp = spacy.load("pt_core_news_lg")
    
    def redact_regex(self, text: str) -> Tuple[str, list]:
        """Layer 1: Regex-based redaction"""
        warnings = []
        
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
                warnings.append(f"regex_detected_{pii_type}")
        
        return text, warnings
    
    def redact_ner(self, text: str) -> Tuple[str, list]:
        """Layer 2: NER-based redaction"""
        if not self.use_ner:
            return text, []
        
        warnings = []
        doc = self.nlp(text)
        
        entities_to_redact = ["PER", "LOC", "ORG"]
        
        for ent in doc.ents:
            if ent.label_ in entities_to_redact:
                text = text.replace(ent.text, f"[{ent.label_}_REDACTED]")
                warnings.append(f"ner_detected_{ent.label_}")
        
        return text, warnings
    
    def redact_full(self, text: str) -> Tuple[str, dict]:
        """Full redaction pipeline"""
        warnings = []
        
        # Layer 1
        text, regex_warnings = self.redact_regex(text)
        warnings.extend(regex_warnings)
        
        # Layer 2
        if self.use_ner:
            text, ner_warnings = self.redact_ner(text)
            warnings.extend(ner_warnings)
        
        return text, {
            "redacted": len(warnings) > 0,
            "warnings": warnings
        }

# Singleton
sanitizer = PIISanitizer()
```

#### 2.3 - Rate Limiter (app/services/rate_limiter.py)

```python
import time
from typing import Tuple
import redis.asyncio as aioredis

class RateLimiter:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def check_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> Tuple[bool, int]:
        """
        Verifica rate limit usando sliding window.
        Retorna (dentro_do_limite, retry_after_seconds)
        """
        now = time.time()
        window_start = now - window_seconds
        
        pipe = self.redis.pipeline()
        
        # Remove entradas antigas
        pipe.zremrangebyscore(key, '-inf', window_start)
        
        # Adiciona entrada atual
        pipe.zadd(key, {str(now): now})
        
        # Conta entradas na janela
        pipe.zcard(key)
        
        # Set TTL
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        count = results[2]
        
        within_limit = count <= limit
        
        # Calcula retry_after
        if not within_limit:
            retry_after = await self.get_retry_after(key, window_seconds)
        else:
            retry_after = 0
        
        return within_limit, retry_after
    
    async def get_retry_after(self, key: str, window_seconds: int) -> int:
        """Calcula segundos até window reset"""
        scores = await self.redis.zrange(key, 0, 0, withscores=True)
        if not scores:
            return window_seconds
        
        oldest_timestamp = scores[0][1]
        window_start = time.time() - window_seconds
        
        if oldest_timestamp < window_start:
            return 0
        
        return int(oldest_timestamp + window_seconds - time.time())
    
    async def check_customer_limit(self, customer_id: str) -> Tuple[bool, int]:
        """10 mensagens/minuto"""
        return await self.check_limit(
            f"rate:customer:{customer_id}",
            limit=10,
            window_seconds=60
        )
    
    async def check_customer_hourly(self, customer_id: str) -> Tuple[bool, int]:
        """100 mensagens/hora"""
        return await self.check_limit(
            f"rate:customer:hourly:{customer_id}",
            limit=100,
            window_seconds=3600
        )
```

---

### FASE 3: AI Agents (Dia 6-8)

#### 3.1 - PydanticAI Agents (app/agents/pydantic_agents.py)

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal

# Output models
class ConversationSummary(BaseModel):
    main_topic: str = Field(description="Tópico principal")
    customer_issue: str = Field(description="Problema do cliente")
    current_status: str = Field(description="Status atual")
    next_steps: list[str] = Field(description="Próximos passos")
    sentiment_score: float = Field(ge=-1, le=1)
    urgency: Literal["low", "medium", "high"]

class ConversationAnalysis(BaseModel):
    sentiment: Literal["satisfied", "neutral", "dissatisfied"]
    urgency: Literal["low", "medium", "high"]
    complexity: Literal["simple", "moderate", "complex"]
    recommended_actions: list[str]
    bot_can_handle: bool
    escalation_reason: str | None = None

class ResponseSuggestion(BaseModel):
    suggested_response: str
    tone: Literal["professional", "empathetic", "apologetic", "casual"]
    confidence: float = Field(ge=0, le=1)
    alternative_approaches: list[str]

# Agents
class SummaryAgent:
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=ConversationSummary,
            system_prompt="""
            Você é um assistente especializado em sumarizar conversas.
            Analise e extraia as informações mais importantes de forma estruturada.
            Seja conciso mas completo.
            """
        )
    
    async def summarize(self, messages: list[dict]) -> ConversationSummary:
        formatted = "\n".join([
            f"[{m['sender_type']}] {m['content']}" 
            for m in messages
        ])
        
        result = await self.agent.run(
            f"Analise e resuma esta conversa:\n\n{formatted}"
        )
        
        return result.data

class AnalysisAgent:
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=ConversationAnalysis,
            system_prompt="""
            Você é um supervisor de atendimento experiente.
            Analise conversas para determinar sentimento, urgência, complexidade
            e se o bot pode continuar ou precisa escalar.
            """
        )
    
    async def analyze(self, messages: list[dict]) -> ConversationAnalysis:
        formatted = "\n".join([
            f"[{m['sender_type']}] {m['content']}" 
            for m in messages
        ])
        
        result = await self.agent.run(
            f"Analise esta conversa:\n\n{formatted}"
        )
        
        return result.data

class SuggestionAgent:
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=ResponseSuggestion,
            system_prompt="""
            Você é um atendente experiente que sugere respostas excelentes.
            Considere o contexto completo, seja empático e profissional.
            """
        )
    
    async def suggest(
        self, 
        context: str,
        last_message: str
    ) -> ResponseSuggestion:
        result = await self.agent.run(
            f"""
            Contexto: {context}
            
            Última mensagem: {last_message}
            
            Sugira uma resposta apropriada.
            """
        )
        
        return result.data
```

#### 3.2 - LangGraph Agent (app/agents/langgraph_agent.py)

```python
from langgraph.graph import StateGraph, END
from composio_langgraph import ComposioToolSet, Action
from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    intent: str
    plan: List[str]
    execution_result: dict
    should_escalate: bool
    final_response: str
    composio_entity_id: str

def build_bot_graph():
    workflow = StateGraph(AgentState)
    composio_toolset = ComposioToolSet()
    
    # Nós
    workflow.add_node("entender", entender_intent)
    workflow.add_node("planejar", criar_plano)
    workflow.add_node("executar", lambda state: executar_acoes(state, composio_toolset))
    workflow.add_node("validar", validar_resultado)
    workflow.add_node("responder", gerar_resposta)
    workflow.add_node("escalar", escalar_para_humano)
    
    # Fluxo
    workflow.set_entry_point("entender")
    workflow.add_edge("entender", "planejar")
    workflow.add_edge("planejar", "executar")
    
    workflow.add_conditional_edges(
        "executar",
        decidir_proximo_passo,
        {
            "validar": "validar",
            "escalar": "escalar"
        }
    )
    
    workflow.add_conditional_edges(
        "validar",
        verificar_validacao,
        {
            "responder": "responder",
            "planejar": "planejar"
        }
    )
    
    workflow.add_edge("responder", END)
    workflow.add_edge("escalar", END)
    
    return workflow.compile()

# Implementação dos nós (simplificado)
def entender_intent(state):
    # LLM classifica intenção
    return {"intent": "track_order"}

def criar_plano(state):
    # LLM escolhe actions
    return {"plan": ["buscar_pedido"]}

def executar_acoes(state, composio):
    # Executa via Composio
    results = {}
    # ... execução
    return {"execution_result": results}

def decidir_proximo_passo(state):
    if state["execution_result"].get("error"):
        return "escalar"
    return "validar"

def validar_resultado(state):
    return {"validation": {"success": True}}

def verificar_validacao(state):
    return "responder" if state["validation"]["success"] else "planejar"

def gerar_resposta(state):
    return {"final_response": "Seu pedido está em trânsito"}

def escalar_para_humano(state):
    return {"should_escalate": True}
```

---

### FASE 4: Command Handler & Timeout Monitor (Dia 9-11)

#### 4.1 - Command Parser (app/services/command_handler.py)

```python
import re
from typing import Tuple, Optional, Dict
from app.agents.pydantic_agents import SummaryAgent, AnalysisAgent, SuggestionAgent
from app.core.database import db
import structlog

logger = structlog.get_logger()

class CommandParser:
    """Parse comandos @... enviados por agentes"""
    
    @staticmethod
    def parse(content: str) -> Tuple[Optional[str], Dict]:
        """
        Parse comando e argumentos.
        Exemplos:
        - "@bot Continue com o cliente" -> ("@bot", {"instruction": "Continue com o cliente"})
        - "@summary last:10" -> ("@summary", {"last": "10"})
        - "@escalate reason:Cliente insatisfeito" -> ("@escalate", {"reason": "Cliente insatisfeito"})
        """
        content = content.strip()
        
        # Verifica se começa com @
        if not content.startswith('@'):
            return None, {}
        
        # Extrai comando (primeira palavra)
        parts = content.split(maxsplit=1)
        command = parts[0].lower()
        
        # Extrai argumentos
        args = {}
        if len(parts) > 1:
            remainder = parts[1]
            
            # Parse key:value pairs
            kv_pattern = r'(\w+):([^\s]+(?:\s+[^\s:]+)*?)(?=\s+\w+:|$)'
            matches = re.findall(kv_pattern, remainder)
            
            if matches:
                for key, value in matches:
                    args[key] = value.strip()
            else:
                # Se não tem key:value, tudo é "instruction" ou "reason"
                if command in ["@bot", "@resume"]:
                    args["instruction"] = remainder.strip()
                elif command in ["@escalate", "@transfer"]:
                    args["reason"] = remainder.strip()
        
        return command, args

class CommandHandler:
    """Executa comandos usando agentes especializados"""
    
    def __init__(self, langgraph_agent, composio_toolset):
        self.agent = langgraph_agent
        self.composio = composio_toolset
        
        # Inicializa PydanticAI agents
        self.summary_agent = SummaryAgent()
        self.analysis_agent = AnalysisAgent()
        self.suggestion_agent = SuggestionAgent()
    
    async def handle(self, conversation_id: str, command: str, args: dict) -> dict:
        """Executa comando e retorna resultado"""
        
        handlers = {
            "@bot": self._handle_bot_takeover,
            "@pause": self._handle_bot_pause,
            "@resume": self._handle_bot_resume,
            "@summary": self._handle_summary,
            "@context": self._handle_context,
            "@status": self._handle_status,
            "@analyze": self._handle_analyze,
            "@suggest": self._handle_suggest,
            "@escalate": self._handle_escalate,
            "@transfer": self._handle_transfer,
            "@help": self._handle_help,
        }
        
        handler = handlers.get(command)
        if not handler:
            return {"error": f"Comando desconhecido: {command}"}
        
        logger.info("executing_command", command=command, conversation_id=conversation_id)
        
        return await handler(conversation_id, args)
    
    # === Handlers ===
    
    async def _handle_bot_takeover(self, conv_id: str, args: dict) -> dict:
        """@bot - Bot assume conversa"""
        
        # Busca mensagens desde último bot
        messages = await db.fetch("""
            SELECT sender_type, content, timestamp
            FROM messages
            WHERE conversation_id = $1
            AND timestamp > (
                SELECT timestamp FROM messages 
                WHERE id = (
                    SELECT last_bot_message_id FROM conversations WHERE id = $1
                )
            )
            ORDER BY timestamp ASC
        """, conv_id)
        
        message_dicts = [dict(m) for m in messages]
        
        # Sumariza contexto
        if len(message_dicts) > 5:
            summary = await self.summary_agent.summarize(message_dicts)
            context = f"Resumo: {summary.main_topic}. {summary.customer_issue}"
        else:
            context = "\n".join([f"[{m['sender_type']}] {m['content']}" for m in message_dicts])
        
        instruction = args.get("instruction", "Continue a conversa")
        
        # Invoca LangGraph
        result = self.agent.invoke({
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": instruction}
            ],
            "composio_entity_id": await self._get_composio_entity(conv_id)
        })
        
        # Atualiza is_hil
        await db.execute("""
            UPDATE conversations 
            SET is_hil = false, version = version + 1
            WHERE id = $1
        """, conv_id)
        
        # Audit log
        await db.execute("""
            INSERT INTO conversation_audit 
            (conversation_id, actor, action, old_is_hil, new_is_hil, reason)
            VALUES ($1, 'system', 'hil_deactivated', true, false, 'bot_takeover')
        """, conv_id)
        
        return {
            "success": True,
            "bot_response": result.get("final_response"),
            "status": "bot_active"
        }
    
    async def _handle_bot_pause(self, conv_id: str, args: dict) -> dict:
        """@pause - Pausa bot"""
        await db.execute("""
            UPDATE conversations 
            SET is_hil = true, version = version + 1
            WHERE id = $1
        """, conv_id)
        
        await db.execute("""
            INSERT INTO conversation_audit 
            (conversation_id, actor, action, old_is_hil, new_is_hil, reason)
            VALUES ($1, 'agent', 'hil_activated', false, true, 'pause_command')
        """, conv_id)
        
        return {
            "success": True,
            "message": "Bot pausado. Use @resume para reativar.",
            "status": "human_active"
        }
    
    async def _handle_bot_resume(self, conv_id: str, args: dict) -> dict:
        """@resume - Resume bot"""
        return await self._handle_bot_takeover(conv_id, {"instruction": "Continue"})
    
    async def _handle_summary(self, conv_id: str, args: dict) -> dict:
        """@summary - Resume conversa"""
        last_n = int(args.get("last", 0)) or None
        
        query = """
            SELECT sender_type, content, timestamp
            FROM messages
            WHERE conversation_id = $1
            ORDER BY timestamp DESC
        """
        
        if last_n:
            query += f" LIMIT {last_n}"
        
        messages = await db.fetch(query, conv_id)
        message_dicts = [dict(m) for m in reversed(messages)]
        
        # Usa PydanticAI agent
        summary = await self.summary_agent.summarize(message_dicts)
        
        return {
            "success": True,
            "summary": summary.model_dump(),
            "messages_analyzed": len(messages),
            "formatted": f"""
📝 **Resumo da Conversa**

**Tópico:** {summary.main_topic}

**Problema do Cliente:** {summary.customer_issue}

**Status Atual:** {summary.current_status}

**Próximos Passos:**
{chr(10).join(f"  • {step}" for step in summary.next_steps)}

**Sentimento:** {summary.sentiment_score:.2f} | **Urgência:** {summary.urgency}
            """
        }
    
    async def _handle_context(self, conv_id: str, args: dict) -> dict:
        """@context - Mostra contexto completo"""
        conv = await db.fetchrow("""
            SELECT * FROM conversations WHERE id = $1
        """, conv_id)
        
        messages = await db.fetch("""
            SELECT sender_type, content, timestamp
            FROM messages
            WHERE conversation_id = $1
            ORDER BY timestamp DESC
            LIMIT 10
        """, conv_id)
        
        return {
            "success": True,
            "conversation_id": str(conv['id']),
            "customer_id": conv['customer_id'],
            "is_hil": conv['is_hil'],
            "started_at": conv['created_at'].isoformat(),
            "last_activity": conv['last_activity_at'].isoformat(),
            "message_count": len(messages),
            "recent_messages": [
                {
                    "sender": m['sender_type'],
                    "content": m['content'][:100],
                    "timestamp": m['timestamp'].isoformat()
                }
                for m in messages
            ]
        }
    
    async def _handle_status(self, conv_id: str, args: dict) -> dict:
        """@status - Status da conversa"""
        conv = await db.fetchrow("""
            SELECT is_hil, last_activity_at FROM conversations WHERE id = $1
        """, conv_id)
        
        status = "🤖 Bot Ativo" if not conv['is_hil'] else "👤 Humano Ativo"
        
        from datetime import datetime, timezone
        inactive_minutes = (
            datetime.now(timezone.utc) - conv['last_activity_at']
        ).total_seconds() / 60
        
        return {
            "success": True,
            "status": status,
            "is_hil": conv['is_hil'],
            "last_activity": conv['last_activity_at'].isoformat(),
            "inactive_minutes": round(inactive_minutes, 1)
        }
    
    async def _handle_analyze(self, conv_id: str, args: dict) -> dict:
        """@analyze - Analisa conversa"""
        messages = await db.fetch("""
            SELECT sender_type, content, timestamp
            FROM messages
            WHERE conversation_id = $1
            ORDER BY timestamp ASC
        """, conv_id)
        
        message_dicts = [dict(m) for m in messages]
        
        # Usa AnalysisAgent
        analysis = await self.analysis_agent.analyze(message_dicts)
        
        return {
            "success": True,
            "analysis": analysis.model_dump(),
            "formatted": f"""
📊 **Análise da Conversa**

**Sentimento:** {analysis.sentiment}
**Urgência:** {analysis.urgency}
**Complexidade:** {analysis.complexity}

**Bot pode resolver?** {"✅ Sim" if analysis.bot_can_handle else "❌ Não"}
{f"**Razão:** {analysis.escalation_reason}" if analysis.escalation_reason else ""}

**Ações Recomendadas:**
{chr(10).join(f"  • {action}" for action in analysis.recommended_actions)}
            """
        }
    
    async def _handle_suggest(self, conv_id: str, args: dict) -> dict:
        """@suggest - Sugere resposta"""
        messages = await db.fetch("""
            SELECT sender_type, content
            FROM messages
            WHERE conversation_id = $1
            ORDER BY timestamp ASC
        """, conv_id)
        
        # Separa contexto e última mensagem
        customer_messages = [m for m in messages if m['sender_type'] == 'customer']
        last_customer_msg = customer_messages[-1]['content'] if customer_messages else ""
        
        context = "\n".join([
            f"[{m['sender_type']}] {m['content']}"
            for m in messages[:-1]
        ])
        
        # Usa SuggestionAgent
        suggestion = await self.suggestion_agent.suggest(context, last_customer_msg)
        
        return {
            "success": True,
            "suggestion": suggestion.model_dump(),
            "formatted": f"""
💡 **Sugestão de Resposta** (Confiança: {suggestion.confidence:.0%})

**Tom:** {suggestion.tone}

**Resposta Sugerida:**
{suggestion.suggested_response}

**Abordagens Alternativas:**
{chr(10).join(f"  • {alt}" for alt in suggestion.alternative_approaches)}

⚠️ _Esta é apenas uma sugestão. Revise antes de enviar._
            """
        }
    
    async def _handle_escalate(self, conv_id: str, args: dict) -> dict:
        """@escalate - Escala para supervisor"""
        reason = args.get("reason", "Requisitado pelo atendente")
        
        # Busca summary
        messages = await db.fetch("""
            SELECT sender_type, content, timestamp
            FROM messages WHERE conversation_id = $1
            ORDER BY timestamp ASC
        """, conv_id)
        
        message_dicts = [dict(m) for m in messages]
        summary = await self.summary_agent.summarize(message_dicts)
        
        # Atualiza conversa
        await db.execute("""
            UPDATE conversations 
            SET is_hil = true, escalated = true
            WHERE id = $1
        """, conv_id)
        
        # Audit
        await db.execute("""
            INSERT INTO conversation_audit 
            (conversation_id, actor, action, reason, details)
            VALUES ($1, 'agent', 'escalated', $2, $3)
        """, conv_id, reason, summary.model_dump_json())
        
        return {
            "success": True,
            "message": "Escalado para supervisor",
            "reason": reason,
            "summary": summary.model_dump()
        }
    
    async def _handle_transfer(self, conv_id: str, args: dict) -> dict:
        """@transfer - Transfere departamento"""
        department = args.get("department", "geral")
        reason = args.get("reason", "Transferência solicitada")
        
        await db.execute("""
            UPDATE conversations 
            SET department = $1
            WHERE id = $2
        """, department, conv_id)
        
        await db.execute("""
            INSERT INTO conversation_audit 
            (conversation_id, actor, action, reason, details)
            VALUES ($1, 'agent', 'transferred', $2, $3)
        """, conv_id, reason, f'{{"department": "{department}"}}')
        
        return {
            "success": True,
            "message": f"Transferido para {department}",
            "reason": reason
        }
    
    async def _handle_help(self, conv_id: str, args: dict) -> dict:
        """@help - Lista comandos"""
        return {
            "success": True,
            "commands": {
                "Controle": {
                    "@bot [instrução]": "Bot assume conversa",
                    "@pause": "Pausa bot",
                    "@resume": "Resume bot",
                },
                "Informações": {
                    "@summary [last:N]": "Resume conversa",
                    "@context": "Contexto completo",
                    "@status": "Status atual",
                },
                "Ações": {
                    "@analyze": "Analisa conversa",
                    "@suggest": "Sugere resposta",
                },
                "Escalação": {
                    "@escalate [reason:X]": "Escala para supervisor",
                    "@transfer department:X": "Transfere departamento",
                },
                "Utilidades": {
                    "@help": "Esta mensagem",
                }
            }
        }
    
    async def _get_composio_entity(self, conv_id: str) -> str:
        """Obtém entity ID do Composio para o customer"""
        customer_id = await db.fetchval("""
            SELECT customer_id FROM conversations WHERE id = $1
        """, conv_id)
        
        # Mapeia customer_id para Composio entity
        # (implementar mapeamento real)
        return f"entity_{customer_id}"
```

#### 4.2 - Timeout Monitor (app/services/timeout_monitor.py)

```python
import asyncio
import time
from datetime import datetime, timedelta
import structlog
from app.core.redis import redis_client
from app.core.database import db
from app.agents.pydantic_agents import SummaryAgent

logger = structlog.get_logger()

class TimeoutMonitor:
    def __init__(self):
        self.summary_agent = SummaryAgent()
    
    async def register_human_takeover(
        self, 
        conversation_id: str,
        timeout_minutes: int = 30
    ):
        """Registra timeout quando humano assume conversa"""
        expire_at = time.time() + (timeout_minutes * 60)
        
        # Adiciona à sorted set (score = expire timestamp)
        await redis_client.client.zadd(
            "timeout_queue",
            {conversation_id: expire_at}
        )
        
        logger.info(
            "timeout_registered",
            conversation_id=conversation_id,
            expires_at=datetime.fromtimestamp(expire_at).isoformat()
        )
    
    async def cancel_timeout(self, conversation_id: str):
        """Cancela timeout (quando bot assume ou humano responde)"""
        removed = await redis_client.client.zrem("timeout_queue", conversation_id)
        
        if removed:
            logger.info(
                "timeout_cancelled",
                conversation_id=conversation_id
            )
    
    async def handle_expired_conversation(self, conversation_id: str):
        """Processa conversa expirada por timeout"""
        
        # Adquire lock distribuído
        lock_key = f"lock:timeout:{conversation_id}"
        lock = await redis_client.client.set(lock_key, "1", nx=True, ex=60)
        
        if not lock:
            logger.debug(
                "timeout_already_being_processed",
                conversation_id=conversation_id
            )
            return
        
        try:
            # Verifica se ainda está em HIL
            conv = await db.fetchrow("""
                SELECT * FROM conversations 
                WHERE id = $1 AND is_hil = true AND closed = false
            """, conversation_id)
            
            if not conv:
                logger.info(
                    "timeout_conversation_already_closed",
                    conversation_id=conversation_id
                )
                return
            
            # Gera summary
            messages = await db.fetch("""
                SELECT sender_type, content, timestamp
                FROM messages 
                WHERE conversation_id = $1
                ORDER BY timestamp ASC
            """, conversation_id)
            
            message_dicts = [dict(m) for m in messages]
            summary = await self.summary_agent.summarize(message_dicts)
            
            # Envia mensagem de encerramento
            await self._send_to_customer(
                conv['customer_id'],
                "Esta conversa foi encerrada por inatividade. "
                "Se precisar de ajuda, inicie uma nova conversa."
            )
            
            # Atualiza conversa
            await db.execute("""
                UPDATE conversations
                SET 
                    is_hil = false,
                    closed = true,
                    closed_at = NOW(),
                    close_reason = 'timeout',
                    summary = $1
                WHERE id = $2
            """, summary.model_dump_json(), conversation_id)
            
            # Audit log
            await db.execute("""
                INSERT INTO conversation_audit
                (conversation_id, actor, action, old_is_hil, new_is_hil, reason, details)
                VALUES ($1, 'system', 'timeout_closed', true, false, 'inactivity_timeout', $2)
            """, conversation_id, summary.model_dump_json())
            
            logger.info(
                "timeout_conversation_closed",
                conversation_id=conversation_id
            )
            
            # Remove da fila
            await redis_client.client.zrem("timeout_queue", conversation_id)
            
        finally:
            # Libera lock
            await redis_client.client.delete(lock_key)
    
    async def worker_loop(self, poll_interval_seconds: int = 10):
        """Worker loop que processa timeouts"""
        logger.info("timeout_monitor_started", poll_interval=poll_interval_seconds)
        
        while True:
            try:
                now = time.time()
                
                # Busca conversas expiradas
                expired = await redis_client.client.zrangebyscore(
                    "timeout_queue",
                    min='-inf',
                    max=now,
                    start=0,
                    num=10
                )
                
                if expired:
                    logger.info(
                        "timeout_processing_batch",
                        count=len(expired)
                    )
                    
                    for conv_id in expired:
                        try:
                            await self.handle_expired_conversation(conv_id)
                        except Exception as e:
                            logger.error(
                                "timeout_processing_error",
                                conversation_id=conv_id,
                                error=str(e),
                                exc_info=True
                            )
                
                await asyncio.sleep(poll_interval_seconds)
                
            except Exception as e:
                logger.error(
                    "timeout_worker_loop_error",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(30)
    
    async def _send_to_customer(self, customer_id: str, message: str):
        """Envia mensagem para customer (implementar integração)"""
        logger.info(
            "sending_to_customer",
            customer_id=customer_id,
            message=message[:50]
        )
        # TODO: Integrar com canal apropriado (WhatsApp, Telegram, etc)

# Singleton
timeout_monitor = TimeoutMonitor()
```

---

### FASE 5: API Endpoints (Dia 12-14)

#### 5.1 - Webhooks (app/api/webhooks.py)

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.conversation import IncomingMessage
from app.services.command_handler import CommandParser, CommandHandler
from app.services.rate_limiter import RateLimiter
from app.services.timeout_monitor import timeout_monitor
from app.core.database import db
from app.core.redis import redis_client
from app.agents.langgraph_agent import build_bot_graph
from composio_langgraph import ComposioToolSet
import structlog
from datetime import datetime, timezone
import os

logger = structlog.get_logger()
router = APIRouter()

# Initialize components
langgraph_agent = build_bot_graph()
composio_toolset = ComposioToolSet()
command_handler = CommandHandler(langgraph_agent, composio_toolset)
rate_limiter = RateLimiter(redis_client.client)

@router.post("/message")
async def receive_message(
    message: IncomingMessage, 
    background_tasks: BackgroundTasks
):
    """
    Webhook principal que recebe mensagens de qualquer canal.
    Retorna 200 imediatamente e processa em background.
    """
    
    # Rate limiting
    if message.sender_type == "customer":
        within_limit, retry_after = await rate_limiter.check_customer_limit(message.customer_id)
        
        if not within_limit:
            logger.warning(
                "customer_rate_limit_exceeded",
                customer_id=message.customer_id
            )
            raise HTTPException(
                status_code=429,
                detail={"error": "Rate limit exceeded", "retry_after": retry_after},
                headers={"Retry-After": str(retry_after)}
            )
    
    # Idempotency check
    if message.origin_id:
        exists = await db.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM messages 
                WHERE conversation_id = $1 AND message_origin_id = $2
            )
        """, message.conversation_id, message.origin_id)
        
        if exists:
            logger.info(
                "duplicate_message_rejected",
                conversation_id=message.conversation_id,
                origin_id=message.origin_id
            )
            return {"status": "duplicate", "processed": False}
    
    # Processa em background
    background_tasks.add_task(process_message, message)
    
    return {"status": "received", "processed": True}

async def process_message(message: IncomingMessage):
    """Processa mensagem de forma assíncrona"""
    
    try:
        # Get ou cria conversa
        conv = await db.fetchrow("""
            INSERT INTO conversations (id, customer_id, last_activity_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (id) 
            DO UPDATE SET last_activity_at = NOW()
            RETURNING *
        """, message.conversation_id, message.customer_id if message.sender_type == "customer" else "unknown")
        
        # Salva mensagem
        await db.execute("""
            INSERT INTO messages 
            (conversation_id, sender_type, sender_id, content, message_origin_id)
            VALUES ($1, $2, $3, $4, $5)
        """, 
            message.conversation_id,
            message.sender_type,
            message.sender_id,
            message.content,
            message.origin_id
        )
        
        # Roteamento
        if message.sender_type == "human_agent":
            await handle_human_agent_message(conv, message)
        elif message.sender_type == "customer":
            await handle_customer_message(conv, message)
        
        logger.info(
            "message_processed",
            conversation_id=message.conversation_id,
            sender_type=message.sender_type
        )
        
    except Exception as e:
        logger.error(
            "message_processing_failed",
            conversation_id=message.conversation_id,
            error=str(e),
            exc_info=True
        )

async def handle_human_agent_message(conv, message: IncomingMessage):
    """Processa mensagem do atendente humano"""
    
    # Tenta parsear comando
    command, args = CommandParser.parse(message.content)
    
    if command:
        # É um comando - executa
        result = await command_handler.handle(
            message.conversation_id,
            command,
            args
        )
        
        # Envia resultado para agente
        await send_to_agent(message.conversation_id, result.get("formatted", result))
        
        # Se foi @bot, cancela timeout
        if command == "@bot":
            await timeout_monitor.cancel_timeout(message.conversation_id)
    else:
        # Mensagem normal - humano assume
        await db.execute("""
            UPDATE conversations 
            SET is_hil = true, version = version + 1
            WHERE id = $1
        """, message.conversation_id)
        
        # Audit
        await db.execute("""
            INSERT INTO conversation_audit 
            (conversation_id, actor, action, old_is_hil, new_is_hil, reason)
            VALUES ($1, $2, 'hil_activated', false, true, 'agent_message')
        """, message.conversation_id, f"agent_{message.sender_id}")
        
        # Inicia timeout monitor
        timeout_minutes = int(os.getenv("HUMAN_TIMEOUT_MINUTES", 30))
        await timeout_monitor.register_human_takeover(
            message.conversation_id,
            timeout_minutes
        )
        
        # Encaminha para customer
        await send_to_customer(conv['customer_id'], message.content)

async def handle_customer_message(conv, message: IncomingMessage):
    """Processa mensagem do cliente"""
    
    if conv['is_hil']:
        # Humano no controle - encaminha para atendente
        await send_to_agent(message.conversation_id, {
            "type": "customer_message",
            "content": message.content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Renova timeout
        timeout_minutes = int(os.getenv("HUMAN_TIMEOUT_MINUTES", 30))
        await timeout_monitor.register_human_takeover(
            message.conversation_id,
            timeout_minutes
        )
    else:
        # Bot no controle - processa via LangGraph
        try:
            # Busca histórico
            history = await db.fetch("""
                SELECT sender_type, content
                FROM messages
                WHERE conversation_id = $1
                ORDER BY timestamp ASC
                LIMIT 50
            """, message.conversation_id)
            
            history_list = [
                {"role": "user" if m['sender_type'] == "customer" else "assistant", 
                 "content": m['content']}
                for m in history
            ]
            
            # Invoca LangGraph
            entity_id = await get_composio_entity(conv['customer_id'])
            
            result = langgraph_agent.invoke({
                "messages": history_list + [{"role": "user", "content": message.content}],
                "composio_entity_id": entity_id
            })
            
            # Verifica escalação
            if result.get("should_escalate"):
                # Bot decidiu escalar
                await db.execute("""
                    UPDATE conversations 
                    SET is_hil = true, escalated = true
                    WHERE id = $1
                """, message.conversation_id)
                
                # Notifica humano
                await notify_human_agent(
                    message.conversation_id,
                    escalation_reason=result.get("escalation_reason", "Bot auto-escalation"),
                    context=result.get("summary", "")
                )
            else:
                # Bot respondeu normalmente
                await send_to_customer(
                    conv['customer_id'],
                    result["final_response"]
                )
                
                # Salva resposta do bot
                bot_msg_id = await db.fetchval("""
                    INSERT INTO messages 
                    (conversation_id, sender_type, sender_id, content)
                    VALUES ($1, 'ai_agent', 'bot', $2)
                    RETURNING id
                """, message.conversation_id, result["final_response"])
                
                await db.execute("""
                    UPDATE conversations 
                    SET last_bot_message_id = $1
                    WHERE id = $2
                """, bot_msg_id, message.conversation_id)
                
        except Exception as e:
            logger.error(
                "bot_processing_error",
                conversation_id=message.conversation_id,
                error=str(e),
                exc_info=True
            )
            
            # Erro no bot - escala para humano
            await db.execute("""
                UPDATE conversations 
                SET is_hil = true
                WHERE id = $1
            """, message.conversation_id)
            
            await notify_human_agent(
                message.conversation_id,
                escalation_reason=f"Bot error: {str(e)}",
                error=True
            )

async def send_to_customer(customer_id: str, content: str):
    """Envia mensagem para customer"""
    logger.info("sending_to_customer", customer_id=customer_id, content_length=len(content))
    # TODO: Implementar integração com canal (WhatsApp, Telegram, Web, etc)
    pass

async def send_to_agent(conversation_id: str, content):
    """Envia mensagem/resultado para atendente"""
    logger.info("sending_to_agent", conversation_id=conversation_id)
    # TODO: Implementar integração (dashboard, Slack, etc)
    pass

async def notify_human_agent(conversation_id: str, escalation_reason: str, **kwargs):
    """Notifica atendente sobre escalação"""
    logger.info(
        "notifying_human_agent",
        conversation_id=conversation_id,
        reason=escalation_reason
    )
    # TODO: Implementar notificação (email, Slack, push, etc)
    pass

async def get_composio_entity(customer_id: str) -> str:
    """Obtém ou cria entity ID do Composio"""
    # Verifica se já existe
    entity_id = await db.fetchval("""
        SELECT composio_entity_id 
        FROM customer_composio_mapping 
        WHERE customer_id = $1
    """, customer_id)
    
    if entity_id:
        return entity_id
    
    # Cria novo entity no Composio
    # entity_id = composio_toolset.create_entity(name=customer_id)
    entity_id = f"entity_{customer_id}"  # Placeholder
    
    # Salva mapeamento
    await db.execute("""
        INSERT INTO customer_composio_mapping (customer_id, composio_entity_id)
        VALUES ($1, $2)
        ON CONFLICT (customer_id) DO NOTHING
    """, customer_id, entity_id)
    
    return entity_id
```

#### 5.2 - Health Checks (app/api/health.py)

```python
from fastapi import APIRouter
from app.core.database import db
from app.core.redis import redis_client
import structlog

logger = structlog.get_logger()
router = APIRouter()

@router.get("/")
async def health_check():
    """Health check básico"""
    return {"status": "healthy", "service": "HIL System"}

@router.get("/ready")
async def readiness_check():
    """Readiness check - verifica dependências"""
    
    checks = {
        "database": False,
        "redis": False
    }
    
    # Check database
    try:
        await db.fetchval("SELECT 1")
        checks["database"] = True
    except Exception as e:
        logger.error("database_health_check_failed", error=str(e))
    
    # Check Redis
    try:
        await redis_client.client.ping()
        checks["redis"] = True
    except Exception as e:
        logger.error("redis_health_check_failed", error=str(e))
    
    all_healthy = all(checks.values())
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": checks
    }

@router.get("/stats")
async def get_stats():
    """Estatísticas do sistema"""
    
    stats = await db.fetchrow("""
        SELECT 
            COUNT(*) FILTER (WHERE NOT closed) as active_conversations,
            COUNT(*) FILTER (WHERE is_hil = false AND NOT closed) as bot_active,
            COUNT(*) FILTER (WHERE is_hil = true AND NOT closed) as human_active,
            COUNT(*) FILTER (WHERE escalated = true) as escalated_total
        FROM conversations
    """)
    
    return {
        "active_conversations": stats['active_conversations'],
        "bot_active": stats['bot_active'],
        "human_active": stats['human_active'],
        "escalated_total": stats['escalated_total']
    }
```

#### 5.3 - GDPR Endpoints (app/api/gdpr.py)

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from app.services.gdpr_service import GDPRService
from app.core.database import db
from app.core.redis import redis_client
import structlog

logger = structlog.get_logger()
router = APIRouter()

gdpr_service = GDPRService(db, redis_client.client)

class GDPRDeleteRequest(BaseModel):
    requester_email: EmailStr
    confirmation_token: str
    reason: str = "customer_request"

@router.delete("/customer/{customer_id}")
async def delete_customer_data(
    customer_id: str,
    request: GDPRDeleteRequest,
    background_tasks: BackgroundTasks
):
    """
    GDPR/LGPD: Delete customer data.
    Requer token de confirmação enviado por email.
    """
    
    # Verifica token
    token_valid = await verify_gdpr_token(customer_id, request.confirmation_token)
    if not token_valid:
        raise HTTPException(403, "Invalid confirmation token")
    
    # Executa deleção em background
    background_tasks.add_task(
        gdpr_service.delete_customer_data,
        customer_id,
        request.requester_email,
        request.reason
    )
    
    return {
        "status": "queued",
        "message": "Deletion request queued. You will receive confirmation email within 24h."
    }

@router.post("/customer/{customer_id}/request-deletion")
async def request_deletion_token(customer_id: str, email: EmailStr):
    """
    Solicita token de confirmação para deleção.
    Envia email com token.
    """
    
    # Gera token
    import secrets
    token = secrets.token_urlsafe(32)
    
    # Salva token (válido por 24h)
    await redis_client.client.setex(
        f"gdpr_token:{customer_id}",
        86400,  # 24 horas
        token
    )
    
    # Envia email
    # TODO: Implementar envio de email
    logger.info(
        "gdpr_token_generated",
        customer_id=customer_id,
        email=email
    )
    
    return {
        "status": "sent",
        "message": "Confirmation token sent to email"
    }

async def verify_gdpr_token(customer_id: str, token: str) -> bool:
    """Verifica se token é válido"""
    stored_token = await redis_client.client.get(f"gdpr_token:{customer_id}")
    return stored_token == token
```

---

### FASE 6: GDPR Service & RQ Workers (Dia 15-17)

#### 6.1 - GDPR Service (app/services/gdpr_service.py)

```python
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class GDPRService:
    def __init__(self, db_pool, redis_client):
        self.db = db_pool
        self.redis = redis_client
    
    async def delete_customer_data(
        self, 
        customer_id: str,
        requester_email: str,
        reason: str = "customer_request"
    ) -> dict:
        """
        Deleta todos dados de um customer (LGPD/GDPR).
        """
        
        logger.info("starting_gdpr_deletion", customer_id=customer_id)
        
        try:
            # 1. Find all conversations
            conversations = await self.db.fetch("""
                SELECT id FROM conversations 
                WHERE customer_id = $1
            """, customer_id)
            
            conv_ids = [str(c['id']) for c in conversations]
            
            if not conv_ids:
                logger.warning("no_data_found", customer_id=customer_id)
                return {
                    "status": "completed",
                    "customer_id": customer_id,
                    "conversations_deleted": 0,
                    "messages_deleted": 0
                }
            
            # 2. Soft delete messages
            deleted_messages = await self.db.fetchval("""
                UPDATE messages 
                SET 
                    deleted_at = NOW(),
                    content = '[DELETED PER GDPR REQUEST]'
                WHERE conversation_id = ANY($1::uuid[])
                AND deleted_at IS NULL
                RETURNING COUNT(*)
            """, conv_ids)
            
            # 3. Anonymize conversations
            await self.db.execute("""
                UPDATE conversations
                SET 
                    customer_id = 'ANONYMIZED_' || id::text,
                    last_activity_at = NOW()
                WHERE id = ANY($1::uuid[])
            """, conv_ids)
            
            # 4. Log deletion request
            await self.db.execute("""
                INSERT INTO gdpr_deletion_requests
                (customer_id, requester_email, reason, conversations_affected, 
                 messages_deleted, completed_at, status)
                VALUES ($1, $2, $3, $4, $5, NOW(), 'completed')
            """, customer_id, requester_email, reason, len(conv_ids), deleted_messages)
            
            # 5. Invalida caches
            for conv_id in conv_ids:
                await self.redis.delete(f"conversation:{conv_id}")
            
            logger.info(
                "gdpr_deletion_completed",
                customer_id=customer_id,
                conversations_affected=len(conv_ids),
                messages_deleted=deleted_messages
            )
            
            return {
                "status": "completed",
                "customer_id": customer_id,
                "conversations_deleted": len(conv_ids),
                "messages_deleted": deleted_messages,
                "completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(
                "gdpr_deletion_failed",
                customer_id=customer_id,
                error=str(e),
                exc_info=True
            )
            
            # Log failure
            await self.db.execute("""
                INSERT INTO gdpr_deletion_requests
                (customer_id, requester_email, reason, status)
                VALUES ($1, $2, $3, 'failed')
            """, customer_id, requester_email, reason)
            
            raise

    async def cleanup_expired_data(self):
        """
        Cleanup automático baseado em políticas de retenção.
        Roda diariamente via RQ scheduler.
        """
        
        logger.info("starting_retention_cleanup")
        
        # Messages - 90 dias
        cutoff_messages = datetime.now() - timedelta(days=90)
        
        deleted_messages = await self.db.fetchval("""
            UPDATE messages 
            SET deleted_at = NOW(), content = '[EXPIRED]'
            WHERE timestamp < $1
            AND deleted_at IS NULL
            RETURNING COUNT(*)
        """, cutoff_messages)
        
        logger.info(
            "retention_cleanup_messages",
            deleted_count=deleted_messages
        )
        
        # Conversations - anonymize após 2 anos
        cutoff_conversations = datetime.now() - timedelta(days=730)
        
        anonymized = await self.db.fetchval("""
            UPDATE conversations
            SET customer_id = 'ANONYMIZED_' || id::text
            WHERE created_at < $1
            AND customer_id NOT LIKE 'ANONYMIZED_%'
            AND closed = true
            RETURNING COUNT(*)
        """, cutoff_conversations)
        
        logger.info(
            "retention_cleanup_conversations",
            anonymized_count=anonymized
        )
        
        return {
            "messages_deleted": deleted_messages,
            "conversations_anonymized": anonymized
        }
```

#### 6.2 - RQ Worker Setup (app/workers/rq_worker.py)

```python
from redis import Redis
from rq import Worker, Queue
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.database import db
import asyncio

# Redis connection
redis_conn = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

# Define queues (ordem = prioridade)
queues = [
    Queue('notifications', connection=redis_conn),  # Alta prioridade
    Queue('reports', connection=redis_conn),        # Média
    Queue('retention', connection=redis_conn)       # Baixa (noturno)
]

if __name__ == '__main__':
    # Inicializa database
    asyncio.run(db.connect())
    
    # Start worker
    worker = Worker(queues, connection=redis_conn)
    worker.work()
```

#### 6.3 - RQ Tasks (app/workers/tasks/retention.py)

```python
import asyncio
from datetime import datetime, timedelta
import logging
from app.services.gdpr_service import GDPRService
from app.core.database import db
from app.core.redis import redis_client

logger = logging.getLogger(__name__)

async def cleanup_expired_data_async():
    """Task assíncrona de cleanup"""
    
    # Conecta ao database
    await db.connect()
    await redis_client.connect()
    
    try:
        gdpr_service = GDPRService(db, redis_client.client)
        result = await gdpr_service.cleanup_expired_data()
        
        logger.info(f"Cleanup completed: {result}")
        return result
        
    finally:
        await db.disconnect()
        await redis_client.disconnect()

def cleanup_expired_data():
    """
    Wrapper síncrono para RQ.
    Cleanup de dados expirados (GDPR/LGPD).
    """
    return asyncio.run(cleanup_expired_data_async())
```

#### 6.4 - RQ Scheduler Setup (app/workers/scheduler.py)

```python
from rq_scheduler import Scheduler
from redis import Redis
from datetime import datetime
import os

# Redis connection
redis_conn = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD")
)

scheduler = Scheduler(connection=redis_conn, queue_name='retention')

# Schedule cleanup job (diário às 3 AM)
scheduler.cron(
    "0 3 * * *",  # Cron expression
    func='app.workers.tasks.retention.cleanup_expired_data',
    timeout='30m'
)

if __name__ == '__main__':
    print("RQ Scheduler started. Cleanup job scheduled for 3 AM daily.")
    scheduler.run()
```

---

### FASE 7: Observability & Metrics (Dia 18-20)

#### 7.1 - Prometheus Metrics (app/core/metrics.py)

```python
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from typing import Literal

class MetricsRegistry:
    """Registry centralizado com cardinality controlada"""
    
    # Modelos permitidos
    ALLOWED_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
    ALLOWED_OUTCOMES = ["success", "error", "timeout", "rate_limit"]
    ALLOWED_COMMANDS = ["bot", "pause", "resume", "summary", "analyze", "suggest"]
    
    def __init__(self):
        # LLM metrics
        self.llm_calls_total = Counter(
            'llm_calls_total',
            'Total LLM API calls',
            ['model', 'outcome']
        )
        
        self.llm_duration = Histogram(
            'llm_call_duration_seconds',
            'LLM call duration',
            ['model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.llm_tokens_total = Counter(
            'llm_tokens_total',
            'Total tokens consumed',
            ['model', 'type']
        )
        
        self.llm_cost_usd = Counter(
            'llm_cost_usd_total',
            'Total LLM cost in USD',
            ['model']
        )
        
        # Command metrics
        self.commands_executed = Counter(
            'commands_executed_total',
            'Total commands executed',
            ['command', 'outcome']
        )
        
        # Conversation metrics
        self.conversations_active = Gauge(
            'conversations_active',
            'Currently active conversations'
        )
        
        self.conversations_bot_active = Gauge(
            'conversations_bot_active',
            'Conversations with bot active'
        )
        
        self.conversations_human_active = Gauge(
            'conversations_human_active',
            'Conversations with human active'
        )
        
        # Escalation metrics
        self.escalations_total = Counter(
            'escalations_total',
            'Total escalations to human',
            ['reason']
        )
        
        # Rate limit metrics
        self.rate_limit_exceeded = Counter(
            'rate_limit_exceeded_total',
            'Rate limit violations',
            ['entity_type', 'limit_type']
        )
    
    def record_llm_call(
        self,
        model: str,
        outcome: str,
        duration: float,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        cost_usd: float = 0.0
    ):
        """Record LLM call com validação"""
        
        if model not in self.ALLOWED_MODELS:
            model = "unknown"
        
        if outcome not in self.ALLOWED_OUTCOMES:
            outcome = "unknown"
        
        self.llm_calls_total.labels(model=model, outcome=outcome).inc()
        self.llm_duration.labels(model=model).observe(duration)
        
        if tokens_prompt > 0:
            self.llm_tokens_total.labels(model=model, type='prompt').inc(tokens_prompt)
        if tokens_completion > 0:
            self.llm_tokens_total.labels(model=model, type='completion').inc(tokens_completion)
        if cost_usd > 0:
            self.llm_cost_usd.labels(model=model).inc(cost_usd)
    
    def record_command(self, command: str, outcome: str):
        """Record command execution"""
        command = command.lstrip('@')
        
        if command not in self.ALLOWED_COMMANDS:
            command = "unknown"
        
        self.commands_executed.labels(command=command, outcome=outcome).inc()
    
    def record_escalation(self, reason: str):
        """Record escalation"""
        self.escalations_total.labels(reason=reason).inc()

# Singleton
metrics = MetricsRegistry()

# Prometheus ASGI app
metrics_app = make_asgi_app()
```

#### 7.2 - Structured Logging (app/core/logging_config.py)

```python
import structlog
import logging
import sys

def setup_logging():
    """Configura logging estruturado"""
    
    # Processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # JSON em produção, console em dev
    import os
    if os.getenv("ENVIRONMENT") == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configura logging padrão
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

# Inicializa
setup_logging()
```

#### 7.3 - OpenTelemetry Tracing (app/core/tracing.py)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import os

def setup_tracing(app):
    """Configura OpenTelemetry tracing"""
    
    if not os.getenv("ENABLE_TRACING", "false").lower() == "true":
        return
    
    # Setup provider
    trace.set_tracer_provider(TracerProvider())
    
    # Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_PORT", 6831)),
    )
    
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    # Instrumenta FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    return trace.get_tracer(__name__)

# Tracer global
tracer = None
```

---

### FASE 8: Docker & Deployment (Dia 21-22)

#### 8.1 - Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download pt_core_news_lg

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 8.2 - docker-compose.yml

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://hiluser:hilpassword@db:5432/hildb
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - COMPOSIO_API_KEY=${COMPOSIO_API_KEY}
      - ENABLE_METRICS=true
      - ENABLE_TRACING=true
      - JAEGER_HOST=jaeger
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - hilnet
    volumes:
      - ./logs:/app/logs

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=hildb
      - POSTGRES_USER=hiluser
      - POSTGRES_PASSWORD=hilpassword
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations:/docker-entrypoint-initdb.d
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hiluser -d hildb"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - hilnet
    ports:
      - "5432:5432"  # Remover em produção

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    networks:
      - hilnet
    ports:
      - "6379:6379"  # Remover em produção

  rq-worker:
    build: .
    command: python app/workers/rq_worker.py
    environment:
      - DATABASE_URL=postgresql://hiluser:hilpassword@db:5432/hildb
      - REDIS_URL=redis://redis:6379
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
      - db
    deploy:
      replicas: 2
    restart: unless-stopped
    networks:
      - hilnet

  rq-scheduler:
    build: .
    command: python app/workers/scheduler.py
    environment:
      - REDIS_URL=redis://redis:6379
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - hilnet

  rq-dashboard:
    image: eoranged/rq-dashboard
    ports:
      - "9181:9181"
    environment:
      - RQ_DASHBOARD_REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    networks:
      - hilnet

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - hilnet

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - hilnet

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "6831:6831/udp"
      - "16686:16686"
    networks:
      - hilnet

networks:
  hilnet:
    driver: bridge

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
```

#### 8.3 - prometheus.yml

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hil-system'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
```

---

### FASE 9: Tests (Dia 23-25)

#### 9.1 - Pytest Configuration (conftest.py)

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from app.models.conversation import Conversation, Message
from app.agents.pydantic_agents import ConversationSummary, ConversationAnalysis

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_db():
    """Mock database"""
    db = AsyncMock()
    db.fetchrow = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    db.fetchval = AsyncMock(return_value=1)
    return db

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.delete = AsyncMock(return_value=1)
    redis.zadd = AsyncMock(return_value=1)
    redis.zrem = AsyncMock(return_value=1)
    return redis

@pytest.fixture
def mock_summary_agent():
    """Mock PydanticAI SummaryAgent"""
    agent = AsyncMock()
    agent.summarize = AsyncMock(return_value=ConversationSummary(
        main_topic="Test topic",
        customer_issue="Test issue",
        current_status="Test status",
        next_steps=["Step 1", "Step 2"],
        sentiment_score=0.5,
        urgency="medium"
    ))
    return agent

@pytest.fixture
def mock_analysis_agent():
    """Mock PydanticAI AnalysisAgent"""
    agent = AsyncMock()
    agent.analyze = AsyncMock(return_value=ConversationAnalysis(
        sentiment="neutral",
        urgency="medium",
        complexity="simple",
        recommended_actions=["Test action"],
        bot_can_handle=True,
        escalation_reason=None
    ))
    return agent

@pytest.fixture
def sample_conversation():
    """Sample conversation data"""
    from uuid import uuid4
    from datetime import datetime
    
    return Conversation(
        id=uuid4(),
        customer_id="cust_123",
        is_hil=False,
        last_activity_at=datetime.now(),
        escalated=False,
        closed=False,
        created_at=datetime.now()
    )
```

#### 9.2 - Test Commands (tests/test_commands.py)

```python
import pytest
from app.services.command_handler import CommandParser, CommandHandler

class TestCommandParser:
    def test_parse_simple_command(self):
        command, args = CommandParser.parse("@summary")
        assert command == "@summary"
        assert args == {}
    
    def test_parse_command_with_args(self):
        command, args = CommandParser.parse("@summary last:10")
        assert command == "@summary"
        assert args == {"last": "10"}
    
    def test_parse_bot_command(self):
        command, args = CommandParser.parse("@bot Continue with customer")
        assert command == "@bot"
        assert args == {"instruction": "Continue with customer"}
    
    def test_parse_escalate_with_reason(self):
        command, args = CommandParser.parse("@escalate reason:Customer unhappy")
        assert command == "@escalate"
        assert args == {"reason": "Customer unhappy"}
    
    def test_parse_non_command(self):
        command, args = CommandParser.parse("Hello customer")
        assert command is None
        assert args == {}

@pytest.mark.asyncio
class TestCommandHandler:
    async def test_summary_command(self, mock_db, mock_summary_agent):
        handler = CommandHandler(None, None)
        handler.summary_agent = mock_summary_agent
        
        mock_db.fetch.return_value = [
            {"sender_type": "customer", "content": "Hello", "timestamp": "2025-01-01"}
        ]
        
        result = await handler._handle_summary("conv_123", {})
        
        assert result["success"] == True
        assert "summary" in result
        assert "formatted" in result
        mock_summary_agent.summarize.assert_called_once()

    async def test_status_command(self, mock_db):
        handler = CommandHandler(None, None)
        
        mock_db.fetchrow.return_value = {
            "is_hil": False,
            "last_activity_at": "2025-01-01T10:00:00"
        }
        
        result = await handler._handle_status("conv_123", {})
        
        assert result["success"] == True
        assert result["is_hil"] == False
        assert "🤖" in result["status"]

    async def test_escalate_command(self, mock_db, mock_summary_agent):
        handler = CommandHandler(None, None)
        handler.summary_agent = mock_summary_agent
        
        mock_db.fetch.return_value = []
        
        result = await handler._handle_escalate("conv_123", {"reason": "Test"})
        
        assert result["success"] == True
        assert result["reason"] == "Test"
        mock_db.execute.assert_called()
```

#### 9.3 - Test Concurrency (tests/test_concurrency.py)

```python
import pytest
import asyncio
from app.services.conversation import ConversationRepository

@pytest.mark.asyncio
class TestConcurrency:
    async def test_pessimistic_lock_prevents_race_condition(self, mock_db):
        """Testa que lock pessimístico previne race conditions"""
        repo = ConversationRepository(mock_db)
        
        # Simula lock conflict
        mock_db.fetchrow.side_effect = Exception("could not obtain lock")
        
        with pytest.raises(Exception):
            async with repo.lock_conversation("conv_123"):
                pass
    
    async def test_concurrent_updates_serialized(self, mock_db):
        """Testa que updates concorrentes são serializados"""
        repo = ConversationRepository(mock_db)
        
        update_count = 0
        
        async def update_conversation():
            nonlocal update_count
            await repo.update_is_hil("conv_123", True, "test", "test")
            update_count += 1
        
        # Executa múltiplos updates concorrentes
        await asyncio.gather(
            update_conversation(),
            update_conversation(),
            update_conversation()
        )
        
        # Todos devem completar
        assert update_count == 3
```

#### 9.4 - Test GDPR (tests/test_gdpr.py)

```python
import pytest
from app.services.gdpr_service import GDPRService

@pytest.mark.asyncio
class TestGDPR:
    async def test_delete_customer_data(self, mock_db, mock_redis):
        service = GDPRService(mock_db, mock_redis)
        
        # Mock conversations
        mock_db.fetch.return_value = [
            {"id": "conv_1"},
            {"id": "conv_2"}
        ]
        
        mock_db.fetchval.return_value = 10  # messages deleted
        
        result = await service.delete_customer_data(
            "cust_123",
            "test@example.com",
            "test"
        )
        
        assert result["status"] == "completed"
        assert result["conversations_deleted"] == 2
        assert result["messages_deleted"] == 10
        
        # Verifica que soft delete foi chamado
        assert mock_db.execute.call_count >= 3  # messages, conversations, log
    
    async def test_cleanup_expired_data(self, mock_db, mock_redis):
        service = GDPRService(mock_db, mock_redis)
        
        mock_db.fetchval.side_effect = [5, 3]  # messages, conversations
        
        result = await service.cleanup_expired_data()
        
        assert result["messages_deleted"] == 5
        assert result["conversations_anonymized"] == 3
```

#### 9.5 - Test Rate Limiting (tests/test_rate_limit.py)

```python
import pytest
from app.services.rate_limiter import RateLimiter

@pytest.mark.asyncio
class TestRateLimiter:
    async def test_within_limit(self, mock_redis):
        limiter = RateLimiter(mock_redis)
        
        # Mock Redis pipeline
        mock_pipeline = mock_redis.pipeline.return_value
        mock_pipeline.execute.return_value = [None, None, 5, None]  # 5 requests
        
        within_limit, retry_after = await limiter.check_limit(
            "test_key",
            limit=10,
            window_seconds=60
        )
        
        assert within_limit == True
        assert retry_after == 0
    
    async def test_exceeded_limit(self, mock_redis):
        limiter = RateLimiter(mock_redis)
        
        # Mock Redis pipeline
        mock_pipeline = mock_redis.pipeline.return_value
        mock_pipeline.execute.return_value = [None, None, 11, None]  # 11 requests
        
        mock_redis.zrange.return_value = [(b"1234567890", 1234567890.0)]
        
        within_limit, retry_after = await limiter.check_limit(
            "test_key",
            limit=10,
            window_seconds=60
        )
        
        assert within_limit == False
        assert retry_after > 0
```

#### 9.6 - Test PII Sanitization (tests/test_pii.py)

```python
import pytest
from app.services.pii_sanitizer import PIISanitizer

class TestPIISanitizer:
    def test_redact_email(self):
        sanitizer = PIISanitizer(use_ner=False)
        
        text = "My email is john@example.com"
        redacted, info = sanitizer.redact_full(text)
        
        assert "john@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert info["redacted"] == True
    
    def test_redact_cpf(self):
        sanitizer = PIISanitizer(use_ner=False)
        
        text = "Meu CPF é 123.456.789-00"
        redacted, info = sanitizer.redact_full(text)
        
        assert "123.456.789-00" not in redacted
        assert "[CPF_REDACTED]" in redacted
    
    def test_redact_phone(self):
        sanitizer = PIISanitizer(use_ner=False)
        
        text = "Telefone: (11) 98765-4321"
        redacted, info = sanitizer.redact_full(text)
        
        assert "98765-4321" not in redacted
        assert "[PHONE_BR_REDACTED]" in redacted
    
    def test_no_pii_detected(self):
        sanitizer = PIISanitizer(use_ner=False)
        
        text = "Hello world"
        redacted, info = sanitizer.redact_full(text)
        
        assert redacted == text
        assert info["redacted"] == False
```

---

### FASE 10: Scripts & Utilities (Dia 26-27)

#### 10.1 - Migration Runner (scripts/run_migrations.py)

```python
import asyncio
import asyncpg
import os
from pathlib import Path

async def run_migrations():
    """Executa migrations SQL"""
    
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    
    migrations_dir = Path(__file__).parent.parent / "migrations"
    
    # Cria tabela de controle de migrations
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    
    # Busca migrations aplicadas
    applied = await conn.fetch("SELECT version FROM schema_migrations")
    applied_versions = {row['version'] for row in applied}
    
    # Lista migrations
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    for migration_file in migration_files:
        version = migration_file.stem
        
        if version in applied_versions:
            print(f"⏭️  Skipping {version} (already applied)")
            continue
        
        print(f"▶️  Applying {version}...")
        
        # Lê SQL
        sql = migration_file.read_text()
        
        # Executa em transaction
        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                version
            )
        
        print(f"✅ Applied {version}")
    
    await conn.close()
    print("✅ All migrations applied")

if __name__ == "__main__":
    asyncio.run(run_migrations())
```

#### 10.2 - Check PII in Logs (scripts/check_pii.py)

```python
import re
import sys
from pathlib import Path

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'cpf': r'\b\d{3}\.?\d{3}\.?\d{3}[-]?\d{2}\b',
}

def scan_logs_for_pii(log_dir='./logs'):
    """Escaneia logs em busca de PII"""
    violations = []
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"⚠️  Log directory {log_dir} not found")
        return violations
    
    for log_file in log_path.glob('*.log'):
        with open(log_file) as f:
            for line_num, line in enumerate(f, 1):
                for pii_type, pattern in PII_PATTERNS.items():
                    if re.search(pattern, line):
                        violations.append({
                            'file': log_file.name,
                            'line': line_num,
                            'type': pii_type,
                            'content': line.strip()[:100]
                        })
    
    if violations:
        print("❌ PII FOUND IN LOGS:")
        for v in violations:
            print(f"  {v['file']}:{v['line']} - {v['type']}")
        sys.exit(1)
    
    print("✅ No PII detected in logs")

if __name__ == '__main__':
    scan_logs_for_pii()
```

#### 10.3 - Check Metric Cardinality (scripts/check_metrics.py)

```python
import re
import sys
from pathlib import Path

def check_metric_cardinality():
    """Verifica cardinality de métricas"""
    
    # Busca por uso de métricas no código
    violations = []
    
    for py_file in Path('app').rglob('*.py'):
        with open(py_file) as f:
            content = f.read()
            
            # Busca por labels com valores variáveis suspeitos
            # Ex: .labels(user_id=user_id) <- RUIM
            suspicious_patterns = [
                r'\.labels\([^)]*user_id[^)]*\)',
                r'\.labels\([^)]*customer_id[^)]*\)',
                r'\.labels\([^)]*conversation_id[^)]*\)',
                r'\.labels\([^)]*message_id[^)]*\)',
            ]
            
            for pattern in suspicious_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    violations.append({
                        'file': str(py_file),
                        'pattern': match.group(),
                        'issue': 'High-cardinality label detected'
                    })
    
    if violations:
        print("❌ HIGH-CARDINALITY METRICS DETECTED:")
        for v in violations:
            print(f"  {v['file']}: {v['pattern']}")
            print(f"    Issue: {v['issue']}")
        sys.exit(1)
    
    print("✅ No high-cardinality metrics detected")

if __name__ == '__main__':
    check_metric_cardinality()
```

#### 10.4 - Seed Data (scripts/seed_data.py)

```python
import asyncio
import asyncpg
import os
from uuid import uuid4
from datetime import datetime

async def seed_data():
    """Popula database com dados de teste"""
    
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    
    print("🌱 Seeding data...")
    
    # Cria agents
    agents = [
        ("agent_1", "João Silva", "joao@example.com", "operator"),
        ("agent_2", "Maria Santos", "maria@example.com", "supervisor"),
        ("agent_3", "Admin User", "admin@example.com", "admin"),
    ]
    
    for agent_id, name, email, role in agents:
        await conn.execute("""
            INSERT INTO agents (id, name, email, role)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO NOTHING
        """, agent_id, name, email, role)
        print(f"  ✅ Created agent: {name} ({role})")
    
    # Cria conversas de teste
    conversations = []
    for i in range(5):
        conv_id = uuid4()
        customer_id = f"cust_{i+1}"
        
        await conn.execute("""
            INSERT INTO conversations 
            (id, customer_id, is_hil, last_activity_at, created_at)
            VALUES ($1, $2, $3, NOW(), NOW())
        """, conv_id, customer_id, i % 2 == 0)
        
        conversations.append((conv_id, customer_id))
        print(f"  ✅ Created conversation: {conv_id}")
        
        # Adiciona mensagens
        messages = [
            ("customer", customer_id, "Olá, preciso de ajuda"),
            ("ai_agent", "bot", "Olá! Como posso ajudar?"),
            ("customer", customer_id, "Onde está meu pedido?"),
        ]
        
        for sender_type, sender_id, content in messages:
            await conn.execute("""
                INSERT INTO messages 
                (conversation_id, sender_type, sender_id, content)
                VALUES ($1, $2, $3, $4)
            """, conv_id, sender_type, sender_id, content)
    
    print("✅ Seed data created")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(seed_data())
```

---

### FASE 11: README & Documentation (Dia 28)

#### 11.1 - README.md

```markdown
# Sistema HIL (Human-in-the-Loop)

Sistema de conversação híbrido bot-humano com transição suave entre atendimento automatizado e manual.

## 🎯 Features

- ✅ Transição suave bot ↔ humano
- ✅ Comandos avançados (@summary, @analyze, @suggest, etc)
- ✅ 150+ integrações via Composio (Shopify, Salesforce, Gmail, etc)
- ✅ Análise de sentimento e contexto (PydanticAI)
- ✅ Timeout automático com Redis Sorted Set
- ✅ Rate limiting por customer e agent
- ✅ PII redaction (Regex + NER)
- ✅ GDPR/LGPD compliance (deleção de dados)
- ✅ Observabilidade completa (Prometheus + Jaeger)
- ✅ Batch jobs com RQ workers

## 🏗️ Arquitetura

```
┌─────────────────────────────────────┐
│   FastAPI (Webhooks + Commands)     │
├─────────────────────────────────────┤
│   LangGraph Agent (Raciocínio)      │
│   PydanticAI Agents (Especializados)│
│   Composio (150+ Integrações)       │
├─────────────────────────────────────┤
│   PostgreSQL + Redis                 │
│   RQ Workers (Batch Jobs)            │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15
- Redis 7

### Setup Local

```bash
# Clone repository
git clone <repo-url>
cd hil-system

# Copy env file
cp .env.example .env

# Edit .env com suas credenciais
nano .env

# Build containers
docker-compose up -d

# Run migrations
docker-compose exec app python scripts/run_migrations.py

# Seed data (opcional)
docker-compose exec app python scripts/seed_data.py

# Check logs
docker-compose logs -f app
```

Acesse:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics
- **RQ Dashboard**: http://localhost:9181
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger**: http://localhost:16686

### Desenvolvimento

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download pt_core_news_lg

# Run tests
pytest -v --cov=app

# Run locally
uvicorn app.main:app --reload
```

## 📡 API Endpoints

### Webhooks

```bash
POST /webhook/message
```

Recebe mensagens de qualquer canal (WhatsApp, Telegram, Web, etc).

**Request:**
```json
{
  "conversation_id": "conv_123",
  "sender_type": "customer",
  "sender_id": "cust_456",
  "content": "Olá, preciso de ajuda",
  "origin_id": "msg_abc",
  "provider": "whatsapp"
}
```

### Comandos de Atendente

Os atendentes podem usar comandos especiais:

- `@bot [instrução]` - Bot assume conversa
- `@pause` - Pausa bot
- `@resume` - Resume bot
- `@summary [last:N]` - Resume conversa
- `@analyze` - Analisa sentimento/urgência
- `@suggest` - Sugere resposta
- `@escalate [reason:X]` - Escala para supervisor
- `@status` - Status da conversa
- `@help` - Lista comandos

### GDPR/LGPD

```bash
# Request deletion token
POST /api/v1/gdpr/customer/{customer_id}/request-deletion

# Delete customer data
DELETE /api/v1/gdpr/customer/{customer_id}
```

## 🔧 Configuration

### Timeouts

```bash
HUMAN_TIMEOUT_MINUTES=30  # Timeout de inatividade humana
BOT_RESPONSE_TIMEOUT_SECONDS=30  # Timeout de resposta do bot
```

### Rate Limits

```bash
CUSTOMER_RATE_LIMIT_PER_MINUTE=10
CUSTOMER_RATE_LIMIT_PER_HOUR=100
AGENT_RATE_LIMIT_PER_MINUTE=50
```

### Data Retention

```bash
DATA_RETENTION_DAYS_MESSAGES=90
DATA_RETENTION_DAYS_SUMMARIES=730
DATA_RETENTION_DAYS_AUDIT=2555
```

## 📊 Monitoring

### Prometheus Metrics

```
# LLM
llm_calls_total{model,outcome}
llm_call_duration_seconds{model}
llm_tokens_total{model,type}
llm_cost_usd_total{model}

# Commands
commands_executed_total{command,outcome}

# Conversations
conversations_active
conversations_bot_active
conversations_human_active

# Escalations
escalations_total{reason}

# Rate Limits
rate_limit_exceeded_total{entity_type,limit_type}
```

### Grafana Dashboards

Dashboards pré-configurados disponíveis em `/grafana/dashboards/`.

### Distributed Tracing

Traces disponíveis no Jaeger (http://localhost:16686).

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_commands.py -v

# Check PII in logs
python scripts/check_pii.py

# Check metric cardinality
python scripts/check_metrics.py
```

## 🔒 Security

### PII Protection

PII é automaticamente redactado em logs usando:
- Regex patterns (email, CPF, phone, etc)
- NER model (nomes, localizações)

### Secrets Management

Em produção, use HashiCorp Vault ou AWS Secrets Manager.

```python
ENVIRONMENT=production
VAULT_ADDR=http://vault:8200
VAULT_TOKEN=...
```

### Rate Limiting

Rate limiting aplicado em:
- Customers: 10 msg/min, 100 msg/hora
- Agents: 50 comandos/min
- Global: 100 req/min/IP

## 📦 Deployment

### Production Checklist

- [ ] Configure secrets manager (Vault/AWS)
- [ ] Setup SSL/TLS certificates
- [ ] Configure reverse proxy (Nginx)
- [ ] Setup monitoring alerts
- [ ] Configure backup strategy
- [ ] Test failover scenarios
- [ ] Load test endpoints
- [ ] Review security settings

### Docker Production

```bash
# Build production image
docker build -t hil-system:latest .

# Run with production config
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes (opcional)

Manifests disponíveis em `/k8s/`.

## 🛠️ Troubleshooting

### Bot não responde

1. Verificar se `is_hil = false`
2. Verificar OpenAI API key
3. Verificar logs: `docker-compose logs app`

### Timeout não funciona

1. Verificar Redis connection
2. Verificar se timeout worker está rodando
3. Verificar logs: `docker-compose logs app | grep timeout`

### Rate limit falso positivo

1. Verificar Redis sorted sets: `redis-cli ZRANGE rate:customer:X 0 -1`
2. Limpar manualmente: `redis-cli DEL rate:customer:X`

### Métricas não aparecem

1. Verificar Prometheus scraping: http://localhost:9090/targets
2. Verificar `/metrics` endpoint: http://localhost:8000/metrics

## 📚 Additional Resources

- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
- [PydanticAI Docs](https://ai.pydantic.dev/)
- [Composio Docs](https://docs.composio.dev/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## 📄 License

MIT

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run tests
4. Submit pull request

## 📞 Support

- Email: support@example.com
- Slack: #hil-system
```

---

### CHECKLIST DE IMPLEMENTAÇÃO COMPLETO

```markdown
# ✅ Checklist de Implementação

## Dia 1-2: Setup Básico
- [ ] Criar estrutura de diretórios
- [ ] Setup Git e .gitignore
- [ ] Criar .env.example
- [ ] Setup requirements.txt
- [ ] Database connection (AsyncPG)
- [ ] Redis connection
- [ ] FastAPI main app

## Dia 3-5: Core Services
- [ ] Pydantic models (Conversation, Message)
- [ ] PII Sanitizer (Regex + NER)
- [ ] Rate Limiter (Redis sliding window)
- [ ] Command Parser
- [ ] Conversation Repository (pessimistic locks)

## Dia 6-8: AI Agents
- [ ] PydanticAI agents (Summary, Analysis, Suggestion)
- [ ] LangGraph agent (entender, planejar, executar)
- [ ] Composio integration
- [ ] Command Handler (todos os @comandos)

## Dia 9-11: Timeout & Commands
- [ ] Timeout Monitor (Redis Sorted Set)
- [ ] Worker loop (asyncio)
- [ ] Command execution (async handlers)
- [ ] Audit logging

## Dia 12-14: API Endpoints
- [ ] Webhook POST /webhook/message
- [ ] Health checks
- [ ] GDPR endpoints
- [ ] Message processing (background tasks)
- [ ] Idempotency check

## Dia 15-17: GDPR & RQ Workers
- [ ] GDPR Service (delete customer data)
- [ ] Retention cleanup
- [ ] RQ worker setup
- [ ] RQ scheduler (cron jobs)
- [ ] RQ tasks (retention, reports, notifications)

## Dia 18-20: Observability
- [ ] Prometheus metrics
- [ ] Structured logging (structlog)
- [ ] OpenTelemetry tracing
- [ ] Grafana dashboards
- [ ] Alert rules

## Dia 21-22: Docker & Deployment
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] Health checks em containers
- [ ] Volume mounts
- [ ] Network configuration
- [ ] prometheus.yml

## Dia 23-25: Tests
- [ ] conftest.py (fixtures)
- [ ] Test commands
- [ ] Test concurrency
- [ ] Test GDPR
- [ ] Test rate limiting
- [ ] Test PII sanitization
- [ ] CI pipeline (.github/workflows/test.yml)

## Dia 26-27: Scripts & Utilities
- [ ] Migration runner
- [ ] PII scanner
- [ ] Metric cardinality checker
- [ ] Seed data script

## Dia 28: Documentation
- [ ] README.md completo
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Troubleshooting guide
- [ ] Contributing guide

## Final: Production Ready
- [ ] All tests passing
- [ ] Zero PII in logs
- [ ] Metrics cardinality < 50
- [ ] Docker compose up works
- [ ] Migrations run successfully
- [ ] Health checks pass
- [ ] Grafana dashboards working
- [ ] Jaeger tracing working
- [ ] RQ workers processing jobs
- [ ] Rate limiting enforced
- [ ] GDPR deletion works
```

---

## 🎯 Comandos Rápidos para Começar

```bash
# 1. Setup inicial
mkdir hil-system && cd hil-system
git init

# 2. Criar estrutura
mkdir -p app/{core,models,agents,services,api,workers/tasks}
mkdir -p tests migrations scripts

# 3. Copiar requirements.txt (do documento)
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
# ... (resto das dependências)
EOF

# 4. Criar .env
cp .env.example .env
# Editar .env com suas credenciais

# 5. Build e run
docker-compose up -d

# 6. Migrations
docker-compose exec app python scripts/run_migrations.py

# 7. Seed data
docker-compose exec app python scripts/seed_data.py

# 8. Test
docker-compose exec app pytest -v

# 9. Check health
curl http://localhost:8000/health

# 10. View logs
docker-compose logs -f app
```

---

Este é o guia completo para implementação! Siga as fases sequencialmente e use o checklist para acompanhar o progresso. Cada arquivo está documentado com comentários explicativos.