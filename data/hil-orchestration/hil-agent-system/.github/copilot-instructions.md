# HIL Agent System - AI Development Guide

## Current State (See ROADMAP.md for full status)
- ✅ **Foundation**: FastAPI + SQLModel + pgvector + Docker setup complete
- ✅ **Simple Agent**: Working implementation with LLM router framework
- ✅ **Workflow System**: YAML loader + registry + execution engine (basic)
- 🟡 **LLM Integration**: Base classes exist, need actual API calls
- ❌ **HIL Meta-Orchestrator**: Comprehensive system from implementation_guideline.md "Human-in-the-Loop (HIL) Meta-Workflow" section (HILOrchestrator, QueueManager, HIL schema, API endpoints)
- ❌ **Reasoning/Code Agents**: Not implemented
- ❌ **Tool Integration**: Composio integration missing

## Dual Execution Modes
The system supports **two execution modes**:

### 1. Standalone Agentic Workflows ✅ (Working)
- Direct workflow execution via `/api/v1/workflows/execute`
- Pure AI processing without human intervention
- Current implementation in `WorkflowIntegrationService.execute_workflow()`

### 2. HIL-Enabled Workflows 🔄 (Planned)  
- Meta-orchestrator with `is_hil` state management
- Agentic workflows terminate at sink nodes:
  - `FINISH`: "I handled this successfully" → conversation complete  
  - `HANDOVER`: "I need human help" → escalate to human queue

## Quick Start
```bash
# Setup (use UV for speed)
uv pip install -r requirements-dev.txt
docker-compose up -d postgres redis
alembic upgrade head
uvicorn app.main:app --reload
```

## Key Commands
- `make test-fast` - Unit tests (fix Python path first)
- `make lint format` - Ruff code quality
- `make all-checks` - Full CI pipeline

## 🚨 Anti-Echo Development Patterns (HIGH PRIORITY)

**Problem**: Agent conversations suffer from echoing/repetition due to (1) raw history resending, (2) concurrent memory writes, (3) missing idempotency controls.

### Required for ALL conversation features:
1. **Turn-based idempotency**: Every user input gets `session_id:turn_id` key
2. **Redis locks**: Use `conversation_lock:{session_id}` for memory updates  
3. **Response hashing**: Check duplicate responses before sending
4. **Background processing**: Move embeddings/summaries to Celery tasks
5. **Incremental context**: Never resend full conversation history

### Mandatory Code Patterns:
```python
# ✅ ALWAYS: Idempotent conversation processing
async def process_user_input(session_id: str, turn_id: str, input_text: str):
    idempotency_key = f"{session_id}:{turn_id}"
    if await redis.exists(f"processed:{idempotency_key}"):
        return await get_cached_response(idempotency_key)
    
    async with redis.lock(f"conversation_lock:{session_id}", timeout=30):
        # Process and cache result atomically
        result = await process_conversation_turn(...)
        await redis.setex(f"processed:{idempotency_key}", 3600, result)
        return result

# ✅ ALWAYS: Check for duplicate responses  
async def send_agent_response(session_id: str, response_content: str):
    content_hash = hashlib.sha256(response_content.encode()).hexdigest()
    duplicate_key = f"response_hash:{session_id}:{content_hash}"
    
    if await redis.exists(duplicate_key):
        logger.warning(f"Prevented duplicate response for {session_id}")
        return None  # Don't send duplicate
        
    await redis.setex(duplicate_key, 3600, "sent")
    return response_content

# ❌ NEVER: Direct memory updates without locking
conversation.memory.append(new_message)  # Risk of corruption!

# ❌ NEVER: Resend full conversation history
context = "\n".join([turn.content for turn in conversation.turns])  # Explosion!
```

### Required Models:
```python
# ADD TO SCHEMA: Turn tracking with idempotency
class ConversationTurn(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id")
    turn_number: int
    user_input_hash: str  
    agent_response_hash: str  
    idempotency_key: str  # session_id:turn_id
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

## Critical Patterns

### Agent Development
- **Simple Agent Pattern**: `app/agents/types/simple_agent.py` - follow this structure
- **LLM Router**: Use `ModelProfile` enum (fast/balanced/powerful) 
- **Error Handling**: Custom exceptions in `app/core/exceptions.py`
- **Structured Output**: Pydantic models for all agent responses

### Database & Config
- **Models**: SQLModel in `app/models/` (SQLAlchemy + Pydantic)
- **Config**: `app/core/config.py` with environment detection
- **Sessions**: Dependency injection `get_session()` in FastAPI
- **Vector Storage**: pgvector with 1536 dimensions (OpenAI embeddings)

### Workflow System
- **YAML Files**: Store in `config/workflows/` with version suffixes
- **Registry**: `app/services/workflow_registry.py` manages definitions
- **Execution**: `app/services/workflow_integration.py` handles DAG processing

### Priority TODOs (Anti-Echo First!)
1. 🚨 **Add ConversationTurn Model** - Turn tracking with `session_id:turn_id` idempotency keys
2. 🚨 **Implement MemoryLockService** - Redis distributed locking for conversation updates
3. 🚨 **Response Deduplication** - Hash-based duplicate prevention middleware
4. 🚨 **Background Memory Tasks** - Celery integration for async embeddings/summaries
5. **Fix LLM Providers** - Complete OpenAI/Anthropic API integration for standalone workflows
6. **HIL Database Schema** - Implement tables from implementation_guideline.md "HIL Database Schema" section (conversations, human_agents, agent_queue)
7. **HILOrchestrator Service** - 7-state conversation management with sink node processing
8. ✅ **Development Setup** - Makefile now working with UV + proper dependencies

### Common Patterns
- Always validate `llm_router is not None` in agent constructors
- **Standalone workflows**: Direct execution with standard return values
- **HIL workflows**: Must terminate at sink nodes (`FINISH` or `HANDOVER`)
- Use Jinja2 for prompt templates with `input_data` 
- Log execution time and costs for observability
- Follow async/await patterns for all database operations
- Version workflows and maintain backward compatibility