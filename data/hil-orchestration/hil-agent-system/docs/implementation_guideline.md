# Complete HIL Agent System Architecture
## Production-Ready AI Workflow Orchestration with Code Agents

**Version:** 2.0  
**Last Updated:** 2025-01-12  
**Status:** Design Complete - Ready for Implementation

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Core Components](#core-components)
4. [Agent Types](#agent-types)
5. [Tool & Integration System](#tool--integration-system)
6. [Memory & Context Management](#memory--context-management)
7. [LLM Routing & Cost Optimization](#llm-routing--cost-optimization)
8. [Security & Sandboxing](#security--sandboxing)
9. [Human-in-the-Loop (HIL) Meta-Workflow](#human-in-the-loop-hil-meta-workflow)
10. [Observability & Monitoring](#observability--monitoring)
11. [Database Schema](#database-schema)
12. [Implementation Guide](#implementation-guide)
13. [API Documentation](#api-documentation)
14. [Cost Analysis](#cost-analysis)
15. [Production Deployment](#production-deployment)
16. [Build vs Buy Analysis](#build-vs-buy-analysis)

---

## 🎯 Executive Summary

### What We're Building

A **production-grade AI workflow orchestration system** that enables:
- **Dynamic workflow execution** via DAG-based orchestration
- **Three agent types**: Simple (classification), Reasoning (ReAct), Code (autonomous)
- **1000+ tool integrations** via Composio (Shopify, Gmail, Slack, etc.)
- **Intelligent LLM routing** for cost optimization (56% savings)
- **Hybrid memory system** (short-term + long-term + episodic)
- **Complete observability** (metrics, traces, costs)

### Key Metrics

| Metric | Target | Impact |
|--------|--------|--------|
| **Agent Types** | 3 (Simple, Reasoning, Code) | Covers 95% of use cases |
| **Tool Integrations** | 1000+ via Composio | Zero custom OAuth coding |
| **Cost Optimization** | 56% reduction | $948/month savings |
| **Latency P95** | <15s for complex tasks | 3x faster than baseline |
| **Error Rate** | <5% for autonomous agents | Self-healing with retry |
| **System Uptime** | >99.5% | Circuit breakers + fallbacks |

### Timeline & Resources

- **Development Time:** 12-14 weeks
- **Team Required:** 2 ML Engineers + 1 Backend Engineer + 1 DevOps
- **Development Cost:** $300-400k
- **Operational Cost:** $700-1,700/month (LLM + infrastructure)

### Alternative: n8n + AgentKit

- **Development Time:** 4 weeks
- **Team Required:** 1-2 Engineers
- **Cost:** $50-100k development + $200-500/month operations

---

## 🏗️ System Architecture Overview

### Complete Hierarchy

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

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  🎯 Workflow Orchestrator                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           📊 Workflow DAG Executor                        │  │
│  │  • Load from WorkflowRegistry (PostgreSQL)                │  │
│  │  • Topological sort & parallel execution                  │  │
│  │  • Conditional branching (JMESPath/Python)                │  │
│  │  • Checkpointing for fault tolerance                      │  │
│  │  • Retry logic with exponential backoff                   │  │
│  └─────────────────────┬─────────────────────────────────────┘  │
└─────────────────────────┼─────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │      🔍 Agent Registry             │
         │  • Dynamic agent discovery         │
         │  • Health checks & fallbacks       │
         │  • Capability-based routing        │
         │  • Metadata management             │
         └────────────┬───────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Simple    │ │Reasoning │ │Code      │
    │Agent     │ │Agent     │ │Agent     │
    │(1-shot)  │ │(ReAct)   │ │(Agentic) │
    │~1.5s     │ │~8s       │ │~25s      │
    │$0.001    │ │$0.03     │ │$0.15     │
    └──────────┘ └──────────┘ └──────────┘
          │           │           │
          └───────────┼───────────┘
                      ▼
         ┌────────────────────────────────────┐
         │       💰 LLM Model Router          │
         │  • Profile-based selection         │
         │    - Fast: GPT-3.5, Claude Haiku   │
         │    - Balanced: GPT-4-turbo, Sonnet │
         │    - Powerful: GPT-4, Claude Opus  │
         │  • Token estimation & budgeting    │
         │  • Circuit breakers per model      │
         │  • Automatic fallback chains       │
         └────────────┬───────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │OpenAI    │ │Anthropic │ │Local LLM │
    │GPT-4/3.5 │ │Claude 3  │ │Ollama    │
    └──────────┘ └──────────┘ └──────────┘
          │           │           │
          └───────────┼───────────┘
                      ▼
         ┌────────────────────────────────────┐
         │    🧠 Memory Manager                │
         │  • Short-term: Recent messages (PG)│
         │  • Long-term: Semantic search (VDB)│
         │  • Episodic: Agent executions      │
         │  • Hybrid retrieval strategies     │
         └────────────┬───────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │   🛠️ Composio Tool Manager         │
         │  • 1000+ actions across 150+ apps  │
         │  • Dynamic action discovery        │
         │  • OAuth flow management           │
         │  • Rate limiting per app           │
         │  • Semantic action search          │
         └────────────┬───────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Shopify   │ │Gmail     │ │Slack     │
    │150 acts  │ │80 acts   │ │60 acts   │
    └──────────┘ └──────────┘ └──────────┘
```

### Request Flow Example

```
1. 📥 Customer Message: "I want to return order #12345"
   └─> Workflow: customer_support_advanced

2. 🔵 Node: classify_intent (Simple Agent)
   ├─> LLM Router: Select GPT-3.5-turbo (fast profile)
   ├─> Execute: Classification
   └─> Output: {intent: "return_request", confidence: 0.95}

3. 🔀 Conditional Edge: intent == "return_request"
   └─> Route to: handle_return node

4. 🔵 Node: handle_return (Code Agent)
   ├─> Agent Registry: Create autonomous_return_agent
   ├─> LLM Router: Select GPT-4 (powerful profile)
   ├─> Memory: Retrieve customer history
   ├─> Composio Tools:
   │   ├─> shopify.get_order
   │   ├─> shopify.create_return
   │   └─> gmail.send_email
   ├─> Agent Plans:
   │   1. Get order details
   │   2. Verify return eligibility
   │   3. Create return request
   │   4. Send confirmation
   └─> Output: {success: true, return_id: "RET-456"}

5. 🔵 Node: assemble_response (Simple Agent)
   ├─> LLM Router: Select GPT-3.5-turbo (fast profile)
   └─> Output: "Hi! Your return has been processed..."

6. 📊 Metrics:
   ├─> Total Cost: $0.1520
   ├─> Duration: 8,450ms
   ├─> LLM Calls: 4 (2x GPT-3.5, 2x GPT-4)
   └─> Composio Actions: 4 (3x Shopify, 1x Gmail)
```

---

## 🔧 Core Components

### 1. Agent Registry

**Purpose:** Centralized discovery and management of all agent types.

**Key Features:**
- Dynamic agent registration with metadata
- Health monitoring (error rate, latency)
- Automatic fallback selection
- Capability-based routing

**Code Structure:**

```python
# app/agents/registry.py

class AgentMetadata(BaseModel):
    name: str
    type: str  # simple, reasoning, code
    description: str
    capabilities: List[AgentCapability]
    supported_models: List[str]
    default_model: str
    cost_tier: str  # low, medium, high
    avg_latency_ms: int
    is_healthy: bool = True
    error_rate: float = 0.0

class AgentRegistry:
    def register(self, metadata: AgentMetadata, factory: AgentFactory)
    async def create_agent(self, agent_name: str, config: dict) -> Agent
    def get_metadata(self, agent_name: str) -> AgentMetadata
    def list_agents(self, capability: AgentCapability = None) -> List[AgentMetadata]
    async def _health_check_loop(self)  # Background task
```

**Registration Example:**

```python
agent_registry.register(
    AgentMetadata(
        name="simple_intent_classifier",
        type="simple",
        capabilities=[AgentCapability.STRUCTURED_OUTPUT],
        supported_models=["gpt-3.5-turbo", "claude-3-haiku"],
        default_model="gpt-3.5-turbo",
        cost_tier="low",
        avg_latency_ms=800
    ),
    SimpleAgentFactory()
)
```

### 2. Workflow Orchestrator

**Purpose:** Execute DAG-based workflows with conditional logic.

**Key Features:**
- Topological sort for execution order
- Parallel execution where possible
- Conditional branching (JMESPath/Python expressions)
- State checkpointing for fault tolerance
- Automatic retry with exponential backoff

**Code Structure:**

```python
# app/agents/orchestrator.py

class AgentOrchestrator:
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: dict,
        context: dict = None
    ) -> dict
    
    async def _execute_dag(
        self,
        workflow_config: dict,
        input_data: dict
    ) -> dict
    
    async def _execute_node(
        self,
        node_config: dict,
        node_inputs: dict
    ) -> Any
    
    def _evaluate_condition(
        self,
        condition: dict,
        node_outputs: dict
    ) -> bool
```

**Workflow Configuration:**

```yaml
# config/workflows/example.yaml
name: customer_support_advanced
version: 2.0.0

workflow:
  nodes:
    - id: classify_intent
      agent: simple_intent_classifier
      config:
        model_profile: fast
    
    - id: handle_return
      agent: autonomous_return_agent
      config:
        model_profile: powerful
        tools: [shopify.*, gmail.send_email]
      condition:
        type: jmespath
        expression: "classify_intent.output.intent == 'return_request'"
  
  edges:
    - from: classify_intent
      to: handle_return
      condition:
        type: jmespath
        expression: "output.intent == 'return_request'"
```

### 3. Workflow Registry Service

**Purpose:** Store and manage workflow definitions.

**Key Features:**
- Version control for workflows
- Workflow templates
- Execution history tracking
- A/B testing support

**Code Structure:**

```python
# app/services/workflow_registry.py

class WorkflowRegistryService:
    async def create_workflow(self, workflow: WorkflowCreate) -> Workflow
    async def get_workflow(self, workflow_id: str) -> Workflow
    async def list_workflows(self, filters: dict = None) -> List[Workflow]
    async def update_workflow(self, workflow_id: str, updates: dict) -> Workflow
    async def delete_workflow(self, workflow_id: str)
    async def record_execution_start(self, workflow_id: str) -> str
    async def record_execution_complete(self, execution_id: str, status: str)
```

---

## 🤖 Agent Types

### 1. Simple Agent (Stateless, Single-Shot)

**Use Cases:**
- Intent classification
- Sentiment analysis
- Entity extraction
- Simple Q&A

**Characteristics:**
- ⚡ Latency: ~1-2s
- 💰 Cost: ~$0.001 per call
- 🎯 Reliability: 99%+
- 🔄 No iteration loop

**Implementation:**

```python
# app/agents/types/simple_agent.py

class SimpleAgent:
    def __init__(
        self,
        llm_router: LLMRouter,
        model_profile: str = "fast",
        output_schema: Type[BaseModel] = None,
        temperature: float = 0.0
    )
    
    async def run(
        self,
        prompt: str,
        input_data: dict,
        system_prompt: str = None
    ) -> BaseModel:
        # 1. Render prompt template
        rendered = Template(prompt).render(**input_data)
        
        # 2. Route to optimal model
        decision = await self.llm_router.route(
            messages=[{"role": "user", "content": rendered}],
            profile=ModelProfile(model_profile)
        )
        
        # 3. Call LLM
        provider = LLMProviderFactory.create(decision.provider, model=decision.model)
        response = await provider.complete_structured(
            messages=messages,
            schema=output_schema.schema()
        )
        
        # 4. Validate & return
        return output_schema(**response)
```

**Example Usage:**

```python
class IntentClassification(BaseModel):
    intent: str
    confidence: float
    entities: dict

agent = SimpleAgent(
    llm_router=llm_router,
    model_profile="fast",
    output_schema=IntentClassification
)

result = await agent.run(
    prompt="Classify intent: {{message}}",
    input_data={"message": "I want to return my order"}
)
# Output: IntentClassification(intent="return_request", confidence=0.95, ...)
```

### 2. Reasoning Agent (ReAct Pattern)

**Use Cases:**
- Multi-step problem solving
- Information gathering across multiple sources
- Dynamic decision making
- Tool orchestration

**Characteristics:**
- ⚡ Latency: ~5-15s
- 💰 Cost: ~$0.01-0.05 per task
- 🎯 Reliability: 95%+
- 🔄 Max 10 iterations (configurable)

**ReAct Flow:**

```
1. Thought: "I need to check order status"
2. Action: call_api("get_order", {"id": "123"})
3. Observation: "Order is shipped"
4. Thought: "Customer wants return, need return policy"
5. Action: search_kb("return policy shipped orders")
6. Observation: "30 day return window..."
7. Thought: "I have enough info to respond"
8. Final Answer: "You can return within 30 days..."
```

**Implementation:**

```python
# app/agents/types/reasoning_agent.py

class ReasoningAgent:
    def __init__(
        self,
        llm_router: LLMRouter,
        tools: List[Tool],
        max_iterations: int = 10,
        model_profile: str = "balanced"
    )
    
    async def run(
        self,
        task: str,
        context: dict = None
    ) -> dict:
        reasoning_chain = []
        
        for iteration in range(max_iterations):
            # Generate next thought/action
            step = await self._generate_step(task, reasoning_chain)
            reasoning_chain.append(step)
            
            # If final answer, done
            if step.action is None:
                return {
                    "answer": step.thought,
                    "reasoning_chain": reasoning_chain,
                    "iterations": iteration + 1
                }
            
            # Execute action (tool call)
            observation = await self._execute_tool(step.action, step.action_input)
            step.observation = observation
        
        return {"error": "Max iterations exceeded"}
```

**System Prompt:**

```
You are a helpful AI assistant that uses tools to solve tasks.

Available tools:
- get_order: Get order details by ID
- search_kb: Search knowledge base
- create_return: Create return request

Format:
Thought: [Your reasoning]
Action: [Tool name]
Action Input: {"param": "value"}
Observation: [Will be provided]
...
Thought: [Final reasoning]
Final Answer: [Complete answer]
```

### 3. Code Agent (Full Autonomy)

**Use Cases:**
- Complex multi-step automation
- Self-healing workflows
- Dynamic workflow generation
- Multi-API orchestration

**Characteristics:**
- ⚡ Latency: ~15-60s
- 💰 Cost: ~$0.05-0.20 per task
- 🎯 Reliability: 85-90%
- 🔄 Plan → Execute → Reflect loop

**Agent Lifecycle:**

```
1. PLANNING Phase:
   - Analyze goal
   - Generate step-by-step plan
   - Estimate duration & tools needed
   - Learn from similar past executions (memory)

2. EXECUTING Phase:
   - Execute plan steps sequentially
   - Handle step outputs as inputs to next steps
   - Record success/failure of each step
   - Continue on success, stop on critical failure

3. REFLECTING Phase:
   - Analyze execution results
   - Determine if goal achieved
   - If failed: identify root cause
   - Generate suggestions for retry

4. RETRY (if needed):
   - Adjust plan based on reflection
   - Execute again with improvements
   - Max 3 retries by default
```

**Implementation:**

```python
# app/agents/types/code_agent.py

class CodeAgent:
    def __init__(
        self,
        llm_router: LLMRouter,
        tools: List[Tool],
        max_retries: int = 3,
        memory_manager: MemoryManager = None,
        model_profile: str = "powerful"
    )
    
    async def run(
        self,
        goal: str,
        context: dict = None
    ) -> dict:
        retry_count = 0
        
        while retry_count < max_retries:
            # Phase 1: Planning
            plan = await self._generate_plan(goal, context)
            
            # Phase 2: Execution
            results = await self._execute_plan(plan)
            
            # Phase 3: Reflection
            reflection = await self._reflect(goal, plan, results)
            
            if reflection["goal_achieved"]:
                # Store success in memory
                await self.memory.store_execution(goal, plan, results, True)
                return {"success": True, "result": reflection["final_output"]}
            
            # Adjust context for retry
            context = {
                **context,
                "previous_attempt": {
                    "plan": plan,
                    "failure_reason": reflection["failure_reason"]
                }
            }
            retry_count += 1
        
        return {"success": False, "error": "Max retries exceeded"}
```

**Planning Prompt:**

```
Generate a detailed execution plan to achieve this goal:

Goal: {{goal}}

Available Tools:
{{#each tools}}
- {{name}}: {{description}}
  Input: {{input_schema}}
{{/each}}

{{#if similar_plans}}
Similar successful plans from memory:
{{#each similar_plans}}
- {{goal}}: {{steps.length}} steps, success rate: {{success_rate}}
{{/each}}
{{/if}}

Generate JSON plan:
{
  "goal": "...",
  "steps": [
    {
      "id": 1,
      "description": "...",
      "tool": "tool_name",
      "input": {"param": "value"},
      "expected_output": "...",
      "critical": true/false
    }
  ],
  "estimated_duration": 30,
  "required_tools": ["tool1", "tool2"]
}
```

---

## 🛠️ Tool & Integration System

### 1. Tool Registry

**Purpose:** Manage tool lifecycle, execution, caching, and metrics.

**Key Features:**
- Timeout enforcement
- Automatic retry with exponential backoff
- Result caching (Redis)
- Rate limiting
- Metrics tracking (Prometheus)

**Implementation:**

```python
# app/tools/registry.py

class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict
    func: Callable
    timeout_seconds: int = 30
    retry_config: dict = {"max_attempts": 3, "backoff": 2}
    cache_ttl: int = None  # seconds

class ToolRegistry:
    def register(self, tool: Tool)
    
    async def execute(
        self,
        tool_name: str,
        input_data: dict
    ) -> ToolExecutionResult:
        # 1. Check cache
        # 2. Execute with timeout
        # 3. Retry on failure
        # 4. Cache result
        # 5. Record metrics
```

**Built-in Tools:**

```python
# HTTP Tools
http_request: Make HTTP requests to external APIs
  Input: {method, url, headers, body}
  Timeout: 30s
  Retry: 3x

# Data Tools
transform_json: Transform JSON using JMESPath
  Input: {data, expression}
  Timeout: 5s

filter_list: Filter list items by condition
  Input: {items, condition}
  Timeout: 10s

# Database Tools
query_database: Execute safe SELECT query
  Input: {query, params}
  Timeout: 10s
  Cache: 5 min
  Safety: SELECT only, parameterized
```

### 2. Composio Tool Manager

**Purpose:** Integrate 1000+ actions across 150+ apps without custom OAuth.

**Key Features:**
- Dynamic action discovery at startup
- OAuth flow management (per entity)
- Rate limiting per app
- Semantic action search
- Action metadata caching

**Supported Apps (Examples):**

| App | Actions | Use Cases |
|-----|---------|-----------|
| **Shopify** | 150+ | Order management, returns, inventory |
| **Gmail** | 80+ | Send emails, read threads, manage labels |
| **Slack** | 60+ | Send messages, create channels, file uploads |
| **Stripe** | 100+ | Payments, refunds, subscriptions |
| **Google Calendar** | 50+ | Event management, scheduling |
| **Notion** | 70+ | Database operations, page creation |

**Implementation:**

```python
# app/tools/composio_manager.py

class ComposioToolManager:
    def __init__(self, api_key: str, redis_client, db_pool)
    
    async def initialize(self):
        """Discover all apps and actions at startup"""
        apps = self.client.get_apps()
        for app in apps:
            for action in app.actions:
                await self._register_action(app.name, action)
    
    async def get_tools(
        self,
        tool_specs: List[str],  # ["shopify.*", "gmail.send_email"]
        entity_id: str = None
    ) -> List[Tool]:
        """Convert Composio actions to Tool objects"""
    
    async def execute_action(
        self,
        app_name: str,
        action_name: str,
        params: dict,
        entity_id: str = None
    ) -> dict:
        """Execute action with rate limiting and OAuth"""
    
    async def connect_app(
        self,
        app_name: str,
        entity_id: str,
        redirect_url: str
    ) -> str:
        """Return OAuth authorization URL"""
    
    async def search_actions(
        self,
        query: str,
        limit: int = 10
    ) -> List[ComposioAction]:
        """Semantic search for relevant actions"""
```

**Usage in Workflow:**

```yaml
# Workflow config
composio:
  apps:
    - name: shopify
      entity_id_source: customer_id  # Use customer_id for OAuth
      actions:
        - get_order
        - create_return
        - update_order_status
    
    - name: gmail
      entity_id_source: agent_id  # Use agent_id for sending emails
      actions:
        - send_email

# Node using Composio tools
nodes:
  - id: process_return
    agent: autonomous_return_agent
    config:
      tools:
        - shopify.*  # All Shopify actions
        - gmail.send_email  # Specific Gmail action
```

**OAuth Flow:**

```
1. User requests connection: POST /composio/connect/shopify
   └─> Response: {authorization_url: "https://..."}

2. User completes OAuth in browser
   └─> Redirect: https://yourapp.com/callback?code=...

3. App exchanges code for token: POST /composio/callback
   └─> Composio stores encrypted token

4. Future tool executions use stored token
   └─> Automatic refresh when expired
```

---

## 🧠 Memory & Context Management

### Architecture

```
Memory Manager
├── Short-term Memory (PostgreSQL)
│   ├── Recent messages (last 50-100)
│   ├── Conversation summaries
│   └── Session context
│
├── Long-term Memory (Vector DB)
│   ├── Knowledge base documents
│   ├── Past conversation chunks
│   └── Semantic search index
│
└── Episodic Memory (PostgreSQL + Vector DB)
    ├── Agent execution history
    ├── Successful plans
    └── Failure analysis
```

### Implementation

```python
# app/memory/manager.py

class MemoryManager:
    def __init__(self, db_pool, vector_store, embeddings_model="text-embedding-3-small")
    
    # Short-term Memory
    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 50
    ) -> List[dict]:
        """Get recent messages from PostgreSQL"""
    
    async def get_conversation_summary(
        self,
        conversation_id: str
    ) -> dict:
        """Get cached conversation summary"""
    
    # Long-term Memory (RAG)
    async def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        filters: dict = None
    ) -> List[dict]:
        """Semantic search in vector store"""
    
    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict
    ):
        """Index document for RAG"""
    
    async def index_conversation_chunk(
        self,
        conversation_id: str,
        chunk_id: str,
        messages: List[dict]
    ):
        """Index conversation for future similarity search"""
    
    # Episodic Memory
    async def store_execution(
        self,
        goal: str,
        plan: Plan,
        results: List[ExecutionResult],
        success: bool
    ):
        """Store agent execution for learning"""
    
    async def search_similar_executions(
        self,
        goal: str,
        limit: int = 3,
        success_only: bool = False
    ) -> List[dict]:
        """Find similar past executions"""
    
    # Hybrid Retrieval
    async def get_context(
        self,
        conversation_id: str,
        current_message: str,
        strategy: str = "hybrid"  # short_only, long_only, hybrid
    ) -> dict:
        """
        Get context using hybrid strategy:
        - Recent messages (short-term)
        - Relevant documents (long-term RAG)
        - Similar past conversations
        """
```

### Context Strategies

| Strategy | Use Case | Latency | Cost |
|----------|----------|---------|------|
| **short_only** | Simple queries | ~50ms | Free |
| **long_only** | Complex research | ~200ms | $0.0001/search |
| **hybrid** | Customer support | ~250ms | $0.0001/search |

**Usage Example:**

```python
# In node configuration
context_sources:
  - conversation_history  # Short-term
  - knowledge_base       # Long-term RAG
  - similar_conversations # Episodic

# Agent receives enriched context
context = await memory.get_context(
    conversation_id="conv_123",
    current_message="How do I return my order?",
    strategy="hybrid"
)

# context = {
#   "recent_messages": [...],
#   "relevant_documents": [
#     {"content": "Return Policy: Items can be...", "score": 0.92},
#     {"content": "Shipping returns: Print label...", "score": 0.87}
#   ],
#   "similar_conversations": [
#     {"summary": "Customer returned damaged item", "outcome": "success"}
#   ]
# }
```

---

## 💰 LLM Routing & Cost Optimization

### Model Catalog

| Model | Provider | Input/1K | Output/1K | Context | Latency | Quality |
|-------|----------|----------|-----------|---------|---------|---------|
| **GPT-4** | OpenAI | $0.03 | $0.06 | 8K | 4000ms | 10.0 |
| **GPT-4-turbo** | OpenAI | $0.01 | $0.03 | 128K | 3000ms | 9.5 |
| **GPT-3.5-turbo** | OpenAI | $0.0005 | $0.0015 | 16K | 800ms | 7.0 |
| **Claude-3-Opus** | Anthropic | $0.015 | $0.075 | 200K | 3500ms | 9.8 |
| **Claude-3-Sonnet** | Anthropic | $0.003 | $0.015 | 200K | 2000ms | 8.5 |
| **Claude-3-Haiku** | Anthropic | $0.00025 | $0.00125 | 200K | 600ms | 7.5 |

### Routing Profiles

```python
# app/agents/llm_router.py

class ModelProfile(str, Enum):
    FAST = "fast"        # Speed priority, cost-effective
    BALANCED = "balanced" # Balance of speed/cost/quality
    POWERFUL = "powerful" # Quality priority, expensive

PROFILES = {
    ModelProfile.FAST: {
        "primary": ["gpt-3.5-turbo", "claude-3-haiku"],
        "fallback": ["gpt-4-turbo", "claude-3-sonnet"],
        "max_cost_per_call": 0.01,
        "max_latency_ms": 1000,
        "weights": {"latency": 0.6, "cost": 0.3, "quality": 0.1}
    },
    ModelProfile.BALANCED: {
        "primary": ["gpt-4-turbo", "claude-3-sonnet"],
        "fallback": ["gpt-4", "claude-3-opus"],
        "max_cost_per_call": 0.10,
        "max_latency_ms": 3000,
        "weights": {"latency": 0.33, "cost": 0.33, "quality": 0.34}
    },
    ModelProfile.POWERFUL: {
        "primary": ["gpt-4", "claude-3-opus"],
        "fallback": ["gpt-4-turbo", "claude-3-sonnet"],
        "max_cost_per_call": 1.00,
        "max_latency_ms": 5000,
        "weights": {"quality": 0.6, "latency": 0.2, "cost": 0.2}
    }
}
```

### Router Algorithm

```python
class LLMRouter:
    async def route(
        self,
        messages: List[dict],
        profile: ModelProfile = ModelProfile.BALANCED,
        required_capabilities: List[str] = None
    ) -> RoutingDecision:
        # 1. Estimate token count
        token_count = self._estimate_tokens(messages)
        
        # 2. Get candidate models from profile
        candidates = PROFILES[profile]["primary"] + PROFILES[profile]["fallback"]
        
        # 3. Filter by capabilities (e.g., tools, vision)
        if required_capabilities:
            candidates = self._filter_by_capabilities(candidates, required_capabilities)
        
        # 4. Score each viable model
        viable_models = []
        for model_name in candidates:
            spec = self.MODELS[model_name]
            
            # Check constraints
            if token_count > spec.context_window * 0.8:
                continue
            
            estimated_cost = self._estimate_cost(token_count, spec)
            if estimated_cost > PROFILES[profile]["max_cost_per_call"]:
                continue
            
            # Check circuit breaker
            if not await self._check_circuit_breaker(model_name):
                continue
            
            # Check budget
            if self.cost_controller:
                if not await self.cost_controller.check_budget(model_name, token_count):
                    continue
            
            # Calculate weighted score
            score = self._calculate_score(spec, estimated_cost, profile)
            viable_models.append({
                "name": model_name,
                "spec": spec,
                "estimated_cost": estimated_cost,
                "score": score
            })
        
        # 5. Select best model
        viable_models.sort(key=lambda x: x["score"], reverse=True)
        best = viable_models[0]
        
        return RoutingDecision(
            provider=best["spec"].provider,
            model=best["spec"].model,
            estimated_cost=best["estimated_cost"],
            estimated_latency_ms=best["spec"].avg_latency_ms,
            reasoning=self._explain_decision(best, profile)
        )
```

### Cost Optimization Features

**1. Circuit Breakers**
```python
class CircuitBreaker:
    """Prevent cascading failures when model is down"""
    states = ["closed", "open", "half-open"]
    
    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= threshold:
            self.state = "open"  # Block requests
            # Automatically transition to half-open after timeout
```

**2. Budget Controls**
```python
class CostController:
    """Enforce spending limits"""
    hourly_limit = $50
    daily_limit = $500
    monthly_limit = $5000
    
    async def check_budget(self, model: str, tokens: int) -> bool:
        hourly_spent = await self._get_spent("hourly")
        estimated_cost = tokens * model_cost
        
        if hourly_spent + estimated_cost > self.hourly_limit:
            logger.warning("hourly_budget_exceeded")
            return False
        
        return True
```

**3. Automatic Fallbacks**
```
Primary model fails → Try fallback model
Fallback fails → Try tertiary model
All models fail → Return error with context
```

### Cost Savings Analysis

**Without Router (naive approach):**
- Always use GPT-4 for quality
- 1000 conversations/day × $0.15 = $150/day
- **$4,500/month**

**With Router (optimized):**
- 60% Fast profile (GPT-3.5): $0.001 × 600 = $0.60/day
- 30% Balanced (GPT-4-turbo): $0.03 × 300 = $9.00/day
- 10% Powerful (GPT-4): $0.15 × 100 = $15.00/day
- **$24.60/day = $738/month**

**Savings: 84% ($3,762/month)**

---

## 🔒 Security & Sandboxing

### Security Layers

```
┌─────────────────────────────────────────┐
│ Layer 1: API Authentication             │
│ • JWT tokens with expiration             │
│ • API key rotation (90 days)             │
│ • Rate limiting (100 req/min per key)    │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Layer 2: Tool Whitelisting               │
│ • Agents can only use approved tools     │
│ • Safe tools by default (read-only)      │
│ • Dangerous tools require explicit grant │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Layer 3: Docker Sandboxing               │
│ • Code agents run in isolated containers │
│ • No network access                      │
│ • Read-only filesystem                   │
│ • Resource limits (CPU, memory)          │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Layer 4: Data Protection                 │
│ • PII detection & sanitization           │
│ • Encryption at rest (AES-256)           │
│ • Encryption in transit (TLS 1.3)        │
│ • GDPR-compliant deletion                │
└─────────────────────────────────────────┘
```

### 1. Tool Whitelisting

```python
# app/agents/security/tool_whitelist.py

class ToolWhitelist:
    # Safe tools (read-only, no side effects)
    SAFE_TOOLS = {
        "http_request",      # With URL whitelist
        "query_database",    # SELECT only
        "search_kb",
        "transform_json",
        "filter_list"
    }
    
    # Dangerous tools (require explicit permission)
    DANGEROUS_TOOLS = {
        "execute_code",
        "modify_database",
        "send_email",
        "make_payment",
        "delete_resource"
    }
    
    def validate_tool_access(self, tool_name: str, agent_type: str) -> bool:
        # Simple agents: only safe tools
        if agent_type == "simple":
            return tool_name in self.SAFE_TOOLS
        
        # Reasoning agents: safe + explicitly allowed
        if agent_type == "reasoning":
            return (
                tool_name in self.SAFE_TOOLS or
                tool_name in self.allowed_tools
            )
        
        # Code agents: must be explicitly allowed
        if agent_type == "code":
            return tool_name in self.allowed_tools
        
        return False
```

### 2. Docker Sandboxing for Code Agents

```python
# app/agents/security/sandbox.py

class AgentSandbox:
    """Execute code agents in isolated Docker containers"""
    
    async def execute_agent(
        self,
        agent_code: str,
        goal: str,
        context: dict,
        allowed_tools: List[str]
    ) -> dict:
        # Create container with strict security
        container = self.docker_client.containers.run(
            image="hil-agent-sandbox:latest",
            command=["python", "/app/agent_runner.py"],
            environment={"AGENT_PAYLOAD": json.dumps(payload)},
            
            # Security restrictions
            network_mode="none",        # No network access
            read_only=True,             # Read-only filesystem
            mem_limit="512m",           # Memory limit
            cpu_quota=50000,            # 0.5 CPU
            cap_drop=["ALL"],           # Drop all capabilities
            security_opt=["no-new-privileges"],
            
            detach=True,
            remove=True
        )
        
        try:
            result = await asyncio.wait_for(
                self._wait_for_container(container),
                timeout=300  # 5 minutes max
            )
            return result
        except asyncio.TimeoutError:
            container.kill()
            raise ValueError("Agent execution timeout")
```

**Dockerfile for Sandbox:**

```dockerfile
# Dockerfile.agent-sandbox
FROM python:3.11-slim

# Minimal dependencies only
RUN pip install --no-cache-dir pydantic==2.5.0 openai==1.3.7

# Non-root user
RUN useradd -m -u 1000 agent && \
    mkdir -p /app && \
    chown agent:agent /app

USER agent
WORKDIR /app

COPY agent_runner.py /app/

# All I/O through environment variables
# No network, no filesystem access beyond /app
```

### 3. PII Protection

```python
# app/security/pii_detector.py

class PIIDetector:
    """Detect and sanitize PII in logs and outputs"""
    
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    }
    
    def sanitize(self, text: str) -> str:
        """Replace PII with [REDACTED]"""
        for pii_type, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
        return text
    
    def detect(self, text: str) -> List[str]:
        """Detect PII types in text"""
        detected = []
        for pii_type, pattern in self.PATTERNS.items():
            if re.search(pattern, text):
                detected.append(pii_type)
        return detected
```

### 4. OAuth Token Encryption

```python
# app/security/encryption.py

from cryptography.fernet import Fernet

class TokenEncryption:
    """Encrypt OAuth tokens at rest"""
    
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt OAuth token"""
        return self.fernet.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted: str) -> str:
        """Decrypt OAuth token"""
        return self.fernet.decrypt(encrypted.encode()).decode()

# Store in environment/secrets manager
ENCRYPTION_KEY = Fernet.generate_key()  # Store securely!
```

---

## � Human-in-the-Loop (HIL) Meta-Workflow

The HIL system functions as a meta-workflow layer above the agentic workflows, providing seamless transitions between autonomous AI agents and human agents when needed.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    🎯 HIL Meta-Workflow Layer                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Conversation Orchestrator (with is_hil flag)             │  │
│  │                                                            │  │
│  │  if is_hil == false:                                       │  │
│  │    ├──> Execute Agent Workflow                            │  │
│  │    └──> Route to sink based on agent decision:            │  │
│  │         ├──> FINISH (conversation complete)               │  │
│  │         └──> HANDOVER (escalate to human)                 │  │
│  │                                                            │  │
│  │  if is_hil == true:                                        │  │
│  │    ├──> Route to Human Agent Queue                        │  │
│  │    ├──> Wait for human response                           │  │
│  │    └──> Update conversation state                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │   🤖 Agentic Workflow Layer        │
         │                                     │
         │   Agent Workflow Execution          │
         │   ├── Classify Intent               │
         │   ├── Process Request               │
         │   ├── Evaluate Confidence           │
         │   └── Decision:                     │
         │       ├── FINISH (success)          │
         │       └── HANDOVER (need human)     │
         └────────────────────────────────────┘
```

### Conversation State Machine

```
┌──────────────┐
│   NEW        │  Conversation created
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ AI_PROCESSING│  is_hil=false, agent workflow executing
└──────┬───────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐   ┌──────────────┐
│  FINISHED    │   │ HANDOVER     │  Agent requests human
└──────────────┘   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ PENDING_HUMAN│  is_hil=true
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ HUMAN_ACTIVE │  Human agent assigned
                   └──────┬───────┘
                          │
                          ├─────────────────┐
                          │                 │
                          ▼                 ▼
                   ┌──────────────┐   ┌──────────────┐
                   │  RESOLVED    │   │ BACK_TO_AI   │  Human re-routes to AI
                   └──────────────┘   └──────┬───────┘
                                             │
                                             └──> Back to AI_PROCESSING
```

### HIL Database Schema

```sql
-- Conversations (Extended)
ALTER TABLE conversations ADD COLUMN is_hil BOOLEAN DEFAULT false;
ALTER TABLE conversations ADD COLUMN hil_reason TEXT;
ALTER TABLE conversations ADD COLUMN assigned_agent_id UUID REFERENCES human_agents(id);
ALTER TABLE conversations ADD COLUMN handover_context JSONB;
ALTER TABLE conversations ADD COLUMN confidence_score FLOAT;

-- Human Agents
CREATE TABLE human_agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN ('online', 'away', 'offline')),
  skills TEXT[] DEFAULT '{}',  -- e.g., ['returns', 'technical', 'billing']
  current_load INT DEFAULT 0,
  max_capacity INT DEFAULT 5,
  last_active TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Agent Workflow with HIL Sink Nodes

```yaml
# config/workflows/customer_support_hil.yaml

name: customer_support_hil
version: 2.0.0

# HIL Configuration
hil_config:
  enabled: true
  handover_triggers:
    - type: confidence_threshold
      threshold: 0.7  # Handover if confidence < 70%
    - type: explicit_request
      patterns: ["speak to human", "talk to agent", "escalate"]
    - type: agent_decision
    - type: error_threshold
      consecutive_errors: 3

workflow:
  nodes:
    # ===== SINK: Finish (Success) =====
    - id: FINISH
      type: sink
      action: finish_conversation
    
    # ===== SINK: Handover to Human =====
    - id: HANDOVER
      type: sink
      action: handover_to_human
      config:
        required_skills: |
          {% if classify_intent.intent == 'return_request' %}['returns']
          {% elif classify_intent.intent == 'technical_issue' %}['technical']
          {% else %}['general']
          {% endif %}
```

### HIL Orchestration Service

```python
# app/services/hil_orchestrator.py

class ConversationStatus(str, Enum):
    NEW = "new"
    AI_PROCESSING = "ai_processing"
    FINISHED = "finished"
    HANDOVER = "handover"
    PENDING_HUMAN = "pending_human"
    HUMAN_ACTIVE = "human_active"
    RESOLVED = "resolved"
    BACK_TO_AI = "back_to_ai"

class HILOrchestrator:
    """
    Meta-orchestrator for Human-in-the-Loop workflows.
    Manages conversation state and routing between AI and humans.
    """
    
    async def handle_message(
        self,
        conversation_id: str,
        message: str,
        sender_type: str = "user"
    ) -> Dict[str, Any]:
        # Route based on state (is_hil flag)
        if conversation["is_hil"]:
            return await self._handle_human_conversation(conversation_id, message, sender_type)
        else:
            return await self._handle_ai_conversation(conversation_id, message)
```

### Provider API Key Security

When integrating LLM providers (OpenAI, Anthropic, etc.), it's crucial to implement a secure approach for API keys:

1. **Consumer-Based API Keys**: Rather than using a centralized API key for all operations, implement a system where consumers (clients) provide their own API keys. This approach:
   - Ensures costs are properly attributed to each consumer
   - Prevents cross-consumer rate limit issues
   - Reduces centralized security risk
   - Eliminates the need to manage billing across consumers

2. **Key Management Implementation**:
   ```python
   # app/services/llm_router.py
   
   class LLMRouter:
       async def route_request(self, request: LLMRequest) -> LLMResponse:
           # Extract consumer API key from request context
           api_key = request.context.get("api_key")
           
           if not api_key:
               raise SecurityError("Missing API key. Consumer must provide their own key.")
               
           # Use consumer-provided key for the API call
           provider = self._select_provider(request)
           return await provider.complete(
               prompt=request.prompt, 
               api_key=api_key  # Pass consumer key to provider
           )
   ```

3. **Benefits**:
   - No need to store sensitive API keys in your system
   - Clear cost attribution for each consumer
   - Simplified billing and accounting
   - Enhanced security with reduced central key exposure

> **Note**: For detailed implementation of secure OAuth token management and storage for third-party service integration (Shopify, Gmail, etc.), see `keys_security_guideline.md` which contains comprehensive security practices including token encryption, rotation, and OAuth 2.0 flow implementation.

---

## �📊 Observability & Monitoring

### Metrics Stack

```
Prometheus (Metrics) → Grafana (Visualization) → Alertmanager (Alerts)
     ↑
OpenTelemetry (Traces)
     ↑
Application Logs → Elasticsearch → Kibana
```

### Key Metrics

**1. Agent Performance**

```python
# app/agents/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Execution metrics
agent_executions_total = Counter(
    'agent_executions_total',
    'Total agent executions',
    ['agent_type', 'workflow', 'status']
)

agent_execution_duration = Histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration',
    ['agent_type', 'workflow'],
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

agent_iterations = Histogram(
    'agent_iterations_total',
    'Number of reasoning iterations',
    ['agent_type'],
    buckets=[1, 2, 3, 5, 10, 15, 20]
)

# Cost metrics
llm_cost_usd = Counter(
    'llm_cost_usd_total',
    'Total LLM cost in USD',
    ['model', 'agent_type']
)

llm_tokens = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'agent_type', 'type']  # type: input/output
)
```

**2. Tool Performance**

```python
tool_executions_total = Counter(
    'tool_executions_total',
    'Total tool executions',
    ['tool_name', 'status']
)

tool_execution_duration = Histogram(
    'tool_execution_duration_seconds',
    'Tool execution duration',
    ['tool_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

tool_cache_hits = Counter(
    'tool_cache_hits_total',
    'Tool cache hits',
    ['tool_name']
)
```

**3. System Health**

```python
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open)',
    ['service']
)

vector_store_size = Gauge(
    'vector_store_size_total',
    'Total vectors in store',
    ['type']  # conversation/knowledge_base/execution
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)
```

### Distributed Tracing

```python
# app/agents/tracing.py

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("agent.execute")
async def execute_agent(agent_type: str, workflow: str):
    span = trace.get_current_span()
    span.set_attribute("agent.type", agent_type)
    span.set_attribute("workflow.name", workflow)
    
    try:
        result = await agent.run(...)
        span.set_attribute("agent.success", True)
        span.set_attribute("agent.iterations", result.get("iterations", 1))
        return result
    except Exception as e:
        span.set_attribute("agent.success", False)
        span.set_attribute("error.type", type(e).__name__)
        span.record_exception(e)
        raise
```

**Trace Example:**

```
Trace: customer_support_workflow [8,450ms]
├── classify_intent [1,200ms]
│   ├── llm.route [50ms]
│   ├── llm.call [1,100ms] (gpt-3.5-turbo)
│   └── parse_output [50ms]
├── handle_return [6,800ms]
│   ├── agent.plan [2,000ms] (gpt-4)
│   ├── tool.shopify.get_order [800ms]
│   ├── tool.shopify.create_return [1,500ms]
│   ├── tool.gmail.send_email [1,200ms]
│   └── agent.reflect [1,300ms] (gpt-4)
└── assemble_response [450ms]
    └── llm.call [400ms] (gpt-3.5-turbo)
```

### Grafana Dashboards

**Dashboard 1: Agent Performance**

```json
{
  "panels": [
    {
      "title": "Agent Executions/sec",
      "query": "sum(rate(agent_executions_total[5m])) by (agent_type)"
    },
    {
      "title": "P95 Latency by Agent Type",
      "query": "histogram_quantile(0.95, rate(agent_execution_duration_seconds_bucket[5m])) by (agent_type)"
    },
    {
      "title": "Success Rate",
      "query": "sum(rate(agent_executions_total{status=\"success\"}[5m])) / sum(rate(agent_executions_total[5m]))"
    }
  ]
}
```

**Dashboard 2: Cost Tracking**

```json
{
  "panels": [
    {
      "title": "LLM Costs (Last 24h)",
      "query": "sum(increase(llm_cost_usd_total[24h])) by (model)"
    },
    {
      "title": "Cost per Conversation",
      "query": "sum(rate(llm_cost_usd_total[5m])) / sum(rate(agent_executions_total[5m]))"
    },
    {
      "title": "Token Usage by Model",
      "query": "sum(rate(llm_tokens_total[5m])) by (model, type)"
    }
  ]
}
```

### Alert Rules

```yaml
# alerts.yaml

groups:
  - name: agent_alerts
    interval: 1m
    rules:
      # High error rate
      - alert: AgentHighErrorRate
        expr: |
          sum(rate(agent_executions_total{status="failed"}[5m])) / 
          sum(rate(agent_executions_total[5m])) > 0.10
        for: 5m
        annotations:
          summary: "Agent error rate > 10%"
        
      # Budget exceeded
      - alert: DailyBudgetExceeded
        expr: sum(increase(llm_cost_usd_total[24h])) > 500
        annotations:
          summary: "Daily LLM budget exceeded ($500)"
      
      # High latency
      - alert: AgentHighLatency
        expr: |
          histogram_quantile(0.95, 
            rate(agent_execution_duration_seconds_bucket[5m])
          ) > 30
        for: 10m
        annotations:
          summary: "P95 agent latency > 30s"
      
      # Circuit breaker open
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 2m
        annotations:
          summary: "Circuit breaker open for {{$labels.service}}"
```

---

## 💾 Database Schema

### Complete Schema

```sql
-- =====================================
-- Core Tables
-- =====================================

-- Conversations
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id TEXT NOT NULL,
  channel TEXT NOT NULL CHECK (channel IN ('web', 'mobile', 'api')),
  status TEXT NOT NULL CHECK (status IN ('active', 'closed', 'escalated')),
  summary JSONB,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  closed_at TIMESTAMPTZ
);

CREATE INDEX idx_conversations_customer ON conversations(customer_id);
CREATE INDEX idx_conversations_status ON conversations(status);
CREATE INDEX idx_conversations_created ON conversations(created_at DESC);

-- Messages
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  sender_type TEXT NOT NULL CHECK (sender_type IN ('user', 'agent', 'system')),
  sender_id TEXT,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  timestamp TIMESTAMPTZ DEFAULT NOW(),
  deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, timestamp DESC);
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);

-- =====================================
-- Workflow System
-- =====================================

-- Workflow Registry
CREATE TABLE workflow_registry (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  version TEXT NOT NULL,
  description TEXT,
  config JSONB NOT NULL,
  is_active BOOLEAN DEFAULT true,
  created_by TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(name, version)
);

CREATE INDEX idx_workflows_name ON workflow_registry(name);
CREATE INDEX idx_workflows_active ON workflow_registry(is_active);

-- Workflow Executions
CREATE TABLE workflow_executions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id UUID NOT NULL REFERENCES workflow_registry(id),
  conversation_id UUID REFERENCES conversations(id),
  status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'timeout')),
  input_data JSONB,
  output_data JSONB,
  execution_trace JSONB,
  error_message TEXT,
  metrics JSONB,
  started_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE INDEX idx_executions_workflow ON workflow_executions(workflow_id);
CREATE INDEX idx_executions_status ON workflow_executions(status);
CREATE INDEX idx_executions_started ON workflow_executions(started_at DESC);

-- =====================================
-- Agent System
-- =====================================

-- Agent Metadata
CREATE TABLE agent_metadata (
  name TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK (type IN ('simple', 'reasoning', 'code')),
  description TEXT,
  capabilities TEXT[] DEFAULT '{}',
  supported_models TEXT[] DEFAULT '{}',
  default_model TEXT NOT NULL,
  cost_tier TEXT NOT NULL CHECK (cost_tier IN ('low', 'medium', 'high')),
  avg_latency_ms INT,
  is_healthy BOOLEAN DEFAULT true,
  error_rate FLOAT DEFAULT 0.0,
  last_health_check TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_metadata_type ON agent_metadata(type);
CREATE INDEX idx_agent_metadata_healthy ON agent_metadata(is_healthy);

-- Agent Executions
CREATE TABLE agent_executions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_type TEXT NOT NULL,
  goal TEXT NOT NULL,
  plan JSONB NOT NULL,
  results JSONB NOT NULL,
  success BOOLEAN NOT NULL,
  cost_usd FLOAT,
  duration_ms INT,
  iterations INT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- For semantic search (pgvector extension)
  goal_embedding vector(1536)
);

CREATE INDEX idx_agent_executions_type ON agent_executions(agent_type);
CREATE INDEX idx_agent_executions_success ON agent_executions(success);
CREATE INDEX idx_agent_executions_created ON agent_executions(created_at DESC);

-- Vector similarity index
CREATE INDEX idx_agent_executions_embedding 
ON agent_executions USING ivfflat (goal_embedding vector_cosine_ops)
WITH (lists = 100);

-- =====================================
-- Tool System
-- =====================================

-- Tool Execution Metrics
CREATE TABLE tool_execution_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tool_name TEXT NOT NULL,
  success BOOLEAN NOT NULL,
  duration_ms INT NOT NULL,
  cached BOOLEAN DEFAULT false,
  error_message TEXT,
  executed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tool_metrics_name ON tool_execution_metrics(tool_name);
CREATE INDEX idx_tool_metrics_executed ON tool_execution_metrics(executed_at DESC);

-- =====================================
-- Composio Integration
-- =====================================

-- Composio Connections (OAuth)
CREATE TABLE composio_connections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_id TEXT NOT NULL,  -- customer_id, agent_id, etc.
  app_name TEXT NOT NULL,
  connection_id TEXT NOT NULL,
  encrypted_token TEXT,  -- Fernet encrypted
  status TEXT NOT NULL CHECK (status IN ('pending', 'connected', 'failed', 'revoked')),
  connected_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE (entity_id, app_name)
);

CREATE INDEX idx_composio_connections_entity ON composio_connections(entity_id);
CREATE INDEX idx_composio_connections_status ON composio_connections(status);

-- Composio Action Usage
CREATE TABLE composio_action_usage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  app_name TEXT NOT NULL,
  action_name TEXT NOT NULL,
  entity_id TEXT,
  success BOOLEAN NOT NULL,
  error_message TEXT,
  duration_ms INT,
  executed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_composio_usage_app ON composio_action_usage(app_name);
CREATE INDEX idx_composio_usage_action ON composio_action_usage(action_name);
CREATE INDEX idx_composio_usage_executed ON composio_action_usage(executed_at DESC);

-- =====================================
-- LLM Routing
-- =====================================

-- LLM Routing Decisions
CREATE TABLE llm_routing_decisions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id UUID REFERENCES workflow_registry(id),
  node_id TEXT,
  requested_profile TEXT NOT NULL CHECK (requested_profile IN ('fast', 'balanced', 'powerful')),
  selected_model TEXT NOT NULL,
  selected_provider TEXT NOT NULL,
  estimated_tokens INT,
  estimated_cost FLOAT,
  actual_input_tokens INT,
  actual_output_tokens INT,
  actual_cost FLOAT,
  latency_ms INT,
  success BOOLEAN NOT NULL,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_llm_routing_model ON llm_routing_decisions(selected_model);
CREATE INDEX idx_llm_routing_profile ON llm_routing_decisions(requested_profile);
CREATE INDEX idx_llm_routing_created ON llm_routing_decisions(created_at DESC);

-- =====================================
-- Analytics & Reporting
-- =====================================

-- Daily Cost Summary (Materialized View)
CREATE MATERIALIZED VIEW daily_cost_summary AS
SELECT 
  DATE(created_at) as date,
  selected_model,
  requested_profile,
  COUNT(*) as total_calls,
  SUM(actual_cost) as total_cost_usd,
  AVG(actual_cost) as avg_cost_usd,
  SUM(actual_input_tokens + actual_output_tokens) as total_tokens,
  AVG(latency_ms) as avg_latency_ms,
  SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM llm_routing_decisions
GROUP BY DATE(created_at), selected_model, requested_profile;

CREATE UNIQUE INDEX idx_daily_cost_summary 
ON daily_cost_summary(date, selected_model, requested_profile);

-- Refresh daily
CREATE OR REPLACE FUNCTION refresh_daily_cost_summary()
RETURNS void AS $
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY daily_cost_summary;
END;
$ LANGUAGE plpgsql;

-- =====================================
-- Functions & Triggers
-- =====================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$ LANGUAGE plpgsql;

CREATE TRIGGER update_conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_workflow_registry_updated_at
BEFORE UPDATE ON workflow_registry
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =====================================
-- Partitioning (for high-volume tables)
-- =====================================

-- Partition messages by month
CREATE TABLE messages_template (
  LIKE messages INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Create partitions for next 12 months
-- (Run monthly via cron)
CREATE TABLE messages_2025_01 PARTITION OF messages_template
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- =====================================
-- Retention Policies
-- =====================================

-- Delete old messages (90 days)
CREATE OR REPLACE FUNCTION cleanup_old_messages()
RETURNS void AS $
BEGIN
  DELETE FROM messages 
  WHERE timestamp < NOW() - INTERVAL '90 days';
  
  DELETE FROM tool_execution_metrics
  WHERE executed_at < NOW() - INTERVAL '30 days';
  
  DELETE FROM composio_action_usage
  WHERE executed_at < NOW() - INTERVAL '90 days';
END;
$ LANGUAGE plpgsql;

-- Run daily via cron
SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_messages()');
```

---

## 🚀 Implementation Guide

### Phase 1: Foundation (Weeks 1-2)

**Goals:**
- Set up project structure
- Implement database schema
- Create base agent classes
- Implement simple agent type

**Deliverables:**

```
✅ Project setup
  ├── FastAPI application
  ├── PostgreSQL with pgvector
  ├── Redis for caching
  └── Docker Compose for local dev

✅ Database migrations
  ├── Alembic setup
  ├── All tables created
  └── Indexes optimized

✅ Simple Agent
  ├── SimpleAgent class
  ├── LLM provider abstraction (OpenAI, Anthropic)
  ├── Prompt template engine (Jinja2)
  ├── Structured output with Pydantic
  └── Unit tests (80%+ coverage)

✅ Tool Registry (Basic)
  ├── Tool registration
  ├── Tool execution with timeout
  ├── Basic retry logic
  └── HTTP tools
```

**Example Implementation:**

```python
# Week 1-2 Milestone: First working agent

from app.agents.types.simple_agent import SimpleAgent
from app.agents.llm_router import LLMRouter
from pydantic import BaseModel

class IntentResult(BaseModel):
    intent: str
    confidence: float

# Create agent
router = LLMRouter()
agent = SimpleAgent(
    llm_router=router,
    model_profile="fast",
    output_schema=IntentResult
)

# Execute
result = await agent.run(
    prompt="Classify: {{message}}",
    input_data={"message": "I want to return my order"}
)

print(f"Intent: {result.intent}, Confidence: {result.confidence}")
# Output: Intent: return_request, Confidence: 0.95
```

### Phase 2: Reasoning & Tools (Weeks 3-5)

**Goals:**
- Implement ReAct reasoning agent
- Build complete tool system
- Integrate Composio
- Create workflow orchestrator

**Deliverables:**

```
✅ Reasoning Agent
  ├── ReasoningAgent class (ReAct pattern)
  ├── Reasoning loop (max iterations)
  ├── Tool calling mechanism
  └── Integration tests

✅ Tool System
  ├── Advanced tool registry
  ├── Tool caching (Redis)
  ├── Rate limiting
  ├── Metrics collection
  ├── Built-in tools (HTTP, data, DB)
  └── Custom tool creation API

✅ Composio Integration
  ├── ComposioToolManager
  ├── Dynamic action discovery
  ├── OAuth flow management
  ├── Action search
  └── Shopify, Gmail, Slack integrations

✅ Workflow System
  ├── WorkflowRegistry service
  ├── DAG executor
  ├── Conditional edges
  ├── Parallel execution
  └── State checkpointing
```

**Example Workflow:**

```yaml
# Week 3-5 Milestone: Multi-step workflow

name: order_lookup
version: 1.0.0

workflow:
  nodes:
    - id: classify
      agent: simple_classifier
    
    - id: get_order
      agent: reasoning_agent
      config:
        tools: [shopify.get_order]
        task: "Get order details for {{order_id}}"
      condition:
        type: jmespath
        expression: "classify.output.intent == 'order_status'"
```

### Phase 3: Memory & Context (Weeks 6-7)

**Goals:**
- Implement hybrid memory system
- Set up vector database
- Create RAG pipeline
- Build episodic memory

**Deliverables:**

```
✅ Memory Manager
  ├── Short-term memory (PostgreSQL)
  ├── Long-term memory (Pinecone/Chroma)
  ├── Hybrid retrieval
  ├── Conversation indexing
  └── Semantic search

✅ Vector Database
  ├── Pinecone/Chroma setup
  ├── Embedding generation (OpenAI)
  ├── Index management
  └── Cleanup policies

✅ RAG System
  ├── Document indexing
  ├── Chunk strategy
  ├── Relevance ranking
  └── Context assembly

✅ Episodic Memory
  ├── Agent execution storage
  ├── Success pattern recognition
  └── Failure analysis
```

### Phase 4: Code Agents & LLM Routing (Weeks 8-10)

**Goals:**
- Implement autonomous code agents
- Build LLM router with cost optimization
- Create agent sandbox
- Implement cost controls

**Deliverables:**

```
✅ Code Agent
  ├── CodeAgent class (Plan→Execute→Reflect)
  ├── Dynamic planning
  ├── Self-correction
  ├── Memory-based learning
  └── Integration with episodic memory

✅ LLM Router
  ├── Profile-based routing (fast/balanced/powerful)
  ├── Token estimation
  ├── Cost calculation
  ├── Circuit breakers
  ├── Automatic fallbacks
  └── Budget enforcement

✅ Agent Sandbox
  ├── Docker-based isolation
  ├── Network restrictions
  ├── Resource limits
  └── Security policies

✅ Cost Controller
  ├── Budget tracking (hourly/daily/monthly)
  ├── Alerts (80% threshold)
  ├── Automatic throttling
  └── Cost analytics
```

### Phase 5: Security & Observability (Weeks 11-12)

**Goals:**
- Implement all security layers
- Set up monitoring stack
- Create dashboards
- Configure alerts

**Deliverables:**

```
✅ Security
  ├── API authentication (JWT)
  ├── Tool whitelisting
  ├── PII detection & sanitization
  ├── OAuth token encryption
  ├── Audit logging
  └── Security testing

✅ Monitoring
  ├── Prometheus metrics
  ├── OpenTelemetry tracing
  ├── Grafana dashboards
  ├── Alert rules
  ├── Log aggregation (ELK)
  └── Cost tracking dashboard

✅ Documentation
  ├── API documentation (OpenAPI)
  ├── Agent development guide
  ├── Workflow creation guide
  ├── Operations runbook
  └── Architecture diagrams
```

### Phase 6: Testing & Launch (Weeks 13-14)

**Goals:**
- Comprehensive testing
- Performance optimization
- Production deployment
- Go-live

**Deliverables:**

```
✅ Testing
  ├── Unit tests (>80% coverage)
  ├── Integration tests
  ├── Load tests (1000 req/s)
  ├── Security audit
  └── Cost validation

✅ Optimization
  ├── Query optimization
  ├── Cache tuning
  ├── Model selection refinement
  └── Latency improvements

✅ Deployment
  ├── Kubernetes manifests
  ├── Helm charts
  ├── CI/CD pipeline
  ├── Blue-green deployment
  └── Rollback procedures

✅ Launch
  ├── Production smoke tests
  ├── Monitoring validation
  ├── Team training
  └── Documentation handoff
```

---

## 📡 API Documentation

### Workflow Execution

**Execute Workflow**

```http
POST /api/v1/workflows/execute
Content-Type: application/json
Authorization: Bearer {jwt_token}

{
  "workflow_id": "customer_support_advanced",
  "input_data": {
    "customer_message": "I want to return order #12345",
    "customer_id": "cust_789"
  },
  "context": {
    "conversation_id": "conv_abc123",
    "customer_email": "john@example.com"
  }
}
```

**Response:**

```json
{
  "execution_id": "exec_xyz789",
  "status": "completed",
  "outputs": {
    "classify_intent": {
      "intent": "return_request",
      "confidence": 0.95
    },
    "handle_return": {
      "success": true,
      "return_id": "RET-456",
      "label_url": "https://...",
      "summary": "Return processed successfully"
    },
    "assemble_response": {
      "text": "Hi John! I've processed your return..."
    }
  },
  "trace": {
    "workflow_id": "customer_support_advanced",
    "started_at": "2025-01-12T10:30:00Z",
    "completed_at": "2025-01-12T10:30:08Z",
    "nodes": [
      {
        "node_id": "classify_intent",
        "duration_ms": 1200,
        "success": true
      },
      {
        "node_id": "handle_return",
        "duration_ms": 6800,
        "success": true
      },
      {
        "node_id": "assemble_response",
        "duration_ms": 450,
        "success": true
      }
    ],
    "costs": {
      "total_usd": 0.1520,
      "by_node": {
        "classify_intent": 0.0008,
        "handle_return": 0.1500,
        "assemble_response": 0.0012
      }
    }
  },
  "total_cost_usd": 0.1520,
  "duration_ms": 8450
}
```

### Agent Management

**List Available Agents**

```http
GET /api/v1/agents?capability=tool_calling&cost_tier=low
Authorization: Bearer {jwt_token}
```

**Response:**

```json
{
  "agents": [
    {
      "name": "simple_intent_classifier",
      "type": "simple",
      "description": "Fast intent classification",
      "capabilities": ["structured_output"],
      "supported_models": ["gpt-3.5-turbo", "claude-3-haiku"],
      "cost_tier": "low",
      "avg_latency_ms": 800,
      "is_healthy": true,
      "error_rate": 0.02
    }
  ]
}
```

### Composio Integration

**List Composio Apps**

```http
GET /api/v1/composio/apps?category=ecommerce
Authorization: Bearer {jwt_token}
```

**Response:**

```json
{
  "apps": [
    {
      "name": "shopify",
      "display_name": "Shopify",
      "description": "E-commerce platform",
      "category": "ecommerce",
      "total_actions": 150,
      "available_actions": ["get_order", "create_return", "..."],
      "is_connected": true,
      "logo_url": "https://..."
    }
  ]
}
```

**Search Actions**

```http
GET /api/v1/composio/actions/search?query=send email&limit=5
Authorization: Bearer {jwt_token}
```

**Response:**

```json
{
  "actions": [
    {
      "app_name": "gmail",
      "action_name": "send_email",
      "display_name": "Send Email",
      "description": "Send an email message",
      "parameters": {
        "to": {"type": "string", "required": true},
        "subject": {"type": "string", "required": true},
        "body": {"type": "string", "required": true}
      },
      "requires_auth": true
    }
  ]
}
```

**Connect App (OAuth)**

```http
POST /api/v1/composio/connect/shopify
Content-Type: application/json
Authorization: Bearer {jwt_token}

{
  "entity_id": "cust_789",
  "redirect_url": "https://yourapp.com/oauth/callback"
}
```

**Response:**

```json
{
  "authorization_url": "https://shopify.com/oauth/authorize?client_id=...",
  "connection_id": "conn_abc123"
}
```

### Cost Tracking

**Get Cost Summary**

```http
GET /api/v1/costs/summary?period=daily
Authorization: Bearer {jwt_token}
```

**Response:**

```json
{
  "period": "daily",
  "summary": {
    "total_calls": 1523,
    "total_cost_usd": 24.56,
    "avg_cost_usd": 0.0161,
    "total_tokens": 2847392,
    "avg_latency_ms": 2340
  },
  "by_model": [
    {
      "selected_model": "gpt-4-turbo",
      "calls": 456,
      "cost_usd": 18.20,
      "avg_latency_ms": 3100
    },
    {
      "selected_model": "gpt-3.5-turbo",
      "calls": 1067,
      "cost_usd": 6.36,
      "avg_latency_ms": 850
    }
  ]
}
```

---

## 💰 Cost Analysis

### Cost Breakdown by Component

| Component | Cost Type | Monthly Cost | Notes |
|-----------|-----------|--------------|-------|
| **LLM APIs** | Variable | $700-1,700 | Based on 1000 conversations/day |
| **PostgreSQL** | Fixed | $50-200 | Managed service (25GB) |
| **Redis** | Fixed | $20-50 | Managed cache (2GB) |
| **Vector DB** | Fixed | $70-150 | Pinecone Starter/Standard |
| **Kubernetes** | Fixed | $100-300 | 3 nodes (managed) |
| **Monitoring** | Fixed | $50-100 | Grafana Cloud + logs |
| **Composio** | Variable | $99-499 | Based on action usage |
| **Total** | - | **$1,089-2,999** | Average: $1,900/month |

### LLM Cost Scenarios

**Scenario 1: Conservative (Mostly Simple Agents)**
- 1000 conversations/day
- 70% simple agents (GPT-3.5): $0.001 × 700 = $0.70/day
- 25% reasoning (GPT-4-turbo): $0.03 × 250 = $7.50/day
- 5% code agents (GPT-4): $0.15 × 50 = $7.50/day
- **Total: $15.70/day = $471/month**

**Scenario 2: Balanced (Recommended)**
- 1000 conversations/day
- 60% simple (GPT-3.5): $0.001 × 600 = $0.60/day
- 30% reasoning (GPT-4-turbo): $0.03 × 300 = $9.00/day
- 10% code agents (GPT-4): $0.15 × 100 = $15.00/day
- **Total: $24.60/day = $738/month**

**Scenario 3: Aggressive (Heavy Autonomous)**
- 1000 conversations/day
- 40% simple (GPT-3.5): $0.001 × 400 = $0.40/day
- 30% reasoning (GPT-4-turbo): $0.03 × 300 = $9.00/day
- 30% code agents (GPT-4): $0.15 × 300 = $45.00/day
- **Total: $54.40/day = $1,632/month**

### ROI Calculation

**Current State (Manual Support):**
- 10 support agents @ $40k/year = $400k/year
- Handles ~500 conversations/day
- Cost per conversation: $400k / (500 × 365) = $2.19

**With HIL System:**
- Development: $350k (one-time)
- Operations: $1,900/month = $22.8k/year
- Handles 1000 conversations/day (2x capacity)
- Cost per conversation: $22.8k / (1000 × 365) = $0.06

**Savings:**
- Year 1: $400k - ($350k + $22.8k) = $27.2k (break-even in ~11 months)
- Year 2+: $400k - $22.8k = **$377.2k/year savings**
- 5-year ROI: **$1.86M**

### Cost Optimization Tips

**1. Use Model Router Aggressively**
```python
# Default to fast profile
default_profile = "fast"  # Use GPT-3.5 by default

# Only use powerful for critical tasks
if task_complexity > 0.8:
    profile = "powerful"
```

**2. Enable Aggressive Caching**
```python
# Cache tool results
cache_ttl = 300  # 5 minutes for frequently accessed data

# Cache LLM responses for identical prompts
llm_cache_enabled = True
```

**3. Batch Operations**
```python
# Process multiple similar requests together
if len(pending_requests) > 10:
    batch_process(pending_requests)  # Reduces overhead
```

**4. Set Budget Alerts**
```python
# Alert at 80% of daily budget
alert_threshold = 0.80
daily_budget = 50.0  # USD

if current_spend >= daily_budget * alert_threshold:
    send_alert("Approaching daily budget")
```

---

## 🚢 Production Deployment

### Infrastructure Requirements

**Minimum (Low Traffic: <100 conversations/day)**
```yaml
Kubernetes Cluster:
  - 2 nodes (4 CPU, 8GB RAM each)
  - Total: 8 CPU, 16GB RAM

PostgreSQL:
  - Shared instance (2 CPU, 4GB RAM)
  - 10GB storage

Redis:
  - Shared instance (1GB RAM)

Vector DB:
  - Pinecone Starter (100k vectors)

Monthly Cost: ~$500
```

**Recommended (Medium Traffic: 1000 conversations/day)**
```yaml
Kubernetes Cluster:
  - 3 nodes (8 CPU, 16GB RAM each)
  - Total: 24 CPU, 48GB RAM

PostgreSQL:
  - Dedicated instance (4 CPU, 8GB RAM)
  - 25GB storage, automated backups

Redis:
  - Dedicated instance (2GB RAM)
  - Persistence enabled

Vector DB:
  - Pinecone Standard (1M vectors)

Load Balancer:
  - HTTPS with SSL termination

Monthly Cost: ~$1,200
```

**Enterprise (High Traffic: 10k+ conversations/day)**
```yaml
Kubernetes Cluster:
  - 5+ nodes (16 CPU, 32GB RAM each)
  - Total: 80+ CPU, 160+ GB RAM
  - Auto-scaling enabled

PostgreSQL:
  - Multi-region replication
  - 8 CPU, 16GB RAM
  - 100GB storage

Redis:
  - Redis Cluster (6GB RAM)
  - High availability

Vector DB:
  - Pinecone Enterprise (10M+ vectors)
  - Multi-index support

CDN:
  - CloudFlare Enterprise

Monthly Cost: ~$5,000+
```

### Kubernetes Deployment

**Namespace Setup:**

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hil-system
  labels:
    name: hil-system
```

**Application Deployment:**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hil-api
  namespace: hil-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hil-api
  template:
    metadata:
      labels:
        app: hil-api
    spec:
      containers:
      - name: api
        image: hil-system/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: hil-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: hil-secrets
              key: openai-key
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**Service & Ingress:**

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: hil-api
  namespace: hil-system
spec:
  selector:
    app: hil-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hil-ingress
  namespace: hil-system
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.hil-system.com
    secretName: hil-tls
  rules:
  - host: api.hil-system.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hil-api
            port:
              number: 80
```

### CI/CD Pipeline

**GitHub Actions:**

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Tests
        run: |
          docker-compose up -d
          pytest tests/ --cov=app --cov-report=xml
          
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker Image
        run: |
          docker build -t hil-system/api:${{ github.sha }} .
          docker tag hil-system/api:${{ github.sha }} hil-system/api:latest
      
      - name: Push to Registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push hil-system/api:${{ github.sha }}
          docker push hil-system/api:latest
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/hil-api api=hil-system/api:${{ github.sha }} -n hil-system
          kubectl rollout status deployment/hil-api -n hil-system
```

### Health Checks

```python
# app/health.py

from fastapi import APIRouter, status
from typing import Dict, Any

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {"status": "healthy"}

@router.get("/ready")
async def readiness_check(
    db = Depends(get_db),
    redis = Depends(get_redis)
) -> Dict[str, Any]:
    """Readiness check (all dependencies available)"""
    
    checks = {}
    
    # Check database
    try:
        await db.fetchval("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        await redis.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
    
    # Check LLM providers
    try:
        await llm_router.route([{"role": "user", "content": "test"}])
        checks["llm"] = "healthy"
    except Exception as e:
        checks["llm"] = f"unhealthy: {str(e)}"
    
    all_healthy = all(v == "healthy" for v in checks.values())
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {"status": "ready" if all_healthy else "not_ready", "checks": checks}
```

---

## 🆚 Build vs Buy Analysis

### Option 1: Build Custom (This Architecture)

**Pros:**
✅ Complete control over agent behavior  
✅ Custom memory system with episodic learning  
✅ Optimized LLM routing (56% cost savings)  
✅ Deep integration with internal systems  
✅ Extensible for future requirements  
✅ Team learns AI engineering skills  

**Cons:**
❌ 12-14 weeks development time  
❌ Requires ML engineering expertise  
❌ $300-400k development cost  
❌ Ongoing maintenance burden  
❌ Higher initial risk  

**Best For:**
- Companies with >$10M revenue
- ML engineering team (2-3 engineers)
- Complex, unique requirements
- Long-term AI strategy
- Need for IP/competitive advantage

### Option 2: n8n + AgentKit

**Pros:**
✅ Quick setup (4 weeks)  
✅ Visual workflow builder  
✅ 400+ pre-built integrations  
✅ Lower upfront cost ($50-100k)  
✅ Smaller team needed (1-2 engineers)  
✅ Battle-tested platform  

**Cons:**
❌ Limited agent autonomy  
❌ No custom memory system  
❌ Less control over LLM selection  
❌ Vendor lock-in  
❌ Monthly licensing costs  
❌ May not scale to enterprise needs  

**Best For:**
- Startups with <$5M revenue
- Need to ship fast (MVP in weeks)
- Standard workflows (80% use cases)
- Limited ML engineering resources
- Testing AI feasibility

### Hybrid Approach (Recommended)

**Phase 1: Start with n8n (Months 1-3)**
- Build workflows using n8n + AgentKit
- Validate use cases and ROI
- Learn what agents need to do
- Keep team small (1-2 engineers)

**Phase 2: Identify Gaps (Month 4)**
- What can't n8n do?
- Where do you need more control?
- What's the cost of n8n at scale?

**Phase 3: Migrate Gradually (Months 5-10)**
- Build custom components only where needed
- Keep n8n for simple workflows
- Custom agents for complex cases
- Gradual team expansion

**Benefits:**
✅ Fast time to market  
✅ Lower risk  
✅ Learn before building  
✅ Avoid over-engineering  
✅ Smooth migration path  

### Decision Matrix

| Criterion | Weight | Build Custom | n8n + AgentKit | Hybrid |
|-----------|--------|--------------|----------------|--------|
| **Time to Market** | 20% | 2/10 | 9/10 | 7/10 |
| **Cost (Year 1)** | 15% | 3/10 | 8/10 | 7/10 |
| **Flexibility** | 20% | 10/10 | 6/10 | 8/10 |
| **Scalability** | 15% | 10/10 | 7/10 | 9/10 |
| **Maintenance** | 10% | 4/10 | 9/10 | 7/10 |
| **Team Size** | 10% | 3/10 | 9/10 | 7/10 |
| **Learning Curve** | 10% | 3/10 | 9/10 | 6/10 |

**Weighted Scores:**
- Build Custom: **5.8/10**
- n8n + AgentKit: **7.9/10**
- Hybrid: **7.6/10**

**Recommendation: Start with n8n + AgentKit, migrate to hybrid as you scale.**

---

## 📚 Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| **Agent** | AI system that can perceive, reason, and act autonomously |
| **ReAct** | Reasoning + Acting pattern for agent decision-making |
| **RAG** | Retrieval-Augmented Generation (memory + LLM) |
| **DAG** | Directed Acyclic Graph (workflow structure) |
| **Circuit Breaker** | Pattern to prevent cascading failures |
| **Episodic Memory** | Memory of past agent executions and outcomes |
| **Tool** | Function an agent can call to interact with external systems |
| **Composio** | Platform providing 1000+ pre-built tool integrations |

### B. Related Resources

**Documentation:**
- LangGraph: https://langchain-ai.github.io/langgraph/
- Composio: https://docs.composio.dev/
- OpenTelemetry: https://opentelemetry.io/
- Prometheus: https://prometheus.io/docs/

**Papers:**
- ReAct: https://arxiv.org/abs/2210.03629
- Tool Learning: https://arxiv.org/abs/2304.08354
- Agent Benchmarks: https://arxiv.org/abs/2308.04026

**Communities:**
- LangChain Discord: discord.gg/langchain
- AI Agents Reddit: r/LocalLLaMA
- Composio Slack: composio.dev/slack

### C. Team Structure

**Recommended Team (Build Custom):**

```
Project Lead (1)
├── ML Engineers (2)
│   ├── Agent architecture
│   ├── LLM integration
│   ├── Memory systems
│   └── Model optimization
│
├── Backend Engineer (1)
│   ├── API development
│   ├── Database design
│   ├── Tool integrations
│   └── Workflow orchestration
│
├── DevOps Engineer (1)
│   ├── K8s deployment
│   ├── CI/CD pipelines
│   ├── Monitoring setup
│   └── Security hardening
│
└── QA Engineer (0.5)
    ├── Test automation
    ├── Load testing
    └── Security testing
```

**Total:** 4.5 FTE for 12-14 weeks

### D. Next Steps

1. **Week 1: Decision**
   - Review this architecture with stakeholders
   - Decide: Build, Buy, or Hybrid
   - Allocate budget and team

2. **Week 2: Planning**
   - Finalize tech stack
   - Set up development environment
   - Create detailed project plan

3. **Week 3-14: Implementation**
   - Follow phase-by-phase guide
   - Weekly demos to stakeholders
   - Continuous testing and iteration

4. **Week 15-16: Launch**
   - Production deployment
   - Team training
   - Monitoring validation
   - Go-live checklist

---

## 📋 Production Checklist

### Pre-Launch Checklist

**Infrastructure:**
- [ ] Kubernetes cluster provisioned and configured
- [ ] PostgreSQL with automated backups (RPO < 1 hour)
- [ ] Redis cluster with persistence enabled
- [ ] Vector database (Pinecone/Chroma) configured
- [ ] Load balancer with SSL certificates
- [ ] CDN configured (if applicable)
- [ ] All secrets stored in secrets manager (not env vars)

**Application:**
- [ ] All database migrations applied
- [ ] Indexes created and optimized
- [ ] Agent registry populated with all agent types
- [ ] Composio apps connected and OAuth tested
- [ ] LLM API keys validated (OpenAI, Anthropic)
- [ ] Tool whitelists configured per agent type
- [ ] Cost budgets set (hourly/daily/monthly)
- [ ] PII detection patterns configured

**Security:**
- [ ] API authentication implemented (JWT)
- [ ] Rate limiting configured (100 req/min per key)
- [ ] API keys have expiration (90 days)
- [ ] Docker sandbox tested for code agents
- [ ] OAuth tokens encrypted at rest (Fernet)
- [ ] Audit logging enabled
- [ ] Security scan passed (OWASP Top 10)
- [ ] Penetration testing completed

**Monitoring:**
- [ ] Prometheus metrics collecting
- [ ] Grafana dashboards created
  - [ ] Agent Performance dashboard
  - [ ] Cost Tracking dashboard
  - [ ] System Health dashboard
  - [ ] Tool Usage dashboard
- [ ] Alert rules configured
  - [ ] High error rate (>10%)
  - [ ] Daily budget exceeded
  - [ ] High latency (P95 > 30s)
  - [ ] Circuit breaker open
- [ ] PagerDuty/Slack integration tested
- [ ] Log aggregation working (ELK/CloudWatch)
- [ ] Distributed tracing enabled (Jaeger/Tempo)

**Testing:**
- [ ] Unit tests passing (>80% coverage)
- [ ] Integration tests passing
- [ ] Load test passed (1000 req/s sustained)
- [ ] Chaos engineering test (node failure recovery)
- [ ] Cost validation test (within budget)
- [ ] Security audit passed

**Documentation:**
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Architecture diagrams updated
- [ ] Agent development guide written
- [ ] Workflow creation guide written
- [ ] Operations runbook completed
- [ ] Incident response procedures documented
- [ ] Team trained on system operations

**Compliance:**
- [ ] GDPR compliance verified
- [ ] Data retention policy implemented (90 days)
- [ ] Right to deletion implemented
- [ ] Privacy policy updated
- [ ] Terms of service updated

---

## 🔥 Common Issues & Solutions

### Issue 1: High LLM Costs

**Symptoms:**
- Daily budget alerts firing
- Cost per conversation > expected
- High token usage

**Root Causes:**
- Using powerful models for simple tasks
- Not leveraging router profiles
- Inefficient prompts (too verbose)
- No caching enabled

**Solutions:**

```python
# 1. Use model router aggressively
agent = SimpleAgent(
    llm_router=llm_router,
    model_profile="fast",  # Default to cheapest
)

# 2. Enable caching
tool_registry.register(Tool(
    name="expensive_api",
    cache_ttl=300,  # Cache for 5 minutes
))

# 3. Optimize prompts
# Bad: Long, verbose prompt
prompt = """
You are a helpful assistant. Please carefully analyze the following 
customer message and provide a detailed classification...
"""

# Good: Concise, direct prompt
prompt = "Classify intent: {{message}}"

# 4. Set strict budgets
cost_controller = CostController(
    hourly_limit=10.0,  # $10/hour max
    daily_limit=100.0,  # $100/day max
)
```

### Issue 2: Agent Getting Stuck in Loops

**Symptoms:**
- Reasoning agent hits max iterations
- Same tool called repeatedly
- No progress toward goal

**Root Causes:**
- Poor tool design (unclear outputs)
- Ambiguous task description
- Missing context
- Tool errors not handled

**Solutions:**

```python
# 1. Set lower max iterations
agent = ReasoningAgent(
    max_iterations=5,  # Fail fast
)

# 2. Improve tool descriptions
Tool(
    name="get_order",
    description="""
    Get order details by ID.
    Returns: {order_id, status, items, total, customer_email}
    Use this when: Customer asks about order status
    """,
)

# 3. Add loop detection
def detect_loop(reasoning_chain: List[ThoughtStep]) -> bool:
    """Detect if agent is repeating same actions"""
    recent_actions = [s.action for s in reasoning_chain[-3:]]
    return len(recent_actions) == len(set(recent_actions))  # All same

# 4. Improve reflection prompt
reflection_prompt = """
Analyze your reasoning:
- Did you make progress toward the goal?
- Are you repeating the same actions?
- Do you have enough information to answer?
"""
```

### Issue 3: Tool Execution Timeouts

**Symptoms:**
- Tool calls failing with timeout errors
- Workflows stuck in "running" state
- Increased latency

**Root Causes:**
- External API slow/down
- Database query not optimized
- Network issues
- Insufficient timeout value

**Solutions:**

```python
# 1. Implement retry with exponential backoff
Tool(
    name="external_api",
    timeout_seconds=10,  # Reasonable timeout
    retry_config={
        "max_attempts": 3,
        "backoff": 2  # 2s, 4s, 8s
    }
)

# 2. Use circuit breakers
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60
)

# 3. Add fallback tools
if not await tool_registry.execute("primary_api", params):
    result = await tool_registry.execute("fallback_api", params)

# 4. Optimize database queries
# Bad: N+1 query
for order_id in order_ids:
    order = await db.fetchrow("SELECT * FROM orders WHERE id = $1", order_id)

# Good: Batch query
orders = await db.fetch(
    "SELECT * FROM orders WHERE id = ANY($1)",
    order_ids
)
```

### Issue 4: Memory/Context Issues

**Symptoms:**
- Agent missing relevant context
- Repeated questions
- Contradictory responses

**Root Causes:**
- RAG not returning relevant docs
- Conversation history too short
- Embeddings not updated
- Wrong retrieval strategy

**Solutions:**

```python
# 1. Increase context window
memory_manager.get_recent_messages(
    conversation_id,
    limit=100  # Increase from 50
)

# 2. Improve RAG relevance
results = memory_manager.search_semantic(
    query=current_message,
    top_k=10,  # Get more candidates
    filters={"type": "knowledge_base", "category": "returns"}
)

# 3. Use hybrid retrieval
context = await memory_manager.get_context(
    conversation_id,
    current_message,
    strategy="hybrid"  # Short + long + episodic
)

# 4. Re-index periodically
# Cron job to refresh embeddings
await memory_manager.reindex_knowledge_base()
```

### Issue 5: OAuth Token Expiration

**Symptoms:**
- Composio actions failing with "unauthorized"
- Customer complaints about broken integrations
- High tool error rate

**Root Causes:**
- Token refresh not working
- Token not encrypted properly
- Connection revoked by user
- OAuth app credentials invalid

**Solutions:**

```python
# 1. Implement automatic token refresh
async def execute_action_with_refresh(
    app_name: str,
    action_name: str,
    entity_id: str
):
    try:
        return await composio.execute_action(app_name, action_name, entity_id)
    except UnauthorizedError:
        # Try to refresh token
        await composio.refresh_connection(app_name, entity_id)
        return await composio.execute_action(app_name, action_name, entity_id)

# 2. Monitor connection health
async def check_connections():
    """Daily job to verify all connections"""
    connections = await db.fetch(
        "SELECT * FROM composio_connections WHERE status = 'connected'"
    )
    
    for conn in connections:
        try:
            await composio.test_connection(conn.app_name, conn.entity_id)
        except Exception:
            # Notify user to reconnect
            await send_reconnect_email(conn.entity_id, conn.app_name)

# 3. Handle graceful degradation
if not connection_available:
    # Fall back to manual process
    await send_email_to_agent(
        "Please manually process this return in Shopify"
    )
```

---

## 📈 Scaling Strategy

### Horizontal Scaling

**Application Tier:**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hil-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hil-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

**Database Scaling:**

```python
# Use connection pooling
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 10

# Read replicas for queries
READ_REPLICA_URLS = [
    "postgresql://read1.db.com/hil",
    "postgresql://read2.db.com/hil",
]

# Route read-only queries to replicas
@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    # Use read replica
    db = await get_read_replica()
    return await db.fetchrow(
        "SELECT * FROM conversations WHERE id = $1",
        conversation_id
    )
```

**Cache Scaling:**

```python
# Redis Cluster for high throughput
redis_cluster = RedisCluster(
    startup_nodes=[
        {"host": "redis1", "port": 6379},
        {"host": "redis2", "port": 6379},
        {"host": "redis3", "port": 6379},
    ]
)

# Implement cache warming
async def warm_cache():
    """Pre-populate cache with frequently accessed data"""
    popular_workflows = await db.fetch(
        "SELECT id, config FROM workflow_registry WHERE is_active = true"
    )
    
    for workflow in popular_workflows:
        await redis.setex(
            f"workflow:{workflow.id}",
            3600,
            json.dumps(workflow.config)
        )
```

### Vertical Scaling

**When to Scale Up:**
- CPU consistently > 80%
- Memory consistently > 85%
- Database query latency > 100ms P95
- Redis latency > 10ms P95

**Resource Allocation:**

| Component | Current | 100 conv/day | 1,000 conv/day | 10,000 conv/day |
|-----------|---------|--------------|----------------|-----------------|
| **API Pods** | 2 CPU/4GB | 3 pods | 8 pods | 20 pods |
| **PostgreSQL** | 2 CPU/4GB | 4 CPU/8GB | 8 CPU/16GB | 16 CPU/32GB |
| **Redis** | 1GB | 2GB | 4GB | 8GB |
| **Vector DB** | 100K vectors | 1M vectors | 5M vectors | 20M vectors |

---

## 🎓 Training Materials

### For Developers

**1. Creating a New Agent Type**

```python
# Step 1: Define agent class
class CustomAgent:
    def __init__(self, llm_router, tools, config):
        self.llm_router = llm_router
        self.tools = tools
        self.config = config
    
    async def run(self, task: str, context: dict) -> dict:
        # Your agent logic here
        pass

# Step 2: Create factory
class CustomAgentFactory:
    async def create(self, config, llm_router, tool_manager):
        return CustomAgent(llm_router, tools, config)

# Step 3: Register
agent_registry.register(
    AgentMetadata(
        name="custom_agent",
        type="custom",
        capabilities=[AgentCapability.CUSTOM],
        default_model="gpt-4-turbo",
        cost_tier="medium"
    ),
    CustomAgentFactory()
)
```

**2. Creating a Custom Tool**

```python
# Step 1: Define tool function
async def custom_api_call(params: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/endpoint",
            json=params
        )
        return response.json()

# Step 2: Register tool
tool_registry.register(Tool(
    name="custom_api",
    description="Call custom API endpoint",
    input_schema={
        "param1": "string",
        "param2": "number"
    },
    func=custom_api_call,
    timeout_seconds=15,
    cache_ttl=300
))
```

**3. Creating a Workflow**

```yaml
# Step 1: Define workflow YAML
name: my_workflow
version: 1.0.0
description: My custom workflow

workflow:
  nodes:
    - id: step1
      agent: simple_agent
      prompt_template: "Process: {{input}}"
    
    - id: step2
      agent: reasoning_agent
      config:
        tools: [custom_api]
      task: "Handle {{step1.output}}"
      condition:
        type: jmespath
        expression: "step1.output.success == true"
  
  edges:
    - from: step1
      to: step2
```

```python
# Step 2: Register workflow
workflow_service = WorkflowRegistryService(db)
await workflow_service.create_workflow(
    name="my_workflow",
    version="1.0.0",
    config=yaml.safe_load(workflow_yaml)
)

# Step 3: Execute workflow
result = await orchestrator.execute_workflow(
    workflow_id="my_workflow",
    input_data={"input": "Hello World"}
)
```

### For Operations Team

**1. Monitoring Checklist**

Daily:
- [ ] Check Grafana dashboards
- [ ] Review error rate (should be <5%)
- [ ] Check daily cost (should be within budget)
- [ ] Verify all circuit breakers closed

Weekly:
- [ ] Review slow queries (>1s)
- [ ] Check disk usage (DB, Redis)
- [ ] Review top cost workflows
- [ ] Update agent performance metrics

Monthly:
- [ ] Review and optimize agent configurations
- [ ] Update LLM model catalog
- [ ] Clean up old data (retention policy)
- [ ] Security patch updates

**2. Incident Response**

```bash
# High error rate
# 1. Check recent deployments
kubectl rollout history deployment/hil-api -n hil-system

# 2. Check logs
kubectl logs -f deployment/hil-api -n hil-system | grep ERROR

# 3. Rollback if needed
kubectl rollout undo deployment/hil-api -n hil-system

# Database issues
# 1. Check connections
psql -h db.example.com -U hil -c "SELECT count(*) FROM pg_stat_activity"

# 2. Check slow queries
psql -h db.example.com -U hil -c "
  SELECT query, calls, total_time/calls as avg_time
  FROM pg_stat_statements
  ORDER BY total_time DESC
  LIMIT 10
"

# LLM provider down
# 1. Check circuit breaker status
curl https://api.hil-system.com/metrics | grep circuit_breaker_state

# 2. Manually close/open circuit
curl -X POST https://api.hil-system.com/admin/circuit-breaker/gpt-4/close
```

---

## 🎯 Success Metrics

### Key Performance Indicators (KPIs)

**Business Metrics:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Automation Rate** | 80% | % of conversations handled without human |
| **Customer Satisfaction** | 4.2/5 | Post-conversation survey |
| **Escalation Rate** | <20% | % of conversations escalated to human |
| **Resolution Time** | <5 min | Average time to resolve issue |
| **Cost per Conversation** | <$0.10 | Total cost / conversations |
| **ROI** | Positive in 12 months | Savings vs investment |

**Technical Metrics:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **System Uptime** | 99.5% | Uptime monitoring |
| **API Latency P95** | <2s | Prometheus metrics |
| **Agent Success Rate** | 90% | % of agent executions successful |
| **LLM Cost** | <$1,000/day | Daily cost tracking |
| **Error Rate** | <5% | % of failed requests |
| **Cache Hit Rate** | >70% | Redis cache hits / total |

**Agent Performance:**

| Agent Type | Latency Target | Cost Target | Success Rate Target |
|------------|----------------|-------------|---------------------|
| **Simple** | <2s | <$0.01 | >99% |
| **Reasoning** | <15s | <$0.05 | >95% |
| **Code** | <60s | <$0.20 | >85% |

### Dashboards

**Executive Dashboard:**
- Total conversations handled
- Automation rate trend
- Cost savings vs manual
- Customer satisfaction trend
- ROI calculator

**Operations Dashboard:**
- System health (uptime, errors)
- Performance metrics (latency, throughput)
- Cost breakdown (by model, by workflow)
- Alert summary

**Engineering Dashboard:**
- Agent performance by type
- Tool usage and success rates
- LLM routing decisions
- Cache hit rates
- Database query performance

---

## 📞 Support & Maintenance Plan

### Support Tiers

**Tier 1: User Support**
- Handle customer inquiries
- Basic troubleshooting
- Escalate to Tier 2 if needed

**Tier 2: Operations**
- Monitor system health
- Respond to alerts
- Perform routine maintenance
- Escalate to Tier 3 for complex issues

**Tier 3: Engineering**
- Debug complex issues
- Deploy fixes and updates
- Optimize performance
- Develop new features

### On-Call Rotation

```yaml
Schedule:
  - Week 1: Engineer A (Primary), Engineer B (Secondary)
  - Week 2: Engineer B (Primary), Engineer C (Secondary)
  - Week 3: Engineer C (Primary), Engineer A (Secondary)

Responsibilities:
  - Respond to PagerDuty alerts within 15 minutes
  - Triage and resolve P1 incidents within 1 hour
  - Document all incidents in runbook
  - Handoff notes to next on-call

Escalation:
  - P1 (System Down): Page primary immediately
  - P2 (Degraded): Page primary within 15 min
  - P3 (Warning): Create ticket for next business day
```

### Maintenance Windows

**Weekly (Sundays 2-4 AM):**
- Database cleanup (old messages)
- Vector store optimization
- Cache warming
- Log rotation

**Monthly (First Sunday):**
- Security updates
- Database statistics update
- Performance testing
- Backup verification

---

## 🎓 Lessons Learned & Best Practices

### 1. Start Simple, Iterate

❌ **Don't:**
- Build all three agent types at once
- Implement every feature in v1
- Optimize prematurely

✅ **Do:**
- Start with Simple Agent only
- Add Reasoning Agent when needed
- Add Code Agent only for specific use cases
- Measure before optimizing

### 2. Observability is Critical

❌ **Don't:**
- Add monitoring as an afterthought
- Rely only on logs
- Ignore cost tracking

✅ **Do:**
- Instrument from day one
- Use metrics, logs, and traces
- Track costs in real-time
- Set up alerts proactively

### 3. Test with Real Data

❌ **Don't:**
- Only test with synthetic data
- Skip load testing
- Ignore edge cases

✅ **Do:**
- Use production-like data in staging
- Load test at 2x expected traffic
- Test failure scenarios (chaos engineering)
- Validate costs with real usage

### 4. Security First

❌ **Don't:**
- Store API keys in code
- Skip input validation
- Allow arbitrary code execution

✅ **Do:**
- Use secrets manager
- Validate all inputs
- Sandbox code agents
- Encrypt sensitive data

### 5. Cost Control is Essential

❌ **Don't:**
- Use GPT-4 for everything
- Ignore caching opportunities
- Run without budgets

✅ **Do:**
- Use LLM router profiles
- Cache aggressively
- Set budget limits
- Monitor costs daily

---

## 🏁 Conclusion

This architecture provides a **comprehensive, production-ready foundation** for building an AI agent system with:

✅ **Three agent types** covering 95% of use cases  
✅ **1000+ tool integrations** via Composio  
✅ **Intelligent LLM routing** for 56% cost savings  
✅ **Hybrid memory system** for context-aware agents  
✅ **Complete observability** for monitoring and debugging  
✅ **Enterprise security** with sandboxing and encryption  
✅ **Scalable architecture** from 100 to 10,000+ conversations/day  

### Quick Start Guide

**Week 1-2: Foundation**
```bash
# Clone repo
git clone https://github.com/yourorg/hil-system
cd hil-system

# Setup environment
docker-compose up -d
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload

# Test simple agent
curl -X POST http://localhost:8000/api/v1/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": "simple_classifier", "input_data": {"message": "Hello"}}'
```

### Decision Time

**Choose Build Custom if:**
- Budget > $300k
- Team: 2+ ML engineers
- Timeline: 3+ months acceptable
- Need: Full control & customization

**Choose n8n + AgentKit if:**
- Budget < $100k
- Team: 1-2 engineers
- Timeline: < 1 month
- Need: Fast MVP

**Choose Hybrid if:**
- Want to de-risk investment
- Learn before committing
- Gradual migration path
- Best of both worlds

---

## 📬 Contact & Support

For questions or support with this architecture:

**Documentation**: [Link to internal wiki]  
**Slack Channel**: #hil-system  
**Email**: ml-team@company.com  
**Office Hours**: Tuesdays 2-3 PM  

---

**Document Version:** 2.0  
**Last Updated:** 2025-01-12  
**Next Review:** 2025-04-12  
**Maintained By:** ML Engineering Team

---

*This architecture is a living document. Please submit PRs for improvements or open issues for questions.*