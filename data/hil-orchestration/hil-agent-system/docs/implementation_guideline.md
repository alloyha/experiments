# Complete HIL Agent System Architecture
## Production-Ready AI Workflow Orchestration with Code Agents

**Version:** 2.1  
**Last Updated:** 2025-10-13
**Status:** Core Design Complete - Production Features Documented (Postponed)
**Maintained By:** ML Engineering Team

> **⚠️ Implementation Note**: Sections 13-16 (Production Hardening) are fully documented but **postponed until core system is functional**. Current focus: Phases 1-6 in `implementation_roadmap.md` (core functionality, HIL system, advanced memory features).

---

##  Production Readiness Status

**Current Readiness**: 72% | **Target**: 95%+ for production launch

Based on industry best practices for production AI agents, the HIL Agent System demonstrates **strong architectural alignment (72%)** with production requirements, with clear paths to address remaining gaps.

### Overall Scores

| Category | Score | Status | Priority |
|----------|-------|--------|----------|
| Outcome-Focused Design | 90% | ✅ Strong | Maintain |
| Tool Security & Guardrails | 85% | ✅ Strong | Maintain |
| Recovery Mechanisms | 70% | 🟡 Good | Phase 4 |
| Observability & Monitoring | 65% | 🟡 Good | Phase 4 |
| Evaluation & Testing | 65% | 🟡 Good | Phase 5 |
| Feature Flags & Rollout | 0% | ❌ Missing | Phase 7 |
| SLO Management | 60% | 🟡 Good | Phase 7 |
| Business KPI Integration | 50% | 🟡 Partial | Phase 8 |
| **Overall Production Readiness** | **72%** | 🟡 **Production-capable with improvements** |

### Key Strengths ✅

1. **Outcome-Focused Architecture** (90%)
   - Dual execution modes (Standalone + HIL)
   - Sink nodes with explicit outcomes (FINISH/HANDOVER)
   - Task completion tracking, not just tool chaining
   - Success metrics built into execution models

2. **Security & Tool Guardrails** (85%)
   - JSON schema validation for all tools
   - Per-entity OAuth (least privilege)
   - Rate limiting per tool/entity
   - Docker sandboxing for code execution

3. **Human Escalation (HIL)** (85%)
   - Automatic handover detection
   - Skills-based agent assignment
   - Queue management with priorities
   - Conversation analytics

4. **Cost Optimization** (80%)
   - Intelligent LLM routing (56% savings)
   - Cost tracking per execution
   - Model selection framework

### Critical Gaps ❌

1. **Feature Flags & Progressive Rollout** (0%)
   - ❌ No shadow mode for safe testing
   - ❌ No progressive rollout strategy
   - ❌ No A/B testing framework
   - **Impact**: Can't deploy new features safely
   - **Priority**: HIGH (Phase 7)

2. **Comprehensive Observability** (65%)
   - ✅ Prometheus metrics
   - ✅ Structured logging
   - ❌ Distributed tracing (OpenTelemetry)
   - ❌ Real-time alerting
   - **Impact**: Limited visibility in production
   - **Priority**: HIGH (Phase 4)

3. **Evaluation Framework** (65%)
   - ✅ Cost tracking
   - ✅ Success rate monitoring
   - ❌ Regression gates in CI/CD
   - ❌ Automated quality checks
   - **Impact**: Risk of regressions
   - **Priority**: HIGH (Phase 5)

4. **SLO Definitions** (60%)
   - ✅ Metrics collection
   - ❌ Per-workflow SLOs (success %, p95, cost)
   - ❌ SLO monitoring and alerting
   - **Impact**: No formal success criteria
   - **Priority**: MEDIUM (Phase 7)

### Implementation Path to 95%+

**Weeks 7-10 (HIGH PRIORITY)**
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] Implement evaluation framework with regression gates
- [ ] Build alerting system (PagerDuty integration)
- [ ] Define SLOs per workflow

**Weeks 11-14 (MEDIUM PRIORITY)**
- [ ] Implement feature flag system
- [ ] Add shadow mode for testing
- [ ] Build A/B testing framework
- [ ] Add business KPI tracking (revenue, time saved)

**Weeks 15+ (LOWER PRIORITY)**
- [ ] Self-healing mechanisms
- [ ] Advanced batching optimizations
- [ ] Predictive scaling

### Current State: Late-Stage Prototype / Early Production 🚙

Using the "kart vs car" analogy:
- ✅ Strong chassis (architecture)
- ✅ Engine works (workflow execution)
- ✅ Safety features designed (HIL, circuit breakers, sandboxing)
- 🟡 Missing dashboard instrumentation (observability gaps)
- ❌ No airbags yet (feature flags, shadow mode)
- 🟡 Needs crash testing (regression gates, evaluation)

**With planned improvements**: **Production-grade vehicle** 🚗✅

> **Full Assessment**: See [`production_readiness_assessment.md`](production_readiness_assessment.md) for detailed gap analysis, scoring rationale, and recommendations.

---

## 📋 Table of Contents

0. [Production Readiness Status](#production-readiness-status) 🆕
1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Core Components](#core-components)
4. [Agent Types](#agent-types)
5. [Tool & Integration System](#tool--integration-system)
6. [Memory & Context Management](#memory--context-management)
7. [Advanced Chunking Strategies](#advanced-chunking-strategies)
8. [Graph Database Integration (Neo4j)](#graph-database-integration-neo4j)
9. [LLM Routing & Cost Optimization](#llm-routing--cost-optimization)
10. [Security & Sandboxing](#security--sandboxing)
11. [Human-in-the-Loop (HIL) Meta-Workflow](#human-in-the-loop-hil-meta-workflow)
12. [Observability & Monitoring](#observability--monitoring)
13. [SLOs & Performance Targets](#slos--performance-targets)
14. [Feature Flags & Progressive Rollout](#feature-flags--progressive-rollout)
15. [Evaluation & Testing Framework](#evaluation--testing-framework)
16. [Production Tuning & Cost Enforcement](#production-tuning--cost-enforcement)
17. [Database Schema](#database-schema)
18. [Implementation Guide](#implementation-guide)
19. [API Documentation](#api-documentation)
20. [Cost Analysis](#cost-analysis)
21. [Production Deployment](#production-deployment)
22. [Build vs Buy Analysis](#build-vs-buy-analysis)

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

## 🧩 Advanced Chunking Strategies

### Overview

Effective chunking is critical for RAG (Retrieval-Augmented Generation) accuracy and cost optimization. Different content types require different chunking approaches to preserve semantic meaning and context.

### Why Chunking Strategies Matter

**Problem**: Naive chunking loses context boundaries, leading to:
- Poor retrieval accuracy (30-40% degradation)
- Wasted tokens on irrelevant context
- Loss of semantic relationships
- Broken conversation flows

**Solution**: Strategy-based chunking that respects content structure.

### Supported Chunking Strategies

#### 1. Fixed-Size Chunking
```python
# app/memory/chunkers/fixed_size.py

class FixedSizeChunker:
    """Token-based chunking with overlap (baseline strategy)"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    async def chunk(self, text: str) -> List[Chunk]:
        tokens = self.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(Chunk(
                content=self.detokenize(chunk_tokens),
                start_idx=i,
                end_idx=i + len(chunk_tokens),
                metadata={"strategy": "fixed", "token_count": len(chunk_tokens)}
            ))
        return chunks
```

**Use Cases**: FAQ documents, general text, product descriptions

#### 2. Semantic Chunking (Recommended)
```python
# app/memory/chunkers/semantic.py

class SemanticChunker:
    """Chunk based on semantic boundaries using embedding similarity"""
    
    def __init__(self, similarity_threshold: float = 0.8, min_chunk_size: int = 100):
        self.threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.embeddings_model = OpenAIEmbeddings()
    
    async def chunk(self, text: str) -> List[Chunk]:
        sentences = self.split_sentences(text)
        embeddings = await self.embeddings_model.embed_documents(sentences)
        
        chunks = []
        current_chunk = []
        current_embedding = embeddings[0]
        
        for sentence, embedding in zip(sentences, embeddings):
            similarity = cosine_similarity(current_embedding, embedding)
            
            if similarity < self.threshold and len(current_chunk) >= self.min_chunk_size:
                # Semantic boundary detected
                chunks.append(Chunk(
                    content=" ".join(current_chunk),
                    metadata={
                        "strategy": "semantic",
                        "boundary_score": similarity
                    }
                ))
                current_chunk = [sentence]
                current_embedding = embedding
            else:
                current_chunk.append(sentence)
                # Rolling average
                current_embedding = (current_embedding + embedding) / 2
        
        return chunks
```

**Use Cases**: Long documentation, technical articles, knowledge base content

#### 3. Conversation-Turn Chunking (HIL-Specific)
```python
# app/memory/chunkers/conversation.py

class ConversationTurnChunker:
    """Chunk conversations by turns, preserving dialogue context"""
    
    def __init__(self, turns_per_chunk: int = 5):
        self.turns_per_chunk = turns_per_chunk
    
    async def chunk(self, conversation: List[Message]) -> List[Chunk]:
        chunks = []
        
        for i in range(0, len(conversation), self.turns_per_chunk):
            turn_window = conversation[i:i + self.turns_per_chunk]
            
            content = self._format_turns(turn_window)
            
            chunks.append(Chunk(
                content=content,
                metadata={
                    "strategy": "conversation_turn",
                    "conversation_id": conversation[0].conversation_id,
                    "turn_start": i,
                    "turn_end": i + len(turn_window),
                    "participants": self._extract_participants(turn_window),
                    "intent": self._detect_intent(turn_window)
                }
            ))
        
        return chunks
    
    def _format_turns(self, turns: List[Message]) -> str:
        return "\n".join([f"{msg.sender_type}: {msg.content}" for msg in turns])
```

**Use Cases**: Customer support history, agent conversation logs, HIL handover context

#### 4. Hierarchical Chunking
```python
# app/memory/chunkers/hierarchical.py

class HierarchicalChunker:
    """Chunk based on document structure (headings, sections)"""
    
    def __init__(self, max_chunk_tokens: int = 1000):
        self.max_chunk_tokens = max_chunk_tokens
    
    async def chunk(self, document: str) -> List[Chunk]:
        sections = self.parse_markdown_structure(document)
        
        chunks = []
        for section in sections:
            if self.token_count(section.content) > self.max_chunk_tokens:
                # Recursively chunk large sections
                subsections = self.split_section(section)
                chunks.extend(subsections)
            else:
                chunks.append(Chunk(
                    content=section.content,
                    metadata={
                        "strategy": "hierarchical",
                        "level": section.level,
                        "heading": section.heading,
                        "parent": section.parent_heading,
                        "path": section.path  # e.g., "API > Endpoints > Workflows"
                    }
                ))
        
        return chunks
```

**Use Cases**: API documentation, technical manuals, structured guides

#### 5. Entity-Based Chunking
```python
# app/memory/chunkers/entity_based.py

class EntityBasedChunker:
    """Chunk around entities (products, orders, customers)"""
    
    def __init__(self, entity_type: str):
        self.entity_type = entity_type
        self.ner_model = spacy.load("en_core_web_lg")
    
    async def chunk(self, text: str) -> List[Chunk]:
        doc = self.ner_model(text)
        
        chunks = []
        current_chunk = []
        current_entity = None
        
        for sent in doc.sents:
            entities = [ent for ent in sent.ents if ent.label_ == self.entity_type]
            
            if entities:
                entity = entities[0]
                if current_entity != entity.text:
                    if current_chunk:
                        chunks.append(Chunk(
                            content=" ".join(current_chunk),
                            metadata={
                                "strategy": "entity_based",
                                "entity_type": self.entity_type,
                                "entity_id": current_entity
                            }
                        ))
                    current_chunk = [sent.text]
                    current_entity = entity.text
                else:
                    current_chunk.append(sent.text)
        
        return chunks
```

**Use Cases**: Product catalogs, order history, customer profiles

### Unified Chunking Service

```python
# app/memory/chunking_service.py

from enum import Enum
from typing import List, Dict, Any

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    CONVERSATION_TURN = "conversation_turn"
    HIERARCHICAL = "hierarchical"
    ENTITY_BASED = "entity_based"

class ChunkingService:
    """Unified service for content chunking"""
    
    def __init__(self, embeddings_model):
        self.strategies = {
            ChunkingStrategy.FIXED: FixedSizeChunker(),
            ChunkingStrategy.SEMANTIC: SemanticChunker(),
            ChunkingStrategy.CONVERSATION_TURN: ConversationTurnChunker(),
            ChunkingStrategy.HIERARCHICAL: HierarchicalChunker(),
            ChunkingStrategy.ENTITY_BASED: EntityBasedChunker("PRODUCT")
        }
        self.embeddings_model = embeddings_model
    
    async def chunk_document(
        self,
        content: str,
        doc_type: str,
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """Automatically select and apply appropriate chunking strategy"""
        
        # Strategy selection based on document type
        strategy_map = {
            "conversation": ChunkingStrategy.CONVERSATION_TURN,
            "documentation": ChunkingStrategy.HIERARCHICAL,
            "product_catalog": ChunkingStrategy.ENTITY_BASED,
            "faq": ChunkingStrategy.SEMANTIC,
            "general": ChunkingStrategy.FIXED
        }
        
        strategy = strategy_map.get(doc_type, ChunkingStrategy.SEMANTIC)
        chunker = self.strategies[strategy]
        
        # Generate chunks
        chunks = await chunker.chunk(content)
        
        # Generate embeddings for all chunks
        embeddings = await self.embeddings_model.embed_documents(
            [c.content for c in chunks]
        )
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            if metadata:
                chunk.metadata.update(metadata)
        
        return chunks
```

### Integration with Memory Manager

```python
# app/memory/manager.py

class MemoryManager:
    def __init__(
        self,
        db_pool,
        vector_store,
        chunking_service: ChunkingService
    ):
        self.db = db_pool
        self.vector_store = vector_store
        self.chunker = chunking_service
    
    async def index_document(
        self,
        doc_id: str,
        content: str,
        doc_type: str,
        metadata: dict = None
    ):
        """Index document with intelligent chunking"""
        
        # Chunk with appropriate strategy
        chunks = await self.chunker.chunk_document(content, doc_type, metadata)
        
        # Store chunks in vector database
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            await self.vector_store.upsert(
                id=chunk_id,
                embedding=chunk.embedding,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "doc_type": doc_type,
                    "strategy": chunk.metadata.get("strategy"),
                    "content": chunk.content,
                    **chunk.metadata
                }
            )
        
        logger.info(
            "document_indexed",
            doc_id=doc_id,
            doc_type=doc_type,
            chunks=len(chunks)
        )
```

### Configuration

```yaml
# config/chunking.yaml

chunking:
  default_strategy: semantic
  
  strategies:
    conversation:
      type: conversation_turn
      turns_per_chunk: 5
      include_metadata: true
    
    documentation:
      type: hierarchical
      max_tokens: 1000
      preserve_structure: true
    
    product_catalog:
      type: entity_based
      entity_type: PRODUCT
      include_context: true
    
    faq:
      type: semantic
      similarity_threshold: 0.8
      min_chunk_size: 100
    
    general:
      type: fixed
      chunk_size: 512
      overlap: 50
  
  embeddings:
    model: "text-embedding-3-small"
    dimensions: 1536
    batch_size: 100
```

### Performance Benefits

| Strategy | Retrieval Accuracy | Token Efficiency | Use Case Fit |
|----------|-------------------|------------------|--------------|
| **Semantic** | +40% | +25% | Long-form content |
| **Conversation-Turn** | +50% | +30% | Support history |
| **Hierarchical** | +35% | +20% | Documentation |
| **Entity-Based** | +45% | +35% | Structured data |
| **Fixed** (baseline) | 0% | 0% | Simple text |

---

## 🕸️ Graph Database Integration (Neo4j)

### Architecture Overview

Neo4j complements PostgreSQL and pgvector by managing complex relationships:

```
┌─────────────────────────────────────────────────────────┐
│              Hybrid Database Architecture                │
└─────────────────────────────────────────────────────────┘

PostgreSQL (Relational)          Neo4j (Relationships)
├── Conversations                ├── Workflow Execution Paths
├── Messages                     │   └── Node dependencies
├── Executions                   │
├── Human Agents                 ├── Agent Learning Graph
├── OAuth Tokens                 │   ├── Similar executions
└── Analytics                    │   └── Success patterns
                                 │
pgvector (Semantic)              ├── Customer Journey
├── Document Embeddings          │   ├── Conversation flows
├── Conversation Chunks          │   └── Handover patterns
└── Execution History            │
                                 ├── Tool Dependencies
                                 │   ├── OAuth requirements
                                 │   └── Rate limit groups
                                 │
                                 └── Skills Network
                                     ├── Agent capabilities
                                     └── Skill relationships
```

### Use Case 1: Workflow Execution Graphs

**Problem**: Understanding all possible execution paths through complex workflows.

```cypher
// Create workflow graph
CREATE (w:Workflow {name: "customer_support_hil", version: "2.0"})
CREATE (n1:Node {id: "classify_intent", type: "simple_agent"})
CREATE (n2:Node {id: "handle_with_ai", type: "reasoning_agent"})
CREATE (n3:Node {id: "FINISH", type: "sink"})
CREATE (n4:Node {id: "HANDOVER", type: "sink"})

CREATE (w)-[:CONTAINS]->(n1)
CREATE (w)-[:CONTAINS]->(n2)
CREATE (n1)-[:NEXT {condition: "confidence > 0.7"}]->(n2)
CREATE (n1)-[:NEXT {condition: "confidence <= 0.7"}]->(n4)
CREATE (n2)-[:NEXT {condition: "success == true"}]->(n3)
CREATE (n2)-[:NEXT {condition: "success == false"}]->(n4)

// Query all paths to HANDOVER
MATCH path = (start:Node)-[:NEXT*]->(end:Node {type: "sink", id: "HANDOVER"})
RETURN path
```

**Benefits**:
- Visualize complex workflows
- Detect circular dependencies
- Optimize execution paths
- Predict handover likelihood

### Use Case 2: Agent Learning Graph

**Problem**: Learning from past executions to improve future performance.

```cypher
// Link similar executions
MATCH (e1:Execution {goal: "return_request", success: true})
MATCH (e2:Execution {goal: "return_request"})
WHERE e1.id <> e2.id
  AND abs(e1.complexity - e2.complexity) < 0.2
CREATE (e1)-[:SIMILAR_TO {score: 0.92}]->(e2)

// Find successful patterns
MATCH (e:Execution {goal: $goal, success: true})
OPTIONAL MATCH (e)-[:SIMILAR_TO]->(similar:Execution {success: true})
WITH e, count(similar) as pattern_count
RETURN e.plan, e.tools_used, e.duration, pattern_count
ORDER BY pattern_count DESC, e.success_rate DESC
LIMIT 5
```

**Benefits**:
- Discover success patterns automatically
- Recommend tools based on similar problems
- Transfer learning between agents
- Identify failure modes

### Use Case 3: Customer Journey Tracking

**Problem**: Understanding multi-step customer interactions across agents.

```cypher
// Track conversation flow
CREATE (c:Customer {id: "cust_123"})
CREATE (conv:Conversation {id: "conv_456", created_at: datetime()})
CREATE (c)-[:HAS_CONVERSATION]->(conv)

CREATE (m1:Message {role: "user", content: "I want to return order #123"})
CREATE (m2:Message {role: "ai", content: "Let me help with that..."})
CREATE (conv)-[:STARTS_WITH]->(m1)
CREATE (m1)-[:FOLLOWED_BY]->(m2)

CREATE (h:Handover {reason: "complex_case", timestamp: datetime()})
CREATE (m2)-[:TRIGGERED]->(h)

CREATE (agent:HumanAgent {id: "agent_007", skill: "returns"})
CREATE (h)-[:ASSIGNED_TO]->(agent)

// Analyze handover patterns
MATCH (c:Customer)-[:HAS_CONVERSATION]->(:Conversation)-[:TRIGGERED]->(h:Handover)
WHERE c.id = $customer_id
RETURN h.reason, h.timestamp, h.resolution_time
ORDER BY h.timestamp DESC
```

**Benefits**:
- Visualize end-to-end customer journeys
- Identify bottlenecks in handover process
- Predict escalation likelihood
- Measure resolution effectiveness

### Use Case 4: Skills-Based Agent Assignment

**Problem**: Intelligently routing work to the best available human agent.

```cypher
// Create skills network
CREATE (returns:Skill {name: "returns", complexity: "advanced"})
CREATE (technical:Skill {name: "technical_support", complexity: "expert"})
CREATE (billing:Skill {name: "billing", complexity: "intermediate"})

CREATE (agent1:HumanAgent {id: "agent_007", name: "Alice", status: "online"})
CREATE (agent2:HumanAgent {id: "agent_008", name: "Bob", status: "online"})

CREATE (agent1)-[:HAS_SKILL {proficiency: 0.95}]->(returns)
CREATE (agent1)-[:HAS_SKILL {proficiency: 0.80}]->(technical)
CREATE (agent2)-[:HAS_SKILL {proficiency: 0.90}]->(billing)

CREATE (returns)-[:RELATED_TO {strength: 0.7}]->(billing)

// Find best agent for multi-skill requirement
MATCH (a:HumanAgent)-[hs:HAS_SKILL]->(s:Skill)
WHERE s.name IN $required_skills
  AND a.status = 'online'
  AND a.current_load < $threshold
WITH a, avg(hs.proficiency) as avg_prof, count(s) as skill_count
WHERE skill_count = size($required_skills)
RETURN a.id, a.name, avg_prof, a.current_load
ORDER BY avg_prof DESC, a.current_load ASC
LIMIT 1
```

**Benefits**:
- Intelligent agent assignment
- Load balancing by expertise
- Skill gap identification
- Training recommendations

### Graph Service Implementation

```python
# app/services/graph_service.py

from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional

class GraphService:
    """Neo4j integration for relationship management"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def close(self):
        await self.driver.close()
    
    # Workflow Operations
    async def create_workflow_graph(self, workflow_config: dict):
        """Create workflow graph from YAML config"""
        async with self.driver.session() as session:
            # Create workflow and nodes
            await session.run("""
                CREATE (w:Workflow {
                    name: $name,
                    version: $version,
                    created_at: datetime()
                })
            """, name=workflow_config["name"], version=workflow_config["version"])
            
            # Create nodes and edges
            for node in workflow_config["workflow"]["nodes"]:
                await session.run("""
                    MATCH (w:Workflow {name: $workflow_name})
                    CREATE (n:Node {id: $node_id, agent_type: $agent_type})
                    CREATE (w)-[:CONTAINS]->(n)
                """, workflow_name=workflow_config["name"], 
                    node_id=node["id"], agent_type=node.get("agent"))
    
    # Learning Operations
    async def link_similar_executions(
        self,
        execution_id: str,
        similar_executions: List[Dict[str, Any]]
    ):
        """Create similarity relationships"""
        async with self.driver.session() as session:
            for similar in similar_executions:
                await session.run("""
                    MATCH (e1:Execution {id: $exec_id})
                    MATCH (e2:Execution {id: $similar_id})
                    MERGE (e1)-[:SIMILAR_TO {
                        score: $score,
                        reason: $reason
                    }]->(e2)
                """, exec_id=execution_id, similar_id=similar["id"],
                    score=similar["similarity_score"], 
                    reason=similar.get("reason"))
    
    async def get_successful_patterns(
        self,
        goal: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find successful execution patterns"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (e:Execution {goal: $goal, success: true})
                OPTIONAL MATCH (e)-[:SIMILAR_TO]->(similar:Execution {success: true})
                WITH e, count(similar) as pattern_count
                RETURN e.id, e.plan, e.tools_used, e.duration, pattern_count
                ORDER BY pattern_count DESC, e.success_rate DESC
                LIMIT $limit
            """, goal=goal, limit=limit)
            return [dict(record) async for record in result]
    
    # Journey Tracking
    async def track_handover(
        self,
        conversation_id: str,
        handover_data: dict
    ):
        """Track handover event"""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (conv:Conversation {id: $conv_id})
                CREATE (h:Handover {
                    id: $handover_id,
                    reason: $reason,
                    timestamp: datetime()
                })
                CREATE (conv)-[:TRIGGERED]->(h)
                
                MATCH (agent:HumanAgent {id: $agent_id})
                CREATE (h)-[:ASSIGNED_TO]->(agent)
            """, conv_id=conversation_id, handover_id=handover_data["id"],
                reason=handover_data["reason"], 
                agent_id=handover_data["assigned_agent_id"])
    
    # Skills Network
    async def find_best_agent(
        self,
        required_skills: List[str],
        current_load_threshold: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Find best agent based on skills"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (a:HumanAgent)-[hs:HAS_SKILL]->(s:Skill)
                WHERE s.name IN $skills
                  AND a.status = 'online'
                  AND a.current_load < $threshold
                WITH a, avg(hs.proficiency) as avg_prof, count(s) as skill_count
                WHERE skill_count = size($skills)
                RETURN a.id, a.name, avg_prof, a.current_load
                ORDER BY avg_prof DESC, a.current_load ASC
                LIMIT 1
            """, skills=required_skills, threshold=current_load_threshold)
            record = await result.single()
            return dict(record) if record else None
```

### Configuration

```yaml
# config/neo4j.yaml

neo4j:
  enabled: true
  uri: "neo4j://localhost:7687"
  user: "neo4j"
  password: "${NEO4J_PASSWORD}"
  database: "hil-agent-system"
  
  # What to track
  track:
    - workflow_execution_paths
    - agent_learning_patterns
    - customer_journeys
    - tool_dependencies
    - skills_network
  
  # Sync strategy
  sync:
    mode: "async"  # async or realtime
    batch_size: 100
    interval_seconds: 60
  
  # Performance
  connection_pool:
    max_size: 50
    acquisition_timeout: 60
```

### Docker Compose Integration

```yaml
# docker-compose.yml

services:
  neo4j:
    image: neo4j:5.15-community
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  neo4j_data:
  neo4j_logs:
```

### Performance Benefits

| Feature | PostgreSQL | Neo4j | Performance Gain |
|---------|-----------|-------|------------------|
| **Relationship Queries** | Complex JOINs | Graph traversal | 100-1000x faster |
| **Pattern Discovery** | Difficult | Native support | Enable new features |
| **Path Finding** | Recursive CTEs | Built-in algorithms | 10-100x faster |
| **Multi-hop Queries** | Multiple queries | Single query | 5-50x faster |

> **Note**: For complete implementation details including all chunking strategies and Neo4j use cases, see `docs/chunking_and_graph_strategy.md`

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

## 🎯 SLOs & Performance Targets

> **📌 Implementation Status**: This section and sections 14-16 (Feature Flags, Evaluation, Cost Enforcement) are **fully documented but postponed** until core system is functional. Focus remains on Phases 1-6 in `implementation_roadmap.md`. These sections provide comprehensive guidance for production hardening when the time comes.

### Service Level Objectives (SLOs)

SLOs define the reliability and performance targets for production AI agents. Each workflow should have explicit success criteria.

### Per-Workflow SLO Configuration

```yaml
# config/workflows/customer_support_hil.yaml
name: customer_support_hil
version: 2.0.0

# Service Level Objectives
slos:
  # Success metrics
  success_rate: 0.95  # 95% task completion rate
  intervention_rate_max: 0.10  # Max 10% require human handover
  
  # Performance metrics
  p50_latency_ms: 5000   # 50th percentile: 5s
  p95_latency_ms: 15000  # 95th percentile: 15s
  p99_latency_ms: 30000  # 99th percentile: 30s
  
  # Cost metrics
  cost_per_request_max: 0.15  # Maximum $0.15 per request
  cost_per_request_target: 0.08  # Target $0.08 per request
  
  # Quality metrics
  customer_satisfaction_min: 4.0  # Minimum 4.0/5.0
  error_rate_max: 0.05  # Maximum 5% error rate
  
  # Alerting thresholds
  alerts:
    - name: low_success_rate
      condition: "success_rate < 0.90"
      severity: warning
      notification_channels: ["slack", "pagerduty"]
    
    - name: critical_success_rate
      condition: "success_rate < 0.85"
      severity: critical
      notification_channels: ["pagerduty", "email"]
    
    - name: high_latency
      condition: "p95_latency_ms > 20000"
      severity: warning
      notification_channels: ["slack"]
    
    - name: cost_overrun
      condition: "cost_per_request_avg > 0.20"
      severity: warning
      notification_channels: ["slack", "email"]
    
    - name: high_intervention_rate
      condition: "intervention_rate > 0.15"
      severity: warning
      notification_channels: ["slack"]
```

### SLO Monitoring Service

```python
# app/services/slo_monitor.py

from typing import Dict, List, Any
from datetime import datetime, timedelta
import asyncio

class SLOMonitor:
    """Monitor and enforce Service Level Objectives"""
    
    def __init__(self, db_pool, redis_client, alerting_service):
        self.db = db_pool
        self.redis = redis_client
        self.alerts = alerting_service
        self.slo_configs = {}
    
    async def load_workflow_slos(self, workflow_id: str):
        """Load SLO configuration for a workflow"""
        workflow_config = await self._get_workflow_config(workflow_id)
        self.slo_configs[workflow_id] = workflow_config.get("slos", {})
    
    async def check_slos(self, workflow_id: str, time_window: timedelta = timedelta(hours=1)):
        """Check if SLOs are being met"""
        
        slos = self.slo_configs.get(workflow_id, {})
        if not slos:
            return None
        
        # Calculate metrics from executions
        metrics = await self._calculate_metrics(workflow_id, time_window)
        
        violations = []
        
        # Check success rate
        if "success_rate" in slos:
            if metrics["success_rate"] < slos["success_rate"]:
                violations.append({
                    "metric": "success_rate",
                    "expected": slos["success_rate"],
                    "actual": metrics["success_rate"],
                    "severity": "high"
                })
        
        # Check latency p95
        if "p95_latency_ms" in slos:
            if metrics["p95_latency_ms"] > slos["p95_latency_ms"]:
                violations.append({
                    "metric": "p95_latency_ms",
                    "expected": slos["p95_latency_ms"],
                    "actual": metrics["p95_latency_ms"],
                    "severity": "medium"
                })
        
        # Check cost per request
        if "cost_per_request_max" in slos:
            if metrics["cost_per_request_avg"] > slos["cost_per_request_max"]:
                violations.append({
                    "metric": "cost_per_request",
                    "expected": slos["cost_per_request_max"],
                    "actual": metrics["cost_per_request_avg"],
                    "severity": "medium"
                })
        
        # Check intervention rate
        if "intervention_rate_max" in slos:
            if metrics["intervention_rate"] > slos["intervention_rate_max"]:
                violations.append({
                    "metric": "intervention_rate",
                    "expected": slos["intervention_rate_max"],
                    "actual": metrics["intervention_rate"],
                    "severity": "medium"
                })
        
        # Trigger alerts
        if violations:
            await self._trigger_alerts(workflow_id, violations, slos.get("alerts", []))
        
        return {
            "workflow_id": workflow_id,
            "time_window": str(time_window),
            "metrics": metrics,
            "slos": slos,
            "violations": violations,
            "compliant": len(violations) == 0
        }
    
    async def _calculate_metrics(self, workflow_id: str, time_window: timedelta) -> Dict[str, float]:
        """Calculate metrics from recent executions"""
        
        since = datetime.now() - time_window
        
        # Query executions
        executions = await self.db.fetch("""
            SELECT 
                status,
                duration_ms,
                total_cost,
                handover_count,
                customer_satisfaction
            FROM executions
            WHERE workflow_id = $1
              AND created_at >= $2
        """, workflow_id, since)
        
        if not executions:
            return {}
        
        total = len(executions)
        successful = sum(1 for e in executions if e["status"] == "completed")
        handovers = sum(1 for e in executions if e["handover_count"] > 0)
        
        latencies = sorted([e["duration_ms"] for e in executions])
        costs = [e["total_cost"] for e in executions]
        satisfactions = [e["customer_satisfaction"] for e in executions if e["customer_satisfaction"]]
        
        return {
            "success_rate": successful / total,
            "intervention_rate": handovers / total,
            "p50_latency_ms": latencies[int(len(latencies) * 0.5)] if latencies else 0,
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
            "cost_per_request_avg": sum(costs) / len(costs) if costs else 0,
            "customer_satisfaction_avg": sum(satisfactions) / len(satisfactions) if satisfactions else 0,
            "error_rate": (total - successful) / total,
            "total_executions": total
        }
    
    async def _trigger_alerts(self, workflow_id: str, violations: List[Dict], alert_configs: List[Dict]):
        """Trigger configured alerts for SLO violations"""
        
        for violation in violations:
            # Find matching alert config
            matching_alerts = [
                alert for alert in alert_configs
                if violation["metric"] in alert.get("condition", "")
            ]
            
            for alert_config in matching_alerts:
                await self.alerts.send_alert(
                    name=alert_config["name"],
                    severity=alert_config["severity"],
                    message=f"SLO violation for {workflow_id}: {violation['metric']} = {violation['actual']:.3f} (expected: {violation['expected']:.3f})",
                    channels=alert_config.get("notification_channels", ["slack"]),
                    metadata={
                        "workflow_id": workflow_id,
                        "violation": violation
                    }
                )
    
    async def start_monitoring_loop(self, check_interval: int = 300):
        """Background task to continuously monitor SLOs"""
        
        while True:
            try:
                for workflow_id in self.slo_configs.keys():
                    result = await self.check_slos(workflow_id)
                    
                    if result and not result["compliant"]:
                        logger.warning(
                            "slo_violations_detected",
                            workflow_id=workflow_id,
                            violations=result["violations"]
                        )
                
                await asyncio.sleep(check_interval)
            
            except Exception as e:
                logger.error("slo_monitoring_error", error=str(e), exc_info=True)
                await asyncio.sleep(60)
```

### Per-Node Budget Enforcement

```python
# app/services/workflow_orchestrator.py

class AgentOrchestrator:
    async def execute_node(
        self,
        node_config: dict,
        node_inputs: dict,
        execution_context: dict
    ) -> Any:
        """Execute workflow node with SLO enforcement"""
        
        node_id = node_config["id"]
        limits = node_config.get("limits", {})
        
        # Timeout enforcement
        timeout_ms = limits.get("timeout_ms", 30000)
        
        # Budget enforcement
        max_cost = limits.get("max_cost", 1.0)
        current_cost = execution_context.get("cumulative_cost", 0)
        
        if current_cost >= max_cost:
            raise BudgetExceededError(
                f"Node {node_id} would exceed budget: {current_cost} >= {max_cost}"
            )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_node_impl(node_config, node_inputs),
                timeout=timeout_ms / 1000
            )
            
            # Track cost
            node_cost = result.get("cost", 0)
            execution_context["cumulative_cost"] = current_cost + node_cost
            
            # Check if cumulative cost exceeds limit
            if execution_context["cumulative_cost"] > max_cost:
                logger.warning(
                    "node_cost_exceeded",
                    node_id=node_id,
                    cost=node_cost,
                    cumulative=execution_context["cumulative_cost"],
                    limit=max_cost
                )
            
            return result
        
        except asyncio.TimeoutError:
            logger.error(
                "node_timeout",
                node_id=node_id,
                timeout_ms=timeout_ms
            )
            raise NodeTimeoutError(f"Node {node_id} exceeded timeout of {timeout_ms}ms")
```

### SLO Dashboard Configuration

```yaml
# config/dashboards/slo_dashboard.yaml

dashboard:
  name: "Workflow SLOs"
  refresh: 30s
  
  panels:
    - title: "Success Rate by Workflow"
      type: graph
      query: |
        sum(rate(workflow_executions_success[5m])) by (workflow_id) /
        sum(rate(workflow_executions_total[5m])) by (workflow_id)
      slo_line: 0.95
    
    - title: "P95 Latency by Workflow"
      type: graph
      query: |
        histogram_quantile(0.95, 
          sum(rate(workflow_duration_ms_bucket[5m])) by (workflow_id, le))
      slo_line: 15000
    
    - title: "Cost per Request"
      type: graph
      query: |
        sum(rate(workflow_cost_total[5m])) by (workflow_id) /
        sum(rate(workflow_executions_total[5m])) by (workflow_id)
      slo_line: 0.15
    
    - title: "Intervention Rate"
      type: graph
      query: |
        sum(rate(workflow_handovers_total[5m])) by (workflow_id) /
        sum(rate(workflow_executions_total[5m])) by (workflow_id)
      slo_line: 0.10
    
    - title: "SLO Compliance"
      type: stat
      query: |
        count(slo_compliant == 1) / count(slo_compliant)
      thresholds:
        - value: 0.95
          color: green
        - value: 0.90
          color: yellow
        - value: 0
          color: red
```

---

## 🚦 Feature Flags & Progressive Rollout

### Feature Flag System

Progressive rollout is critical for production AI agents. Deploy safely with shadow mode, gradual rollout, and automatic rollback.

### Feature Flag Service

```python
# app/services/feature_flags.py

import hashlib
from typing import Dict, Set, Optional
from enum import Enum

class RolloutStrategy(str, Enum):
    PERCENTAGE = "percentage"
    WHITELIST = "whitelist"
    GRADUAL = "gradual"

class FeatureFlagService:
    """Manage feature flags for progressive rollout"""
    
    def __init__(self, redis_client, db_pool):
        self.redis = redis_client
        self.db = db_pool
    
    async def is_enabled(
        self,
        feature_name: str,
        entity_id: str,
        context: Dict = None
    ) -> bool:
        """Check if feature is enabled for entity"""
        
        # Get feature config
        config = await self._get_feature_config(feature_name)
        
        if not config or not config.get("enabled", False):
            return False
        
        # Check whitelist
        if await self.redis.sismember(f"feature:{feature_name}:whitelist", entity_id):
            return True
        
        # Check blacklist
        if await self.redis.sismember(f"feature:{feature_name}:blacklist", entity_id):
            return False
        
        # Apply rollout strategy
        strategy = config.get("rollout_strategy", RolloutStrategy.PERCENTAGE)
        rollout_percentage = config.get("rollout_percentage", 0)
        
        if strategy == RolloutStrategy.PERCENTAGE:
            return self._check_percentage_rollout(entity_id, rollout_percentage)
        
        elif strategy == RolloutStrategy.GRADUAL:
            # Gradual rollout based on time
            rollout_start = config.get("rollout_start_time")
            rollout_duration_hours = config.get("rollout_duration_hours", 168)  # 7 days
            
            return self._check_gradual_rollout(
                entity_id,
                rollout_start,
                rollout_duration_hours,
                rollout_percentage
            )
        
        return False
    
    def _check_percentage_rollout(self, entity_id: str, percentage: int) -> bool:
        """Deterministic percentage-based rollout"""
        entity_hash = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
        return (entity_hash % 100) < percentage
    
    def _check_gradual_rollout(
        self,
        entity_id: str,
        start_time: datetime,
        duration_hours: int,
        target_percentage: int
    ) -> bool:
        """Gradual rollout over time"""
        
        if not start_time:
            return False
        
        now = datetime.now()
        elapsed_hours = (now - start_time).total_seconds() / 3600
        
        if elapsed_hours < 0:
            # Not started yet
            return False
        
        if elapsed_hours >= duration_hours:
            # Fully rolled out
            current_percentage = target_percentage
        else:
            # Linear ramp
            current_percentage = int((elapsed_hours / duration_hours) * target_percentage)
        
        return self._check_percentage_rollout(entity_id, current_percentage)
    
    async def add_to_whitelist(self, feature_name: str, entity_id: str):
        """Add entity to feature whitelist"""
        await self.redis.sadd(f"feature:{feature_name}:whitelist", entity_id)
        logger.info("feature_whitelist_added", feature=feature_name, entity=entity_id)
    
    async def add_to_blacklist(self, feature_name: str, entity_id: str):
        """Add entity to feature blacklist (disable feature)"""
        await self.redis.sadd(f"feature:{feature_name}:blacklist", entity_id)
        logger.info("feature_blacklist_added", feature=feature_name, entity=entity_id)
    
    async def update_rollout_percentage(self, feature_name: str, percentage: int):
        """Update rollout percentage (0-100)"""
        await self.redis.hset(
            f"feature:{feature_name}:config",
            "rollout_percentage",
            percentage
        )
        logger.info(
            "feature_rollout_updated",
            feature=feature_name,
            percentage=percentage
        )
```

### Shadow Mode Implementation

```python
# app/services/shadow_mode.py

class ShadowModeOrchestrator:
    """Execute workflows in shadow mode for safe testing"""
    
    def __init__(
        self,
        prod_orchestrator: AgentOrchestrator,
        shadow_orchestrator: AgentOrchestrator,
        feature_flags: FeatureFlagService,
        metrics_collector
    ):
        self.prod = prod_orchestrator
        self.shadow = shadow_orchestrator
        self.flags = feature_flags
        self.metrics = metrics_collector
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: dict,
        context: dict
    ) -> dict:
        """Execute workflow with optional shadow mode"""
        
        # Always execute production version
        prod_result = await self.prod.execute_workflow(workflow_id, input_data, context)
        
        # Check if shadow mode enabled for this workflow
        shadow_enabled = await self.flags.is_enabled(
            f"shadow_mode_{workflow_id}",
            context.get("entity_id", "")
        )
        
        if shadow_enabled:
            # Execute shadow version asynchronously (don't wait)
            asyncio.create_task(
                self._execute_shadow(workflow_id, input_data, context, prod_result)
            )
        
        return prod_result
    
    async def _execute_shadow(
        self,
        workflow_id: str,
        input_data: dict,
        context: dict,
        expected_result: dict
    ):
        """Execute shadow version and compare results"""
        
        try:
            shadow_result = await self.shadow.execute_workflow(
                workflow_id,
                input_data,
                context
            )
            
            # Compare results
            comparison = self._compare_results(expected_result, shadow_result)
            
            # Log comparison
            logger.info(
                "shadow_mode_execution",
                workflow_id=workflow_id,
                matches=comparison["matches"],
                differences=comparison["differences"]
            )
            
            # Store comparison for analysis
            await self.metrics.record_shadow_comparison(
                workflow_id=workflow_id,
                prod_result=expected_result,
                shadow_result=shadow_result,
                comparison=comparison
            )
        
        except Exception as e:
            logger.error(
                "shadow_mode_error",
                workflow_id=workflow_id,
                error=str(e),
                exc_info=True
            )
            
            await self.metrics.record_shadow_error(
                workflow_id=workflow_id,
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    def _compare_results(self, prod: dict, shadow: dict) -> dict:
        """Compare production and shadow results"""
        
        differences = []
        
        # Compare status
        if prod.get("status") != shadow.get("status"):
            differences.append({
                "field": "status",
                "prod": prod.get("status"),
                "shadow": shadow.get("status")
            })
        
        # Compare cost (should be similar)
        prod_cost = prod.get("total_cost", 0)
        shadow_cost = shadow.get("total_cost", 0)
        if abs(prod_cost - shadow_cost) / max(prod_cost, 0.001) > 0.1:  # >10% difference
            differences.append({
                "field": "cost",
                "prod": prod_cost,
                "shadow": shadow_cost,
                "diff_percent": abs(prod_cost - shadow_cost) / prod_cost * 100
            })
        
        # Compare duration
        prod_duration = prod.get("duration_ms", 0)
        shadow_duration = shadow.get("duration_ms", 0)
        if abs(prod_duration - shadow_duration) / max(prod_duration, 1) > 0.2:  # >20% difference
            differences.append({
                "field": "duration",
                "prod": prod_duration,
                "shadow": shadow_duration,
                "diff_percent": abs(prod_duration - shadow_duration) / prod_duration * 100
            })
        
        return {
            "matches": len(differences) == 0,
            "differences": differences
        }
```

### Progressive Rollout Stages

```yaml
# config/rollout/workflow_v2_rollout.yaml

feature_name: "customer_support_workflow_v2"
description: "New workflow with improved AI routing"

rollout_stages:
  # Stage 1: Shadow mode (no user impact)
  - name: shadow
    duration_days: 7
    rollout_percentage: 0
    mode: shadow_only
    metrics_collection: true
    success_criteria:
      - metric: error_rate
        threshold: 0.05
      - metric: cost_difference
        threshold: 0.10
  
  # Stage 2: Suggest mode (AI suggests, human chooses)
  - name: suggest
    duration_days: 7
    rollout_percentage: 5
    mode: suggest_with_fallback
    metrics_collection: true
    success_criteria:
      - metric: suggestion_acceptance_rate
        threshold: 0.70
      - metric: error_rate
        threshold: 0.05
  
  # Stage 3: Review mode (AI acts, human reviews)
  - name: review
    duration_days: 7
    rollout_percentage: 25
    mode: auto_with_review
    human_review_sampling: 0.10  # Review 10% of executions
    success_criteria:
      - metric: success_rate
        threshold: 0.95
      - metric: review_approval_rate
        threshold: 0.90
  
  # Stage 4: Auto mode with monitoring
  - name: auto
    duration_days: ongoing
    rollout_percentage: 100
    mode: fully_automatic
    monitoring:
      - metric: success_rate
        alert_threshold: 0.93
      - metric: cost_per_request
        alert_threshold: 0.20
      - metric: intervention_rate
        alert_threshold: 0.12
    
    auto_rollback:
      enabled: true
      triggers:
        - metric: error_rate
          threshold: 0.10
          window_minutes: 15
        - metric: success_rate
          threshold: 0.85
          window_minutes: 30

# Rollback plan
rollback:
  strategy: immediate
  preserve_data: true
  notification_channels: ["pagerduty", "slack"]
```

### Feature Flag API

```python
# app/api/v1/endpoints/feature_flags.py

from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/api/v1/feature-flags", tags=["feature-flags"])

@router.post("/{feature_name}/enable")
async def enable_feature(
    feature_name: str,
    config: FeatureFlagConfig,
    flags: FeatureFlagService = Depends(get_feature_flags)
):
    """Enable a feature flag with rollout configuration"""
    
    await flags.create_or_update_feature(feature_name, config)
    
    return {
        "feature": feature_name,
        "status": "enabled",
        "config": config
    }

@router.post("/{feature_name}/rollout")
async def update_rollout(
    feature_name: str,
    percentage: int,
    flags: FeatureFlagService = Depends(get_feature_flags)
):
    """Update rollout percentage (0-100)"""
    
    if not 0 <= percentage <= 100:
        raise HTTPException(status_code=400, detail="Percentage must be 0-100")
    
    await flags.update_rollout_percentage(feature_name, percentage)
    
    return {
        "feature": feature_name,
        "rollout_percentage": percentage
    }

@router.post("/{feature_name}/whitelist/{entity_id}")
async def add_to_whitelist(
    feature_name: str,
    entity_id: str,
    flags: FeatureFlagService = Depends(get_feature_flags)
):
    """Add entity to feature whitelist"""
    
    await flags.add_to_whitelist(feature_name, entity_id)
    
    return {
        "feature": feature_name,
        "entity_id": entity_id,
        "status": "whitelisted"
    }

@router.get("/{feature_name}/status")
async def get_feature_status(
    feature_name: str,
    flags: FeatureFlagService = Depends(get_feature_flags)
):
    """Get feature flag status and rollout info"""
    
    config = await flags._get_feature_config(feature_name)
    stats = await flags.get_rollout_stats(feature_name)
    
    return {
        "feature": feature_name,
        "config": config,
        "stats": stats
    }
```

---

## 🧪 Evaluation & Testing Framework

### Automated Evaluation System

```python
# app/services/evaluation_framework.py

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    execution_id: str
    workflow_id: str
    task_success: bool
    intervention_needed: bool
    intervention_rate: float
    cost: float
    latency_ms: int
    customer_satisfaction: Optional[float]
    errors: List[str]
    
    def passes_threshold(self, thresholds: Dict[str, float]) -> bool:
        """Check if execution meets threshold requirements"""
        
        checks = []
        
        if "task_success_min" in thresholds:
            checks.append(self.task_success or thresholds["task_success_min"] == 0)
        
        if "intervention_rate_max" in thresholds:
            checks.append(self.intervention_rate <= thresholds["intervention_rate_max"])
        
        if "cost_max" in thresholds:
            checks.append(self.cost <= thresholds["cost_max"])
        
        if "latency_max_ms" in thresholds:
            checks.append(self.latency_ms <= thresholds["latency_max_ms"])
        
        return all(checks)

class WorkflowEvaluator:
    """Evaluate workflow executions against success criteria"""
    
    def __init__(self, db_pool, redis_client):
        self.db = db_pool
        self.redis = redis_client
    
    async def evaluate_execution(
        self,
        execution_id: str,
        ground_truth: Optional[Dict] = None
    ) -> EvaluationResult:
        """Evaluate a single execution"""
        
        # Fetch execution data
        execution = await self.db.fetchrow("""
            SELECT *
            FROM executions
            WHERE id = $1
        """, execution_id)
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        # Calculate metrics
        task_success = execution["status"] == "completed"
        intervention_needed = execution["handover_count"] > 0
        
        # Get conversation for intervention rate
        conversation = await self._get_conversation(execution["conversation_id"])
        total_turns = len(conversation.get("messages", []))
        intervention_rate = execution["handover_count"] / max(total_turns, 1)
        
        # Check against ground truth if provided
        errors = []
        if ground_truth:
            errors = await self._compare_with_ground_truth(execution, ground_truth)
        
        return EvaluationResult(
            execution_id=execution_id,
            workflow_id=execution["workflow_id"],
            task_success=task_success,
            intervention_needed=intervention_needed,
            intervention_rate=intervention_rate,
            cost=execution["total_cost"],
            latency_ms=execution["duration_ms"],
            customer_satisfaction=execution.get("customer_satisfaction"),
            errors=errors
        )
    
    async def evaluate_batch(
        self,
        execution_ids: List[str],
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate multiple executions and calculate aggregate metrics"""
        
        results = []
        for exec_id in execution_ids:
            result = await self.evaluate_execution(exec_id)
            results.append(result)
        
        # Calculate aggregates
        total = len(results)
        successful = sum(1 for r in results if r.task_success)
        interventions = sum(1 for r in results if r.intervention_needed)
        avg_cost = sum(r.cost for r in results) / total
        avg_latency = sum(r.latency_ms for r in results) / total
        latencies = sorted([r.latency_ms for r in results])
        
        aggregate_metrics = {
            "total_executions": total,
            "success_rate": successful / total,
            "intervention_rate": interventions / total,
            "avg_cost": avg_cost,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": latencies[int(total * 0.5)] if total > 0 else 0,
            "p95_latency_ms": latencies[int(total * 0.95)] if total > 0 else 0,
            "p99_latency_ms": latencies[int(total * 0.99)] if total > 0 else 0,
        }
        
        # Check if batch passes thresholds
        passes = (
            aggregate_metrics["success_rate"] >= thresholds.get("success_rate_min", 0.95) and
            aggregate_metrics["intervention_rate"] <= thresholds.get("intervention_rate_max", 0.10) and
            aggregate_metrics["avg_cost"] <= thresholds.get("cost_max", 0.15) and
            aggregate_metrics["p95_latency_ms"] <= thresholds.get("p95_latency_max_ms", 15000)
        )
        
        return {
            "aggregate_metrics": aggregate_metrics,
            "thresholds": thresholds,
            "passes_thresholds": passes,
            "individual_results": results
        }
    
    async def _compare_with_ground_truth(
        self,
        execution: Dict,
        ground_truth: Dict
    ) -> List[str]:
        """Compare execution output with ground truth"""
        
        errors = []
        
        # Compare final status
        if ground_truth.get("expected_status") != execution["status"]:
            errors.append(
                f"Status mismatch: expected {ground_truth['expected_status']}, "
                f"got {execution['status']}"
            )
        
        # Compare tools used
        expected_tools = set(ground_truth.get("expected_tools", []))
        actual_tools = set(execution.get("tools_used", []))
        
        missing_tools = expected_tools - actual_tools
        if missing_tools:
            errors.append(f"Missing expected tools: {missing_tools}")
        
        extra_tools = actual_tools - expected_tools
        if extra_tools:
            errors.append(f"Used unexpected tools: {extra_tools}")
        
        return errors

### Regression Testing Integration

```python
# tests/regression/test_workflow_regression.py

import pytest
from app.services.evaluation_framework import WorkflowEvaluator

@pytest.fixture
async def workflow_evaluator():
    # Setup evaluator with test database
    return WorkflowEvaluator(test_db, test_redis)

@pytest.fixture
async def regression_test_set():
    """Load regression test set with ground truth"""
    return await load_test_cases("tests/data/regression_test_set.json")

@pytest.mark.asyncio
async def test_customer_support_workflow_regression(
    workflow_evaluator,
    regression_test_set
):
    """Regression test for customer support workflow"""
    
    # Execute workflow for each test case
    execution_ids = []
    for test_case in regression_test_set:
        result = await execute_workflow(
            "customer_support_hil",
            test_case["input"]
        )
        execution_ids.append(result["execution_id"])
    
    # Evaluate batch
    thresholds = {
        "success_rate_min": 0.95,
        "intervention_rate_max": 0.10,
        "cost_max": 0.15,
        "p95_latency_max_ms": 15000
    }
    
    evaluation = await workflow_evaluator.evaluate_batch(
        execution_ids,
        thresholds
    )
    
    # Assert thresholds are met
    assert evaluation["passes_thresholds"], \
        f"Regression test failed: {evaluation['aggregate_metrics']}"
    
    # Individual assertions
    assert evaluation["aggregate_metrics"]["success_rate"] >= 0.95
    assert evaluation["aggregate_metrics"]["intervention_rate"] <= 0.10
    assert evaluation["aggregate_metrics"]["p95_latency_ms"] <= 15000
    assert evaluation["aggregate_metrics"]["avg_cost"] <= 0.15

@pytest.mark.asyncio
async def test_no_critical_regressions(workflow_evaluator, regression_test_set):
    """Ensure no critical functionality is broken"""
    
    # Test cases that must NEVER fail
    critical_test_cases = [
        tc for tc in regression_test_set
        if tc.get("critical", False)
    ]
    
    for test_case in critical_test_cases:
        result = await execute_workflow(
            "customer_support_hil",
            test_case["input"]
        )
        
        evaluation = await workflow_evaluator.evaluate_execution(
            result["execution_id"],
            ground_truth=test_case["expected_output"]
        )
        
        assert evaluation.task_success, \
            f"Critical test case failed: {test_case['name']}"
        
        assert len(evaluation.errors) == 0, \
            f"Errors in critical test case: {evaluation.errors}"
```

### CI/CD Integration

```yaml
# .github/workflows/regression_tests.yml

name: Regression Tests

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main]

jobs:
  regression_tests:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: pgvector/pgvector:latest
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --all-extras
      
      - name: Run regression tests
        run: |
          pytest tests/regression/ \
            --cov=app \
            --cov-report=xml \
            --cov-fail-under=80
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
      
      - name: Quality gate check
        run: |
          python scripts/check_quality_gates.py \
            --success-rate-min 0.95 \
            --intervention-rate-max 0.10 \
            --p95-latency-max 15000 \
            --cost-max 0.15
```

---

## 💰 Production Tuning & Cost Enforcement

### Cost Enforcement Service

```python
# app/services/cost_controller.py

class CostController:
    """Enforce cost budgets and caps"""
    
    def __init__(self, redis_client, db_pool):
        self.redis = redis_client
        self.db = db_pool
    
    async def check_and_reserve_budget(
        self,
        entity_id: str,
        workflow_id: str,
        estimated_cost: float
    ) -> bool:
        """Check if budget available and reserve it"""
        
        # Check entity daily budget
        entity_budget_key = f"budget:entity:{entity_id}:daily"
        entity_remaining = await self.redis.get(entity_budget_key)
        
        if entity_remaining is not None:
            entity_remaining = float(entity_remaining)
            if entity_remaining < estimated_cost:
                logger.warning(
                    "entity_budget_exceeded",
                    entity_id=entity_id,
                    remaining=entity_remaining,
                    requested=estimated_cost
                )
                return False
        
        # Check workflow budget
        workflow_budget_key = f"budget:workflow:{workflow_id}:daily"
        workflow_remaining = await self.redis.get(workflow_budget_key)
        
        if workflow_remaining is not None:
            workflow_remaining = float(workflow_remaining)
            if workflow_remaining < estimated_cost:
                logger.warning(
                    "workflow_budget_exceeded",
                    workflow_id=workflow_id,
                    remaining=workflow_remaining,
                    requested=estimated_cost
                )
                return False
        
        # Reserve budget (decrement)
        if entity_remaining is not None:
            await self.redis.decrbyfloat(entity_budget_key, estimated_cost)
        
        if workflow_remaining is not None:
            await self.redis.decrbyfloat(workflow_budget_key, estimated_cost)
        
        return True
    
    async def refund_budget(
        self,
        entity_id: str,
        workflow_id: str,
        amount: float
    ):
        """Refund unused budget"""
        
        entity_budget_key = f"budget:entity:{entity_id}:daily"
        workflow_budget_key = f"budget:workflow:{workflow_id}:daily"
        
        await self.redis.incrbyfloat(entity_budget_key, amount)
        await self.redis.incrbyfloat(workflow_budget_key, amount)
        
        logger.info(
            "budget_refunded",
            entity_id=entity_id,
            workflow_id=workflow_id,
            amount=amount
        )
    
    async def set_daily_budget(
        self,
        entity_id: str,
        budget: float,
        ttl: int = 86400  # 24 hours
    ):
        """Set daily budget for entity"""
        
        budget_key = f"budget:entity:{entity_id}:daily"
        await self.redis.set(budget_key, budget, ex=ttl)
        
        logger.info(
            "daily_budget_set",
            entity_id=entity_id,
            budget=budget
        )
    
    async def get_usage_stats(
        self,
        entity_id: str,
        time_window: timedelta = timedelta(days=1)
    ) -> Dict[str, Any]:
        """Get cost usage statistics"""
        
        since = datetime.now() - time_window
        
        usage = await self.db.fetchrow("""
            SELECT 
                COUNT(*) as total_executions,
                SUM(total_cost) as total_cost,
                AVG(total_cost) as avg_cost_per_execution,
                MAX(total_cost) as max_cost,
                MIN(total_cost) as min_cost
            FROM executions
            WHERE entity_id = $1
              AND created_at >= $2
        """, entity_id, since)
        
        # Get remaining budget
        budget_key = f"budget:entity:{entity_id}:daily"
        remaining_budget = await self.redis.get(budget_key)
        remaining_budget = float(remaining_budget) if remaining_budget else None
        
        return {
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_executions": usage["total_executions"],
            "total_cost": float(usage["total_cost"] or 0),
            "avg_cost_per_execution": float(usage["avg_cost_per_execution"] or 0),
            "max_cost": float(usage["max_cost"] or 0),
            "min_cost": float(usage["min_cost"] or 0),
            "remaining_daily_budget": remaining_budget
        }
```

### Request Batching

```python
# app/services/batch_processor.py

class BatchProcessor:
    """Batch similar requests for efficiency"""
    
    def __init__(self, llm_router, batch_size: int = 10, batch_timeout: float = 1.0):
        self.llm_router = llm_router
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[Dict] = []
        self.lock = asyncio.Lock()
    
    async def submit_request(self, request: LLMRequest) -> LLMResponse:
        """Submit request to batch processor"""
        
        # Create future for this request
        future = asyncio.Future()
        
        async with self.lock:
            self.pending_requests.append({
                "request": request,
                "future": future
            })
            
            # If batch is full, process immediately
            if len(self.pending_requests) >= self.batch_size:
                await self._process_batch()
        
        # Wait for result (with timeout)
        try:
            result = await asyncio.wait_for(future, timeout=self.batch_timeout * 2)
            return result
        except asyncio.TimeoutError:
            # Fallback to individual processing
            return await self.llm_router.route_request(request)
    
    async def _process_batch(self):
        """Process pending batch"""
        
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:]
        self.pending_requests.clear()
        
        try:
            # Group by model
            model_groups = {}
            for item in batch:
                model = item["request"].model_name
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(item)
            
            # Process each model group
            for model, items in model_groups.items():
                requests = [item["request"] for item in items]
                futures = [item["future"] for item in items]
                
                # Batch API call
                results = await self.llm_router.batch_complete(model, requests)
                
                # Resolve futures
                for future, result in zip(futures, results):
                    future.set_result(result)
        
        except Exception as e:
            # Resolve all futures with error
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(e)
    
    async def start_batch_timer(self):
        """Background task to process batches periodically"""
        
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            async with self.lock:
                if self.pending_requests:
                    await self._process_batch()
```

### Timeout Configuration

```yaml
# config/workflows/customer_support_hil.yaml
workflow:
  # Global workflow timeout
  timeout_ms: 60000  # 1 minute max
  
  nodes:
    - id: classify_intent
      agent: simple_intent_classifier
      limits:
        timeout_ms: 2000  # 2 seconds
        max_retries: 2
        retry_backoff: exponential
        max_cost: 0.01
    
    - id: handle_with_ai
      agent: reasoning_support_agent
      limits:
        timeout_ms: 30000  # 30 seconds
        max_retries: 1
        retry_backoff: exponential
        max_cost: 0.10
        
        # Per-tool timeouts
        tool_timeouts:
          shopify.get_order: 5000
          shopify.create_return: 10000
          gmail.send_email: 8000
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

*This architecture is a living document. Please submit PRs for improvements or open issues for questions.*