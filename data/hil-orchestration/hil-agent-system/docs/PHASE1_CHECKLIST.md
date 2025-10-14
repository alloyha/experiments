# Phase 1 Implementation Checklist
## Anti-Echo & Core LLM (Weeks 1-2) 🚨 HIGH PRIORITY

**Goal**: Build robust foundation preventing repetitive responses and enable actual LLM execution.

---

## Week 1: Anti-Echo & Memory Foundation

### Day 1-2: Conversation Turn Tracking ✅

**Files to create:**
- [ ] `app/models/conversation_turn.py` - ConversationTurn model with idempotency
- [ ] `app/services/turn_manager.py` - Turn tracking service
- [ ] `tests/test_turn_manager.py` - Unit tests

**Implementation:**
```python
# app/models/conversation_turn.py
class ConversationTurn(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id")
    turn_number: int  # Sequential turn tracking
    idempotency_key: str = Field(unique=True, index=True)  # session_id:turn_id
    user_input_hash: str  # SHA-256 for duplicate detection
    agent_response_hash: Optional[str]
    processing_status: str  # PROCESSING, COMPLETED, FAILED
    created_at: datetime = Field(default_factory=datetime.now)
```

**Acceptance Criteria:**
- ✅ Can create turn with session_id:turn_id format
- ✅ Prevents duplicate turn processing (idempotency)
- ✅ Tracks hash of user input and response

---

### Day 3-4: Redis Distributed Locking 🔒

**Files to create:**
- [ ] `app/services/memory_lock.py` - MemoryLockService with Redis
- [ ] `tests/test_memory_lock.py` - Lock behavior tests

**Implementation:**
```python
# app/services/memory_lock.py
class MemoryLockService:
    async def acquire_conversation_lock(
        self,
        session_id: str,
        timeout: int = 30
    ) -> AsyncContextManager:
        """
        Acquire distributed lock using Redis SET NX
        Prevents concurrent memory corruption
        """
        
    async def ensure_idempotent_execution(
        self,
        idempotency_key: str
    ) -> bool:
        """
        Check if operation already processed
        Returns True if safe to proceed
        """
```

**Acceptance Criteria:**
- ✅ Multiple requests with same session_id block correctly
- ✅ Lock auto-releases after timeout
- ✅ Idempotency check prevents duplicate work

---

### Day 5: Response Deduplication 🚫

**Files to create:**
- [ ] `app/services/anti_echo.py` - AntiEchoMemory service
- [ ] `tests/test_anti_echo.py` - Deduplication tests

**Implementation:**
```python
# app/services/anti_echo.py
class AntiEchoMemory:
    async def should_suppress_response(
        self,
        conversation_id: str,
        proposed_response: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if response is too similar to recent responses
        Returns: (should_suppress, reason)
        """
```

**Acceptance Criteria:**
- ✅ Detects identical responses (100% match)
- ✅ Detects highly similar responses (>90% similarity)
- ✅ Stores last 10 responses per conversation

---

## Week 2: LLM Integration & Workflow Execution

### Day 6-7: OpenAI API Integration 🤖

**Files to update:**
- [ ] `app/core/llm_providers.py` - Add actual OpenAI API calls
- [ ] `app/core/llm_router.py` - Wire up OpenAI provider
- [ ] `tests/test_llm_providers.py` - Integration tests

**Implementation:**
```python
# app/core/llm_providers.py
class OpenAIProvider(LLMProvider):
    async def complete(
        self,
        messages: List[dict],
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> LLMResponse:
        """Actually call OpenAI API"""
        client = AsyncOpenAI(api_key=self.api_key)
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            usage=response.usage.model_dump(),
            finish_reason=response.choices[0].finish_reason
        )
```

**Acceptance Criteria:**
- ✅ Can call OpenAI API successfully
- ✅ Handles rate limits gracefully
- ✅ Tracks token usage and cost
- ✅ Supports streaming responses

---

### Day 8-9: Sink Node Support (FINISH/HANDOVER) 🎯

**Files to update:**
- [ ] `app/services/workflow_integration.py` - Add sink node handling
- [ ] `app/models/workflow.py` - Add SinkNodeType enum
- [ ] `tests/test_sink_nodes.py` - Sink routing tests

**Implementation:**
```python
# app/services/workflow_integration.py
async def execute_sink_node(
    self,
    node: WorkflowNode,
    execution_context: dict
) -> ExecutionResult:
    """
    Handle FINISH or HANDOVER sink nodes
    """
    if node.node_type == SinkNodeType.FINISH:
        return ExecutionResult(
            status="completed",
            outcome="success",
            final_response=execution_context.get("last_response")
        )
    
    elif node.node_type == SinkNodeType.HANDOVER:
        # Extract handover context
        handover_context = self._extract_handover_context(execution_context)
        
        # Create handover event
        await self.hil_service.create_handover(
            conversation_id=execution_context["conversation_id"],
            reason=node.config.get("reason", "agent_decision"),
            context=handover_context
        )
        
        return ExecutionResult(
            status="handover_initiated",
            outcome="escalated",
            handover_id=handover_context["handover_id"]
        )
```

**Acceptance Criteria:**
- ✅ FINISH sink completes workflow successfully
- ✅ HANDOVER sink creates handover event
- ✅ Context properly extracted and passed

---

### Day 10: Cost Tracking & Metrics 💰

**Files to update:**
- [ ] `app/services/cost_tracker.py` - Create cost tracking service
- [ ] `app/models/execution.py` - Add cost fields
- [ ] Update all LLM calls to track costs

**Implementation:**
```python
# app/services/cost_tracker.py
class CostTracker:
    # Pricing per 1K tokens (as of Oct 2025)
    PRICING = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for LLM call"""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
```

**Acceptance Criteria:**
- ✅ All LLM calls tracked with costs
- ✅ Per-execution cost aggregation
- ✅ Cost metrics in Prometheus

---

## Testing & Integration

### End of Week 2: Integration Testing

**Create:**
- [ ] `tests/integration/test_phase1_flow.py` - Full flow test

**Test Scenario:**
```python
async def test_complete_conversation_flow():
    """Test anti-echo + LLM + sink nodes working together"""
    
    # 1. Send message
    response1 = await client.post("/api/v1/message", json={
        "session_id": "test-session",
        "message": "Hello, I need help"
    })
    assert response1.status_code == 200
    
    # 2. Send duplicate message - should be deduplicated
    response2 = await client.post("/api/v1/message", json={
        "session_id": "test-session",
        "message": "Hello, I need help"
    })
    assert "duplicate detected" in response2.json()["message"]
    
    # 3. Continue conversation
    response3 = await client.post("/api/v1/message", json={
        "session_id": "test-session",
        "message": "I want to return an item"
    })
    
    # 4. Verify cost tracked
    execution = await get_execution(response3.json()["execution_id"])
    assert execution.total_cost > 0
```

---

## Success Criteria (Phase 1 Complete)

### Must Have ✅
- [ ] No repetitive responses (anti-echo working)
- [ ] LLM API calls succeed with OpenAI
- [ ] Sink nodes route correctly (FINISH/HANDOVER)
- [ ] Cost tracking accurate
- [ ] Distributed locking prevents race conditions
- [ ] Idempotency prevents duplicate processing

### Performance Targets
- [ ] Response time < 2s P95
- [ ] Lock acquisition < 50ms
- [ ] Deduplication check < 10ms
- [ ] Zero duplicate responses in tests

### Code Quality
- [ ] 80%+ test coverage
- [ ] All tests passing
- [ ] No linting errors
- [ ] Documentation complete

---

## Phase 1 Deliverables

**New Services:**
1. ✅ TurnManager - Conversation turn tracking
2. ✅ MemoryLockService - Distributed locking
3. ✅ AntiEchoMemory - Response deduplication
4. ✅ CostTracker - LLM cost calculation

**Enhanced Services:**
1. ✅ OpenAIProvider - Actual API integration
2. ✅ WorkflowIntegration - Sink node support

**New Models:**
1. ✅ ConversationTurn - Turn tracking with idempotency

**Tests:**
1. ✅ Unit tests for all new services
2. ✅ Integration test for full flow

---

## Ready to Start?

**Next Steps:**
1. Create `app/models/conversation_turn.py`
2. Create `app/services/turn_manager.py`
3. Run tests to verify foundation

Let me know when you're ready and I'll help you implement each component! 🚀
