# Coverage Gaps Analysis - Path to 100%

## Executive Summary

**Current Coverage:** 91% (351 tests)  
**Gap to 100%:** 9 percentage points  
**Total Missing Items:** 36 coverage gaps

## Gap Classification

### 🚫 Not Worth Testing (2 items - 5.6%)
**Lines:** 4 total  
**Impact:** Skip these - would require uninstalling dependencies

- `sage/storage/redis.py:26-28` - Import fallback for missing Redis
- `sage/storage/postgresql.py:26-28` - Import fallback for missing asyncpg

**Recommendation:** Exclude from coverage requirements (pragma: no cover)

---

### 🐛 Error Paths & Exception Handling (10 items - 27.8%)
**Lines:** ~30 total  
**Impact:** HIGH - These account for most of the gap to 95%

**PostgreSQL Storage:**
- Line 79: Connection pool creation failure
- Lines 194-195, 199, 201: Database operation errors
- Line 322: Update step state error
- Lines 387, 403-404, 417: Cleanup operation errors

**Redis Storage:**
- Line 47: Connection ping failure
- Lines 139-140, 150: Delete operation errors
- Lines 189-196: List sagas error handling
- Lines 245, 255, 260: Update step state errors
- Lines 328-330, 347, 361-362: Health check errors

**Logging:**
- Lines 55, 57, 59, 319-323: Log formatting errors

**Test Strategy:**
```python
# Mock connection failures
async def test_redis_connection_failure():
    storage = RedisSagaStorage(redis_url="redis://invalid:9999")
    with pytest.raises(SagaStorageConnectionError):
        async with storage:
            await storage.save_saga_state(...)

# Mock operation failures  
async def test_redis_delete_error(mocker):
    mocker.patch('redis.asyncio.Redis.delete', side_effect=RedisError)
    # Test error handling
```

---

### 🔀 Branch Coverage (7 items - 19.4%)
**Lines:** ~15 total  
**Impact:** MEDIUM - Easy wins for coverage percentage

- `sage/core.py:103-104` - Test step without compensation
- `sage/core.py:284->282` - DAG batch building edge case
- `sage/monitoring/metrics.py:34->38` - Metrics branch
- `sage/storage/base.py:200->203` - Error branch
- `sage/storage/memory.py` - Multiple conditional branches
- `sage/strategies/*.py` - Exception handling branches

**Test Strategy:**
```python
# Add step without compensation
await saga.add_step("no_comp_step", action)  # No compensation param

# Test empty dependencies
saga = DAGSaga()
saga._build_execution_batches()  # Test empty case
```

---

### 📝 Helper Methods & Properties (9 items - 25%)
**Lines:** ~12 total  
**Impact:** MEDIUM - Simple to test

- `sage/core.py:108` - `.name` property
- `sage/core.py:254-255` - `set_failure_strategy()` method
- `sage/core.py:791` - Result property
- `sage/types.py:57, 61` - `is_completed`, `is_rolled_back` properties
- `sage/orchestrator.py:64` - Orchestrator method
- `sage/state_machine.py:23` - State machine init
- `sage/strategies/wait_all.py:37` - Exception re-raise

**Test Strategy:**
```python
# Test properties
assert saga.name == "TestSaga"
result = SagaResult(success=True, ...)
assert result.is_completed == True

# Test helper methods
saga.set_failure_strategy(ParallelFailureStrategy.WAIT_ALL)
```

---

### 🧪 Advanced Testing Scenarios (6 items - 16.7%)
**Lines:** ~25 total  
**Impact:** LOW-MEDIUM - Complex scenarios

- `sage/monitoring/tracing.py:166-181` - Trace context extraction
- `sage/monitoring/tracing.py:198-212` - Span attributes
- Redis/PostgreSQL complex branching paths
- Async context manager cleanup
- Timeout exception paths

---

### 🔌 Optional Dependencies (2 items - 5.6%)
**Lines:** ~25 total  
**Impact:** LOW - External dependency

- `sage/monitoring/tracing.py:21-26` - OTLP imports
- `sage/monitoring/tracing.py:351-372` - OTLP exporter config

**Recommendation:** Skip or mark as optional (requires opentelemetry-exporter-otlp)

---

## Priority Roadmap to >95%

### Phase 1: Storage Backend Error Paths (Highest Impact)
**Target:** 91% → 94% (+3%)  
**Effort:** Medium (2-3 hours)

Add 10-15 tests for:
- Connection failure scenarios
- Operation error handling
- Cleanup error paths
- Health check failures

### Phase 2: Helper Methods & Properties (Quick Wins)
**Target:** 94% → 95% (+1%)  
**Effort:** Low (30 minutes)

Add 5-10 simple tests for:
- Property getters
- Helper method calls
- Result properties

### Phase 3: Branch Coverage (Fill Remaining Gaps)
**Target:** 95% → 96% (+1%)  
**Effort:** Low (1 hour)

Add 5-8 tests for:
- Steps without compensation
- Edge cases in batch building
- Conditional branches

### Phase 4: Advanced Scenarios (Optional)
**Target:** 96% → 98%+ (+2%)  
**Effort:** High (3-4 hours)

Add complex integration tests for:
- Trace context propagation
- Complex async cleanup scenarios
- Advanced error recovery

---

## Expected Outcomes

### Conservative Goal (Phase 1-2)
- **Coverage:** 91% → 95%
- **New Tests:** +15-20
- **Time Investment:** 3-4 hours

### Aggressive Goal (Phase 1-3)
- **Coverage:** 91% → 96%
- **New Tests:** +25-30
- **Time Investment:** 4-5 hours

### Maximum Goal (Phase 1-4)
- **Coverage:** 91% → 98%
- **New Tests:** +35-45
- **Time Investment:** 7-8 hours

---

## Summary Statistics

| Category | Items | Lines | Priority | Testable |
|----------|-------|-------|----------|----------|
| Import Errors | 2 | 4 | Skip | ❌ No |
| Error Paths | 10 | ~30 | HIGH | ✅ Yes |
| Branches | 7 | ~15 | MEDIUM | ✅ Yes |
| Helpers | 9 | ~12 | MEDIUM | ✅ Yes |
| Advanced | 6 | ~25 | LOW | ✅ Yes |
| Optional Deps | 2 | ~25 | SKIP | ❌ No |
| **TOTAL** | **36** | **~111** | - | **32/36** |

**Realistic Target:** 95-96% coverage by testing 32 of 36 gaps
