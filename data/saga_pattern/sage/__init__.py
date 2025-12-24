# ============================================
# FILE: saga/__init__.py
# ============================================

"""
Sage - Enterprise Saga Pattern Implementation

A production-ready implementation of the Saga pattern for distributed transactions
with support for:
- Declarative saga definitions with @step and @compensate decorators
- Flexible compensation ordering with dependency graphs
- Multiple storage backends (Memory, Redis, PostgreSQL)
- OpenTelemetry distributed tracing
- Prometheus metrics

Quick Start (Declarative API - Recommended):
    >>> from sage import Saga, step, compensate
    >>> 
    >>> class OrderSaga(Saga):
    ...     @step(name="create_order")
    ...     async def create_order(self, ctx):
    ...         return await OrderService.create(ctx)
    ...     
    ...     @compensate("create_order")
    ...     async def cancel_order(self, ctx):
    ...         await OrderService.delete(ctx["order_id"])
    >>> 
    >>> saga = OrderSaga()
    >>> result = await saga.run({"items": [...], "amount": 99.99})

Classic API (Imperative):
    >>> from sage import ClassicSaga
    >>> 
    >>> saga = ClassicSaga(name="OrderSaga")
    >>> saga.add_step("create_order", create_order, compensate=cancel_order)
    >>> result = await saga.execute(context={"order_id": "123"})
"""

# Import the classic imperative Saga as ClassicSaga
from sage.compensation_graph import (
    CircularDependencyError,
    CompensationGraphError,
    CompensationNode,
    CompensationType,
    SagaCompensationGraph,
)
from sage.core import Saga as ClassicSaga
from sage.core import SagaContext, SagaStep

# Import the declarative Saga as the primary Saga class
from sage.decorators import (
    DeclarativeSaga,  # Backward compatibility alias
    Saga,
    SagaStepDefinition,
    compensate,
    step,
)
from sage.exceptions import (
    MissingDependencyError,
    SagaCompensationError,
    SagaError,
    SagaExecutionError,
    SagaStepError,
    SagaTimeoutError,
)
from sage.orchestrator import SagaOrchestrator
from sage.types import ParallelFailureStrategy, SagaResult, SagaStatus, SagaStepStatus

# Backward compatibility aliases
DAGSaga = ClassicSaga

__version__ = "2.1.0"

__all__ = [
    # Declarative API (recommended)
    "Saga",
    "step",
    "compensate",
    "SagaStepDefinition",

    # Classic/Imperative API
    "ClassicSaga",
    "SagaContext",
    "SagaStep",
    "SagaOrchestrator",

    # Compensation graph
    "SagaCompensationGraph",
    "CompensationType",
    "CompensationNode",

    # Types
    "SagaStatus",
    "SagaStepStatus",
    "SagaResult",
    "ParallelFailureStrategy",

    # Exceptions
    "SagaError",
    "SagaStepError",
    "SagaCompensationError",
    "SagaTimeoutError",
    "SagaExecutionError",
    "MissingDependencyError",
    "CompensationGraphError",
    "CircularDependencyError",

    # Backward compatibility
    "DAGSaga",
    "DeclarativeSaga",
]
