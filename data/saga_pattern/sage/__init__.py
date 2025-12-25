# ============================================
# FILE: saga/__init__.py
# ============================================

"""
Sage - Enterprise Saga Pattern Implementation

A production-ready implementation of the Saga pattern for distributed transactions
with support for:
- Declarative saga definitions with @action and @compensate decorators
- Flexible compensation ordering with dependency graphs
- Multiple storage backends (Memory, Redis, PostgreSQL)
- OpenTelemetry distributed tracing
- Prometheus metrics
- Saga lifecycle listeners for cross-cutting concerns

Quick Start (Declarative API - Recommended):
    >>> from sage import Saga, action, compensate
    >>> 
    >>> class OrderSaga(Saga):
    ...     saga_name = "order-processing"
    ...     
    ...     @action("create_order")
    ...     async def create_order(self, ctx):
    ...         return await OrderService.create(ctx)
    ...     
    ...     @compensate("create_order")
    ...     async def cancel_order(self, ctx):
    ...         await OrderService.delete(ctx["order_id"])
    >>> 
    >>> saga = OrderSaga()
    >>> result = await saga.run({"items": [...], "amount": 99.99})

With Listeners (Metrics, Logging, Outbox):
    >>> from sage.listeners import MetricsSagaListener, OutboxSagaListener
    >>> 
    >>> class OrderSaga(Saga):
    ...     saga_name = "order-processing"
    ...     listeners = [MetricsSagaListener(), OutboxSagaListener(storage)]

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
    action,  # Preferred terminology
    compensate,
    step,  # Alias for backward compatibility
)
from sage.exceptions import (
    MissingDependencyError,
    SagaCompensationError,
    SagaError,
    SagaExecutionError,
    SagaStepError,
    SagaTimeoutError,
)

# Import listeners
from sage.listeners import (
    LoggingSagaListener,
    MetricsSagaListener,
    OutboxSagaListener,
    SagaListener,
    TracingSagaListener,
    default_listeners,
)

from sage.orchestrator import SagaOrchestrator
from sage.types import ParallelFailureStrategy, SagaResult, SagaStatus, SagaStepStatus

# Backward compatibility aliases
DAGSaga = ClassicSaga

__version__ = "2.2.0"

__all__ = [
    # Declarative API (recommended)
    "Saga",
    "action",  # Preferred
    "step",    # Alias for backward compat
    "compensate",
    "SagaStepDefinition",

    # Listeners (cross-cutting concerns)
    "SagaListener",
    "LoggingSagaListener",
    "MetricsSagaListener",
    "TracingSagaListener",
    "OutboxSagaListener",
    "default_listeners",

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
