# ============================================
# FILE: saga/__init__.py
# ============================================

"""
Main package exports - clean public API
"""

from sage.core import Saga, SagaContext, SagaStep
from sage.orchestrator import SagaOrchestrator
from sage.types import SagaStatus, SagaStepStatus, SagaResult, ParallelFailureStrategy
from sage.exceptions import (
    SagaError,
    SagaStepError,
    SagaCompensationError,
    SagaTimeoutError,
    SagaExecutionError,
)

# Backward compatibility - DAGSaga is now just Saga
DAGSaga = Saga

__version__ = "2.0.0"

__all__ = [
    # Core classes
    "Saga",
    "SagaContext",
    "SagaStep",
    "SagaOrchestrator",
    
    # DAG support
    "DAGSaga",
    "ParallelFailureStrategy",
    
    # Types
    "SagaStatus",
    "SagaStepStatus",
    "SagaResult",
    
    # Exceptions
    "SagaError",
    "SagaStepError",
    "SagaCompensationError",
    "SagaTimeoutError",
    "SagaExecutionError",
]