# ============================================
# FILE: saga/exceptions.py
# ============================================

"""
All saga-related exceptions
"""

class SagaError(Exception):
    """Base saga error"""
    pass


class SagaStepError(SagaError):
    """Error executing saga step"""
    pass


class SagaCompensationError(SagaError):
    """Error executing compensation"""
    pass


class SagaTimeoutError(SagaError):
    """Saga step timeout"""
    pass


class SagaExecutionError(SagaError):
    """Error during saga execution"""
    pass


class SagaDependencyError(SagaError):
    """Invalid dependency configuration"""
    pass

