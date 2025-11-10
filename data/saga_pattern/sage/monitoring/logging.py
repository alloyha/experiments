"""
Structured logging for saga execution

Provides structured logging utilities specifically designed for saga pattern tracing,
with proper context propagation and correlation IDs for distributed systems.
"""

import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
from contextvars import ContextVar
from sage.types import SagaStatus, SagaStepStatus

# Context variables for propagating saga context
saga_context: ContextVar[Dict[str, Any]] = ContextVar("saga_context", default={})


class SagaJsonFormatter(logging.Formatter):
    """
    JSON formatter for saga logs with structured fields
    
    Ensures all saga-related logs include correlation IDs and context
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Get saga context if available
        context = saga_context.get({})
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add saga context if available
        if context:
            log_entry.update({
                "saga_id": context.get("saga_id"),
                "saga_name": context.get("saga_name"),
                "step_name": context.get("step_name"),
                "correlation_id": context.get("correlation_id"),
            })
        
        # Add any extra fields from the log record
        if hasattr(record, "saga_id"):
            log_entry["saga_id"] = record.saga_id
        if hasattr(record, "saga_name"):
            log_entry["saga_name"] = record.saga_name
        if hasattr(record, "step_name"):
            log_entry["step_name"] = record.step_name
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "retry_count"):
            log_entry["retry_count"] = record.retry_count
        if hasattr(record, "error_type"):
            log_entry["error_type"] = record.error_type
            
        return json.dumps(log_entry)


class SagaContextFilter(logging.Filter):
    """
    Logging filter that adds saga context to log records
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add saga context to log record"""
        
        context = saga_context.get({})
        
        # Add saga context fields to the record
        record.saga_id = context.get("saga_id", "unknown")
        record.saga_name = context.get("saga_name", "unknown")
        record.step_name = context.get("step_name", "")
        record.correlation_id = context.get("correlation_id", "")
        
        return True


class SagaLogger:
    """
    Saga-aware logger with automatic context propagation
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
        # Add saga context filter
        saga_filter = SagaContextFilter()
        self.logger.addFilter(saga_filter)
    
    def set_saga_context(
        self, 
        saga_id: str, 
        saga_name: str, 
        step_name: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Set saga context for current execution"""
        
        context = {
            "saga_id": saga_id,
            "saga_name": saga_name,
            "step_name": step_name,
            "correlation_id": correlation_id or saga_id,
        }
        saga_context.set(context)
    
    def clear_saga_context(self) -> None:
        """Clear saga context"""
        saga_context.set({})
    
    def saga_started(
        self, 
        saga_id: str, 
        saga_name: str, 
        total_steps: int,
        correlation_id: Optional[str] = None
    ) -> None:
        """Log saga start"""
        
        self.set_saga_context(saga_id, saga_name, correlation_id=correlation_id)
        self.logger.info(
            f"Saga started: {saga_name}",
            extra={
                "saga_id": saga_id,
                "saga_name": saga_name,
                "total_steps": total_steps,
                "correlation_id": correlation_id or saga_id,
            }
        )
    
    def saga_completed(
        self, 
        saga_id: str, 
        saga_name: str, 
        status: SagaStatus,
        duration_ms: float,
        completed_steps: int,
        total_steps: int
    ) -> None:
        """Log saga completion"""
        
        log_level = logging.INFO if status == SagaStatus.COMPLETED else logging.WARNING
        
        self.logger.log(
            log_level,
            f"Saga finished: {saga_name} - Status: {status.value}",
            extra={
                "saga_id": saga_id, 
                "saga_name": saga_name,
                "status": status.value,
                "duration_ms": duration_ms,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
            }
        )
    
    def step_started(
        self, 
        saga_id: str, 
        saga_name: str, 
        step_name: str
    ) -> None:
        """Log step start"""
        
        self.set_saga_context(saga_id, saga_name, step_name)
        self.logger.info(
            f"Step started: {step_name}",
            extra={
                "saga_id": saga_id,
                "saga_name": saga_name, 
                "step_name": step_name,
            }
        )
    
    def step_completed(
        self,
        saga_id: str,
        saga_name: str, 
        step_name: str,
        duration_ms: float,
        retry_count: int = 0
    ) -> None:
        """Log step completion"""
        
        self.logger.info(
            f"Step completed: {step_name}",
            extra={
                "saga_id": saga_id,
                "saga_name": saga_name,
                "step_name": step_name,
                "duration_ms": duration_ms,
                "retry_count": retry_count,
            }
        )
    
    def step_failed(
        self,
        saga_id: str,
        saga_name: str,
        step_name: str, 
        error: Exception,
        retry_count: int = 0
    ) -> None:
        """Log step failure"""
        
        self.logger.error(
            f"Step failed: {step_name} - {str(error)}",
            extra={
                "saga_id": saga_id,
                "saga_name": saga_name,
                "step_name": step_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "retry_count": retry_count,
            },
            exc_info=True
        )
    
    def compensation_started(
        self,
        saga_id: str,
        saga_name: str,
        step_name: str
    ) -> None:
        """Log compensation start"""
        
        self.logger.warning(
            f"Compensation started: {step_name}",
            extra={
                "saga_id": saga_id,
                "saga_name": saga_name,
                "step_name": step_name,
            }
        )
    
    def compensation_completed(
        self,
        saga_id: str,
        saga_name: str,
        step_name: str,
        duration_ms: float
    ) -> None:
        """Log compensation completion"""
        
        self.logger.warning(
            f"Compensation completed: {step_name}",
            extra={
                "saga_id": saga_id,
                "saga_name": saga_name,
                "step_name": step_name,
                "duration_ms": duration_ms,
            }
        )
    
    def compensation_failed(
        self,
        saga_id: str,
        saga_name: str,
        step_name: str,
        error: Exception
    ) -> None:
        """Log compensation failure - critical error"""
        
        self.logger.critical(
            f"Compensation FAILED: {step_name} - {str(error)}",
            extra={
                "saga_id": saga_id,
                "saga_name": saga_name,
                "step_name": step_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            exc_info=True
        )


def setup_saga_logging(
    log_level: str = "INFO",
    json_format: bool = True,
    include_console: bool = True
) -> SagaLogger:
    """
    Set up structured logging for sagas
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting for structured logs
        include_console: Include console handler
        
    Returns:
        Configured SagaLogger instance
    """
    
    # Configure root logger
    root_logger = logging.getLogger("saga")
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if include_console:
        console_handler = logging.StreamHandler()
        
        if json_format:
            console_handler.setFormatter(SagaJsonFormatter())
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(saga_id)s:%(step_name)s] - %(message)s"
            )
            console_handler.setFormatter(formatter)
        
        root_logger.addHandler(console_handler)
    
    return SagaLogger("saga")


# Default saga logger instance
saga_logger = SagaLogger("saga")