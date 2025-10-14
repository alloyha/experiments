"""
Structured logging configuration.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.typing import EventDict

from app.core.config import get_settings


def add_correlation_id(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add correlation ID to log entries."""
    # This will be enhanced later with request context
    return event_dict


def add_service_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service information to log entries."""
    event_dict["service"] = "hil-agent-system"
    event_dict["version"] = "0.1.0"
    return event_dict


def setup_logging() -> None:
    """Setup structured logging."""
    settings = get_settings()

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_service_info,
        add_correlation_id,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        # Removed format_exc_info to allow pretty exception rendering
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_production:
        # JSON formatting for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty formatting for development
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
