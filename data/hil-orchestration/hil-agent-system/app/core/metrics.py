"""
Prometheus metrics configuration.
"""

from typing import Any, Dict, Optional

import structlog
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Info,
    generate_latest,
)

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# Metrics registry
METRICS: dict[str, any] = {}


def setup_metrics(app: FastAPI) -> None:
    """Setup Prometheus metrics."""
    settings = get_settings()

    if not settings.METRICS_ENABLED:
        logger.info("Metrics disabled, skipping setup")
        return

    logger.info("Setting up Prometheus metrics")

    # Application info
    METRICS["app_info"] = Info(
        "hil_agent_system_info",
        "Application information",
    )
    METRICS["app_info"].info(
        {
            "version": "0.1.0",
            "environment": settings.ENVIRONMENT,
        }
    )

    # HTTP metrics
    METRICS["http_requests_total"] = Counter(
        "hil_agent_system_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    METRICS["http_request_duration_seconds"] = Histogram(
        "hil_agent_system_http_request_duration_seconds",
        "HTTP request duration",
        ["method", "endpoint"],
    )

    # Agent metrics
    METRICS["agent_executions_total"] = Counter(
        "hil_agent_system_agent_executions_total",
        "Total agent executions",
        ["agent_type", "status"],
    )

    METRICS["agent_execution_duration_seconds"] = Histogram(
        "hil_agent_system_agent_execution_duration_seconds",
        "Agent execution duration",
        ["agent_type"],
    )

    # Tool metrics
    METRICS["tool_executions_total"] = Counter(
        "hil_agent_system_tool_executions_total",
        "Total tool executions",
        ["tool_name", "status"],
    )

    METRICS["tool_execution_duration_seconds"] = Histogram(
        "hil_agent_system_tool_execution_duration_seconds",
        "Tool execution duration",
        ["tool_name"],
    )

    # LLM metrics
    METRICS["llm_requests_total"] = Counter(
        "hil_agent_system_llm_requests_total",
        "Total LLM requests",
        ["provider", "model", "status"],
    )

    METRICS["llm_tokens_total"] = Counter(
        "hil_agent_system_llm_tokens_total",
        "Total LLM tokens used",
        ["provider", "model", "type"],  # type: input/output
    )

    METRICS["llm_cost_total"] = Counter(
        "hil_agent_system_llm_cost_total",
        "Total LLM cost in USD",
        ["provider", "model"],
    )

    # Workflow metrics
    METRICS["workflow_executions_total"] = Counter(
        "hil_agent_system_workflow_executions_total",
        "Total workflow executions",
        ["workflow_id", "status"],
    )

    METRICS["workflow_execution_duration_seconds"] = Histogram(
        "hil_agent_system_workflow_execution_duration_seconds",
        "Workflow execution duration",
        ["workflow_id"],
    )

    # Memory metrics
    METRICS["memory_retrievals_total"] = Counter(
        "hil_agent_system_memory_retrievals_total",
        "Total memory retrievals",
        ["memory_type", "status"],
    )

    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    logger.info("Prometheus metrics setup complete")


def get_metric(name: str) -> Any | None:
    """Get a metric by name."""
    return METRICS.get(name)


def increment_counter(metric_name: str, labels: dict[str, str] | None = None) -> None:
    """Increment a counter metric."""
    metric = get_metric(metric_name)
    if metric and hasattr(metric, "labels"):
        if labels:
            metric.labels(**labels).inc()
        else:
            metric.inc()


def observe_histogram(
    metric_name: str, value: float, labels: dict[str, str] | None = None
) -> None:
    """Observe a histogram metric."""
    metric = get_metric(metric_name)
    if metric and hasattr(metric, "labels"):
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
