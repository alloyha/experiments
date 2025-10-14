"""
Tests for Prometheus metrics functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Info

from app.core.metrics import (
    setup_metrics,
    get_metric,
    increment_counter,
    observe_histogram,
    METRICS,
)


@pytest.fixture
def app():
    """Create a FastAPI app for testing."""
    return FastAPI()


@pytest.fixture(autouse=True)
def clear_metrics():
    """Clear metrics before and after each test."""
    from prometheus_client import REGISTRY
    METRICS.clear()
    # Clear Prometheus collectors to avoid duplicate registration
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    METRICS.clear()
    # Clear again after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def test_setup_metrics_disabled(app):
    """Test that metrics setup is skipped when disabled."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = False
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # No metrics should be registered
        assert len(METRICS) == 0


def test_setup_metrics_enabled(app):
    """Test that metrics are properly initialized when enabled."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = True
        settings_mock.ENVIRONMENT = "test"
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # Verify all metrics are registered
        assert "app_info" in METRICS
        assert "http_requests_total" in METRICS
        assert "http_request_duration_seconds" in METRICS
        assert "agent_executions_total" in METRICS
        assert "agent_execution_duration_seconds" in METRICS
        assert "tool_executions_total" in METRICS
        assert "tool_execution_duration_seconds" in METRICS
        assert "llm_requests_total" in METRICS
        assert "llm_tokens_total" in METRICS
        assert "llm_cost_total" in METRICS
        assert "workflow_executions_total" in METRICS
        assert "workflow_execution_duration_seconds" in METRICS
        assert "memory_retrievals_total" in METRICS
        
        # Verify metric types
        assert isinstance(METRICS["app_info"], Info)
        assert isinstance(METRICS["http_requests_total"], Counter)
        assert isinstance(METRICS["http_request_duration_seconds"], Histogram)


def test_setup_metrics_creates_endpoint(app):
    """Test that /metrics endpoint is created."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = True
        settings_mock.ENVIRONMENT = "test"
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # Check that metrics endpoint exists
        routes = [route.path for route in app.routes]
        assert "/metrics" in routes


def test_get_metric_existing():
    """Test getting an existing metric."""
    mock_metric = Mock()
    METRICS["test_metric"] = mock_metric
    
    result = get_metric("test_metric")
    
    assert result == mock_metric


def test_get_metric_non_existing():
    """Test getting a non-existing metric returns None."""
    result = get_metric("non_existing_metric")
    
    assert result is None


def test_increment_counter_with_labels():
    """Test incrementing a counter with labels."""
    mock_counter = MagicMock()
    mock_labeled = MagicMock()
    mock_counter.labels.return_value = mock_labeled
    METRICS["test_counter"] = mock_counter
    
    increment_counter("test_counter", {"status": "success"})
    
    mock_counter.labels.assert_called_once_with(status="success")
    mock_labeled.inc.assert_called_once()


def test_increment_counter_without_labels():
    """Test incrementing a counter without labels."""
    mock_counter = MagicMock()
    METRICS["test_counter"] = mock_counter
    
    increment_counter("test_counter")
    
    mock_counter.inc.assert_called_once()


def test_increment_counter_non_existing():
    """Test incrementing a non-existing counter does nothing."""
    # Should not raise an error
    increment_counter("non_existing_counter", {"status": "fail"})


def test_increment_counter_without_labels_method():
    """Test incrementing a metric that doesn't have labels method."""
    mock_metric = Mock(spec=[])  # No labels attribute
    METRICS["bad_metric"] = mock_metric
    
    # Should not raise an error
    increment_counter("bad_metric", {"status": "test"})


def test_observe_histogram_with_labels():
    """Test observing a histogram with labels."""
    mock_histogram = MagicMock()
    mock_labeled = MagicMock()
    mock_histogram.labels.return_value = mock_labeled
    METRICS["test_histogram"] = mock_histogram
    
    observe_histogram("test_histogram", 0.5, {"method": "GET"})
    
    mock_histogram.labels.assert_called_once_with(method="GET")
    mock_labeled.observe.assert_called_once_with(0.5)


def test_observe_histogram_without_labels():
    """Test observing a histogram without labels."""
    mock_histogram = MagicMock()
    METRICS["test_histogram"] = mock_histogram
    
    observe_histogram("test_histogram", 1.23)
    
    mock_histogram.observe.assert_called_once_with(1.23)


def test_observe_histogram_non_existing():
    """Test observing a non-existing histogram does nothing."""
    # Should not raise an error
    observe_histogram("non_existing_histogram", 0.75, {"endpoint": "/api"})


def test_observe_histogram_without_labels_method():
    """Test observing a metric that doesn't have labels method."""
    mock_metric = Mock(spec=[])  # No labels attribute
    METRICS["bad_metric"] = mock_metric
    
    # Should not raise an error
    observe_histogram("bad_metric", 1.0, {"test": "value"})


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_format(app):
    """Test that metrics endpoint returns Prometheus format data."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = True
        settings_mock.ENVIRONMENT = "test"
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # Find the metrics endpoint
        metrics_route = None
        for route in app.routes:
            if route.path == "/metrics":
                metrics_route = route
                break
        
        assert metrics_route is not None
        
        # Call the endpoint function
        response = await metrics_route.endpoint()
        
        # Verify response (prometheus version may vary)
        assert "text/plain" in response.media_type
        assert response.body is not None


def test_http_metrics_registered(app):
    """Test that HTTP metrics are properly configured."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = True
        settings_mock.ENVIRONMENT = "production"
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # Verify HTTP metrics exist and are the right type
        http_requests = METRICS["http_requests_total"]
        assert isinstance(http_requests, Counter)
        # Counter names may have _total stripped by prometheus_client
        assert "http_requests" in http_requests._name


def test_agent_metrics_registered(app):
    """Test that agent metrics are properly configured."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = True
        settings_mock.ENVIRONMENT = "test"
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # Verify agent metrics
        agent_executions = METRICS["agent_executions_total"]
        agent_duration = METRICS["agent_execution_duration_seconds"]
        
        assert isinstance(agent_executions, Counter)
        assert isinstance(agent_duration, Histogram)


def test_llm_metrics_registered(app):
    """Test that LLM metrics are properly configured."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = True
        settings_mock.ENVIRONMENT = "test"
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # Verify LLM metrics
        assert "llm_requests_total" in METRICS
        assert "llm_tokens_total" in METRICS
        assert "llm_cost_total" in METRICS
        
        assert isinstance(METRICS["llm_requests_total"], Counter)
        assert isinstance(METRICS["llm_tokens_total"], Counter)
        assert isinstance(METRICS["llm_cost_total"], Counter)


def test_workflow_metrics_registered(app):
    """Test that workflow metrics are properly configured."""
    with patch("app.core.metrics.get_settings") as mock_settings:
        settings_mock = Mock()
        settings_mock.METRICS_ENABLED = True
        settings_mock.ENVIRONMENT = "test"
        mock_settings.return_value = settings_mock
        
        setup_metrics(app)
        
        # Verify workflow metrics
        assert "workflow_executions_total" in METRICS
        assert "workflow_execution_duration_seconds" in METRICS
        
        assert isinstance(METRICS["workflow_executions_total"], Counter)
        assert isinstance(METRICS["workflow_execution_duration_seconds"], Histogram)
