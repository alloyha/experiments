"""
Pytest configuration and shared fixtures for saga pattern tests
"""

import pytest
from examples.actions.notification import NotificationService


@pytest.fixture(autouse=True)
def deterministic_notifications():
    """
    Automatically set deterministic behavior for notification failures in tests.
    
    This fixture runs automatically for all tests (autouse=True) and ensures
    that notification services don't randomly fail, making tests deterministic.
    """
    # Set failure rates to 0 for deterministic tests
    NotificationService.set_failure_rates(email=0.0, sms=0.0)
    
    yield
    
    # Reset to defaults after test
    NotificationService.reset_failure_rates()


@pytest.fixture
def enable_notification_failures():
    """
    Fixture to explicitly enable notification failures for specific tests.
    
    Use this when you want to test failure scenarios:
        def test_with_failures(enable_notification_failures):
            # SMS/email may fail randomly
            ...
    """
    NotificationService.reset_failure_rates()
    yield
    # Reset after test
    NotificationService.set_failure_rates(email=0.0, sms=0.0)
