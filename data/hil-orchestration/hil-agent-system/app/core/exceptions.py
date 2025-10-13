"""
Core exceptions for the HIL Agent System.
"""


class HILAgentError(Exception):
    """Base exception for HIL Agent System."""


class LLMRoutingError(HILAgentError):
    """Exception raised when LLM routing fails."""


class LLMProviderError(HILAgentError):
    """Exception raised when LLM provider fails."""


class AgentExecutionError(HILAgentError):
    """Exception raised during agent execution."""


class ValidationError(HILAgentError):
    """Exception raised during data validation."""
